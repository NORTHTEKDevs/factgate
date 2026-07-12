"""Near-miss contradiction generator.

Takes a functional-relation triple (s1, r, o1) and corrupts the
object with a *sibling* value -- the object of a different subject
under the same relation (o2 != o1) -- so the false premise is
plausible (same relation, same "shape" of answer) rather than
nonsensical. Renders the corrupted fact as a user-stated premise; the
target is a correction that queries the live KB and cites its actual
answer, so training data always corrects TOWARD what the tool
currently says, not toward the generator's ground truth.

Restricted to relations that are both functional
(`rck.contradiction.FUNCTIONAL_RELATIONS`) and have statement
templates (`templates.TOP_RELATIONS`) -- currently {isa, locatedin}.
"""
from __future__ import annotations

import random
from typing import Iterator

from rck.contradiction import FUNCTIONAL_RELATIONS
from rck.knowledge_base import ShardedKnowledgeBase

from factgate.datagen.protocol import kb_cite, make_example
from factgate.datagen.templates import STATEMENT_TEMPLATES, TOP_RELATIONS

CORRUPT_RELATIONS = tuple(sorted(FUNCTIONAL_RELATIONS & set(TOP_RELATIONS)))
CONFIDENT_THRESHOLD = 0.15


def _build_functional_map(triples: list[dict], relations: tuple[str, ...]
                           ) -> dict[str, dict[str, str]]:
    """relation -> {subject: object}, first occurrence wins (source data
    is deduplicated per (s, r) in practice; ties are not adjudicated)."""
    out: dict[str, dict[str, str]] = {r: {} for r in relations}
    for t in triples:
        r = t["r"]
        if r not in out:
            continue
        out[r].setdefault(t["s"], t["o"])
    return out


def generate(kb: ShardedKnowledgeBase, triples: list[dict],
             *, n: int, seed: int = 0, top_k: int = 5,
             max_attempts_factor: int = 10) -> Iterator[dict]:
    rng = random.Random(seed)
    fmap = _build_functional_map(triples, CORRUPT_RELATIONS)
    # Only relations with >=2 distinct subjects can produce a sibling swap.
    usable = {r: subj_map for r, subj_map in fmap.items() if len(subj_map) >= 2}
    if not usable:
        return
    relations = list(usable.keys())
    emitted = 0
    attempts = 0
    max_attempts = max(n * max_attempts_factor, 50)
    idx = 0
    while emitted < n and attempts < max_attempts:
        attempts += 1
        r = relations[rng.randrange(len(relations))]
        subj_map = usable[r]
        subjects = list(subj_map.keys())
        s1 = subjects[rng.randrange(len(subjects))]
        o1 = subj_map[s1]
        s2 = subjects[rng.randrange(len(subjects))]
        o2 = subj_map[s2]
        if o2 == o1:
            continue  # need a genuinely different sibling value

        tid = rng.randrange(len(STATEMENT_TEMPLATES[r]))
        false_premise = STATEMENT_TEMPLATES[r][tid].format(s=s1, o=o2)

        candidates = kb.query({"S": s1, "R": r}, "O", top_k=top_k)
        kb_q = {"s": s1, "r": r, "unknown": "O", "top_k": top_k}
        kb_r = {"results": [[str(sym), round(float(sc), 4)] for sym, sc in candidates]}
        if not candidates or float(candidates[0][1]) < CONFIDENT_THRESHOLD:
            continue  # need a confident correction

        kb_answer = str(candidates[0][0])
        if kb_answer == o2:
            continue  # KB agrees with the corrupted premise -- not a contradiction

        question = f"I read that: {false_premise} Is that correct?"
        final_answer = (
            f"No, that's not correct. According to the knowledge base, "
            f"{STATEMENT_TEMPLATES[r][tid].format(s=s1, o=kb_answer)} "
            f"(confidence {float(candidates[0][1]):.2f}), not {o2!r}. "
            f"{kb_cite(s1, r, kb_answer)}"
        )

        idx += 1
        yield make_example(
            example_id=f"corrupt-{seed}-{idx}",
            generator="corrupt",
            question=question,
            kb_q=kb_q,
            kb_r=kb_r,
            final_answer=final_answer,
            meta={"relation": r, "subject": s1, "corrupted_object": o2,
                  "sibling_subject": s2, "kb_answer": kb_answer,
                  "true_object_from_source": o1,
                  "top_score": float(candidates[0][1])},
        )
        emitted += 1

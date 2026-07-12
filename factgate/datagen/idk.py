"""Calibrated-refusal ("I don't know") generator, two modes:

* `generate_absent` -- sample random (S, R) pairs from the entity /
  relation vocabulary and keep only the ones the KB genuinely doesn't
  know (state == IDK per `rck.idk_detection`).
* `generate_holdout` -- deletion-holdout mode. Given a list of triples
  that were reserved BEFORE the KB was built (see
  scripts/holdout_split.py), ask about them; since the fact was never
  stored the KB should answer IDK. We still verify with
  `ask_with_idk` rather than assuming, since crosstalk / synonymy
  could accidentally resurrect the answer from unrelated facts.
"""
from __future__ import annotations

import random
from typing import Iterable, Iterator

from rck.idk_detection import EpistemicState, IDKPolicy, ask_with_idk
from rck.knowledge_base import ShardedKnowledgeBase

from factgate.datagen.protocol import KB_IDK_CITE, make_example
from factgate.datagen.templates import QUESTION_TEMPLATES


def _forward_template(r: str, rng: random.Random):
    tmpls = [t for t in QUESTION_TEMPLATES.get(r, []) if t.direction == "forward"]
    return rng.choice(tmpls) if tmpls else None


def _raw_candidates_block(kb: ShardedKnowledgeBase, known: dict,
                           unknown_role: str, top_k: int) -> dict:
    candidates = kb.query(known, unknown_role, top_k=top_k)
    return {"results": [[str(sym), round(float(sc), 4)] for sym, sc in candidates]}


def generate_absent(kb: ShardedKnowledgeBase, entities: list[str],
                     *, n: int, seed: int = 0,
                     relations: Iterable[str] | None = None,
                     top_k: int = 5, policy: IDKPolicy | None = None,
                     max_attempts_factor: int = 20) -> Iterator[dict]:
    rng = random.Random(seed)
    relations = list(relations) if relations else list(QUESTION_TEMPLATES.keys())
    if not entities or not relations:
        return
    emitted = 0
    attempts = 0
    max_attempts = max(n * max_attempts_factor, 200)
    idx = 0
    while emitted < n and attempts < max_attempts:
        attempts += 1
        s = entities[rng.randrange(len(entities))]
        r = relations[rng.randrange(len(relations))]
        tmpl = _forward_template(r, rng)
        if tmpl is None:
            continue
        known = {"S": s, "R": r}
        ea = ask_with_idk(kb, known, "O", top_k=top_k, policy=policy)
        if ea.state != EpistemicState.IDK:
            continue

        question = tmpl.text.format(s=s)
        kb_q = {"s": s, "r": r, "unknown": "O", "top_k": top_k}
        kb_r = _raw_candidates_block(kb, known, "O", top_k)
        final_answer = f"I don't know. {ea.explanation} {KB_IDK_CITE}"

        idx += 1
        yield make_example(
            example_id=f"idk-absent-{seed}-{idx}",
            generator="idk",
            question=question,
            kb_q=kb_q,
            kb_r=kb_r,
            final_answer=final_answer,
            meta={"mode": "absent", "relation": r, "subject": s,
                  "state": ea.state.value, "top_score": ea.top_score},
        )
        emitted += 1


def generate_holdout(kb: ShardedKnowledgeBase, holdout_triples: list[dict],
                      *, n: int, seed: int = 0, top_k: int = 5,
                      policy: IDKPolicy | None = None,
                      max_attempts_factor: int = 10) -> Iterator[dict]:
    rng = random.Random(seed)
    pool = [t for t in holdout_triples if t["r"] in QUESTION_TEMPLATES]
    if not pool:
        return
    emitted = 0
    attempts = 0
    max_attempts = max(n * max_attempts_factor, 100)
    idx = 0
    while emitted < n and attempts < max_attempts:
        attempts += 1
        fact = pool[rng.randrange(len(pool))]
        s, r, o = fact["s"], fact["r"], fact["o"]
        tmpl = _forward_template(r, rng)
        if tmpl is None:
            continue
        known = {"S": s, "R": r}
        ea = ask_with_idk(kb, known, "O", top_k=top_k, policy=policy)
        if ea.state != EpistemicState.IDK:
            continue  # holdout fact accidentally recoverable -- skip, don't mislabel

        question = tmpl.text.format(s=s)
        kb_q = {"s": s, "r": r, "unknown": "O", "top_k": top_k}
        kb_r = _raw_candidates_block(kb, known, "O", top_k)
        final_answer = f"I don't know. {ea.explanation} {KB_IDK_CITE}"

        idx += 1
        yield make_example(
            example_id=f"idk-holdout-{seed}-{idx}",
            generator="idk",
            question=question,
            kb_q=kb_q,
            kb_r=kb_r,
            final_answer=final_answer,
            meta={"mode": "holdout", "relation": r, "subject": s,
                  "held_out_object": o, "state": ea.state.value,
                  "top_score": ea.top_score},
        )
        emitted += 1

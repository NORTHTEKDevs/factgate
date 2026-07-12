"""Direct QA generator: sample a stored (s, r, o), ask a natural-language
question about it, query the KB, and render a cited answer.

Only emits examples where the KB's own retrieval is confident (top
score >= KNOWN_THRESHOLD) -- the point of this generator is clean,
grounded direct-answer training. Under-confident retrievals are the
job of idk.py, not qa.py.
"""
from __future__ import annotations

import random
from typing import Iterable, Iterator

from rck.knowledge_base import ShardedKnowledgeBase

from factgate.datagen.protocol import kb_cite, make_example
from factgate.datagen.templates import QUESTION_TEMPLATES

KNOWN_THRESHOLD = 0.15


def generate(kb: ShardedKnowledgeBase, triples: list[dict],
             *, n: int, seed: int = 0, top_k: int = 5,
             max_attempts_factor: int = 8) -> Iterator[dict]:
    """Yield up to `n` QA examples sampled from `triples`.

    `triples` are dicts with keys s/r/o, restricted to relations that
    have question templates (callers should pre-filter via
    templates.TOP_RELATIONS).
    """
    rng = random.Random(seed)
    pool = [t for t in triples if t["r"] in QUESTION_TEMPLATES]
    if not pool:
        return
    emitted = 0
    attempts = 0
    max_attempts = max(n * max_attempts_factor, 50)
    idx = 0
    while emitted < n and attempts < max_attempts:
        attempts += 1
        fact = pool[rng.randrange(len(pool))]
        s, r, o = fact["s"], fact["r"], fact["o"]
        tmpl = rng.choice(QUESTION_TEMPLATES[r])
        if tmpl.direction == "forward":
            question = tmpl.text.format(s=s)
            known, unknown_role = {"S": s, "R": r}, "O"
            kb_q = {"s": s, "r": r, "unknown": "O", "top_k": top_k}
        else:
            question = tmpl.text.format(o=o)
            known, unknown_role = {"R": r, "O": o}, "S"
            kb_q = {"o": o, "r": r, "unknown": "S", "top_k": top_k}

        candidates = kb.query(known, unknown_role, top_k=top_k)
        kb_r = {"results": [[str(sym), round(float(sc), 4)]
                             for sym, sc in candidates]}

        if not candidates or float(candidates[0][1]) < KNOWN_THRESHOLD:
            continue  # not confident enough for the direct-QA generator

        top_sym, top_score = str(candidates[0][0]), float(candidates[0][1])
        if tmpl.direction == "forward":
            answer_s, answer_o = s, top_sym
        else:
            answer_s, answer_o = top_sym, o
        final_answer = (
            f"{answer_o if tmpl.direction == 'forward' else answer_s} "
            f"(confidence {top_score:.2f}). {kb_cite(answer_s, r, answer_o)}"
        )

        idx += 1
        yield make_example(
            example_id=f"qa-{seed}-{idx}",
            generator="qa",
            question=question,
            kb_q=kb_q,
            kb_r=kb_r,
            final_answer=final_answer,
            meta={"relation": r, "direction": tmpl.direction,
                  "top_score": top_score,
                  "source_triple": {"s": s, "r": r, "o": o}},
        )
        emitted += 1

"""Multi-hop (2-4 hop) chain QA generator.

Samples real hop-paths from the source triples (so every hop actually
exists in the corpus), then re-derives the chain through the live KB
via `rck.chain_walker.walk_chain` so the recorded scores are genuine
HRR retrieval confidences, not synthetic numbers. Renders a
derivation-trace answer that lists every hop with its score.
"""
from __future__ import annotations

import random
from typing import Iterator

from rck.chain_walker import Hop, walk_chain
from rck.knowledge_base import ShardedKnowledgeBase

from factgate.datagen.protocol import kb_cite, make_example
from factgate.datagen.templates import TOP_RELATIONS

MIN_HOP_SCORE = 0.10


def build_adjacency(triples: list[dict],
                     relations: tuple[str, ...] = TOP_RELATIONS
                     ) -> dict[str, list[tuple[str, str]]]:
    """s -> [(r, o), ...] restricted to `relations`."""
    adj: dict[str, list[tuple[str, str]]] = {}
    for t in triples:
        if t["r"] not in relations:
            continue
        adj.setdefault(t["s"], []).append((t["r"], t["o"]))
    return adj


def _sample_path(adj: dict[str, list[tuple[str, str]]], rng: random.Random,
                  *, min_hops: int, max_hops: int, max_tries: int = 30
                  ) -> tuple[str, list[Hop]] | None:
    starts = [s for s, edges in adj.items() if edges]
    if not starts:
        return None
    for _ in range(max_tries):
        start = rng.choice(starts)
        target_hops = rng.randint(min_hops, max_hops)
        node = start
        hops: list[Hop] = []
        visited = {start}
        ok = True
        for _ in range(target_hops):
            edges = adj.get(node)
            if not edges:
                ok = False
                break
            r, o = rng.choice(edges)
            if o in visited:  # avoid trivial cycles
                ok = False
                break
            hops.append(Hop(relation=r, min_score=MIN_HOP_SCORE))
            visited.add(o)
            node = o
        if ok and len(hops) >= min_hops:
            return start, hops
    return None


def generate(kb: ShardedKnowledgeBase, triples: list[dict],
             *, n: int, seed: int = 0, min_hops: int = 2, max_hops: int = 4,
             max_attempts_factor: int = 10) -> Iterator[dict]:
    rng = random.Random(seed)
    adj = build_adjacency(triples)
    emitted = 0
    attempts = 0
    max_attempts = max(n * max_attempts_factor, 50)
    idx = 0
    while emitted < n and attempts < max_attempts:
        attempts += 1
        sampled = _sample_path(adj, rng, min_hops=min_hops, max_hops=max_hops)
        if sampled is None:
            continue
        start, hops = sampled
        result = walk_chain(kb, start, hops)
        if result.answer is None or result.aborted_at != -1:
            continue
        if result.confidence < MIN_HOP_SCORE:
            continue

        question = (
            f"Starting from {start!r}, follow "
            f"{' then '.join(h.relation for h in hops)} -- what do you "
            f"end up at?"
        )
        kb_q = [{"s": s, "r": r, "unknown": "O"} for s, r, _o, _sc in result.trace]
        kb_r = [{"results": [[o, round(float(sc), 4)]]}
                for _s, _r, o, sc in result.trace]

        steps = []
        for i, (s, r, o, sc) in enumerate(result.trace, start=1):
            steps.append(f"Step {i}: {s} {r} {o} (score {sc:.2f}). "
                         f"{kb_cite(s, r, o)}")
        final_answer = (
            " ".join(steps) +
            f" Therefore, starting from {start!r} via "
            f"{len(result.trace)} hop(s), the answer is "
            f"{result.answer!r} (overall confidence {result.confidence:.2f})."
        )

        idx += 1
        yield make_example(
            example_id=f"chain-{seed}-{idx}",
            generator="chains",
            question=question,
            kb_q=kb_q,
            kb_r=kb_r,
            final_answer=final_answer,
            meta={"start": start, "hops": [h.relation for h in hops],
                  "hop_count": len(hops), "confidence": result.confidence,
                  "answer": result.answer},
        )
        emitted += 1

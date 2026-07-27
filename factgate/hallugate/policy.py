"""Ephemeral per-example KB + the fail-closed sentence policy.

The original gate is fail-OPEN: `bench/runner.py`'s gated_hallucination_rate counts only
claims that were successfully extracted, so an unextracted prose claim contributes to
neither numerator nor denominator and is invisible rather than counted as a leak.

Fail-CLOSED inverts that. Every sentence lands in exactly one state:

    PASS   -- carried >=1 claim, all corroborated by the source
    BLOCK  -- carried a claim the source contradicts
    HELD   -- carried a claim the source cannot corroborate, OR looked like a factual
              assertion the extractor could not resolve into a checkable claim
    SKIP   -- no factual assertion detected (hedges, meta-statements, questions)

The HELD state is what preserves the structural guarantee: extraction failures become a
coverage cost (over-blocking) rather than a safety failure (a leak). That trade is only
honest if HELD is reported, so `score()` reports over-block and leak together, never
one without the other.
"""
from __future__ import annotations

import re

from rck.bulk_ingest import bulk_load_triples
from rck.knowledge_base import ShardedKnowledgeBase

from factgate.gate import GateStatus, verify_claim
from factgate.stats import wilson  # noqa: F401  (re-export)

PASS, BLOCK, HELD, SKIP = "PASS", "BLOCK", "HELD", "SKIP"

# A sentence is treated as asserting a fact unless it is purely a hedge, a refusal, or a
# question. Deliberately permissive: misclassifying an assertion as a non-claim is the
# one path that can still leak, so the bar to earn SKIP is high.
_NON_ASSERTION = re.compile(
    r"^\s*(?:i\s+(?:am\s+unable|cannot|can't|don't\s+know)"
    r"|(?:the\s+)?(?:passages?|context|documents?|sources?)\s+(?:do(?:es)?\s+not|don't)"
    r"|unfortunately|sorry|there\s+is\s+no\s+information"
    r"|based\s+on\s+the\s+(?:provided\s+)?(?:passages?|context))\b",
    re.IGNORECASE)


def is_assertion(sentence: str) -> bool:
    """False only for hedges/refusals/questions; everything else is treated as a claim."""
    s = sentence.strip()
    if len(s) < 12 or s.endswith("?"):
        return False
    return _NON_ASSERTION.match(s) is None


def build_ephemeral_kb(triples, *, dim: int = 4096, n_shards: int = 8, seed: int = 0):
    """A throwaway KB holding only this example's source facts.

    n_shards is small by design: a per-example source yields tens of facts, not 90k, so
    the shard sizing that the 90k KB needs would put ~0 facts per shard here.
    """
    kb = ShardedKnowledgeBase(dim=dim, n_shards=n_shards, seed=seed)
    bulk_load_triples(kb, triples, symmetrize=False)
    return kb


def classify(sentence: str, triples, kb) -> str:
    """Decide one sentence's fate against the source KB."""
    if not is_assertion(sentence):
        return SKIP
    if not triples:
        return HELD                      # assertion the extractor could not resolve
    verdicts = [verify_claim(kb, None, s, r, o).status for s, r, o in triples]
    if any(v is GateStatus.CONTRADICTED for v in verdicts):
        return BLOCK
    if all(v is GateStatus.VERIFIED for v in verdicts):
        return PASS
    return HELD                          # OUT_OF_KB / UNRESOLVED -> not corroborated




def score(rows) -> dict:
    """rows: iterable of (state, is_hallucinated).

    leak_rate       = hallucinated sentences allowed through / hallucinated sentences
    over_block_rate = faithful sentences held or blocked / faithful sentences

    Both are reported together. Either alone is meaningless: a gate that blocks
    everything has a 0% leak rate.
    """
    rows = [r for r in rows if r[0] != SKIP]
    hal = [s for s, h in rows if h]
    fai = [s for s, h in rows if not h]
    leaks = sum(1 for s in hal if s == PASS)
    blocked_faithful = sum(1 for s in fai if s in (BLOCK, HELD))
    return {
        "n_scored": len(rows), "n_hallucinated": len(hal), "n_faithful": len(fai),
        "leak_rate": leaks / len(hal) if hal else None,
        "leak_ci95": wilson(leaks, len(hal)),
        "over_block_rate": blocked_faithful / len(fai) if fai else None,
        "over_block_ci95": wilson(blocked_faithful, len(fai)),
        "leaks": leaks, "blocked_faithful": blocked_faithful,
    }

"""Tests for the fail-closed policy and the paired leak/over-block scoring."""
import pytest

from factgate.hallugate.policy import (
    BLOCK, HELD, PASS, SKIP, build_ephemeral_kb, classify, is_assertion, score, wilson)


# ------------------------------------------------------------- assertions
@pytest.mark.parametrize("s", [
    "Based on the provided passages, technicians are paid hourly.",
    "The passages do not provide specific information on pay.",
    "I am unable to answer the question based on the given passages.",
    "Unfortunately, that is not stated.",
    "What is the average pay?",
    "Yes.",
])
def test_non_assertions_are_skipped(s):
    assert is_assertion(s) is False


@pytest.mark.parametrize("s", [
    "A portacaval shunt is a type of surgical procedure.",
    "The highest average pay is in Alaska at $23.70 per hour.",
])
def test_real_assertions_are_detected(s):
    assert is_assertion(s) is True


# ----------------------------------------------------------- fail-closed
def test_unresolvable_assertion_is_held_not_passed():
    """The core fail-closed property: no triples extracted from a factual-looking
    sentence must NOT silently pass. This is the leak the original design allowed."""
    kb = build_ephemeral_kb([("dog", "isa", "mammal")])
    assert classify("A dog is a kind of mammal.", [], kb) == HELD


def test_non_assertion_with_no_triples_is_skipped_not_held():
    kb = build_ephemeral_kb([("dog", "isa", "mammal")])
    assert classify("I am unable to answer that question here.", [], kb) == SKIP


def test_uncorroborated_claim_is_held():
    kb = build_ephemeral_kb([("dog", "isa", "mammal")])
    assert classify("A zibbler is a kind of glorptron.",
                    [("zibbler", "isa", "glorptron")], kb) == HELD


def test_corroborated_claim_passes():
    kb = build_ephemeral_kb([("dog", "isa", "mammal")])
    assert classify("A dog is a kind of mammal.",
                    [("dog", "isa", "mammal")], kb) == PASS


# ---------------------------------------------------------------- scoring
def test_score_reports_both_rates():
    rows = [(PASS, True), (HELD, True), (PASS, False), (HELD, False)]
    out = score(rows)
    assert out["leak_rate"] == 0.5           # 1 of 2 hallucinated passed
    assert out["over_block_rate"] == 0.5     # 1 of 2 faithful held
    assert out["n_scored"] == 4


def test_score_excludes_skipped_from_denominators():
    rows = [(SKIP, False), (SKIP, True), (PASS, False)]
    out = score(rows)
    assert out["n_scored"] == 1
    assert out["n_hallucinated"] == 0
    assert out["leak_rate"] is None


def test_block_everything_gets_zero_leak_but_total_overblock():
    """The degenerate policy must be visibly degenerate, not flattering."""
    rows = [(HELD, True), (HELD, True), (HELD, False), (HELD, False)]
    out = score(rows)
    assert out["leak_rate"] == 0.0
    assert out["over_block_rate"] == 1.0


def test_wilson_bounds_zero_numerator():
    lo, hi = wilson(0, 100)
    assert lo == 0.0 and 0 < hi < 0.05

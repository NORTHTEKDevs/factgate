"""Property-based search over the safety invariants.

Every bug found in this project so far came from a hand-written case, which only covers
what someone thought of. This generates inputs nobody chose and asserts the properties
that must hold for all of them.

Seeded, so a failure is reproducible. Absence of failures here is not proof of
correctness -- it is a much wider search than the example tests.

THE INVARIANTS

  I1  VERIFIED requires provable equality. If the gate says VERIFIED, re-parsing both
      sides must agree on magnitude and unit (or on exact text). A VERIFIED that cannot
      be re-derived is a leak.
  I2  BLOCK requires provable difference. If the gate says BLOCK, the values must not be
      equal and must not be comparable-but-uncertain. A BLOCK on an equal value tells a
      user their correct figure is wrong.
  I3  Total. Every input returns one of the three verdicts. No exception escapes.
  I4  MATCH is symmetric EXCEPT for the documented range/point asymmetry: a point inside
      a declared range MATCHES, but a range never confirms a declared point. The invariant
      is therefore "symmetric unless exactly one side is a range".
  I5  Reflexive. compare(a, a) is MATCH for any parseable value.
  6   Absence never verifies. No absence marker is ever VERIFIED against a real value.
"""
import random
import re

import pytest

from factgate.domain.factset import FactSet, ValidationError
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim
from factgate.domain.quantity import (
    DIFFER, INCOMPARABLE, MATCH, compare_values, parse_quantity, parse_range)

SEED = 20260729
UNITS = ["mg/kg", "mg", "ml/kg", "percent", "%", "months", "weeks", "days", "hours",
         "breaths per minute", "degrees celsius", "beats per minute", ""]
CURRENCIES = ["$", "£", "€", "US$"]
MAGNITUDES = ["", "k", "K", "M", "B", " million", " billion"]
JUNK = ["", " PO", " IV", " per day", " per query", " (approx)", " or above", " q6h",
        " every 6 hours", " divided", "/mo", "/year", " download", " flat",
        " after v0.5", ", revised", " to Series B", "!!", " µg", " ~"]
ABSENCE = ["NONE", "none", "N/A", "not stated", "unknown", "unclear", "Not provided",
           "Not applicable", "nil", "missing", "", "   "]


def _num(rng):
    return rng.choice([
        f"{rng.randint(1, 999)}",
        f"{rng.randint(1, 99)}.{rng.randint(0, 99):02d}",
        f"{rng.randint(1, 9)},{rng.randint(100, 999)}",
        f"0.{rng.randint(1, 99)}",
    ])


def _value(rng):
    """A plausible declared or claimed value."""
    kind = rng.random()
    if kind < 0.35:                                    # plain quantity
        return f"{_num(rng)} {rng.choice(UNITS)}".strip()
    if kind < 0.60:                                    # currency
        return f"{rng.choice(CURRENCIES)}{_num(rng)}{rng.choice(MAGNITUDES)}"
    if kind < 0.80:                                    # range
        a, b = sorted([rng.randint(1, 500), rng.randint(1, 500)])
        if a == b:
            b = a + rng.randint(1, 50)
        sep = rng.choice(["-", " to ", "-"])
        if rng.random() < 0.5:
            return f"{rng.choice(CURRENCIES)}{a}{sep}{rng.choice(CURRENCIES)}{b}"
        return f"{a}{sep}{b} {rng.choice(UNITS)}".strip()
    if kind < 0.90:                                    # open range
        return f"{rng.choice(CURRENCIES)}{_num(rng)}{rng.choice(MAGNITUDES)}+"
    return rng.choice(["oral", "intravenous", "IM", "twice daily", "unspecified"])


def _claim(rng, declared):
    """A claimed value: sometimes the declared one, sometimes mutated, sometimes junk."""
    r = rng.random()
    if r < 0.30:
        return declared + rng.choice(JUNK)
    if r < 0.45:
        return rng.choice(ABSENCE)
    if r < 0.60:                                        # scale the numbers
        return re.sub(r"[0-9][0-9,]*(?:\.[0-9]+)?",
                      lambda m: f"{float(m.group(0).replace(',', '')) * rng.choice([2, 0.5, 10]):g}",
                      declared)
    if r < 0.75:
        return _value(rng)
    return declared


def _equal_by_reparse(a: str, b: str) -> bool:
    """Independent re-derivation of equality, deliberately not the code under test."""
    qa, qb = parse_quantity(a), parse_quantity(b)
    if qa is not None and qb is not None:
        return qa.value == qb.value and qa.unit == qb.unit
    ra, rb = parse_range(a), parse_range(b)
    if ra is not None and rb is not None:
        return (ra.low, ra.high, ra.unit) == (rb.low, rb.high, rb.unit)
    if ra is not None and qb is not None:
        return ra.unit == qb.unit and ra.low <= qb.value <= ra.high
    if qa is None and qb is None and ra is None and rb is None:
        return " ".join(a.lower().split()) == " ".join(b.lower().split())
    return False


N = 4000


def test_i3_total_no_input_raises():
    """Every generated pair returns a verdict. An exception in a gate is a fail-open in
    practice, because callers wrap it and carry on."""
    rng = random.Random(SEED)
    for _ in range(N):
        d, c = _value(rng), None
        c = _claim(rng, d)
        out = compare_values(d, c)
        assert out in (MATCH, DIFFER, INCOMPARABLE), (d, c, out)


def test_i1_verified_implies_provable_equality():
    """The leak invariant. MATCH must be re-derivable without the code under test."""
    rng = random.Random(SEED + 1)
    for _ in range(N):
        d = _value(rng)
        c = _claim(rng, d)
        if compare_values(d, c) == MATCH:
            assert _equal_by_reparse(d, c), f"MATCH not re-derivable: {d!r} vs {c!r}"


def test_i2_block_implies_not_equal():
    """The false-BLOCK invariant. DIFFER must never fire on values that are equal."""
    rng = random.Random(SEED + 2)
    for _ in range(N):
        d = _value(rng)
        c = _claim(rng, d)
        if compare_values(d, c) == DIFFER:
            assert not _equal_by_reparse(d, c), f"DIFFER on equal values: {d!r} vs {c!r}"


def test_i5_reflexive():
    rng = random.Random(SEED + 3)
    for _ in range(N):
        v = _value(rng)
        if parse_quantity(v) is None and parse_range(v) is None:
            continue
        assert compare_values(v, v) == MATCH, f"not reflexive: {v!r}"


def test_i4_match_is_symmetric_except_range_versus_point():
    """The first version of this invariant was WRONG and the fuzzer was right to break it.
    "$5,173 million+" vs "$41.72B" matches one way only, because a point inside a declared
    range is supported by the document while a range cannot confirm a declared point. That
    asymmetry is the design; symmetry is only required when both sides are the same kind."""
    rng = random.Random(SEED + 4)
    for _ in range(N):
        a, b = _value(rng), _value(rng)
        if compare_values(a, b) != MATCH:
            continue
        one_sided = (parse_range(a) is None) != (parse_range(b) is None)
        if one_sided:
            continue
        assert compare_values(b, a) == MATCH, f"asymmetric MATCH: {a!r} vs {b!r}"


def test_i4b_range_point_asymmetry_is_the_documented_direction():
    """And the asymmetry only ever runs one way: range-declared accepts a contained point,
    point-declared never accepts a range."""
    assert compare_values("5-10 mg/kg", "7 mg/kg") == MATCH
    assert compare_values("7 mg/kg", "5-10 mg/kg") == INCOMPARABLE


def test_i6_absence_never_verifies():
    rng = random.Random(SEED + 5)
    for _ in range(N // 4):
        d = _value(rng)
        for marker in ABSENCE:
            assert compare_values(d, marker) != MATCH, f"{marker!r} verified against {d!r}"


def test_gate_is_total_over_generated_domains():
    """The same search one level up: through a real FactSet and gate_claim."""
    rng = random.Random(SEED + 6)
    checked = 0
    for _ in range(600):
        declared = _value(rng)
        spec = {"domain": "fuzz", "entities": {"thing": ["the thing"]},
                "relations": {"prop": {"kind": "text"}},
                "facts": [{"s": "thing", "r": "prop", "o": declared, "source": "q"}]}
        try:
            fs = FactSet.from_dict(spec)
        except ValidationError:
            continue
        v = gate_claim(fs, rng.choice(["thing", "the thing", "other", None]), "prop",
                       _claim(rng, declared))
        assert v.status in (VERIFIED, BLOCK, HELD)
        if v.status == VERIFIED:
            assert _equal_by_reparse(declared, fs.normalise_value(v.claimed) or "")
        checked += 1
    assert checked > 100, "generator produced too few loadable domains to be meaningful"

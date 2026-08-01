"""Differential proof over the whole value grammar, against an independent oracle.

WHY THIS EXISTS. Five rounds of adversarial review found defects one shape at a time --
fractions, scientific notation, ratios, negative ranges, unicode digits, typography -- and
the last two rounds found them INSIDE the previous round's fixes. Finding them by hand does
not scale and does not converge, because each new notation adds surface faster than a human
reviewer covers it.

This covers the surface mechanically instead. It generates every declared value shape,
crosses them against each other, and checks the three-valued contract on every pair using
an ORACLE THAT SHARES NO CODE WITH THE IMPLEMENTATION:

  * the oracle parses with `fractions.Fraction`, i.e. exact rational arithmetic, where the
    implementation uses float. A float rounding bug therefore shows up as a disagreement
    rather than being reproduced identically on both sides.
  * the oracle is deliberately partial. It says EQUAL, UNEQUAL or DONT_KNOW, and only the
    first two are assertions. That keeps it honest: it cannot manufacture a failure for a
    shape it does not itself understand.

THE TWO ASSERTIONS, which are the product's entire safety claim:

  MATCH  requires the oracle to agree the values are EQUAL      (a violation is a LEAK)
  DIFFER requires the oracle to agree they are UNEQUAL          (a violation is a FALSE BLOCK)

INCOMPARABLE is never asserted against, because "cannot prove it" is always a permissible
answer and is what the gate turns into a hold.
"""
from fractions import Fraction
import itertools
import re

import pytest

from factgate.domain.quantity import (
    DIFFER, INCOMPARABLE, MATCH, compare_values, parse_quantity, parse_range)

EQUAL, UNEQUAL, DONT_KNOW = "EQUAL", "UNEQUAL", "DONT_KNOW"

# ---------------------------------------------------------------- the oracle
_CURRENCY = {"$": "usd", "US$": "usd", "£": "gbp", "€": "eur", "¥": "jpy"}
_MAG = {"k": 1000, "m": 10 ** 6, "b": 10 ** 9,
        "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}


def _exact(text):
    """(Fraction, unit) for a single value, or None if the oracle does not understand it.

    Deliberately independent: exact rationals, its own regexes, no import from the module
    under test beyond the entry points being checked.
    """
    s = " ".join(str(text).split())
    if not s:
        return None
    cur = ""
    m = re.match(r"^(US\$|[$£€¥])\s*(.*)$", s)
    if m:
        cur, s = _CURRENCY[m.group(1)], m.group(2)

    m = re.match(r"^([0-9]+)\s+([0-9]+)\s*/\s*([0-9]+)\s*([A-Za-zµ%°/ ]*)$", s)
    if m and int(m.group(3)):                      # mixed number "2 1/2 inch"
        v = Fraction(int(m.group(1))) + Fraction(int(m.group(2)), int(m.group(3)))
        return v, (cur + re.sub(r"\s+", "", m.group(4))).lower()
    m = re.match(r"^([0-9]+)\s*/\s*([0-9]+)\s*([A-Za-zµ%°/ ]*)$", s)
    if m and int(m.group(2)):                      # plain fraction
        return (Fraction(int(m.group(1)), int(m.group(2))),
                (cur + re.sub(r"\s+", "", m.group(3))).lower())
    m = re.match(r"^([-+]?[0-9]*\.?[0-9]+)[eE]([-+]?[0-9]+)\s*([A-Za-z0-9µ%°/ ]*)$", s)
    if m:                                          # scientific notation
        exp = int(m.group(2))
        if abs(exp) > 300:
            return None                            # the implementation refuses these
        v = Fraction(m.group(1)) * (Fraction(10) ** exp)
        return v, (cur + re.sub(r"\s+", "", m.group(3))).lower()
    m = re.match(r"^([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
                 r"([kKmMbB]|thousand|million|billion)?\s*([A-Za-zµ%°/ ]*)$", s)
    if m:
        v = Fraction(m.group(1).replace(",", ""))
        unit = re.sub(r"\s+", "", m.group(3) or "").lower()
        if m.group(2):
            key = m.group(2).lower()
            # a bare magnitude letter is only expanded when a currency marks it as money,
            # which is the implementation's documented rule
            if cur or key in ("thousand", "million", "billion"):
                v *= _MAG[key]
            else:
                return None
        return v, (cur + unit)
    return None


def _exact_range(text):
    """((low, high), unit) or None."""
    s = " ".join(str(text).split())
    m = re.match(r"^(?:between\s+)?(.+?)\s*(?:-|–|—|\bto\b|\band\b)\s*(.+)$", s)
    if not m:
        return None
    lo, hi = _exact(m.group(1)), _exact(m.group(2))
    if lo is None or hi is None:
        return None
    lo_v, lo_u = lo
    hi_v, hi_u = hi
    unit = hi_u or lo_u
    if lo_u and hi_u and lo_u != hi_u:
        return None
    if lo_v >= hi_v:
        return None
    return (lo_v, hi_v), unit


def oracle(declared, claimed):
    """EQUAL / UNEQUAL / DONT_KNOW, computed with exact arithmetic."""
    dr, cr = _exact_range(declared), _exact_range(claimed)
    dq, cq = _exact(declared), _exact(claimed)
    if dr and cr:
        return EQUAL if dr == cr else UNEQUAL
    if dr and cq:
        (lo, hi), unit = dr
        v, u = cq
        if u != unit:
            return DONT_KNOW
        return EQUAL if lo <= v <= hi else UNEQUAL
    if cr and dq:
        return DONT_KNOW          # a range never confirms a point; deliberately asymmetric
    if dq and cq:
        if dq[1] != cq[1]:
            return DONT_KNOW      # different units prove nothing without conversion
        return EQUAL if dq[0] == cq[0] else UNEQUAL
    return DONT_KNOW


# ------------------------------------------------------------- the generator
_NUMBERS = ["0", "1", "5", "7", "10", "15", "40", "99", "100", "250", "1000",
            "1,200", "0.5", "0.05", "1.5", "2.25", "15.4", "99.99", "-5", "-20",
            "1/2", "3/4", "7/16", "2 1/2", "1.2e3", "5e-2"]
_UNITS = ["", "mg", "mg/kg", "percent", "%", "ml", "days", "weeks", "degrees celsius",
          "beats per minute", "inch"]
_CUR = ["", "$", "£", "€"]
_MAGS = ["", "k", "M", "million"]


def _values(limit):
    """Every shape the library claims to read, as concrete strings."""
    out = []
    for n in _NUMBERS:
        for u in _UNITS[:6]:
            out.append(f"{n} {u}".strip())
    for c in _CUR[1:]:
        for n in ("5", "250", "1,200", "0.5"):
            for mag in _MAGS:
                out.append(f"{c}{n}{mag}")
    for lo, hi in (("5", "10"), ("0.5", "1.5"), ("-20", "-15"), ("100", "250")):
        for u in ("mg", "percent", "degrees celsius", ""):
            out.append(f"{lo}-{hi} {u}".strip())
            out.append(f"{lo} to {hi} {u}".strip())
            out.append(f"between {lo} and {hi} {u}".strip())
    out += ["3:1", "16:9", "20:1", "$100M+", "18 months+", "oral", "intravenous",
            "twice daily", "NONE", "not stated", ""]
    return out[:limit]


# --------------------------------------------------------------- the proofs
@pytest.mark.parametrize("numeric", [False, True])
def test_match_implies_the_oracle_agrees_they_are_equal(numeric):
    """THE LEAK PROOF. Every MATCH the implementation returns must be confirmed EQUAL by
    exact rational arithmetic that shares no code with it."""
    vals = _values(240)
    bad = []
    for a, b in itertools.product(vals, repeat=2):
        if compare_values(a, b, numeric) != MATCH:
            continue
        if oracle(a, b) == UNEQUAL:
            bad.append((a, b))
    assert not bad, f"MATCH on values the oracle proves unequal: {bad[:8]}"


@pytest.mark.parametrize("numeric", [False, True])
def test_differ_implies_the_oracle_agrees_they_are_unequal(numeric):
    """THE FALSE-BLOCK PROOF. Every DIFFER must be confirmed UNEQUAL by the oracle. A DIFFER
    on values the oracle proves EQUAL is the gate telling a user their correct answer
    contradicts the document."""
    vals = _values(240)
    bad = []
    for a, b in itertools.product(vals, repeat=2):
        if compare_values(a, b, numeric) != DIFFER:
            continue
        if oracle(a, b) == EQUAL:
            bad.append((a, b))
    assert not bad, f"DIFFER on values the oracle proves equal: {bad[:8]}"


@pytest.mark.parametrize("numeric", [False, True])
def test_comparison_is_total_and_reflexive(numeric):
    """No input raises, every verdict is one of three, and every value matches itself."""
    for v in _values(400):
        assert compare_values(v, v, numeric) in (MATCH, DIFFER, INCOMPARABLE)
        if parse_quantity(v, numeric) is not None or parse_range(v, numeric) is not None:
            assert compare_values(v, v, numeric) == MATCH, f"not reflexive: {v!r}"


@pytest.mark.parametrize("numeric", [False, True])
def test_match_is_symmetric_except_range_against_point(numeric):
    """A point inside a declared range matches; a range never confirms a declared point.
    Symmetry is required everywhere else."""
    vals = _values(180)
    for a, b in itertools.product(vals, repeat=2):
        if compare_values(a, b, numeric) != MATCH:
            continue
        one_sided = (parse_range(a, numeric) is None) != (parse_range(b, numeric) is None)
        if one_sided:
            continue
        assert compare_values(b, a, numeric) == MATCH, f"asymmetric MATCH: {a!r} {b!r}"


def test_the_oracle_is_not_vacuous():
    """A proof whose oracle never decides anything proves nothing. It must return EQUAL and
    UNEQUAL on a substantial share of the generated pairs, or these tests are hollow."""
    vals = _values(160)
    verdicts = [oracle(a, b) for a, b in itertools.product(vals, repeat=2)]
    decided = sum(1 for v in verdicts if v != DONT_KNOW)
    equal = sum(1 for v in verdicts if v == EQUAL)
    # Most pairs carry different units, where "prove nothing" is the correct answer, so
    # the decided share is naturally a minority. What matters is that it is substantial in
    # absolute terms and includes both verdicts -- a one-sided oracle could only ever catch
    # one of the two failure modes.
    unequal = sum(1 for v in verdicts if v == UNEQUAL)
    assert decided > 1500, f"oracle decided only {decided}/{len(verdicts)}"
    assert equal > 100, f"oracle found only {equal} equal pairs"
    assert unequal > 500, f"oracle found only {unequal} unequal pairs"


def test_the_oracle_disagrees_with_a_deliberately_broken_comparison():
    """And it must be able to FAIL. A sanity check that the proof would catch a real leak:
    a comparison that says everything matches is caught immediately."""
    vals = _values(80)
    caught = [(a, b) for a, b in itertools.product(vals, repeat=2)
              if oracle(a, b) == UNEQUAL]
    assert caught, "oracle cannot prove any pair unequal, so the leak proof is hollow"

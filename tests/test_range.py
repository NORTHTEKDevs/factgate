"""Range values.

Real documents state ranges constantly -- "$1,500-3,000 setup", "$300-500/mo",
"5 to 10 mg/kg". The schema could not express one, so every such fact was either omitted
or held, and ranges were the single largest coverage gap measured on real files.

The semantics are a deliberate choice, not an obvious one:

  declared RANGE, claimed POINT inside   -> VERIFIED   the document supports that figure
  declared RANGE, claimed POINT outside  -> BLOCK      provably unsupported
  declared RANGE, claimed same RANGE     -> VERIFIED
  declared RANGE, claimed other RANGE    -> DIFFER if disjoint, else HELD
  declared POINT, claimed RANGE          -> HELD       a range cannot confirm a point

The last row is the safety-relevant one: if the protocol gives a single dose and the model
answers with a range, that is not the declared value even when it contains it.
"""
import pytest

from factgate.domain.factset import FactSet
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim
from factgate.domain.quantity import DIFFER, INCOMPARABLE, MATCH, compare_values, parse_range


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("raw,lo,hi,unit", [
    ("5-10 mg/kg", 5.0, 10.0, "mg/kg"),
    ("5 to 10 mg/kg", 5.0, 10.0, "mg/kg"),
    ("5 - 10 mg/kg", 5.0, 10.0, "mg/kg"),
    ("$1,500-3,000", 1500.0, 3000.0, "usd"),
    ("$300-500", 300.0, 500.0, "usd"),
    ("$25M-$40M", 25_000_000.0, 40_000_000.0, "usd"),
    ("15-25%", 15.0, 25.0, "%"),
])
def test_range_parsing(raw, lo, hi, unit):
    r = parse_range(raw)
    assert r is not None
    assert (r.low, r.high, r.unit) == (lo, hi, unit)


@pytest.mark.parametrize("raw", ["15 mg/kg", "$199", "oral", "", "10-", "-10"])
def test_non_ranges_do_not_parse_as_ranges(raw):
    assert parse_range(raw) is None


def test_reversed_bounds_are_rejected():
    """"10-5" is malformed, not a range from 10 to 5. Silently swapping would accept a
    typo as a valid declaration."""
    assert parse_range("10-5 mg/kg") is None


# --------------------------------------------------------------- comparison
def test_point_inside_declared_range_matches():
    assert compare_values("5-10 mg/kg", "7 mg/kg") == MATCH
    assert compare_values("$1,500-3,000", "$2,000") == MATCH


def test_point_on_the_boundary_matches():
    assert compare_values("5-10 mg/kg", "5 mg/kg") == MATCH
    assert compare_values("5-10 mg/kg", "10 mg/kg") == MATCH


def test_point_outside_declared_range_differs():
    """The safety case: a value the document does not support must block."""
    assert compare_values("5-10 mg/kg", "20 mg/kg") == DIFFER
    assert compare_values("$1,500-3,000", "$500") == DIFFER


def test_point_with_a_different_unit_is_incomparable_not_a_match():
    assert compare_values("5-10 mg/kg", "7 mg") == INCOMPARABLE


def test_identical_range_matches():
    assert compare_values("5-10 mg/kg", "5-10 mg/kg") == MATCH
    assert compare_values("5-10 mg/kg", "5 to 10 mg/kg") == MATCH


def test_disjoint_ranges_differ():
    assert compare_values("5-10 mg/kg", "20-30 mg/kg") == DIFFER


def test_overlapping_but_different_ranges_are_incomparable():
    """Neither provably the same nor provably different, so hold."""
    assert compare_values("5-10 mg/kg", "7-15 mg/kg") == INCOMPARABLE


def test_declared_point_against_a_claimed_range_is_incomparable():
    """SAFETY. The protocol gives one dose; the model answered with a range that happens
    to contain it. That is not the declared value, and confirming it would let a reader
    infer the whole range is protocol-supported."""
    assert compare_values("15 mg/kg", "10-20 mg/kg") == INCOMPARABLE


# --------------------------------------------------------------- end to end
RANGE_FACTS = {
    "domain": "r",
    "entities": {"setup": [], "acetaminophen": []},
    "relations": {"price": {"kind": "quantity"}, "dose": {"kind": "quantity"}},
    "facts": [
        {"s": "setup", "r": "price", "o": "$1,500-3,000", "source": "Setup is $1,500-3,000."},
        {"s": "acetaminophen", "r": "dose", "o": "5-10 mg/kg",
         "source": "Give acetaminophen 5-10 mg/kg."},
    ],
}


def test_range_facts_load_for_a_quantity_relation():
    """A kind=quantity relation must accept a range value, or the fact cannot be declared
    at all -- which is what forced ranges to be dropped from real domains."""
    fs = FactSet.from_dict(RANGE_FACTS)
    assert len(fs.facts) == 2


def test_gate_verifies_a_supported_figure_and_blocks_an_unsupported_one():
    fs = FactSet.from_dict(RANGE_FACTS)
    assert gate_claim(fs, "setup", "price", "$2,000").status == VERIFIED
    assert gate_claim(fs, "setup", "price", "$9,000").status == BLOCK
    assert gate_claim(fs, "acetaminophen", "dose", "7 mg/kg").status == VERIFIED
    assert gate_claim(fs, "acetaminophen", "dose", "25 mg/kg").status == BLOCK


def test_range_fact_still_holds_on_an_uncomparable_claim():
    fs = FactSet.from_dict(RANGE_FACTS)
    assert gate_claim(fs, "acetaminophen", "dose", "7 mg").status == HELD
    assert gate_claim(fs, "acetaminophen", "dose", "Not provided").status == HELD


# ------------------------------------------- corrupting a range-valued fact
def test_corrupting_a_range_must_land_outside_it():
    """MEASURED false leak. Scaling the first number of "$4M-$8M" produced "$8M-$8M",
    whose leading value is $8M -- still inside the declared range, so the gate verified
    it correctly and the harness scored a leak. A corruption of a range has to leave the
    range."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "rdb", Path(__file__).resolve().parents[1] / "scripts" / "run_domain_bench.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k in range(4):
        wrong = m.corrupt("$4M-$8M", k)
        assert compare_values("$4M-$8M", wrong) == DIFFER, (
            f"corrupt(k={k}) produced {wrong!r}, which is not provably wrong")


def test_corrupting_a_point_still_lands_outside():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "rdb", Path(__file__).resolve().parents[1] / "scripts" / "run_domain_bench.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for v in ("15 mg/kg", "$199", "$150M"):
        for k in range(4):
            assert compare_values(v, m.corrupt(v, k)) == DIFFER


# ----------------------------- reported by a first-time author, unprompted
def test_currency_may_carry_trailing_words_like_a_plain_quantity():
    """Reported as an unpredictable asymmetry: "12 weeks engineering" parsed but
    "$5k cloud credit" was rejected at load. Both are a number with descriptive words;
    there is no reason for currency to be the stricter case."""
    from factgate.domain.quantity import parse_quantity
    q = parse_quantity("$5k cloud credit")
    assert q is not None and q.value == 5000.0
    # trailing words are part of the unit, exactly as for plain quantities
    assert parse_quantity("$5k cloud credit") != parse_quantity("$5k")


def test_leading_approximation_marker_is_accepted():
    """The document writes "~$2,000". Rejecting it at load forced the author to strip the
    tilde, discarding the "approximate" signal the source was careful to convey. The gate
    has no tolerance by design, so the marker is accepted and ignored -- but the file
    loads."""
    from factgate.domain.quantity import parse_quantity
    assert parse_quantity("~$2,000") == parse_quantity("$2,000")
    assert parse_quantity("~15 mg/kg") == parse_quantity("15 mg/kg")


def test_trailing_plus_is_an_open_range_not_a_point():
    """"$100M+" means "at least $100M". Parsing it as exactly $100M would verify that one
    figure and block every larger one, inverting the document's meaning."""
    r = parse_range("$100M+")
    assert r is not None and r.low == 100_000_000.0 and r.high == float("inf")
    assert compare_values("$100M+", "$150M") == MATCH
    assert compare_values("$100M+", "$50M") == DIFFER


def test_open_range_still_respects_units():
    assert compare_values("$100M+", "150 mg/kg") == INCOMPARABLE


def test_claimed_range_with_trailing_text_is_incomparable_like_a_point():
    """The range fallback recovers a clean leading range, but unexplained trailing text
    still blocks a MATCH -- consistent with the point path. A permissive version verified
    "EUR306+ to Series B" against a declared "EUR306+", found by property testing."""
    # Corrected after fuzzing: unexplained trailing text is INCOMPARABLE for a range for
    # the same reason it is for a point. Declaring "after v0.5" as a qualifier is how an
    # author recovers this; ignoring the words is not.
    assert compare_values("12-16 weeks", "12-16 weeks after v0.5") == INCOMPARABLE
    assert compare_values("12-16 weeks", "40-50 weeks after v0.5") == DIFFER


def test_leading_range_fallback_does_not_rescue_an_ambiguous_claim():
    """If the declared range's own bounds appear elsewhere in the claim it is not a clean
    single value, and the safe answer is to hold."""
    assert compare_values("12-16 weeks", "20-30 weeks, revised from 12-16 weeks") \
        == INCOMPARABLE


# ------------------------------------------- ranges written as prose
@pytest.mark.parametrize("raw,lo,hi,unit", [
    ("between $500K and $2M", 500_000.0, 2_000_000.0, "usd"),
    ("between $5K and $50K", 5_000.0, 50_000.0, "usd"),
    ("between 5 and 10 mg/kg", 5.0, 10.0, "mg/kg"),
    ("between 15 and 25%", 15.0, 25.0, "%"),
])
def test_between_x_and_y_is_a_range(raw, lo, hi, unit):
    """Measured: the model writes ranges as prose ("between $500K and $2M") where the
    document writes them with a dash. Three of one document's eight held values were this
    single unrecognised form."""
    r = parse_range(raw)
    assert r is not None and (r.low, r.high, r.unit) == (lo, hi, unit)


def test_prose_range_compares_like_any_other():
    assert compare_values("$500K-$2M", "between $500K and $2M") == MATCH
    assert compare_values("$500K-$2M", "between $5M and $9M") == DIFFER

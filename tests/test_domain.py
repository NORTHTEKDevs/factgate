"""Tests for the bounded-domain gate.

This is the architecture the HalluGate negative result pointed to (docs/HALLUGATE.md):
a parameter-free verdict needs a canonical vocabulary on both sides, which a bounded
domain supplies and free text does not.

The logic-bearing pieces are quantity parsing and value comparison, because that is where
a hallucinated dose is caught or missed. A tolerance bug here is a patient-safety bug.
"""
import json

import pytest

from factgate.domain.factset import FactSet, ValidationError
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim
from factgate.domain.quantity import (
    DIFFER, INCOMPARABLE, MATCH, Quantity, compare_values, parse_quantity, values_match)

FACTS = {
    "domain": "test-dosing",
    "entities": {"acetaminophen": ["tylenol", "paracetamol"],
                 "ibuprofen": ["advil"]},
    "relations": {"pediatric_dose": {"kind": "quantity"},
                  "max_daily": {"kind": "quantity"},
                  "route": {"kind": "text"}},
    "facts": [
        {"s": "acetaminophen", "r": "pediatric_dose", "o": "15 mg/kg",
         "source": "Give acetaminophen 15 mg/kg PO q4-6h."},
        {"s": "acetaminophen", "r": "route", "o": "oral",
         "source": "Give acetaminophen 15 mg/kg PO q4-6h."},
        {"s": "ibuprofen", "r": "pediatric_dose", "o": "10 mg/kg",
         "source": "Ibuprofen 10 mg/kg PO q6h for children over 6 months."},
    ],
}


# --------------------------------------------------------------- quantities
@pytest.mark.parametrize("raw,val,unit", [
    ("15 mg/kg", 15.0, "mg/kg"),
    ("15mg/kg", 15.0, "mg/kg"),
    ("  15.5 mg/kg ", 15.5, "mg/kg"),
    ("1,200 mg", 1200.0, "mg"),
    # multi-word units are normal in clinical thresholds; rejecting them made the
    # demo fact set fail to load, which is how this was found.
    ("60 breaths per minute", 60.0, "breathsperminute"),
    ("38 degrees celsius", 38.0, "degreescelsius"),
    ("92 percent", 92.0, "percent"),
    ("0.15 mg/kg", 0.15, "mg/kg"),
])
def test_parse_quantity(raw, val, unit):
    q = parse_quantity(raw)
    assert q == Quantity(val, unit)


def test_multi_word_units_still_discriminate():
    assert values_match("60 breaths per minute", "60 breaths per minute") is True
    assert values_match("60 breaths per minute", "80 breaths per minute") is False
    assert values_match("38 degrees celsius", "38 degrees fahrenheit") is False


def test_parse_quantity_rejects_non_quantity():
    assert parse_quantity("oral") is None
    assert parse_quantity("") is None


def test_values_match_identical_quantity():
    assert values_match("15 mg/kg", "15 mg/kg") is True
    assert values_match("15 mg/kg", "15mg/kg") is True


def test_values_match_rejects_different_magnitude():
    """The whole point: 20 mg/kg must never match a declared 15 mg/kg."""
    assert values_match("15 mg/kg", "20 mg/kg") is False
    assert values_match("15 mg/kg", "25 mg/kg") is False


def test_values_match_rejects_unit_mismatch():
    """Same number, different unit is NOT a match -- 15 mg != 15 mg/kg."""
    assert values_match("15 mg/kg", "15 mg") is False


def test_values_match_text_is_case_insensitive():
    assert values_match("oral", "Oral") is True
    assert values_match("oral", "intravenous") is False


def test_compare_trailing_annotation_is_incomparable_not_differ():
    """Found in a live run: the model stated the CORRECT dose as "10 mg/kg PO" and the
    gate emitted BLOCK, i.e. it told a clinician the right dose contradicted the
    protocol. Same magnitude with an unrecognised unit tail must be INCOMPARABLE (held),
    never DIFFER (blocked). It is also not a MATCH: "10 mg/kg per day" would prefix the
    same way and mean something else."""
    assert compare_values("10 mg/kg", "10 mg/kg PO") == INCOMPARABLE


def test_compare_different_magnitude_still_differs_despite_annotation():
    """The safety case must survive the fix: a wrong dose with a route suffix still
    blocks, because the magnitudes genuinely differ."""
    assert compare_values("10 mg/kg", "20 mg/kg PO") == DIFFER


@pytest.mark.parametrize("claimed", [
    "5 to 10 mg/kg",
    "10 to 20 mg/kg",
    "20 mg is wrong, the correct dose is 10 mg/kg",
])
def test_ambiguous_multi_number_claim_is_incomparable_not_differ(claimed):
    """Found by adversarial review. The leading-quantity fallback read only the FIRST
    number, so a range containing the declared dose, or a correction naming it, was
    reported as a contradiction. Same false-BLOCK class as the "10 mg/kg PO" bug:
    the gate told a clinician a correct dose conflicted with the protocol."""
    assert compare_values("10 mg/kg", claimed) == INCOMPARABLE


def test_single_wrong_number_with_annotation_still_differs():
    """The safety case must survive: one unambiguous wrong value still blocks."""
    assert compare_values("10 mg/kg", "20 mg/kg PO") == DIFFER
    assert compare_values("10 mg/kg", "20 mg/kg every 6 hours") == DIFFER


@pytest.mark.parametrize("declared,claimed", [
    ("5 mg", "5 Mg"),     # milli- vs mega-: a factor of 10^9
    ("5 mg", "5 MG"),
    ("5 mL", "5 ML"),
])
def test_unit_case_is_significant(declared, claimed):
    """Found by adversarial review. Units were case-folded before comparison, so a
    megagram verified against a milligram. SI prefixes are case-significant; folding
    them is not a normalisation, it is a 10^9 error."""
    assert compare_values(declared, claimed) == INCOMPARABLE


def test_compare_exact_is_match():
    assert compare_values("10 mg/kg", "10 mg/kg") == MATCH


def test_compare_unit_mismatch_is_incomparable():
    assert compare_values("15 mg/kg", "15 mg") == INCOMPARABLE


def test_compare_missing_claim_is_incomparable():
    assert compare_values("15 mg/kg", None) == INCOMPARABLE


def test_values_match_does_not_use_float_equality_for_near_values():
    """15.0 vs 15.4 must not pass. No silent rounding tolerance."""
    assert values_match("15 mg/kg", "15.4 mg/kg") is False


# ------------------------------------------------------------------ factset
def test_factset_resolves_alias_to_canonical_entity():
    fs = FactSet.from_dict(FACTS)
    assert fs.resolve_entity("Tylenol") == "acetaminophen"
    assert fs.resolve_entity("PARACETAMOL") == "acetaminophen"
    assert fs.resolve_entity("acetaminophen") == "acetaminophen"


def test_factset_returns_none_for_unknown_entity():
    fs = FactSet.from_dict(FACTS)
    assert fs.resolve_entity("morphine") is None


def test_factset_lookup_returns_declared_value():
    fs = FactSet.from_dict(FACTS)
    assert fs.lookup("acetaminophen", "pediatric_dose").o == "15 mg/kg"


def test_factset_lookup_missing_pair_returns_none():
    fs = FactSet.from_dict(FACTS)
    assert fs.lookup("ibuprofen", "max_daily") is None


def test_factset_rejects_fact_with_undeclared_entity():
    bad = {**FACTS, "facts": FACTS["facts"] + [
        {"s": "morphine", "r": "pediatric_dose", "o": "1 mg/kg", "source": "x"}]}
    with pytest.raises(ValidationError, match="morphine"):
        FactSet.from_dict(bad)


def test_factset_rejects_fact_with_undeclared_relation():
    bad = {**FACTS, "facts": FACTS["facts"] + [
        {"s": "ibuprofen", "r": "half_life", "o": "2 h", "source": "x"}]}
    with pytest.raises(ValidationError, match="half_life"):
        FactSet.from_dict(bad)


def test_factset_rejects_quantity_relation_holding_non_quantity():
    bad = {**FACTS, "facts": [
        {"s": "ibuprofen", "r": "pediatric_dose", "o": "lots", "source": "x"}]}
    with pytest.raises(ValidationError, match="quantity"):
        FactSet.from_dict(bad)


def test_factset_rejects_contradictory_duplicate_facts():
    """Two different declared values for the same (s, r) makes the gate incoherent."""
    bad = {**FACTS, "facts": FACTS["facts"] + [
        {"s": "acetaminophen", "r": "pediatric_dose", "o": "20 mg/kg", "source": "x"}]}
    with pytest.raises(ValidationError, match="conflict"):
        FactSet.from_dict(bad)


def test_factset_validates_source_quotes_against_corpus():
    fs = FactSet.from_dict(FACTS)
    corpus = ("Give acetaminophen 15 mg/kg PO q4-6h. "
              "Ibuprofen 10 mg/kg PO q6h for children over 6 months.")
    assert fs.validate_sources(corpus) == (3, [])


def test_factset_reports_unquoted_facts():
    fs = FactSet.from_dict(FACTS)
    ok, missing = fs.validate_sources("Ibuprofen 10 mg/kg PO q6h for children over 6 months.")
    assert ok == 1
    assert len(missing) == 2


# --------------------------------------------------------------------- gate
def test_gate_verifies_correct_claim():
    fs = FactSet.from_dict(FACTS)
    v = gate_claim(fs, "Tylenol", "pediatric_dose", "15 mg/kg")
    assert v.status == VERIFIED


def test_gate_blocks_wrong_dose():
    """The demo moment: a wrong dose must be BLOCKED, never merely unverified."""
    fs = FactSet.from_dict(FACTS)
    v = gate_claim(fs, "acetaminophen", "pediatric_dose", "20 mg/kg")
    assert v.status == BLOCK
    assert v.declared == "15 mg/kg"


def test_gate_holds_unresolvable_entity():
    fs = FactSet.from_dict(FACTS)
    assert gate_claim(fs, "morphine", "pediatric_dose", "1 mg/kg").status == HELD


def test_gate_holds_undeclared_relation_for_known_entity():
    fs = FactSet.from_dict(FACTS)
    assert gate_claim(fs, "ibuprofen", "max_daily", "40 mg/kg").status == HELD


def test_gate_holds_when_entity_is_none():
    """Extraction that could not link is HELD, never PASS. Fail-closed."""
    fs = FactSet.from_dict(FACTS)
    assert gate_claim(fs, None, "pediatric_dose", "15 mg/kg").status == HELD


def test_gate_verdict_carries_source_quote_for_audit():
    fs = FactSet.from_dict(FACTS)
    v = gate_claim(fs, "acetaminophen", "pediatric_dose", "15 mg/kg")
    assert "15 mg/kg" in v.source


# ------------------------------------------- domain-declared normalisation
NORM_FACTS = {
    "domain": "n",
    "entities": {"ibuprofen": [], "oxygen saturation": []},
    "relations": {"pediatric_dose": {"kind": "quantity"},
                  "escalation_threshold": {"kind": "quantity"}},
    # Qualifiers the DOMAIN declares irrelevant to a value. Route and frequency do not
    # change what "10 mg/kg" means; "per day" would, so it is deliberately absent.
    "value_qualifiers": ["PO", "IV", "IM", r"every \d+ (?:to \d+ )?hours?", r"q\d+h",
                         "over 1 hour", r"for children over \d+ months?"],
    "unit_aliases": {"%": "percent"},
    "facts": [
        {"s": "ibuprofen", "r": "pediatric_dose", "o": "10 mg/kg", "source": "x"},
        {"s": "oxygen saturation", "r": "escalation_threshold", "o": "92 percent",
         "source": "y"},
    ],
}


def test_declared_qualifier_is_stripped_so_a_correct_value_verifies():
    """Measured: the model answered "10 mg/kg PO every 6 hours" -- correct -- and the
    gate held it, because the route and frequency corrupted the parsed unit. A domain
    may declare which trailing qualifiers are irrelevant to its values."""
    fs = FactSet.from_dict(NORM_FACTS)
    v = gate_claim(fs, "ibuprofen", "pediatric_dose", "10 mg/kg PO every 6 hours")
    assert v.status == VERIFIED


def test_undeclared_qualifier_is_still_held_not_verified():
    """"per day" is NOT declared, and it changes the meaning, so it must stay HELD.
    Stripping unknown trailing text would be the unsafe direction."""
    fs = FactSet.from_dict(NORM_FACTS)
    assert gate_claim(fs, "ibuprofen", "pediatric_dose", "10 mg/kg per day").status == HELD


def test_declared_qualifier_does_not_rescue_a_wrong_value():
    fs = FactSet.from_dict(NORM_FACTS)
    assert gate_claim(fs, "ibuprofen", "pediatric_dose", "20 mg/kg PO").status == BLOCK


def test_declared_unit_alias_verifies():
    """Measured: the model answered "92%" against a declared "92 percent"."""
    fs = FactSet.from_dict(NORM_FACTS)
    assert gate_claim(fs, "oxygen saturation", "escalation_threshold", "92%").status \
        == VERIFIED


def test_unit_alias_does_not_collapse_different_magnitudes():
    fs = FactSet.from_dict(NORM_FACTS)
    assert gate_claim(fs, "oxygen saturation", "escalation_threshold", "88%").status \
        == BLOCK


def test_wording_quoted_from_the_source_verifies_without_being_declared():
    """BEHAVIOUR CHANGE, deliberate. This previously asserted HELD, on the principle that
    a fact set declaring no qualifiers should gain no new stripping behaviour.

    It is not stripping. The source reads "Give acetaminophen 15 mg/kg PO q4-6h." and the
    claim is "15 mg/kg PO" -- the model quoted the document. Requiring the author to
    declare "PO" before a correct reading could be confirmed is precisely the authoring
    burden that produced the measured over-block, and the fact's own source sentence is
    already a declared, validated part of the fact set.

    The verdict records which route it took, so an auditor can tell the two apart."""
    fs = FactSet.from_dict(FACTS)
    v = gate_claim(fs, "acetaminophen", "pediatric_dose", "15 mg/kg PO")
    assert v.status == VERIFIED
    assert "quoted from" in v.reason


def test_wording_the_source_does_not_state_is_still_held():
    """The other half of the contract, and the reason the change is safe: a route the
    source never mentions is not admitted just because it is short and plausible."""
    fs = FactSet.from_dict(FACTS)
    assert gate_claim(fs, "acetaminophen", "pediatric_dose", "15 mg/kg IV").status == HELD
    assert gate_claim(fs, "acetaminophen", "pediatric_dose",
                      "15 mg/kg per day").status == HELD


# ---------------------------------------------------------- fact-set linting
def test_lint_flags_a_qualifier_that_collapses_two_declared_values():
    """A declared qualifier that makes two DISTINCT facts normalise to the same string is
    provably unsafe: the gate would verify one against the other. This is the concrete
    form of the "per day" footgun -- stripping it turns a daily total into a per-dose
    value."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"drug": []},
        "relations": {"dose": {"kind": "quantity"}, "daily_max": {"kind": "quantity"}},
        "value_qualifiers": ["per day"],
        "facts": [{"s": "drug", "r": "dose", "o": "10 mg/kg", "source": "x"},
                  {"s": "drug", "r": "daily_max", "o": "10 mg/kg per day", "source": "y"}]})
    errors = [p for p in fs.lint() if p["level"] == "error"]
    assert errors, "collapsing qualifier must be an error"
    assert "per day" in errors[0]["message"]


def test_lint_warns_on_rate_bearing_qualifiers():
    """Time words usually change what a value means per unit time, so declaring one
    irrelevant deserves a warning even when nothing collides today."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"drug": []},
        "relations": {"dose": {"kind": "quantity"}},
        "value_qualifiers": ["daily", "PO"],
        "facts": [{"s": "drug", "r": "dose", "o": "10 mg/kg", "source": "x"}]})
    warns = [p for p in fs.lint() if p["level"] == "warning"]
    assert any("daily" in w["message"] for w in warns)
    assert not any("PO" in w["message"] for w in warns)


def test_lint_is_clean_for_a_safe_domain():
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"drug": []},
        "relations": {"dose": {"kind": "quantity"}},
        "value_qualifiers": ["PO", "IV"],
        "facts": [{"s": "drug", "r": "dose", "o": "10 mg/kg", "source": "x"}]})
    assert fs.lint() == []


def test_shipped_demo_domains_have_no_lint_errors():
    """The published demos must not ship a provably unsafe qualifier."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data" / "domains"
    for f in sorted(root.glob("*.json")):
        fs = FactSet.from_dict(json.loads(f.read_text(encoding="utf-8")))
        errors = [p for p in fs.lint() if p["level"] == "error"]
        assert errors == [], f"{f.name}: {errors}"


# ------------------------------------------------------------- audit trail
def test_factset_fingerprint_is_stable_and_order_independent():
    """Compliance needs to prove which fact set produced a verdict. Reordering facts is
    not a change; altering a value is."""
    a = FactSet.from_dict(FACTS)
    reordered = {**FACTS, "facts": list(reversed(FACTS["facts"]))}
    assert FactSet.from_dict(reordered).fingerprint == a.fingerprint


def test_factset_fingerprint_changes_when_a_value_changes():
    changed = json.loads(json.dumps(FACTS))
    changed["facts"][0]["o"] = "16 mg/kg"
    assert FactSet.from_dict(changed).fingerprint != FactSet.from_dict(FACTS).fingerprint


def test_factset_fingerprint_changes_when_a_qualifier_changes():
    """Qualifiers change verdicts, so they must be inside the fingerprint."""
    a = FactSet.from_dict({**FACTS, "value_qualifiers": ["PO"]})
    b = FactSet.from_dict({**FACTS, "value_qualifiers": ["PO", "per day"]})
    assert a.fingerprint != b.fingerprint


def test_verdict_carries_the_fingerprint():
    fs = FactSet.from_dict(FACTS)
    v = gate_claim(fs, "acetaminophen", "pediatric_dose", "15 mg/kg")
    assert v.factset_fingerprint == fs.fingerprint
    assert len(v.factset_fingerprint) == 16


# --------------------------------------------------------------- currency
@pytest.mark.parametrize("raw,val,unit", [
    ("$199", 199.0, "usd"),
    ("$1,200", 1200.0, "usd"),
    ("$14.50", 14.5, "usd"),
    ("199 dollars", 199.0, "usd"),
    ("199 USD", 199.0, "usd"),
])
def test_currency_parses_with_symbol_or_word(raw, val, unit):
    """MEASURED FAILURE on a real business document: every price is written "$199", the
    parser required the number first, so every comparison came back INCOMPARABLE and the
    gate held 11 of 11 correct values. Currency is universal in business prose."""
    assert parse_quantity(raw) == Quantity(val, unit)


@pytest.mark.parametrize("raw,val", [
    ("$150M", 150_000_000.0),
    ("$32.9M", 32_900_000.0),
    ("$471M", 471_000_000.0),
    ("$5k", 5_000.0),
    ("150 million dollars", 150_000_000.0),
    ("32.9 million dollars", 32_900_000.0),
])
def test_magnitude_suffixes_expand(raw, val):
    """"$150M" and "150 million dollars" must compare equal, or a declared value can
    never match how the document writes it."""
    q = parse_quantity(raw)
    assert q is not None and q.value == val and q.unit == "usd"


def test_bare_magnitude_suffix_is_not_expanded():
    """"150M" without a currency or a spelled-out magnitude is ambiguous (metres? milli?)
    and must NOT be silently multiplied."""
    q = parse_quantity("150M")
    assert q is None or q.value == 150.0


def test_currency_still_discriminates():
    assert values_match("$199", "$199") is True
    assert values_match("$199", "$198") is False
    assert values_match("$150M", "150 million dollars") is True
    assert values_match("$150M", "$150") is False


def test_currency_relation_accepts_symbol_form_at_load():
    """The natural form a business author writes must load, not error."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"x": []},
        "relations": {"price": {"kind": "quantity"}},
        "facts": [{"s": "x", "r": "price", "o": "$199", "source": "q"}]})
    assert gate_claim(fs, "x", "price", "$199").status == VERIFIED
    assert gate_claim(fs, "x", "price", "$398").status == BLOCK


# ------------------------------------------------- corruption surface form
def test_corrupt_preserves_the_written_notation():
    """Currency expansion broke the corruption generator: "$150M" scaled to 3e+08 and was
    emitted as "3e+08usd", which the model could not substitute, so it rewrote a
    DIFFERENT number and left the value under test correct. Eight trials were then
    mislabeled as leaks. Corruption must stay in the document's own notation."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "rdb", Path(__file__).resolve().parents[1] / "scripts" / "run_domain_bench.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.corrupt("$150M", 0) == "$300M"
    assert m.corrupt("$199", 0) == "$398"
    assert m.corrupt("15 mg/kg", 0) == "30 mg/kg"
    assert "e+" not in m.corrupt("$32.9M", 2)


# ------------------------------------- currency with trailing rate qualifiers
def test_currency_with_trailing_text_falls_back_to_the_leading_quantity():
    """"$15/mo" must not be unparseable. The currency regex is anchored, so any trailing
    text killed it AND the leading-quantity fallback (which required a digit first),
    leaving every rate-suffixed price INCOMPARABLE."""
    assert compare_values("$15", "$15/mo") == INCOMPARABLE   # undeclared suffix -> hold
    assert compare_values("$15", "$30/mo") == DIFFER         # wrong magnitude -> block


def test_domain_may_declare_a_non_word_qualifier():
    """"/mo" starts with a non-word character, so a \b-anchored pattern could never
    match it -- every monthly price stayed held no matter what the domain declared."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"newsletter": []},
        "relations": {"monthly_price": {"kind": "quantity"}},
        "value_qualifiers": ["/mo", "/month", "per user", "one-time", "download"],
        "facts": [{"s": "newsletter", "r": "monthly_price", "o": "$15", "source": "q"}]})
    assert gate_claim(fs, "newsletter", "monthly_price", "$15/mo").status == VERIFIED
    assert gate_claim(fs, "newsletter", "monthly_price", "$15/month").status == VERIFIED
    assert gate_claim(fs, "newsletter", "monthly_price", "$30/mo").status == BLOCK


def test_qualifier_stripping_cleans_up_its_own_empty_brackets():
    """Stripping a qualifier out of "$79(download)" left "$79( )", which parses as
    nothing. Removing an emptied bracket is tidying the strip's own residue, not
    inferring meaning."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"pack": []},
        "relations": {"price": {"kind": "quantity"}},
        "value_qualifiers": ["download"],
        "facts": [{"s": "pack", "r": "price", "o": "$79", "source": "q"}]})
    assert fs.normalise_value("$79(download)") == "$79"
    assert gate_claim(fs, "pack", "price", "$79(download)").status == VERIFIED
    assert gate_claim(fs, "pack", "price", "$99(download)").status == BLOCK


@pytest.mark.parametrize("qualifier", ["+ benefits", "*starred", "?maybe"])
def test_qualifiers_that_are_not_valid_regex_are_treated_as_literals(qualifier):
    """Found while declaring a real financial model: "+ benefits" is ordinary English but
    an invalid regex ("nothing to repeat"), and the domain refused to load with a regex
    parser error. An author writing plain text should not have to know the field is a
    regex; an uncompilable pattern falls back to a literal, which can never be wrong.

    A string that IS a valid regex (e.g. "C++") stays a pattern -- intent is
    undecidable there -- and raises a qualifier_warning instead."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"x": []},
        "relations": {"price": {"kind": "quantity"}},
        "value_qualifiers": [qualifier],
        "facts": [{"s": "x", "r": "price", "o": "$5", "source": "q"}]})
    assert fs.normalise_value(f"$5 {qualifier}") == "$5"


def test_valid_regex_qualifiers_still_behave_as_patterns():
    """The fallback must not turn working patterns into literals."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"x": []},
        "relations": {"price": {"kind": "quantity"}},
        "value_qualifiers": [r"every \d+ hours?"],
        "facts": [{"s": "x", "r": "price", "o": "$5", "source": "q"}]})
    assert fs.normalise_value("$5 every 6 hours") == "$5"
    assert fs.normalise_value("$5 every 12 hour") == "$5"


def test_metacharacter_qualifier_that_compiles_raises_a_warning():
    """"C++" is a valid regex meaning "C then one-or-more +", so it strips only "C+" from
    "$5 C++". Intent is undecidable, so the author is told rather than guessed at."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"x": []},
        "relations": {"price": {"kind": "quantity"}},
        "value_qualifiers": ["C++"],
        "facts": [{"s": "x", "r": "price", "o": "$5", "source": "q"}]})
    assert any("C++" in w for w in fs.qualifier_warnings)


def test_plain_qualifiers_raise_no_warning():
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"x": []},
        "relations": {"price": {"kind": "quantity"}},
        "value_qualifiers": ["one-time", "download"],
        "facts": [{"s": "x", "r": "price", "o": "$5", "source": "q"}]})
    assert fs.qualifier_warnings == []


# ------------------------------------------- wrong-slot answers are category errors
def test_a_duration_answer_against_a_currency_fact_is_incomparable():
    """Measured: asked "how much is the raise?" on a passage about runway, the extractor
    answered "18 months" -- a category error, not a competing amount. Comparing it as a
    value is meaningless; the honest verdict is that it cannot be compared."""
    assert compare_values("$5M-$10M", "18 months") == INCOMPARABLE
    assert compare_values("$199", "18 months") == INCOMPARABLE


def test_a_currency_answer_against_a_non_currency_fact_is_incomparable():
    assert compare_values("18 months", "$199") == INCOMPARABLE
    assert compare_values("60 breaths per minute", "$30") == INCOMPARABLE


def test_same_category_comparisons_are_unaffected():
    """The guard must not touch the cases that already worked."""
    assert compare_values("$199", "$398") == DIFFER
    assert compare_values("$199", "$199") == MATCH
    assert compare_values("15 mg/kg", "30 mg/kg") == DIFFER

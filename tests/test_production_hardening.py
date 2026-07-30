"""Failures that only appear when someone other than the author operates the gate.

A fact set is DATA. In any deployment where the person who writes it is not the person who
runs it -- a customer uploading their own domain, a fact set pulled from a registry, a
config file edited by an analyst -- it is untrusted input, and every defect below was
reachable by writing an ordinary-looking JSON file.

Every case here was reproduced before it was fixed; the docstrings record the measurement.
"""
import time

import pytest

from factgate.domain.factset import FactSet, ValidationError
from factgate.domain.gate import VERIFIED, gate_claim
from factgate.domain.link import respell_from_passage, value_is_grounded

BASE = {"domain": "d", "entities": {"e": ["e"]},
        "relations": {"p": {"kind": "quantity"}},
        "facts": [{"s": "e", "r": "p", "o": "5 mg", "source": "It is 5 mg."}]}


# --------------------------------------------------------------- denial of service
@pytest.mark.parametrize("pattern", ["(a+)+b", "(a*)*b", r"(\d+)+", "([a-z]+)*x",
                                     r"(\w+\s?)+$"])
def test_exponential_qualifier_patterns_are_refused_at_load(pattern):
    """MEASURED before the fix: the qualifier "(a+)+b" took 4.84 SECONDS to normalise a
    26-character value, doubling per added character -- 40 characters is hours. Author
    strings went straight into re.compile, so a fact set was a denial-of-service payload
    delivered as data."""
    with pytest.raises(ValidationError, match="unbounded quantifier"):
        FactSet.from_dict({**BASE, "value_qualifiers": [pattern]})


@pytest.mark.parametrize("pattern", [r"every \d+ (?:to \d+ )?hours?", r"q\d+h", "PO",
                                     "/mo", "per year", "C++", "(ab|cd)+", "a{2,4}"])
def test_legitimate_qualifier_patterns_still_load(pattern):
    """The power is deliberate -- real clinical domains declare frequency patterns -- so
    the fix rejects the dangerous SHAPE, not regexes in general. "(?:to \\d+ )?" is safe
    because `?` matches at most once and cannot compound."""
    FactSet.from_dict({**BASE, "value_qualifiers": [pattern]})


def test_a_rejected_pattern_costs_nothing_to_reject():
    """The check is a linear scan that never executes the pattern."""
    start = time.perf_counter()
    for _ in range(200):
        with pytest.raises(ValidationError):
            FactSet.from_dict({**BASE, "value_qualifiers": ["(a+)+b"]})
    assert time.perf_counter() - start < 1.0


def test_an_absurdly_long_value_does_not_run_qualifier_matching():
    fs = FactSet.from_dict({**BASE, "value_qualifiers": ["PO", "per day", "monthly"]})
    huge = "5 mg " + "x" * 100_000
    start = time.perf_counter()
    assert fs.normalise_value(huge) == huge
    assert time.perf_counter() - start < 0.5


# ------------------------------------------------- structurally wrong fact sets
def test_a_bare_string_where_aliases_belong_is_refused():
    """SILENTLY ACCEPTED before the fix, and a correctness bug rather than an ergonomic
    one: Python iterates the string, so entities={"acetaminophen": "paracetamol"} gave the
    drug the aliases 'a','c','e','l','m','o','p','r','t'. Measured:
    gate_claim(fs, "a", "dose", "15 mg/kg") returned VERIFIED. One stray letter in a
    response resolved to a specific drug."""
    with pytest.raises(ValidationError, match="character by character"):
        FactSet.from_dict({**BASE, "entities": {"acetaminophen": "paracetamol"}})


def test_a_bare_string_where_qualifiers_belong_is_refused():
    """Also silently accepted, compiling the qualifiers \\bP\\b and \\bO\\b and stripping
    standalone letters out of every value in the domain."""
    with pytest.raises(ValidationError, match="character by character"):
        FactSet.from_dict({**BASE, "value_qualifiers": "PO"})


@pytest.mark.parametrize("spec,field", [
    ({"entities": ["e"]}, "entities"),
    ({"relations": {"p": "quantity"}}, "relations"),
    ({"facts": {"s": "e"}}, "facts"),
    ({"unit_aliases": ["x"]}, "unit_aliases"),
    ({"conditions": "tier"}, "conditions"),
    ({"domain": 7}, "domain"),
])
def test_malformed_structure_raises_validation_error_naming_the_field(spec, field):
    """Previously AttributeError/TypeError from inside the library ("'list' object has no
    attribute 'items'"), which a service wrapping this cannot catch by contract or explain
    to whoever sent the request."""
    with pytest.raises(ValidationError, match=field):
        FactSet.from_dict({**BASE, **spec})


def test_a_non_object_fact_set_is_refused():
    with pytest.raises(ValidationError):
        FactSet.from_dict("not a fact set")


def test_empty_spec_still_loads_as_an_empty_gate():
    """The shape check must not break the deliberate empty-gate default, which is safe:
    no facts means everything is HELD."""
    fs = FactSet.from_dict({})
    assert fs.facts == [] and fs.resolve_entity("anything") is None


# ---------------------------------------------------------- caller-facing types
@pytest.mark.parametrize("bad", [5, ["5 mg"], {"v": 1}, b"5 mg", 3.4])
def test_non_string_claim_raises_type_error_not_an_internal_crash(bad):
    """A service passes whatever its JSON contained. These previously surfaced as
    "object of type 'int' has no len()" from inside the quantity parser."""
    fs = FactSet.from_dict(BASE)
    with pytest.raises(TypeError, match="claimed_value must be a string"):
        gate_claim(fs, "e", "p", bad)


def test_non_dict_context_raises_type_error():
    fs = FactSet.from_dict(BASE)
    with pytest.raises(TypeError, match="context must be a dict"):
        gate_claim(fs, "e", "p", "5 mg", context=["tier", "standard"])


# ------------------------------------------------------- grounding the UNIT too
def test_a_fabricated_unit_is_not_grounded_by_a_real_number():
    """Found by probing the guard directly. It checked only that the NUMBER occurred in
    the passage, so any unit at all rode along. Harmless on its own -- a fabricated unit
    does not match the declared one -- but in a domain declaring unit_aliases, a passage
    reading "500 units of stock", an extractor answering "500 usd", and an alias
    usd -> dollars against a declared "500 dollars" VERIFIES a currency the document never
    mentions."""
    passage = "The account holds 500 units of stock."
    assert value_is_grounded("500 zorkmids", passage) is False
    assert value_is_grounded("500 dollars", passage) is False


@pytest.mark.parametrize("value,passage", [
    ("92 percent", "Escalate below 92% saturation."),
    ("150 million dollars", "Acme Corp ($150M ARR)"),
    ("$79", "the Skill Pack is $79 download."),
    ("7.5 mg/kg", "Give acetaminophen 7.5 mg/kg PO now."),
    ("38 degrees celsius", "Record 38 degrees celsius or above."),
    ("12 weeks", "It ships in 12 weeks."),
])
def test_units_spelled_differently_are_still_grounded(value, passage):
    """Requiring the unit is only safe because spelling genuinely varies and the check is
    alias-aware: "$" for dollars, "%" for percent, singular for plural."""
    assert value_is_grounded(value, passage) is True


# -------------------------------------------------- the document's own spelling
def test_a_glued_extraction_is_respelled_from_the_passage():
    """MEASURED, three runs out of three at temperature 0: asked for the minimum balance,
    qwen2.5:14b answers "500 dollarsmonthly" -- the right words with the space gone. The
    qualifier patterns are word-bounded, so nothing could strip "monthly" out of
    "dollarsmonthly", and a correct reading was held over one missing space."""
    passage = ("The savings account requires a minimum balance of 500 dollars monthly "
               "to avoid the service charge.")
    assert respell_from_passage("500 dollarsmonthly", passage) == "500 dollars monthly"
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"savings account": []},
        "relations": {"minimum_balance": {"kind": "quantity"}},
        "value_qualifiers": ["monthly"],
        "facts": [{"s": "savings account", "r": "minimum_balance", "o": "500 dollars",
                   "source": passage}]})
    assert gate_claim(fs, "savings account", "minimum_balance",
                      respell_from_passage("500 dollarsmonthly", passage)).status == VERIFIED


@pytest.mark.parametrize("value", ["500 euros", "900 dollarsmonthly", "50 dollars"])
def test_respelling_never_invents_a_value_the_passage_lacks(value):
    """Only whitespace may differ; every other character must match in order, so this can
    never turn one value into a different one."""
    passage = "The savings account requires a minimum balance of 500 dollars monthly."
    assert respell_from_passage(value, passage) == value


# ---------------------------------------------------------------- authoring aid
def test_lint_flags_a_fact_its_own_source_states_more_precisely():
    """REAL, from the shipped bench data: declared "$100M" against the source "GPT-4 cost
    ~$100M+.". The document says AT LEAST $100M; the declaration says exactly $100M. A
    model that reads the document correctly and answers "$100M+" is then held, because a
    range never confirms a point. The gate is right and the fact set is wrong -- the
    hardest kind of over-block for an author to diagnose, so the library says it."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"gpt-4": []},
        "relations": {"training_cost": {"kind": "quantity"}},
        "facts": [{"s": "gpt-4", "r": "training_cost", "o": "$100M",
                   "source": "GPT-4 cost ~$100M+."}]})
    warned = [p for p in fs.lint() if "open-ended range" in p["message"]]
    assert warned and "$100M+" in warned[0]["message"]


def test_lint_does_not_flag_a_declaration_that_already_matches_its_source():
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"gpt-4": []},
        "relations": {"training_cost": {"kind": "quantity"}},
        "facts": [{"s": "gpt-4", "r": "training_cost", "o": "$100M+",
                   "source": "GPT-4 cost ~$100M+."}]})
    assert [p for p in fs.lint() if "open-ended range" in p["message"]] == []


def test_a_currency_amount_with_a_worded_unit_is_still_grounded():
    """REGRESSION, caught by re-running the benchmark rather than by a test. Requiring the
    unit to appear in the passage broke currency values carrying worded units: parse_quantity
    renders "$30 per million tokens" as the single glued unit "usdpermilliontokens", where
    "usd" is a canonical code the document never spells. Two verdicts went from VERIFIED to
    no-claim-at-all. The currency and the wording are now checked separately."""
    assert value_is_grounded(
        "$30 per million tokens",
        "LLM inference costs approximately $30 per million tokens.") is True
    assert value_is_grounded(
        "$0.50 per million tokens",
        "Widget inference costs approximately $0.50 per million tokens.") is True
    # and the hole this whole check exists to close stays closed
    assert value_is_grounded(
        "$30 per billion tokens",
        "LLM inference costs approximately $30 per million tokens.") is False

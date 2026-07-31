"""Tests for qualifier suggestion.

This is tooling, not a verdict path: it must never change an outcome, and it must not
propose text whose removal would not actually rescue the claim.
"""
from factgate.domain.factset import FactSet
from factgate.domain.suggest import suggest_qualifiers

FACTS = {
    "domain": "d",
    "entities": {"rain": [], "llm": []},
    "relations": {"daily_cost": {"kind": "quantity"}, "dose": {"kind": "quantity"}},
    "facts": [
        {"s": "rain", "r": "daily_cost", "o": "$1.50", "source": "q"},
        {"s": "llm", "r": "daily_cost", "o": "$90", "source": "q"},
    ],
}


def test_suggests_the_trailing_text_that_caused_a_hold():
    fs = FactSet.from_dict(FACTS)
    out = suggest_qualifiers(fs, [("rain", "daily_cost", "$1.50 per customer query")])
    assert out and out[0]["qualifier"] == "per customer query"
    assert out[0]["example_slot"] == "rain/daily_cost"


def test_counts_repeats_so_the_common_ones_surface_first():
    fs = FactSet.from_dict(FACTS)
    out = suggest_qualifiers(fs, [
        ("rain", "daily_cost", "$1.50 per customer query"),
        ("llm", "daily_cost", "$90 per customer query"),
        ("rain", "daily_cost", "$1.50 approximately"),
    ])
    assert out[0]["qualifier"] == "per customer query"
    assert out[0]["occurrences"] == 2


def test_does_not_suggest_text_whose_removal_would_not_rescue_the_claim():
    """A wrong value is not a qualifier problem. "$9.00 per query" against a declared
    "$1.50" must not produce a suggestion, or the author would be nudged toward declaring
    away a real contradiction."""
    fs = FactSet.from_dict(FACTS)
    assert suggest_qualifiers(fs, [("rain", "daily_cost", "$9.00 per query")]) == []


def test_flags_time_and_basis_wording_as_risky():
    """"per day" is exactly the footgun: declaring it irrelevant on a per-dose value makes
    a wrong value verify. Suggested, but never silently."""
    fs = FactSet.from_dict(FACTS)
    out = suggest_qualifiers(fs, [("rain", "daily_cost", "$1.50 per day")])
    assert out and out[0]["warning"] is not None


def test_plain_wording_is_not_flagged():
    fs = FactSet.from_dict(FACTS)
    out = suggest_qualifiers(fs, [("rain", "daily_cost", "$1.50 flat")])
    assert out and out[0]["warning"] is None


def test_already_matching_claims_produce_nothing():
    fs = FactSet.from_dict(FACTS)
    assert suggest_qualifiers(fs, [("rain", "daily_cost", "$1.50")]) == []


def test_unknown_slot_is_ignored():
    fs = FactSet.from_dict(FACTS)
    assert suggest_qualifiers(fs, [("nobody", "daily_cost", "$1.50 per query")]) == []


def test_suggestions_render_as_a_declarable_block():
    """Production use means copy-paste, not a hand-written join script. The renderer
    emits the JSON fragment to paste into value_qualifiers, with risky items commented."""
    from factgate.domain.suggest import render_suggestions
    fs = FactSet.from_dict(FACTS)
    out = render_suggestions(fs, [
        ("rain", "daily_cost", "$1.50 per customer query"),
        ("rain", "daily_cost", "$1.50 flat"),
    ])
    assert '"flat"' in out
    assert "per customer query" in out
    assert "REVIEW" in out          # risky items are flagged, not silently included


def test_renderer_says_so_when_there_is_nothing_to_suggest():
    from factgate.domain.suggest import render_suggestions
    fs = FactSet.from_dict(FACTS)
    assert "no qualifier" in render_suggestions(fs, []).lower()


def test_suggests_a_leading_qualifier():
    """Text BEFORE the value is a qualifier too, and a common one: "within 1 hour",
    "up to 30 days", "at least 90 percent". Only trailing residue was proposed, so a SaaS
    contract answered "within 1 hour of submission" produced NO suggestion at all even
    though one word stood between it and a verdict."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"sev1": []},
        "relations": {"response_time": {"kind": "quantity"}},
        "value_qualifiers": ["of submission"],
        "facts": [{"s": "sev1", "r": "response_time", "o": "1 hour",
                   "source": "Severity 1 response time is within 1 hour of submission."}]})
    out = suggest_qualifiers(fs, [("sev1", "response_time",
                                   "within 1 hour of submission")])
    assert [i["qualifier"] for i in out] == ["within"]


def test_suggests_both_sides_when_both_are_needed():
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"sev1": []},
        "relations": {"response_time": {"kind": "quantity"}},
        "facts": [{"s": "sev1", "r": "response_time", "o": "1 hour",
                   "source": "Severity 1 response time is within 1 hour of submission."}]})
    out = {i["qualifier"] for i in suggest_qualifiers(
        fs, [("sev1", "response_time", "within 1 hour of submission")])}
    assert out == {"within", "of submission"}


def test_prefers_the_smallest_declaration_that_works():
    """Proposing more than is needed asks the author to declare wording irrelevant that
    never mattered, and every such declaration is a chance to make a wrong value verify."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"x": []},
        "relations": {"r": {"kind": "quantity"}},
        "facts": [{"s": "x", "r": "r", "o": "5 mg", "source": "Give 5 mg once."}]})
    out = [i["qualifier"] for i in suggest_qualifiers(fs, [("x", "r", "5 mg once")])]
    assert out == ["once"]

"""Adversarial and edge-case probe of the public API.

Everything here reaches the gate from an LLM, i.e. from untrusted input. The bar is:
never crash, never silently VERIFY something unproven, and never spend unbounded time.
A crash in a gate is a fail-open in practice, because callers wrap it in try/except and
carry on.
"""
import time

import pytest

from factgate.domain.factset import FactSet, ValidationError
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim
from factgate.domain.link import mentioned_entities, normalise_slot_answer
from factgate.domain.quantity import INCOMPARABLE, compare_values, parse_quantity

BASE = {
    "domain": "d",
    "entities": {"aspirin": ["asa"]},
    "relations": {"dose": {"kind": "quantity"}},
    "facts": [{"s": "aspirin", "r": "dose", "o": "5 mg", "source": "Give aspirin 5 mg."}],
}


@pytest.fixture
def fs():
    return FactSet.from_dict(BASE)


# ------------------------------------------------------------ malformed spec
def test_missing_required_key_raises_validation_error_not_keyerror():
    """A malformed fact file must fail with the library's own error type, so callers
    can distinguish 'your data is wrong' from 'the library crashed'."""
    bad = {**BASE, "facts": [{"s": "aspirin", "r": "dose"}]}   # no "o"
    with pytest.raises(ValidationError):
        FactSet.from_dict(bad)


def test_empty_spec_loads_as_empty_gate():
    fs = FactSet.from_dict({})
    assert fs.facts == []
    assert fs.resolve_entity("anything") is None


def test_alias_collision_between_entities_is_rejected():
    """Two entities sharing an alias makes resolution silently arbitrary, which would
    attach a claim to the wrong drug. That must be refused at load time."""
    bad = {**BASE, "entities": {"aspirin": ["asa"], "acetylsalicylic acid": ["asa"]}}
    with pytest.raises(ValidationError, match="asa"):
        FactSet.from_dict(bad)


def test_alias_colliding_with_another_canonical_name_is_rejected():
    bad = {**BASE, "entities": {"aspirin": ["ibuprofen"], "ibuprofen": []}}
    with pytest.raises(ValidationError):
        FactSet.from_dict(bad)


# ------------------------------------------------------------- hostile input
@pytest.mark.parametrize("claimed", [
    "", "   ", None, "\x00", "nan", "inf", "-inf", "1e400",
    "5 mg" + chr(0), "５ mg", "٥ mg",
    "<script>alert(1)</script>", "'; DROP TABLE facts;--",
    "5 mg" * 500,
])
def test_gate_never_crashes_and_never_verifies_on_hostile_value(fs, claimed):
    v = gate_claim(fs, "aspirin", "dose", claimed)
    assert v.status in (BLOCK, HELD)      # crucially, never VERIFIED


@pytest.mark.parametrize("mention", ["", None, "\x00", "a" * 10000, "aspirin" * 100])
def test_gate_never_crashes_on_hostile_entity(fs, mention):
    assert gate_claim(fs, mention, "dose", "5 mg").status in (BLOCK, HELD)


def test_gate_holds_on_unknown_relation_rather_than_raising(fs):
    assert gate_claim(fs, "aspirin", "no_such_relation", "5 mg").status == HELD


def test_nan_and_inf_never_compare_equal():
    """float('nan') == float('nan') is False, so a nan on both sides must not slip
    through as DIFFER-then-BLOCK confusion, and must certainly never MATCH."""
    assert compare_values("5 mg", "nan mg") != "MATCH"
    assert compare_values("5 mg", "inf mg") != "MATCH"


# ------------------------------------------------------------------ no ReDoS
def test_quantity_parse_is_linear_on_adversarial_input():
    """The unit pattern nests quantifiers; a pathological string must not hang.
    An unbounded parse in a request path is a denial of service."""
    evil = "1 " + "a/" * 2000 + "!"
    t0 = time.time()
    parse_quantity(evil)
    assert time.time() - t0 < 1.0


def test_mention_detection_is_linear_on_long_text(fs):
    t0 = time.time()
    mentioned_entities("lorem ipsum " * 20000, fs)
    assert time.time() - t0 < 2.0


# -------------------------------------------------------------- slot answers
@pytest.mark.parametrize("raw", [None, "", "   ", "\n\n", '""'])
def test_empty_slot_answers_are_none(raw):
    assert normalise_slot_answer(raw) is None


def test_slot_answer_of_pure_punctuation_is_none():
    assert normalise_slot_answer("...") is None


# ------------------------------------------------------------ verdict safety
def test_verified_requires_exact_agreement(fs):
    assert gate_claim(fs, "aspirin", "dose", "5 mg").status == VERIFIED
    assert gate_claim(fs, "asa", "dose", "5 mg").status == VERIFIED
    assert gate_claim(fs, "aspirin", "dose", "5 mg/kg").status == HELD
    assert gate_claim(fs, "aspirin", "dose", "50 mg").status == BLOCK


# ------------------------------- found by adversarial review (BLOCKER + MAJOR)
def test_sentence_like_claim_never_blocks_a_text_fact():
    """BLOCKER. For a kind=text relation, a refusal that slipped the slot-answer filter
    became a "value", and the text branch returned DIFFER for any mismatch -> false
    BLOCK. A gate must never report a contradiction on the basis of a hedge."""
    from factgate.domain.quantity import INCOMPARABLE, compare_values
    for hedge in ["The passage never mentions a route",
                  "It is not specified in the provided protocol.",
                  "I could not find that information anywhere"]:
        assert compare_values("oral", hedge) == INCOMPARABLE


def test_genuine_text_contradiction_still_differs():
    """The safety case must survive: a real, short, conflicting value still blocks."""
    from factgate.domain.quantity import DIFFER, compare_values
    assert compare_values("oral", "intravenous") == DIFFER
    assert compare_values("oral", "IV") == DIFFER


def test_slot_answer_rejects_sentence_shaped_text():
    from factgate.domain.link import normalise_slot_answer
    assert normalise_slot_answer("The passage never mentions a route") is None
    assert normalise_slot_answer("It is not specified in the protocol provided") is None
    assert normalise_slot_answer("oral") == "oral"
    assert normalise_slot_answer("15 mg/kg") == "15 mg/kg"
    assert normalise_slot_answer("60 breaths per minute") == "60 breaths per minute"


def test_mention_detection_survives_linebreaks_and_hyphens():
    """MAJOR. Line-wrapped and hyphenated entity names are ordinary in real documents;
    missing them is a silent coverage hole that surfaces as HELD, not as an error."""
    from factgate.domain.factset import FactSet
    from factgate.domain.link import mentioned_entities
    fs = FactSet.from_dict({"domain": "d",
                            "entities": {"oxygen saturation": [], "fluid resuscitation": []},
                            "relations": {}, "facts": []})
    assert mentioned_entities("the oxygen\nsaturation fell", fs) == {"oxygen saturation"}
    assert mentioned_entities("fluid-resuscitation protocol", fs) == {"fluid resuscitation"}
    assert mentioned_entities("oxygen   saturation", fs) == {"oxygen saturation"}


@pytest.mark.parametrize("refusal", [
    "Not provided.", "Not available.", "Not applicable.", "Not given.",
    "Unclear.", "Unspecified.", "No value.", "Unknown.", "N/A",
    "not stated in the passage", "None of the above",
])
def test_short_refusals_never_become_values(refusal):
    """BLOCKER, found by adversarial review AFTER a first fix. Two-word refusals like
    "Not provided." were short enough to pass the value-shape whitelist and did not match
    the absence blacklist, so they reached the comparator and BLOCKED a text-kind fact:
    the gate asserting the passage CONTRADICTS the protocol when the model simply
    declined to answer."""
    from factgate.domain.link import normalise_slot_answer
    assert normalise_slot_answer(refusal) is None


@pytest.mark.parametrize("refusal", [
    "Not provided", "Not applicable", "Unclear", "Unknown", "No value"])
def test_absence_phrases_are_incomparable_not_a_contradiction(refusal):
    """Defence in depth: compare_values is public API, so a caller passing raw text must
    not get a contradiction verdict out of an absence marker either."""
    from factgate.domain.quantity import INCOMPARABLE, compare_values
    assert compare_values("oral", refusal) == INCOMPARABLE


def test_legitimate_short_text_values_still_pass():
    from factgate.domain.link import normalise_slot_answer
    for good in ["oral", "intravenous", "IM", "twice daily", "15 mg/kg"]:
        assert normalise_slot_answer(good) == good

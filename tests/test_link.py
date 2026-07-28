"""Tests for entity mention detection and targeted slot-filling.

Open linking was measured to fail: asked to emit (s, r, o) freely, llama3.2:3b returned a
null relation on 5 of 7 failures even when it had stated the value correctly. It was not
choosing a wrong identifier (only 1 aliasable string, "dose", appeared), it was declining
to link at all. Since the declared (entity, relation) pairs are known up front, the task
is slot filling with the slot supplied, not open information extraction.

Mention detection is deterministic and therefore tested exhaustively here: it decides
which slots get asked about, so a miss here is a silent coverage hole.
"""
import pytest

from factgate.domain.factset import FactSet
from factgate.domain.link import mentioned_entities, normalise_slot_answer

FACTS = {
    "domain": "t",
    "entities": {"acetaminophen": ["tylenol", "apap"],
                 "epinephrine": ["epi"],
                 "respiratory rate": ["rr"],
                 "ibuprofen": []},
    "relations": {"pediatric_dose": {"kind": "quantity"}},
    "facts": [{"s": "acetaminophen", "r": "pediatric_dose", "o": "15 mg/kg",
               "source": "s"}],
}


@pytest.fixture
def fs():
    return FactSet.from_dict(FACTS)


def test_detects_canonical_mention(fs):
    assert mentioned_entities("Give acetaminophen 15 mg/kg.", fs) == {"acetaminophen"}


def test_detects_alias_mention(fs):
    assert mentioned_entities("Give Tylenol 15 mg/kg.", fs) == {"acetaminophen"}


def test_detection_is_case_insensitive(fs):
    assert mentioned_entities("give APAP now", fs) == {"acetaminophen"}


def test_multiword_entity_is_detected(fs):
    assert mentioned_entities("The respiratory rate is 60.", fs) == {"respiratory rate"}


def test_short_alias_does_not_match_inside_another_word(fs):
    """"rr" must not fire on "carrot", and "epi" must not fire on "epidural".
    A spurious mention wastes a model call and can attach a value to the wrong entity."""
    assert mentioned_entities("The carrot was epidural-adjacent.", fs) == set()


def test_alias_that_is_a_prefix_of_its_own_canonical_still_resolves(fs):
    """"epi" is an alias of "epinephrine"; the full word must resolve to the entity once,
    not be missed because the alias is a substring of it."""
    assert mentioned_entities("Epinephrine 0.01 mg/kg IM.", fs) == {"epinephrine"}


def test_unknown_entity_is_not_detected(fs):
    assert mentioned_entities("Give morphine 1 mg/kg.", fs) == set()


def test_multiple_entities_detected(fs):
    assert mentioned_entities("Tylenol and ibuprofen were both given.", fs) == {
        "acetaminophen", "ibuprofen"}


def test_empty_text_detects_nothing(fs):
    assert mentioned_entities("", fs) == set()


# ------------------------------------------------------- slot answer parsing
@pytest.mark.parametrize("raw", ["NONE", "none", " None ", "", "N/A",
                                 "not stated", "The passage does not state it."])
def test_absent_slot_answers_normalise_to_none(raw):
    assert normalise_slot_answer(raw) is None


@pytest.mark.parametrize("raw,want", [
    ("15 mg/kg", "15 mg/kg"),
    ("  15 mg/kg  ", "15 mg/kg"),
    ('"15 mg/kg"', "15 mg/kg"),
    ("VALUE: 15 mg/kg", "15 mg/kg"),
    ("15 mg/kg.", "15 mg/kg"),
])
def test_present_slot_answers_are_cleaned(raw, want):
    assert normalise_slot_answer(raw) == want


def test_slot_answer_keeps_internal_punctuation():
    assert normalise_slot_answer("0.15 mg/kg") == "0.15 mg/kg"


def test_slot_answer_does_not_invent_on_refusal_phrasing():
    """A chatty refusal must become None, never a value. Treating prose as a value would
    send garbage to the comparator, which could only ever produce HELD or a false BLOCK."""
    assert normalise_slot_answer("I'm sorry, that is not mentioned.") is None


@pytest.mark.parametrize("verbose", [
    "10 mg/kg PO every 6 hours",
    "15 mg/kg every 4 to 6 hours",
    "60 breaths per minute or higher",
])
def test_verbose_but_quantity_led_answers_are_kept(verbose):
    """Measured regression: the value-shape whitelist rejected anything over 3 words, so
    a CORRECT but verbose answer was discarded before the gate saw it -- 4 of the 5
    over-blocked cases in the benchmark. An answer that leads with a number is never a
    refusal, and the gate's own domain normalisation handles the trailing qualifiers."""
    assert normalise_slot_answer(verbose) == verbose


def test_verbose_prose_without_a_leading_number_is_still_rejected():
    assert normalise_slot_answer("The protocol does not appear to state this value") is None


# ------------------------------------------------- extractor grounding check
def test_value_absent_from_passage_is_rejected():
    """MEASURED LEAK. The passage said "7.5 mg/kg"; the extractor returned "15 mg/kg" --
    the declared value, hallucinated into existence -- and the gate then correctly
    VERIFIED a claim the text never made. A parameter-free verdict cannot protect against
    a fabricated input, so the extracted value must be shown to occur in the passage."""
    from factgate.domain.link import value_is_grounded
    passage = "Give acetaminophen 7.5 mg/kg PO every 4 to 6 hours."
    assert value_is_grounded("15 mg/kg", passage) is False
    assert value_is_grounded("7.5 mg/kg", passage) is True


def test_grounding_tolerates_different_unit_spelling():
    """The number must be present; the unit may be spelled differently ("92%" vs
    "92 percent"), which is what the domain's unit_aliases exist to reconcile."""
    from factgate.domain.link import value_is_grounded
    assert value_is_grounded("92 percent", "Escalate below 92% saturation.") is True


def test_grounding_does_not_match_a_number_inside_a_longer_number():
    from factgate.domain.link import value_is_grounded
    assert value_is_grounded("15 mg", "The total was 115 mg.") is False
    assert value_is_grounded("5 mg", "Dose 15 mg daily.") is False


def test_grounding_of_text_values_is_substring_based():
    from factgate.domain.link import value_is_grounded
    assert value_is_grounded("oral", "Give it by the ORAL route.") is True
    assert value_is_grounded("intravenous", "Give it by the oral route.") is False


# ---------------------------------------------------- slot question building
def test_domain_declared_question_is_used():
    """Phrasing is domain knowledge: "what dose" reads naturally, "what is the
    pediatric_dose of" does not, and the difference was measured to matter (1/6 -> 3/6
    on llama3.2:3b, 4/6 -> 6/6 on qwen2.5:14b)."""
    from factgate.domain.factset import FactSet
    from factgate.domain.link import slot_question
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"ibuprofen": []},
        "relations": {"pediatric_dose": {"kind": "quantity",
                                         "question": "what is the dose of {entity}?"}},
        "facts": [{"s": "ibuprofen", "r": "pediatric_dose", "o": "1 mg", "source": "x"}]})
    assert slot_question(fs, "ibuprofen", "pediatric_dose") == "what is the dose of ibuprofen?"


def test_question_falls_back_to_a_generic_phrasing():
    from factgate.domain.factset import FactSet
    from factgate.domain.link import slot_question
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"x": []},
        "relations": {"half_life": {"kind": "quantity"}},
        "facts": [{"s": "x", "r": "half_life", "o": "1 h", "source": "s"}]})
    assert slot_question(fs, "x", "half_life") == "what is the half life of x?"


@pytest.mark.parametrize("answer", [
    "$199 one-time download", "$39 per user monthly", "$79 download", "$15/mo",
])
def test_currency_led_verbose_answers_are_kept(answer):
    """MEASURED on a real business document: the leads-with-a-number check required a
    DIGIT, so "$39 per user monthly" (4 words) was dropped by the length rule before the
    gate saw it. Currency is symbol-first; a value starting with $ is no more a refusal
    than one starting with 3."""
    assert normalise_slot_answer(answer) == answer


def test_slot_answer_takes_the_first_line_before_any_commentary():
    """Measured: qwen2.5:14b answered "$79(download)\n\nNote: Since the Pro tier was
    cancelled..." -- a correct value followed by unsolicited commentary, which the prose
    filter then killed entirely. The value is on the first line; the essay after it is
    not part of the answer."""
    assert normalise_slot_answer("$79 (download)\n\nNote: Since the tier was cancelled.") \
        == "$79 (download)"
    assert normalise_slot_answer("15 mg/kg\nThis is the standard dose.") == "15 mg/kg"


def test_first_line_rule_does_not_rescue_a_refusal():
    assert normalise_slot_answer("NONE\n\nThe passage does not state it.") is None
    assert normalise_slot_answer("Not provided.\nSee section 2.") is None


# ------------------- entity matching, from a first-time author's declaration
def _fs(entities):
    from factgate.domain.factset import FactSet
    return FactSet.from_dict({
        "domain": "d", "entities": entities,
        "relations": {"p": {"kind": "quantity"}},
        "facts": [{"s": list(entities)[0], "r": "p", "o": "$5", "source": "q"}]})


def test_spacing_around_punctuation_is_normalised():
    """The author copied "Agency A / Agency B Grant" from a table cell; the model wrote
    "Agency A/Agency B Grant". Same name, different spacing around a slash -- the same class as the
    hyphen and line-break normalisation already applied."""
    fs = _fs({"grant_programme": ["Agency A / Agency B Grant"]})
    assert mentioned_entities("The ask from Agency A/Agency B Grant is $250k.", fs) == {"grant_programme"}
    assert mentioned_entities("The ask from Agency A / Agency B Grant is $250k.", fs) == {"grant_programme"}


def test_trailing_parenthetical_in_an_alias_is_optional():
    """"Pre-seed angels (early stage)" -- the parenthetical is the author disambiguating
    a table row, not part of what anyone calls the thing."""
    fs = _fs({"early_investors": ["Pre-seed angels (early stage)"]})
    assert mentioned_entities("Pre-seed angels take 5-15% equity.", fs) == {"early_investors"}
    assert mentioned_entities("Pre-seed angels (early stage) take 5%.", fs) == {"early_investors"}


def test_multipart_alias_matches_when_all_parts_are_present():
    """"Alpha Fund / Beta Fund / Gamma Partners" vs "Alpha Fund, Beta Fund, and Gamma
    Partners" -- a list joined differently. ALL parts must appear, so a stray mention of one
    common word cannot fire."""
    fs = _fs({"seed_funds": ["Alpha Fund / Beta Fund / Gamma Partners"]})
    assert mentioned_entities(
        "Alpha Fund, Beta Fund, and Gamma Partners ask for $500k-2M.", fs) == {"seed_funds"}


def test_multipart_alias_does_not_fire_on_one_part_alone():
    """The safety side of the previous rule: "alpha" in its ordinary English sense
    must not resolve to a fund."""
    fs = _fs({"seed_funds": ["Alpha Fund / Beta Fund / Gamma Partners"]})
    assert mentioned_entities("The alpha channel was empty.", fs) == set()


def test_a_genuinely_missing_alias_is_still_missed():
    """"AX" declared, "Accelerator X" written. No library rule can bridge that, and
    pretending otherwise would mean guessing at names."""
    fs = _fs({"yc": ["AX"]})
    assert mentioned_entities("Accelerator X takes 7% equity.", fs) == set()

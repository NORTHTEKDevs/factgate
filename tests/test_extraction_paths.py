"""Construction-based proofs for extraction -- the last component deciding verdicts.

The gate is only as good as the claim it is handed. `link_targeted` decides WHICH claims
reach it, using a local model plus four deterministic guards, and a measured leak once came
from the extractor inventing the declared value outright: asked for the dose in "Give
acetaminophen 7.5 mg/kg PO", it answered "15 mg/kg" -- the DECLARED value -- and the gate
verified it correctly, because the gate can only adjudicate what it is given.

Everything here uses a SCRIPTED model, so the tests are deterministic and offline. The
model is treated as hostile: it fabricates, translates, glues words together, refuses,
returns prose, and copies the decoy rather than the answer. The property asserted is not
that the model behaves, but that the PIPELINE never emits a claim its own guards reject:

    every claim link_targeted returns is grounded in the passage it came from,
    is unambiguous there, and contains no word the passage does not use.

That property holds whatever the model says, which is the only useful form of it.
"""
import itertools

import pytest

import factgate.domain.link as link
from factgate.domain.factset import FactSet
from factgate.domain.link import (ambiguous_candidates, foreign_words,
                                  mentioned_entities, normalise_slot_answer,
                                  respell_from_passage, value_is_grounded)


def _fs(declared="15 mg/kg", entity="acetaminophen", relation="dose", **kw):
    return FactSet.from_dict({
        "domain": "d", "entities": {entity: ["the drug"]},
        "relations": {relation: {"kind": "quantity"}},
        "facts": [{"s": entity, "r": relation, "o": declared,
                   "source": f"Give {entity} {declared} PO now."}], **kw})


def _scripted(*answers):
    """A model that returns the given strings in order, then repeats the last."""
    box = {"i": 0}

    def fake(model, prompt, timeout, **kw):
        i = min(box["i"], len(answers) - 1)
        box["i"] += 1
        return answers[i]
    return fake


# --------------------------------------------------------------- grounding
_VALUES = ["15 mg/kg", "7.5 mg/kg", "$1.50", "92 percent", "12 weeks", "-20 degrees C",
           "1,200 mg", "0.5 mg"]


@pytest.mark.parametrize("value", _VALUES)
def test_a_value_the_passage_states_is_grounded(value):
    assert value_is_grounded(value, f"The protocol says {value} exactly.") is True


@pytest.mark.parametrize("stated,claimed", [
    ("15 mg/kg", "30 mg/kg"), ("$1.50", "$2.50"), ("92 percent", "93 percent"),
    ("12 weeks", "12 months"), ("0.5 mg", "5 mg"),
])
def test_a_value_the_passage_does_not_state_is_never_grounded(stated, claimed):
    """CONSTRUCTED: the passage states one value, the claim asserts another. Every pair
    differs in magnitude, so no reading of the passage supports the claim."""
    assert value_is_grounded(claimed, f"The protocol says {stated} exactly.") is False


def test_grounding_is_necessary_but_not_sufficient():
    """A deliberate boundary, recorded because it looks like a hole. "15 mg" IS grounded by
    a passage stating "15 mg/kg" -- the magnitude appears and "mg" appears inside "mg/kg".
    Grounding asks whether the extractor INVENTED the value, not whether it is the declared
    one; the unit comparison is what settles that, and it holds."""
    passage = "Give acetaminophen 15 mg/kg PO now."
    assert value_is_grounded("15 mg", passage) is True
    fs = _fs(declared="15 mg/kg")
    from factgate.domain.gate import HELD, gate_claim
    assert gate_claim(fs, "acetaminophen", "dose", "15 mg").status == HELD


def test_the_measured_fabrication_is_refused():
    """The real leak: asked for the dose in a passage stating 7.5 mg/kg, the extractor
    answered 15 mg/kg -- the DECLARED value, invented. The gate cannot help here; only the
    guard can."""
    passage = "Give acetaminophen 7.5 mg/kg PO now."
    assert value_is_grounded("15 mg/kg", passage) is False
    assert value_is_grounded("7.5 mg/kg", passage) is True


# --------------------------------------------------------------- respelling
@pytest.mark.parametrize("value,passage,expected", [
    ("500 dollarsmonthly", "a minimum balance of 500 dollars monthly to avoid", "500 dollars monthly"),
    ("15mg/kg", "Give 15 mg/kg PO now.", "15 mg/kg"),
    ("500 dollars", "a minimum balance of 500 dollars monthly", "500 dollars"),
])
def test_respelling_recovers_the_documents_own_spacing(value, passage, expected):
    assert respell_from_passage(value, passage) == expected


@pytest.mark.parametrize("value", ["900 dollars", "500 euros", "1 mg", "zzz"])
def test_respelling_never_invents_a_value(value):
    """Only whitespace may differ. Every other character must match in order, so this can
    never turn one value into a different one."""
    passage = "a minimum balance of 500 dollars monthly to avoid the charge"
    out = respell_from_passage(value, passage)
    assert out == value or "".join(out.split()) == "".join(value.split())


# ------------------------------------------------------------ foreign words
@pytest.mark.parametrize("foreign", ["после", "aproximadamente", "ungefähr", "zzzz"])
def test_a_word_the_passage_never_uses_is_detected(foreign):
    passage = "The wall time for v0.5 is 12-16 weeks after v0."
    assert foreign_words(f"12-16 weeks {foreign} v0", passage) == [foreign]
    assert foreign_words("12-16 weeks after v0", passage) == []


# ---------------------------------------------------------------- ambiguity
def test_a_competing_value_for_the_same_slot_is_ambiguous():
    """CONSTRUCTED: a passage stating two candidate doses cannot resolve to one."""
    fs = _fs()
    passage = "Give 15 mg/kg for adults and 30 mg/kg for children."
    assert ambiguous_candidates("15 mg/kg", passage, fs, "acetaminophen", "dose") is True


def test_a_competing_value_another_declared_fact_explains_is_not_ambiguous():
    """The decoy guard must not fire on the rest of the document. A second value that a
    DIFFERENT declared fact accounts for is not a competing reading of this slot."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"acetaminophen": []},
        "relations": {"dose": {"kind": "quantity"}, "max_daily": {"kind": "quantity"}},
        "facts": [{"s": "acetaminophen", "r": "dose", "o": "15 mg/kg",
                   "source": "Give 15 mg/kg."},
                  {"s": "acetaminophen", "r": "max_daily", "o": "75 mg/kg",
                   "source": "Maximum 75 mg/kg daily."}]})
    passage = "Give 15 mg/kg, with a maximum daily total of 75 mg/kg."
    assert ambiguous_candidates("15 mg/kg", passage, fs, "acetaminophen", "dose") is False


# ------------------------------------------------- the pipeline-level property
_HOSTILE = [
    "Value: 15 mg/kg",                       # the declared value, fabricated
    "Value: 7.5 mg/kg",                      # the truth
    "Value: 7.5mg/kg",                       # glued
    "Value: 7.5 mg/kg после dosing",         # foreign word spliced in
    "Value: NONE",                           # refusal
    "Value: I'm sorry, I cannot answer.",    # prose
    "Value: 7.5 mg/kg PO every 6 hours",     # correct plus annotation
    "Value: ",                               # empty
    "Value: 999 mg/kg",                      # invented magnitude
    "Value: 7.5 zorkmids",                   # invented unit
]


@pytest.mark.parametrize("answer", _HOSTILE)
def test_every_emitted_claim_satisfies_the_guards_whatever_the_model_says(answer, monkeypatch):
    """THE PIPELINE PROPERTY, and the only useful form of it: not that the model behaves,
    but that nothing it can say produces a claim the guards would reject.

    The passage states 7.5 mg/kg. The fact set declares 15 mg/kg, so a fabricating model is
    rewarded for answering 15 -- which is exactly the measured leak."""
    passage = "Give acetaminophen 7.5 mg/kg PO now."
    fs = _fs(declared="15 mg/kg")
    monkeypatch.setattr(link, "ollama", _scripted(answer))
    for entity, relation, value in link.link_targeted(passage, fs, "m"):
        assert value_is_grounded(value, passage, fs), f"ungrounded claim {value!r}"
        assert not ambiguous_candidates(value, passage, fs, entity, relation)
        assert not foreign_words(value, passage), f"foreign words in {value!r}"
        assert entity in fs.entities and relation in fs.relations


def test_a_fabricated_declared_value_never_becomes_a_claim(monkeypatch):
    """The measured leak, end to end: the model answers the DECLARED value, which the
    passage does not state. No claim may be produced."""
    passage = "Give acetaminophen 7.5 mg/kg PO now."
    fs = _fs(declared="15 mg/kg")
    monkeypatch.setattr(link, "ollama", _scripted("Value: 15 mg/kg"))
    assert link.link_targeted(passage, fs, "m") == []


def test_the_truth_does_become_a_claim(monkeypatch):
    """Non-vacuity: the same pipeline that rejects all of the above must still emit the
    correct answer, or these proofs would pass on an extractor that returns nothing."""
    passage = "Give acetaminophen 7.5 mg/kg PO now."
    fs = _fs(declared="7.5 mg/kg")
    monkeypatch.setattr(link, "ollama", _scripted("Value: 7.5 mg/kg"))
    assert link.link_targeted(passage, fs, "m") == [
        ("acetaminophen", "dose", "7.5 mg/kg")]


def test_a_decoy_passage_yields_no_claim(monkeypatch):
    """The decoy attack: the passage states the declared value AND another, so the slot
    cannot be resolved even though the model answers correctly."""
    passage = "Give acetaminophen 15 mg/kg for adults and 30 mg/kg for children."
    fs = _fs(declared="15 mg/kg")
    monkeypatch.setattr(link, "ollama", _scripted("Value: 15 mg/kg"))
    assert link.link_targeted(passage, fs, "m") == []


# ------------------------------------------------------------ entity matching
@pytest.mark.parametrize("surface", ["acetaminophen", "the drug", "Acetaminophen",
                                     "ACETAMINOPHEN", "aceta-\nminophen"])
def test_a_declared_surface_form_is_matched(surface):
    fs = _fs()
    assert "acetaminophen" in mentioned_entities(f"Give {surface} now.", fs)


@pytest.mark.parametrize("absent", ["ibuprofen", "paracetamol", "the medicine", ""])
def test_an_undeclared_name_never_resolves(absent):
    """An entity the fact set does not declare must never resolve, or a claim would attach
    to the wrong subject -- in a dosing domain, the wrong drug."""
    fs = _fs()
    assert mentioned_entities(f"Give {absent} now.", fs) == set()


# ------------------------------------------------------------- answer shapes
@pytest.mark.parametrize("refusal", [
    "NONE", "Not provided", "N/A", "unknown", "unclear", "I'm sorry, I cannot answer",
    "The passage does not specify a dose for this drug at all", "",
])
def test_a_refusal_never_becomes_a_value(refusal):
    """A refusal that reaches the comparator once BLOCKED a text fact -- the gate reporting
    a contradiction because the model declined to answer."""
    assert normalise_slot_answer(refusal) is None


@pytest.mark.parametrize("value", ["15 mg/kg", "$39 per user monthly", "between $5K and $2M",
                                   "12-16 weeks", "92%"])
def test_a_real_value_survives_the_shape_filter(value):
    assert normalise_slot_answer(value) == value


# --------------------------------------------------- numbers spelled as words
@pytest.mark.parametrize("spelled,digits", [
    ("a four-hour observation period", "a 4-hour observation period"),
    ("twelve weeks", "12 weeks"),
    ("observe for six hours", "observe for 6 hours"),
])
def test_number_words_become_digits(spelled, digits):
    """MEASURED ON A ROUTING RUN, and the reason routing coverage was not 100%: asked about
    an observation period declared as "4 hours", the model answered "a four-hour observation
    period" -- the correct value, spelled out. The digit never appears, grounding failed, no
    claim was produced, and a correct clinical statement reached the user with NOTHING
    adjudicating it. A bypass is worse than a hold: the user is not even told to check."""
    from factgate.domain.link import digits_for_words
    assert digits_for_words(spelled) == digits


@pytest.mark.parametrize("text", ["money and honey", "someone atoned", "a stone wall",
                                  "no numbers here"])
def test_a_number_word_inside_another_word_is_not_substituted(text):
    """Without word boundaries "money" became "m1y". The substitution is bounded on both
    sides, so only a whole word counts."""
    from factgate.domain.link import digits_for_words
    assert digits_for_words(text) == text


def test_a_spelled_number_is_grounded_but_a_wrong_one_still_is_not():
    """Coverage rises; nothing new verifies. The claim reaches the gate, where it is
    adjudicated on its merits."""
    answer = "the dose is 0.01 mg/kg IM and requires a four-hour observation period"
    assert value_is_grounded("4 hours", answer) is True
    assert value_is_grounded("9 hours", answer) is False

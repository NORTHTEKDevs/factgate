"""Conditional facts: one (entity, relation) holding different values by condition.

Real protocols are conditional -- amoxicillin is 45 mg/kg standard and 90 mg/kg for
otitis media; a lending rate depends on credit tier. Until now the schema rejected that
outright (`conflict: declared as X and Y`), which ruled out most real source documents.

The safety-critical rule is what happens when the condition is NOT established: the gate
must HOLD, never pick a variant. Verifying "90 mg/kg" without knowing the indication would
confirm an overdose in a standard-indication context.
"""
import pytest

from factgate.domain.factset import FactSet, ValidationError
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim

COND = {
    "domain": "cond",
    "entities": {"amoxicillin": [], "aspirin": []},
    "relations": {"pediatric_dose": {"kind": "quantity"}},
    "conditions": ["indication"],
    "facts": [
        {"s": "amoxicillin", "r": "pediatric_dose", "o": "45 mg/kg",
         "when": {"indication": "standard"}, "source": "Amoxicillin 45 mg/kg standard."},
        {"s": "amoxicillin", "r": "pediatric_dose", "o": "90 mg/kg",
         "when": {"indication": "otitis media"},
         "source": "Amoxicillin 90 mg/kg for otitis media."},
        {"s": "aspirin", "r": "pediatric_dose", "o": "5 mg/kg",
         "source": "Aspirin 5 mg/kg."},
    ],
}


@pytest.fixture
def fs():
    return FactSet.from_dict(COND)


def test_conditional_variants_load_without_conflict(fs):
    """The old schema rejected this outright, which excluded most real protocols."""
    assert len(fs.facts) == 3


def test_unconditioned_claim_on_a_conditional_slot_is_held(fs):
    """THE safety rule. Without the indication, "90 mg/kg" cannot be confirmed -- it is
    correct for otitis media and a 2x overdose otherwise."""
    assert gate_claim(fs, "amoxicillin", "pediatric_dose", "90 mg/kg").status == HELD
    assert gate_claim(fs, "amoxicillin", "pediatric_dose", "45 mg/kg").status == HELD


def test_matching_condition_verifies(fs):
    v = gate_claim(fs, "amoxicillin", "pediatric_dose", "90 mg/kg",
                   context={"indication": "otitis media"})
    assert v.status == VERIFIED


def test_wrong_value_under_a_known_condition_blocks(fs):
    v = gate_claim(fs, "amoxicillin", "pediatric_dose", "45 mg/kg",
                   context={"indication": "otitis media"})
    assert v.status == BLOCK
    assert v.declared == "90 mg/kg"


def test_value_matching_no_variant_blocks_even_unconditioned(fs):
    """A value that is none of the declared variants is a provable contradiction
    regardless of which condition applies."""
    assert gate_claim(fs, "amoxicillin", "pediatric_dose", "500 mg/kg").status == BLOCK


def test_unknown_condition_value_is_held(fs):
    assert gate_claim(fs, "amoxicillin", "pediatric_dose", "90 mg/kg",
                      context={"indication": "sepsis"}).status == HELD


def test_context_is_ignored_for_unconditional_facts(fs):
    """A fact with no `when` still verifies regardless of supplied context."""
    assert gate_claim(fs, "aspirin", "pediatric_dose", "5 mg/kg",
                      context={"indication": "otitis media"}).status == VERIFIED
    assert gate_claim(fs, "aspirin", "pediatric_dose", "5 mg/kg").status == VERIFIED


def test_condition_key_must_be_declared():
    """An undeclared condition key is a typo waiting to silently never match."""
    bad = {**COND, "facts": COND["facts"] + [
        {"s": "aspirin", "r": "pediatric_dose", "o": "9 mg/kg",
         "when": {"indicaton": "typo"}, "source": "x"}]}
    with pytest.raises(ValidationError, match="indicaton"):
        FactSet.from_dict(bad)


def test_duplicate_condition_with_different_value_still_conflicts():
    """Two different values under the SAME condition is still incoherent."""
    bad = {**COND, "facts": COND["facts"] + [
        {"s": "amoxicillin", "r": "pediatric_dose", "o": "60 mg/kg",
         "when": {"indication": "standard"}, "source": "x"}]}
    with pytest.raises(ValidationError, match="conflict"):
        FactSet.from_dict(bad)


def test_verdict_names_the_condition_for_audit(fs):
    v = gate_claim(fs, "amoxicillin", "pediatric_dose", "90 mg/kg")
    assert "indication" in v.reason

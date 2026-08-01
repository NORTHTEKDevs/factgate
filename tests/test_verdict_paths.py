"""Construction-based proofs for the two verdict paths no value oracle can reach.

`compare_values` is covered by tests/test_value_grammar.py against an exact-arithmetic
oracle. Two other things decide verdicts and were covered only by hand-written examples --
which is exactly the regime that produced five rounds of one-at-a-time defects:

  residue.source_grounded   turns INCOMPARABLE into MATCH when the fact's own source
                            sentence states the whole claim. Five leaks were constructed
                            against its first version by an adversarial reviewer.
  factset.lookup            selects among conditional variants. A leak lived here for the
                            whole project's life: an unconditional default silently
                            answered a query that supplied no context.

Neither can be checked by an arithmetic oracle, because both are decisions about TEXT
STRUCTURE and DECLARED CONDITIONS rather than about numbers. So the ground truth comes
from CONSTRUCTION instead: each case is assembled from parts whose correct verdict is known
because of how it was built, not because of what the implementation says.

That is the strongest oracle available here -- it cannot agree with a bug, because it never
consults the code.
"""
import itertools

import pytest

from factgate.domain.factset import FactSet
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim
from factgate.domain.residue import source_grounded

# ------------------------------------------------------------------ residue
# Parts chosen so the correct answer follows from the assembly, not from the library.
_DECLARED = ["35 dollars", "2 percent", "5 mg", "$1.50", "12 weeks", "0.5 mg"]
_SCOPE = ["per occurrence", "of the amount advanced", "per statement cycle",
          "monthly", "per customer query", "at closing"]
_NEGATION = ["not", "never", "except for", "unless", "excluding", "waived for",
             "free for", "contraindicated for", "exempt for"]


def _one_clause(declared, scope):
    """A source stating the declared value and its scope in ONE clause. Admissible."""
    return f"The rate is {declared} {scope}, reviewed annually."


def _split_clause(declared, scope):
    """The scope moved into a DIFFERENT sentence. Not admissible: the residue would be
    harvested from a clause that does not state the value."""
    return f"The rate is {declared}. Charges {scope} may also apply."


def _negated_clause(declared, scope, negation):
    """The clause carries a negation, so nothing drawn from it can be trusted."""
    return f"The rate is {declared} {scope}, {negation} premium members."


@pytest.mark.parametrize("declared", _DECLARED)
@pytest.mark.parametrize("scope", _SCOPE)
def test_residue_is_admitted_exactly_when_one_clause_states_the_whole_claim(declared, scope):
    """CONSTRUCTED GROUND TRUTH. The same declared value and the same scope phrase, placed
    once inside the value's own clause and once in a separate sentence. The first is the
    document saying it; the second is not, and the difference is the whole rule."""
    claim = f"{declared} {scope}"
    assert source_grounded(declared, claim, _one_clause(declared, scope)) is True
    assert source_grounded(declared, claim, _split_clause(declared, scope)) is False


@pytest.mark.parametrize("negation", _NEGATION)
def test_a_negated_clause_never_admits_its_residue(negation):
    """A clause that says the value does NOT apply to someone cannot be the authority for a
    claim that it does."""
    for declared, scope in itertools.product(_DECLARED[:3], _SCOPE[:3]):
        src = _negated_clause(declared, scope, negation)
        assert source_grounded(declared, f"{declared} {scope}", src) is False, src


@pytest.mark.parametrize("declared", _DECLARED)
def test_a_residue_carrying_a_second_value_is_never_admitted(declared):
    """A second number in the residue is a second VALUE, not a basis for the first, however
    faithfully the source states it."""
    for other in ("40 dollars", "99 mg", "3 percent", "٥٠ dollars"):
        claim = f"{declared} up to {other} in some cases"
        src = f"The rate is {claim}, reviewed annually."
        assert source_grounded(declared, claim, src) is False, claim


@pytest.mark.parametrize("declared", _DECLARED)
def test_a_value_appearing_in_several_clauses_is_never_admitted(declared):
    """With the value in more than one clause there is no single clause that states it, so
    the residue could be taken from the wrong one."""
    claim = f"{declared} per occurrence"
    src = (f"The base rate is {declared}. Penalty rates of {declared} per occurrence "
           f"apply after default.")
    assert source_grounded(declared, claim, src) is False


def test_residue_admission_implies_the_claim_extends_the_declared_value():
    """Whatever else it does, an admitted claim must START with the declared value and add
    only non-numeric text. Checked over the whole generated cross-product."""
    for declared, scope in itertools.product(_DECLARED, _SCOPE):
        claim = f"{declared} {scope}"
        if not source_grounded(declared, claim, _one_clause(declared, scope)):
            continue
        assert claim.lower().startswith(declared.lower())
        residue = claim[len(declared):]
        assert not any(ch.isdigit() or ch.isnumeric() for ch in residue)


# -------------------------------------------------------------- conditional
def _conditional(variants, condition="tier"):
    """A fact set whose slot has one value per condition value."""
    return FactSet.from_dict({
        "domain": "d", "entities": {"e": ["e"]},
        "relations": {"p": {"kind": "quantity"}}, "conditions": [condition],
        "facts": [{"s": "e", "r": "p", "o": value,
                   "source": f"For {key} the value is {value}.",
                   "when": {condition: key}} for key, value in variants]})


_VARIANT_SETS = [
    [("standard", "10 mg"), ("premium", "20 mg")],
    [("a", "5 percent"), ("b", "10 percent"), ("c", "15 percent")],
    [("male", "13.5-17.5 g/dL"), ("female", "12.0-15.5 g/dL")],
    [("summer", "$16.75"), ("winter", "$11.40")],
]


@pytest.mark.parametrize("variants", _VARIANT_SETS)
def test_a_conditional_slot_never_verifies_without_its_condition(variants):
    """CONSTRUCTED GROUND TRUTH. Every declared value of the slot, claimed with no context.
    None can be confirmed, because the document says which applies and the caller did not.

    A leak lived here for the project's life: an unconditional default matched every query,
    so a slot mixing a default with conditional variants confirmed the default silently."""
    fs = _conditional(variants)
    for _, value in variants:
        assert gate_claim(fs, "e", "p", value).status == HELD


@pytest.mark.parametrize("variants", _VARIANT_SETS)
def test_the_right_condition_verifies_exactly_its_own_variant(variants):
    """With the condition supplied, the matching variant verifies and every OTHER variant's
    value is a provable contradiction of it."""
    fs = _conditional(variants)
    for key, value in variants:
        ctx = {"tier": key}
        assert gate_claim(fs, "e", "p", value, ctx).status == VERIFIED
        for other_key, other in variants:
            if other == value:
                continue
            status = gate_claim(fs, "e", "p", other, ctx).status
            assert status in (BLOCK, HELD)
            assert status != VERIFIED, f"{other!r} verified under {key!r}"


@pytest.mark.parametrize("variants", _VARIANT_SETS)
def test_a_value_no_variant_states_is_blocked_or_held_never_verified(variants):
    """Constructed to differ from every declared variant, with and without context."""
    fs = _conditional(variants)
    unit = variants[0][1].split()[-1] if " " in variants[0][1] else ""
    for bogus in (f"999 {unit}".strip(), f"0.001 {unit}".strip(), "nonsense value"):
        assert gate_claim(fs, "e", "p", bogus).status != VERIFIED
        for key, _ in variants:
            assert gate_claim(fs, "e", "p", bogus, {"tier": key}).status != VERIFIED


@pytest.mark.parametrize("variants", _VARIANT_SETS)
def test_an_unmatched_condition_falls_back_only_when_a_default_exists(variants):
    """Supplying a condition value the fact set never declares selects nothing. With no
    unconditional default there is nothing to fall back to, so nothing verifies."""
    fs = _conditional(variants)
    for _, value in variants:
        assert gate_claim(fs, "e", "p", value, {"tier": "undeclared"}).status != VERIFIED


@pytest.mark.parametrize("variants", _VARIANT_SETS)
def test_a_default_beside_conditional_variants_still_needs_context(variants):
    """The exact shape of the measured leak: an unconditional default declared alongside
    conditional variants. Without context the slot is ambiguous whichever value is claimed;
    the default applies only once the condition is supplied and rules the others out."""
    spec = {
        "domain": "d", "entities": {"e": ["e"]},
        "relations": {"p": {"kind": "quantity"}}, "conditions": ["tier"],
        "facts": ([{"s": "e", "r": "p", "o": "1 mg", "source": "By default it is 1 mg."}]
                  + [{"s": "e", "r": "p", "o": v,
                      "source": f"For {k} the value is {v}.", "when": {"tier": k}}
                     for k, v in variants])}
    fs = FactSet.from_dict(spec)
    assert gate_claim(fs, "e", "p", "1 mg").status == HELD
    for key, value in variants:
        assert gate_claim(fs, "e", "p", value).status == HELD
        assert gate_claim(fs, "e", "p", value, {"tier": key}).status == VERIFIED
    # a condition the fact set declares, but which no variant claims, selects the default
    assert gate_claim(fs, "e", "p", "1 mg", {"tier": "unlisted"}).status == VERIFIED


def test_the_conditional_proofs_are_not_vacuous():
    """These assert mostly that things do NOT verify, which a gate that never verified
    anything would satisfy. Confirm the same fact sets DO verify when they should."""
    verified = 0
    for variants in _VARIANT_SETS:
        fs = _conditional(variants)
        for key, value in variants:
            if gate_claim(fs, "e", "p", value, {"tier": key}).status == VERIFIED:
                verified += 1
    assert verified >= 8, f"only {verified} verifications; the proofs would pass on a gate "

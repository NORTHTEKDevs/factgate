"""The five leaks an adversarial review constructed against the residue rule.

Every test below is a counterexample someone built to break this feature BEFORE it
shipped. The first version of the rule admitted a claim two ways -- contiguous substring
of the source, OR "every residue word appears somewhere in the source" -- and the second
leg turned out to be catastrophically unsafe. These pin the closed versions so a later
simplification cannot quietly reopen them.

If one of these ever fails, the rule has regressed to something that will confirm a
hallucinated basis, a contraindicated dose, or the wrong customer tier.
"""
import pytest

from factgate.domain.factset import FactSet
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim
from factgate.domain.residue import source_grounded


def _fs(declared: str, source: str, **kw):
    return FactSet.from_dict({
        "domain": "d", "entities": {"e": ["e"]},
        "relations": {"p": {"kind": "quantity"}},
        "facts": [{"s": "e", "r": "p", "o": declared, "source": source}], **kw})


def _status(declared, source, claimed, **kw):
    return gate_claim(_fs(declared, source, **kw), "e", "p", claimed).status


# ------------------------------------------------------------------ the leaks
def test_contraindication_is_not_reassigned():
    """LEAK 1. "for pregnant patients" appears in the source -- in the clause that FORBIDS
    the drug for them. A token-membership test admitted it. Confirming a dose for a
    population the document contraindicates is the worst failure this project can have."""
    assert _status("5 mg",
                   "This drug is intended for adults at 5 mg; it is contraindicated "
                   "for pregnant patients.",
                   "5 mg for pregnant patients") == HELD


def test_residue_cannot_be_assembled_from_a_negating_clause():
    """LEAK 2. Every word of "per customer query" appears in the source -- inside "does not
    vary per customer or per query". Token membership has no notion of polarity."""
    assert _status("$550",
                   "Enterprise support costs $550 per year. Overage billing does not vary "
                   "per customer or per query; it is included in the flat annual rate.",
                   "$550 per customer query") == HELD


def test_contiguity_does_not_run_across_a_sentence_boundary():
    """LEAK 3, and the one that killed the "obviously safe" leg. Normalising punctuation
    away made "35 dollars per occurrence" contiguous across a full stop, joining a flat
    setup fee to a different, conditional surcharge."""
    assert _status("35 dollars",
                   "The setup fee is 35 dollars. Per-occurrence monitoring surcharges may "
                   "apply for premium accounts only.",
                   "35 dollars per occurrence") == HELD


def test_residue_containing_a_second_quantity_is_refused():
    """LEAK 4. The claim is contiguous in the source by construction, but it asserts a
    second dose rather than a basis for the first. It must go through range/point
    comparison, which is deliberately stricter."""
    assert _status("7 mg/kg",
                   "Start at 7 mg/kg and titrate to 14 mg/kg as tolerated.",
                   "7 mg/kg and titrate to 14 mg/kg as tolerated") == HELD


def test_residue_from_another_tier_clause_is_refused():
    """LEAK 5. "premium cardholders pay" is lifted from the clause stating that premium
    cardholders pay NOTHING."""
    assert _status("25 dollars",
                   "Standard cardholders pay 25 dollars per statement cycle, or the "
                   "accrued interest if greater; premium cardholders pay no late fee "
                   "at all.",
                   "25 dollars premium cardholders pay") == HELD


def test_fabricated_basis_is_refused():
    """MEASURED, not constructed: the document says per day, the model said per query.
    The words "customer" and "query" do occur elsewhere in that corpus, which is why the
    check is scoped to the fact's own source clause."""
    assert _status("$1.50", "Widget unit cost: ~$1.50/day = ~$550/year",
                   "$1.50 per customer query") == HELD


# --------------------------------------------------------- what it does admit
@pytest.mark.parametrize("declared,source,claimed", [
    ("35 dollars",
     "The overdraft fee is 35 dollars per occurrence, assessed at end of business day.",
     "35 dollars per occurrence"),
    ("2 percent",
     "The personal loan origination fee is 2 percent of the amount advanced, deducted "
     "at closing.",
     "2 percent of the amount advanced, deducted at closing"),
    ("25 dollars",
     "The credit line minimum payment is 25 dollars per statement cycle, or the accrued "
     "interest if greater.",
     "25 dollars per statement cycle, or the accrued interest if greater"),
    ("500 dollars",
     "The savings account requires a minimum balance of 500 dollars monthly to avoid the "
     "service charge.",
     "500 dollars monthly"),
])
def test_real_measured_over_blocks_now_verify(declared, source, claimed):
    """All four are real extractions from the benchmark, held before this rule existed.
    In each the document itself states the whole claim in the clause declaring the value."""
    assert _status(declared, source, claimed) == VERIFIED


def test_a_provable_difference_is_still_blocked():
    """The rule only ever converts INCOMPARABLE to MATCH. It must not soften BLOCK."""
    assert _status("35 dollars", "The overdraft fee is 35 dollars per occurrence.",
                   "70 dollars") == BLOCK


def test_a_plus_sign_residue_does_not_confirm_a_point_with_a_range():
    """Found by review against REAL shipped data, not a constructed case. The residue is a
    single "+", which carries no digit but turns the claim into an open-ended range. The
    codebase deliberately holds that direction (a range never confirms a declared point,
    test_i4b); admitting it here would have reversed that guarantee through a side door."""
    assert source_grounded("$100M", "$100M+", "GPT-4 cost ~$100M+.") is False
    assert _status("$100M", "GPT-4 cost ~$100M+.", "$100M+") == HELD


def test_both_comparison_sites_agree():
    """gate.py compares in two places -- the primary path and the conditional-variant
    search. Wiring the residue rule into only one is precisely the split-path divergence
    that caused this project's one fuzz-caught leak (a duplicated grounding check drifted
    from the original). Same triple, same answer, whichever path reaches it."""
    src = "The overdraft fee is 35 dollars per occurrence, assessed at end of business day."
    plain = FactSet.from_dict({
        "domain": "d", "entities": {"e": ["e"]},
        "relations": {"p": {"kind": "quantity"}},
        "facts": [{"s": "e", "r": "p", "o": "35 dollars", "source": src}]})
    conditional = FactSet.from_dict({
        "domain": "d", "entities": {"e": ["e"]},
        "relations": {"p": {"kind": "quantity"}}, "conditions": ["tier"],
        "facts": [{"s": "e", "r": "p", "o": "35 dollars", "source": src,
                   "when": {"tier": "standard"}},
                  {"s": "e", "r": "p", "o": "10 dollars", "source": src,
                   "when": {"tier": "premium"}}]})
    assert gate_claim(plain, "e", "p", "35 dollars per occurrence").status == VERIFIED
    # Unconditioned: the residue-matching variant is found, so this is HELD pending the
    # condition -- NOT blocked as "matches none of the declared values".
    assert gate_claim(conditional, "e", "p", "35 dollars per occurrence").status == HELD
    assert gate_claim(conditional, "e", "p", "35 dollars per occurrence",
                      {"tier": "standard"}).status == VERIFIED


@pytest.mark.parametrize("claimed", [
    "35 dollars and 40 dollars", "35 dollars to 70 dollars", "35 dollars+",
    "35 dollars or 12 more", "35 dollars-70 dollars",
])
def test_residue_carrying_a_second_value_never_admits(claimed):
    """Any residue that is itself a value must fall through to the ordinary comparator."""
    src = ("The overdraft fee is 35 dollars and 40 dollars to 70 dollars+ or 12 more, "
           "35 dollars-70 dollars per occurrence.")
    assert source_grounded("35 dollars", claimed, src) is False


# ------------------------------------------------------------------ unit-level
def test_rule_has_no_opinion_unless_the_claim_extends_the_declared_value():
    assert source_grounded("35 dollars", "35 dollars", "The fee is 35 dollars.") is False
    assert source_grounded("35 dollars", "40 dollars per occurrence",
                           "The fee is 35 dollars per occurrence.") is False


def test_rule_refuses_empty_and_missing_inputs():
    assert source_grounded("", "x", "y") is False
    assert source_grounded("x", "", "y") is False
    assert source_grounded("x", "y", "") is False


def test_qualifier_normalisation_does_not_hide_the_residue():
    """MEASURED across two sister domains. Both declare the same fact with the same source
    sentence; the only difference is that one declares MORE value_qualifiers. In the
    domain that declares "of the amount advanced" and "deducted at closing", the claim
    normalised to "2 percent" and the residue rule could no longer find the claim in its
    own source -- so an identical claim verified in one domain and was held in the other.

    The caller passes a normalised claim but the source is raw document text, so both
    spellings have to be tried against it."""
    src = ("The personal loan origination fee is 2 percent of the amount advanced, "
           "deducted at closing.")
    claim = "2 percent of the amount advanced, deducted at closing"
    plain = _fs("2 percent", src)
    tuned = _fs("2 percent", src,
                value_qualifiers=["of the amount advanced", "deducted at closing"])
    assert tuned.normalise_value(claim) == "2 percent"      # heavily normalised
    assert gate_claim(plain, "e", "p", claim).status == VERIFIED
    assert gate_claim(tuned, "e", "p", claim).status == VERIFIED


def test_stripping_a_qualifier_does_not_leave_a_dangling_separator():
    """The other half of the same measurement: stripping the two qualifiers above left
    "2 percent ," -- a floating comma that stops the value parsing, so even ordinary
    comparison could not verify it. Only punctuation the strip orphaned is cleaned; a
    comma between digits is a thousands separator and must survive."""
    fs = _fs("2 percent", "The fee is 2 percent of the amount advanced, deducted at closing.",
             value_qualifiers=["of the amount advanced", "deducted at closing"])
    assert fs.normalise_value(
        "2 percent of the amount advanced, deducted at closing") == "2 percent"
    assert fs.normalise_value("5,000 mg") == "5,000 mg"
    assert fs.normalise_value("$1,250,000") == "$1,250,000"


# ------------------------------------------- the conditional path's false BLOCK
def _cond(*variants, condition="sex"):
    return FactSet.from_dict({
        "domain": "d", "entities": {"e": ["e"]},
        "relations": {"p": {"kind": "quantity"}}, "conditions": [condition],
        "facts": [{"s": "e", "r": "p", "o": o, "source": src,
                   "when": {condition: w}} for o, src, w in variants]})


def test_a_claim_stating_every_variant_is_held_not_blocked():
    """THE WORST VERDICT THIS PRODUCT CAN EMIT, and it was reachable.

    A reference-range sheet declares hemoglobin 13.5-17.5 g/dL for males and 12.0-15.5
    g/dL for females. Asked for the range, the model answered "13.5-17.5 g/dL and
    12.0-15.5 g/dL" -- exactly what the document says. The conditional path blocked it,
    because it blocked whenever no variant MATCHED, folding INCOMPARABLE into DIFFER: the
    one collapse the three-valued design exists to prevent.

    It stayed hidden only because conditional slots were never extracted at all. The
    moment they were, a faithful clinical answer started being reported as contradicting
    the protocol."""
    fs = _cond(("13.5-17.5 g/dL", "Hemoglobin is 13.5-17.5 g/dL in adult males.", "male"),
               ("12.0-15.5 g/dL", "Hemoglobin is 12.0-15.5 g/dL in adult females.",
                "female"))
    v = gate_claim(fs, "e", "p", "13.5-17.5 g/dL and 12.0-15.5 g/dL")
    assert v.status == HELD, "a faithful answer must never be reported as a contradiction"
    assert "neither confirmed nor contradicted" in v.reason


def test_a_conditional_claim_that_every_variant_contradicts_still_blocks():
    """The other half. Softening BLOCK to HELD everywhere would be its own failure: a value
    that provably differs from every declared variant is a contradiction and must be said
    so."""
    fs = _cond(("13.5-17.5 g/dL", "Hemoglobin is 13.5-17.5 g/dL in adult males.", "male"),
               ("12.0-15.5 g/dL", "Hemoglobin is 12.0-15.5 g/dL in adult females.",
                "female"))
    assert gate_claim(fs, "e", "p", "99-100 g/dL").status == BLOCK


def test_a_conditional_claim_matching_one_variant_is_held_then_verified():
    fs = _cond(("13.5-17.5 g/dL", "Hemoglobin is 13.5-17.5 g/dL in adult males.", "male"),
               ("12.0-15.5 g/dL", "Hemoglobin is 12.0-15.5 g/dL in adult females.",
                "female"))
    assert gate_claim(fs, "e", "p", "13.5-17.5 g/dL").status == HELD
    assert gate_claim(fs, "e", "p", "13.5-17.5 g/dL", {"sex": "male"}).status == VERIFIED


@pytest.mark.parametrize("variants,claimed", [
    # a range spanning both declared credits -- faithful to a tiered SLA table
    ((("10 percent", "The service credit is 10 percent.", "t1"),
      ("50 percent", "The service credit is 50 percent.", "t2")),
     "10% to 50% of the fees paid for a month"),
    # residue carrying a second quantity, against the cap it actually belongs to
    ((("12 months of fees paid", "The cap is 12 months of fees paid.", "t1"),
      ("24 months of fees paid", "The cap is 24 months of fees paid.", "t2")),
     "12 months of fees paid by Customer in the 12 months preceding the claim"),
    # a percentage OF another limit, against the deductibles actually declared
    ((("2 percent", "The inland wind and hail deductible is 2 percent.", "inland"),
      ("5 percent", "The coastal wind and hail deductible is 5 percent.", "coastal")),
     "2 percent of the building limit ($50,000)"),
])
def test_uncomparable_conditional_claims_are_held(variants, claimed):
    """All three are real extractions from blind domains that were being BLOCKED. None is
    a provable contradiction of its own declared values; each is text the comparator cannot
    line up, which is the definition of HELD."""
    fs = _cond(*variants, condition="tier")
    assert gate_claim(fs, "e", "p", claimed).status == HELD


# ------------------------------------------- false BLOCK from the comparator itself
def test_different_units_are_not_a_provable_contradiction():
    """The number was tested BEFORE the unit, so two quantities in different units were
    reported as a provable contradiction on the strength of their digits alone:

        declared "5 g"  claimed "5000 mg"  ->  DIFFER, and the gate BLOCKED

    5000 mg IS 5 g. The gate told the user a correct dose contradicted the protocol. The
    same happened for "5000 zz", and nothing in the library knows what zz is.
    """
    assert _status("5 g", "The maximum dose is 5 g per day.", "5000 mg") == HELD
    assert _status("5 g", "The maximum dose is 5 g per day.", "5000 zz") == HELD


def test_a_wrong_value_with_a_route_suffix_still_blocks():
    """The safety case the fix had to preserve, and briefly broke: a WRONG dose carrying an
    annotation is still a provable contradiction, because the annotation sits on the
    declared unit rather than changing it."""
    from factgate.domain.quantity import DIFFER, INCOMPARABLE, compare_values
    assert compare_values("10 mg/kg", "20 mg/kg PO") == DIFFER
    assert compare_values("10 mg/kg", "20 mg/kg every 6 hours") == DIFFER
    # ...and the same annotation on the RIGHT value still may not confirm it. Returning
    # MATCH here reintroduced the leak this project already fixed once.
    assert compare_values("10 mg/kg", "10 mg/kg PO") == INCOMPARABLE


@pytest.mark.parametrize("declared,claimed", [
    ("450 inch-pounds (51 Nm)", "450 inch-pounds"),   # omits a parenthetical conversion
    ("Board Certified", "is Board Certified"),        # carries a leading verb
])
def test_one_text_value_containing_the_other_is_not_a_contradiction(declared, claimed):
    """Both BLOCKED, and both claims are faithful. A difference in completeness is not a
    contradiction; a genuinely competing value shares no containment and still DIFFERs."""
    def text_fs(o, src):
        return FactSet.from_dict({
            "domain": "d", "entities": {"e": ["e"]},
            "relations": {"p": {"kind": "text"}},
            "facts": [{"s": "e", "r": "p", "o": o, "source": src}]})
    fs = text_fs(declared, f"The record states {declared}.")
    assert gate_claim(fs, "e", "p", claimed).status == HELD
    other = text_fs("oral", "Give it by the oral route.")
    assert gate_claim(other, "e", "p", "intravenous").status == BLOCK


# --------------------------------------------------- leaks found by adversarial hunt
def test_an_exemption_word_the_list_missed_is_now_caught():
    """The list had "waived" but not "free", so two sentences meaning the same thing got
    opposite verdicts: "35 dollars, waived for premium members" was correctly HELD while
    "35 dollars, free for premium members" was VERIFIED, confirming a fee the very same
    clause says some customers do not pay."""
    for word in ("free", "waived", "complimentary", "reduced", "refunded"):
        src = f"The fee is 35 dollars, {word} for premium members."
        assert _status("35 dollars", src, f"35 dollars, {word} for premium members") == HELD


def test_a_second_value_written_in_another_script_is_still_seen():
    """The residue may not contain a second value, but the check read ASCII digits only,
    so a numeral in another script was invisible to it. The parser is deliberately
    ASCII-only; this check has the OPPOSITE job and must see everything."""
    for numeral in ("٥٠", "５０", "50"):
        src = f"The fee is 35 dollars, up to {numeral} dollars for expedited processing."
        claim = f"35 dollars, up to {numeral} dollars for expedited processing"
        assert _status("35 dollars", src, claim) == HELD, f"leaked on {numeral!r}"

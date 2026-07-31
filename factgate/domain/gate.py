"""The deterministic verdict layer.

No learned parameters, no model call, no scoring threshold: given a linked claim and a
declared fact set, the verdict is a comparison. That is the property competitors using a
neural judge cannot offer, because a neural judge can itself hallucinate a verdict.

Fail-closed: anything that does not resolve to a declared fact is HELD, never passed.
"""
from __future__ import annotations

from dataclasses import dataclass

from factgate import __version__
from factgate.domain.factset import FactSet
from factgate.domain.quantity import DIFFER, MATCH, compare_values
from factgate.domain.residue import source_grounded

VERIFIED, BLOCK, HELD = "VERIFIED", "BLOCK", "HELD"


@dataclass(frozen=True)
class Verdict:
    status: str
    entity: str | None
    relation: str
    claimed: str | None
    declared: str | None
    source: str
    reason: str
    factset_fingerprint: str = ""
    # A regulated caller has to reconstruct months later WHY a claim was verified. The
    # fact set's fingerprint pins the data; these pin the code that read it. Without them
    # an audit can prove which facts were declared but not which comparison rules applied
    # -- and the rules do change (the residue rule below altered verdicts for existing
    # fact sets without altering the fact sets themselves).
    library_version: str = __version__


def gate_claim(fs: FactSet, entity_mention: str | None, relation: str,
               claimed_value: str | None,
               context: dict[str, str] | None = None) -> Verdict:
    """Adjudicate one linked claim against the declared fact set.

    `context` supplies the conditions under which the claim is being made (e.g.
    {"indication": "otitis media"}). A slot with several conditional values cannot be
    verified without it: confirming "90 mg/kg" with the indication unknown would confirm
    a 2x overdose in the standard case.
    """
    # A caller wrapping this in a service passes whatever its JSON contained. Non-string
    # inputs previously escaped as TypeError/AttributeError from inside the parser
    # ("object of type 'int' has no len()"), which is neither catchable by contract nor
    # explicable to whoever sent the request.
    for name, val in (("entity_mention", entity_mention), ("relation", relation),
                      ("claimed_value", claimed_value)):
        if val is not None and not isinstance(val, str):
            raise TypeError(f"{name} must be a string or None, got "
                            f"{type(val).__name__}: {val!r:.60}")
    if context is not None and not isinstance(context, dict):
        raise TypeError(f"context must be a dict or None, got {type(context).__name__}")

    entity = fs.resolve_entity(entity_mention)
    if entity is None:
        return Verdict(HELD, None, relation, claimed_value, None, "",
                       f"entity {entity_mention!r} is not in domain {fs.domain!r}", factset_fingerprint=fs.fingerprint)

    fact = fs.lookup(entity, relation, context)
    if fact is None:
        variants = fs.variants(entity, relation)
        if not variants:
            return Verdict(HELD, entity, relation, claimed_value, None, "",
                           f"no declared fact for {entity!r}/{relation!r}", factset_fingerprint=fs.fingerprint)
        # The slot IS declared but the condition did not select one value. A claim that
        # matches none of the variants is still a provable contradiction; a claim that
        # matches one is unconfirmable until the condition is supplied.
        norm = fs.normalise_value(claimed_value)
        # Same helper as the primary path below. A residue rule wired into only one of
        # the two comparison sites is exactly the split-path divergence that produced this
        # project's one fuzz-caught leak.
        hit = next((v for v in variants
                    if compare_values(fs.normalise_value(v.o), norm) == MATCH
                    or source_grounded(v.o, norm or "", v.source, raw_claimed=claimed_value)),
                   None)
        keys = sorted({k for v in variants for k, _ in v.when})
        if hit is None:
            return Verdict(BLOCK, entity, relation, claimed_value,
                           " | ".join(v.o for v in variants), variants[0].source,
                           f"matches none of the {len(variants)} declared values for "
                           f"{entity!r}/{relation!r}", factset_fingerprint=fs.fingerprint)
        return Verdict(HELD, entity, relation, claimed_value, hit.o, hit.source,
                       f"{entity!r}/{relation!r} is conditional on {keys}; that "
                       f"condition was not established, so {hit.o!r} cannot be confirmed", factset_fingerprint=fs.fingerprint)

    # Normalisation is the DOMAIN's declaration, not an inference by the gate, and it must
    # be applied to BOTH sides. It was applied only to the claim, so a domain whose unit
    # aliases matched its own declared values could not verify an exact answer:
    #
    #   declared "20 K/uL"  claimed "20 K/uL"  with unit_aliases {"K/uL": "thousand per
    #   microliter"} -> the claim normalised and the declared value did not, so the gate
    #   compared "20 K/uL" against "20 thousand per microliter" and HELD an identical
    #   string. Three of seven holds on a fresh clinical-lab domain were this.
    #
    # Safe because lint() already refuses a fact set whose normalisation collapses two
    # DISTINCT declared values onto the same string -- that check normalises declared
    # values, so it covers unit aliases as well as qualifiers.
    outcome = compare_values(fs.normalise_value(fact.o),
                             fs.normalise_value(claimed_value))
    if outcome == MATCH:
        return Verdict(VERIFIED, entity, relation, claimed_value, fact.o, fact.source,
                       "claimed value matches the declared fact", factset_fingerprint=fs.fingerprint)
    if outcome == DIFFER:
        return Verdict(BLOCK, entity, relation, claimed_value, fact.o, fact.source,
                       f"declared {fact.o!r}, claimed {claimed_value!r}", factset_fingerprint=fs.fingerprint)
    # INCOMPARABLE, but the claim may be the declared value plus wording the document
    # itself supplies ("35 dollars" declared, "35 dollars per occurrence" claimed, source
    # "The overdraft fee is 35 dollars per occurrence"). Admitted only when the source's
    # own clause states the whole claim; see residue.py for the five leaks that shaped it.
    if source_grounded(fact.o, fs.normalise_value(claimed_value) or "", fact.source,
                       raw_claimed=claimed_value):
        return Verdict(VERIFIED, entity, relation, claimed_value, fact.o, fact.source,
                       "claimed value matches the declared fact, with additional wording "
                       "quoted from the fact's own source sentence", factset_fingerprint=fs.fingerprint)
    # Only a provable difference earns a BLOCK. Asserting that a correct value contradicts
    # the protocol is a worse failure than declining to confirm it.
    return Verdict(HELD, entity, relation, claimed_value, fact.o, fact.source,
                   f"cannot compare claimed {claimed_value!r} to declared {fact.o!r}", factset_fingerprint=fs.fingerprint)

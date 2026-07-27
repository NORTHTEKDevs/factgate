"""The deterministic verdict layer.

No learned parameters, no model call, no scoring threshold: given a linked claim and a
declared fact set, the verdict is a comparison. That is the property competitors using a
neural judge cannot offer, because a neural judge can itself hallucinate a verdict.

Fail-closed: anything that does not resolve to a declared fact is HELD, never passed.
"""
from __future__ import annotations

from dataclasses import dataclass

from factgate.domain.factset import FactSet
from factgate.domain.quantity import DIFFER, MATCH, compare_values

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


def gate_claim(fs: FactSet, entity_mention: str | None, relation: str,
               claimed_value: str | None,
               context: dict[str, str] | None = None) -> Verdict:
    """Adjudicate one linked claim against the declared fact set.

    `context` supplies the conditions under which the claim is being made (e.g.
    {"indication": "otitis media"}). A slot with several conditional values cannot be
    verified without it: confirming "90 mg/kg" with the indication unknown would confirm
    a 2x overdose in the standard case.
    """
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
        hit = next((v for v in variants if compare_values(v.o, norm) == MATCH), None)
        keys = sorted({k for v in variants for k, _ in v.when})
        if hit is None:
            return Verdict(BLOCK, entity, relation, claimed_value,
                           " | ".join(v.o for v in variants), variants[0].source,
                           f"matches none of the {len(variants)} declared values for "
                           f"{entity!r}/{relation!r}", factset_fingerprint=fs.fingerprint)
        return Verdict(HELD, entity, relation, claimed_value, hit.o, hit.source,
                       f"{entity!r}/{relation!r} is conditional on {keys}; that "
                       f"condition was not established, so {hit.o!r} cannot be confirmed", factset_fingerprint=fs.fingerprint)

    # Normalisation is the DOMAIN's declaration, not an inference by the gate.
    outcome = compare_values(fact.o, fs.normalise_value(claimed_value))
    if outcome == MATCH:
        return Verdict(VERIFIED, entity, relation, claimed_value, fact.o, fact.source,
                       "claimed value matches the declared fact", factset_fingerprint=fs.fingerprint)
    if outcome == DIFFER:
        return Verdict(BLOCK, entity, relation, claimed_value, fact.o, fact.source,
                       f"declared {fact.o!r}, claimed {claimed_value!r}", factset_fingerprint=fs.fingerprint)
    # INCOMPARABLE: only a provable difference earns a BLOCK. Asserting that a correct
    # value contradicts the protocol is a worse failure than declining to confirm it.
    return Verdict(HELD, entity, relation, claimed_value, fact.o, fact.source,
                   f"cannot compare claimed {claimed_value!r} to declared {fact.o!r}", factset_fingerprint=fs.fingerprint)

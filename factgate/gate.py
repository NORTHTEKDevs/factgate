"""FactGate core: verify LLM claims against an RCK knowledge base.

Every claim (subject, relation, object) is checked against the KB and
classified into one of four verdicts:

  VERIFIED     -- KB agrees with the claim and self-verification
                  (roundtrip) corroborates it.
  CONTRADICTED -- KB has a different, confident answer for a functional
                  relation, or an explicit negative fact denies the claim.
  OUT_OF_KB    -- the KB genuinely doesn't know (IDK).
  UNRESOLVED   -- ambiguous evidence, or self-verification could not
                  corroborate a nominally-KNOWN answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from rck.chain_walker import Hop, walk_chain
from rck.contradiction import FUNCTIONAL_RELATIONS
from rck.idk_detection import EpistemicState, IDKPolicy, ask_with_idk
from rck.knowledge_base import ShardedKnowledgeBase
from rck.negative_facts import denied_pairs_for
from rck.provenance import ProvenanceStore
from rck.self_verify import verify as self_verify
from rck.synonyms import canonical_entity, canonical_relation


class GateStatus(str, Enum):
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    OUT_OF_KB = "OUT_OF_KB"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class GateVerdict:
    status: GateStatus
    subject: str
    relation: str
    obj: str
    kb_answer: Optional[str] = None
    kb_score: float = 0.0
    provenance_summary: Optional[str] = None
    explanation: str = ""
    alternatives: list[tuple[str, float]] = field(default_factory=list)


def verify_claim(kb: ShardedKnowledgeBase, prov: Optional[ProvenanceStore],
                  s: str, r: str, o: str,
                  *, policy: IDKPolicy | None = None) -> GateVerdict:
    """Check a single (subject, relation, object) claim against the KB."""
    cs = canonical_entity(s)
    cr = canonical_relation(r)
    co = canonical_entity(o)

    answer = ask_with_idk(kb, {"S": cs, "R": cr}, "O", top_k=5, policy=policy)
    prov_summary = prov.summarize(cs, cr, co) if prov is not None else None

    if answer.state is EpistemicState.IDK:
        return GateVerdict(
            status=GateStatus.OUT_OF_KB, subject=cs, relation=cr, obj=co,
            kb_answer=None, kb_score=answer.top_score,
            provenance_summary=prov_summary,
            explanation=f"KB has no confident answer: {answer.explanation}",
        )

    if answer.state is EpistemicState.AMBIGUOUS:
        return GateVerdict(
            status=GateStatus.UNRESOLVED, subject=cs, relation=cr, obj=co,
            kb_answer=(str(answer.top_symbol)
                       if answer.top_symbol is not None else None),
            kb_score=answer.top_score,
            provenance_summary=prov_summary,
            alternatives=[(str(sym), float(sc))
                          for sym, sc in answer.alternatives],
            explanation=f"KB is ambiguous: {answer.explanation}",
        )

    # KNOWN.
    top = str(answer.top_symbol) if answer.top_symbol is not None else None
    if top == co:
        corroboration = self_verify(kb, cs, cr, co,
                                     methods=["roundtrip", "reverse"])
        roundtrip = corroboration["by_method"].get("roundtrip", {})
        if not roundtrip.get("verified", False):
            return GateVerdict(
                status=GateStatus.UNRESOLVED, subject=cs, relation=cr, obj=co,
                kb_answer=top, kb_score=answer.top_score,
                provenance_summary=prov_summary,
                explanation=(
                    "KB shows a KNOWN match but roundtrip self-verification "
                    f"failed to corroborate it: {roundtrip.get('notes', '')}"
                ),
            )
        return GateVerdict(
            status=GateStatus.VERIFIED, subject=cs, relation=cr, obj=co,
            kb_answer=top, kb_score=answer.top_score,
            provenance_summary=prov_summary,
            explanation=(
                f"KB confirms ({cs}, {cr}, {co}); corroborated by "
                f"{corroboration['verification_count']}/"
                f"{corroboration['total_methods']} self-verification "
                "methods."
            ),
        )

    # KNOWN, but top answer differs from the claimed object.
    if cr in FUNCTIONAL_RELATIONS:
        return GateVerdict(
            status=GateStatus.CONTRADICTED, subject=cs, relation=cr, obj=co,
            kb_answer=top, kb_score=answer.top_score,
            provenance_summary=prov_summary,
            explanation=(
                f"Functional relation {cr!r}: KB says the answer is "
                f"{top!r} (score {answer.top_score:.3f}), not {co!r} as "
                "claimed."
            ),
        )

    # Non-functional relation: only a contradiction if an explicit
    # negative fact denies the claimed object.
    denied = denied_pairs_for(kb, cs, cr)
    denied_objs = {str(sym).lower() for sym, _ in denied}
    if co in denied_objs:
        return GateVerdict(
            status=GateStatus.CONTRADICTED, subject=cs, relation=cr, obj=co,
            kb_answer=top, kb_score=answer.top_score,
            provenance_summary=prov_summary,
            explanation=(
                f"Explicit negative fact denies ({cs}, {cr}, {co}); KB's "
                f"own top pick for this relation is {top!r}."
            ),
        )

    return GateVerdict(
        status=GateStatus.UNRESOLVED, subject=cs, relation=cr, obj=co,
        kb_answer=top, kb_score=answer.top_score,
        provenance_summary=prov_summary,
        explanation=(
            f"Non-functional relation {cr!r}: KB's top answer is {top!r}, "
            f"claim was {co!r}; no explicit contradiction found so both "
            "may coexist."
        ),
    )


def verify_chain_claim(kb: ShardedKnowledgeBase, s: str,
                        relation_path: list[str], o: str,
                        *, min_score: float = 0.10) -> GateVerdict:
    """Check an explicit multi-hop claim by walking `relation_path` from
    `s` (forward hops only) and comparing the final answer to `o`."""
    cs = canonical_entity(s)
    co = canonical_entity(o)
    path = [canonical_relation(r) for r in relation_path]
    hops = [Hop(relation=r, min_score=min_score) for r in path]
    result = walk_chain(kb, cs, hops)
    joined_relation = "->".join(path)

    if result.answer is None:
        return GateVerdict(
            status=GateStatus.OUT_OF_KB, subject=cs, relation=joined_relation,
            obj=co, kb_answer=None, kb_score=0.0,
            explanation=(
                f"Chain broke at hop {result.aborted_at}: {result.verbalize()}"
            ),
        )

    if result.answer == co:
        return GateVerdict(
            status=GateStatus.VERIFIED, subject=cs, relation=joined_relation,
            obj=co, kb_answer=result.answer, kb_score=result.confidence,
            explanation=f"Chain confirmed: {result.verbalize()}",
        )

    return GateVerdict(
        status=GateStatus.CONTRADICTED, subject=cs, relation=joined_relation,
        obj=co, kb_answer=result.answer, kb_score=result.confidence,
        explanation=(
            f"Chain resolved to {result.answer!r}, not {co!r} as claimed. "
            f"{result.verbalize()}"
        ),
    )

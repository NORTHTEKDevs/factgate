"""Suggest qualifier declarations from held claims.

The residual over-block cost is vocabulary: a value the model stated correctly but with
trailing text the domain has not declared irrelevant ("$1.50 per customer query" against a
declared "$1.50"). The library must not strip such text by inference -- declaring "per day"
irrelevant on a per-dose value would silently make a wrong value verify -- so the decision
belongs to the author.

What the library CAN do is stop making that decision expensive. This reports the exact
trailing text that caused each hold, as candidate `value_qualifiers`, for a human to
approve or reject. Nothing here changes a verdict; it only tells you what to look at.
"""
from __future__ import annotations

import re
from collections import Counter

from factgate.domain.factset import FactSet
from factgate.domain.quantity import MATCH, compare_values


def _residue(declared: str, claimed: str) -> str | None:
    """The text left over once the declared value is removed from the claim."""
    if not declared or not claimed:
        return None
    lead = re.search(r"[0-9][0-9,]*(?:\.[0-9]+)?", declared)
    if lead is None:
        return None
    idx = claimed.find(lead.group(0))
    if idx < 0:
        return None
    tail = claimed[idx + len(lead.group(0)):]
    # Strip only the DECLARED value's own unit if the claim repeats it. Stripping any
    # leading letters instead ate the "per" out of "per customer query", proposing
    # "customer query" -- a qualifier that does not rescue anything.
    declared_unit = declared[lead.end():].strip()
    if declared_unit and tail.lstrip().lower().startswith(declared_unit.lower()):
        tail = tail.lstrip()[len(declared_unit):]
    return tail.strip(" ,.;:") or None


def suggest_qualifiers(fs: FactSet, held_claims) -> list[dict]:
    """Candidate `value_qualifiers` from claims that were held.

    `held_claims` is an iterable of (entity, relation, claimed_value). Only claims whose
    value would MATCH the declared one once the residue is removed are suggested -- a
    residue that does not rescue the claim is not a qualifier, it is a different value.

    Every suggestion is a proposal. Declaring one that genuinely changes meaning (a period,
    a basis, a population) will make a wrong value verify, which is why this returns a list
    to read rather than a patch to apply.
    """
    counts: Counter = Counter()
    examples: dict[str, tuple[str, str]] = {}
    for entity, relation, claimed in held_claims:
        fact = fs.lookup(entity, relation)
        if fact is None or compare_values(fact.o, fs.normalise_value(claimed)) == MATCH:
            continue
        residue = _residue(fact.o, claimed)
        if not residue:
            continue
        # Only propose it if removing the residue actually rescues the claim.
        rescued = fs.normalise_value(claimed.replace(residue, "").strip(" ,.;:"))
        if compare_values(fact.o, rescued) != MATCH:
            continue
        counts[residue] += 1
        examples.setdefault(residue, (f"{entity}/{relation}", claimed))

    out = []
    for residue, n in counts.most_common():
        slot, claimed = examples[residue]
        out.append({
            "qualifier": residue,
            "occurrences": n,
            "example_slot": slot,
            "example_value": claimed,
            "warning": ("contains time or basis wording; declaring it irrelevant may "
                        "change what the value means")
            if re.search(r"\b(per|day|daily|week|month|year|annual|hour|dose|user|query)\b",
                         residue, re.IGNORECASE) else None,
        })
    return out


def render_suggestions(fs: FactSet, held_claims) -> str:
    """Human-readable block a domain author can act on directly.

    Safe items are emitted ready to paste into `value_qualifiers`. Items containing time
    or basis wording are listed separately and NOT pasted, because declaring one of those
    irrelevant is what makes a wrong value verify.
    """
    items = suggest_qualifiers(fs, held_claims)
    if not items:
        return "no qualifier suggestions -- remaining holds are not trailing-text issues"

    safe = [i for i in items if not i["warning"]]
    risky = [i for i in items if i["warning"]]
    lines = ["suggested value_qualifiers (review before declaring):"]
    if safe:
        lines.append("  paste into value_qualifiers:")
        lines.append("    " + ", ".join(f'"{i["qualifier"]}"' for i in safe))
        for i in safe:
            lines.append(f'      {i["qualifier"]!r} x{i["occurrences"]}'
                         f'  e.g. {i["example_slot"]} -> {i["example_value"]!r}')
    if risky:
        lines.append("  REVIEW CAREFULLY -- these carry time or basis wording, and")
        lines.append("  declaring one irrelevant can make a wrong value verify:")
        for i in risky:
            lines.append(f'      {i["qualifier"]!r} x{i["occurrences"]}'
                         f'  e.g. {i["example_slot"]} -> {i["example_value"]!r}')
    return "\n".join(lines)

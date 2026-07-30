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


# A capitalised run is how documents name things: "Y Combinator", "Series A". Sentence
# openers are excluded because the first word of every sentence is capitalised regardless.
_CAPS_RUN = re.compile(r"\b([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,3})")
_COMMON_OPENER = {"the", "a", "an", "we", "it", "this", "that", "they", "our", "based",
                  "according", "there", "these", "for", "in", "on", "at", "as"}


def suggest_entity_aliases(fs: FactSet, unlinked) -> list[dict]:
    """Propose entity aliases for declared facts the extractor never linked.

    MEASURED, and the largest remaining category of hold once trailing text was handled:
    asked what equity an accelerator takes, the model answered "Y Combinator typically
    takes around 7% equity" against an entity declared only as "yc"/"YC". The value was
    stated correctly and the gate never saw it, because nothing matched the entity. That
    reads as an over-block and is really a one-line gap in the fact set.

    `unlinked` is an iterable of (entity, relation, answer_text) for slots that produced no
    claim. A suggestion is offered only when the answer actually STATES the declared value
    -- otherwise the model may simply not have found it, and inventing an alias would make
    a non-answer linkable.

    Suggestions are for a human to accept, never applied automatically. An alias attaches
    claims to an entity, and a wrong one attaches a dose to the wrong drug.
    """
    from factgate.domain.link import mentioned_entities, value_is_grounded

    out: list[dict] = []
    for entity, relation, text in unlinked:
        fact = fs.lookup(entity, relation)
        if fact is None or not text:
            continue
        if not value_is_grounded(fact.o, text, fs):
            continue                    # the answer does not state the value; not an alias gap
        if mentioned_entities(text, fs):
            continue                    # something WAS matched; the miss is elsewhere
        declared = {a.casefold() for a in (entity, *fs.entities.get(entity, []))}
        # Stem each WORD, not the whole name: "the raise"[:4] is "the ", which matches
        # nothing useful, and the near-miss this exists to catch is "raising" -> "raise".
        stems = {w[:4] for a in declared for w in re.findall(r"[a-z]{4,}", a.casefold())}
        taken = {a.casefold() for name, al in fs.entities.items()
                 for a in (name, *al) if name != entity}

        candidates: list[str] = []
        for m in _CAPS_RUN.finditer(text):
            phrase = m.group(1).strip(".,;:")
            if (phrase.casefold() in declared or phrase.casefold() in taken
                    or phrase.casefold() in _COMMON_OPENER or len(phrase) < 2):
                continue
            candidates.append(phrase)
        # Morphological near-misses: "We are raising ..." against an entity called
        # "the raise". A shared four-character stem is a weak signal on its own, which is
        # exactly why this is a suggestion and not a rule.
        for word in re.findall(r"[a-zA-Z]{4,}", text):
            if (word.casefold()[:4] in stems and word.casefold() not in declared
                    and word.casefold() not in taken):
                candidates.append(word)

        seen, ordered = set(), []
        for c in candidates:
            if c.casefold() not in seen:
                seen.add(c.casefold())
                ordered.append(c)
        if ordered:
            out.append({"entity": entity, "relation": relation,
                        "candidates": ordered[:4], "declared_value": fact.o,
                        "answer": " ".join(text.split())[:140]})
    return out


def render_entity_suggestions(fs: FactSet, unlinked) -> str:
    items = suggest_entity_aliases(fs, unlinked)
    if not items:
        return "no entity-alias suggestions -- remaining holds are not naming gaps"
    lines = ["entities the extractor could not match, though the value WAS stated:"]
    for i in items:
        lines.append(f'  {i["entity"]!r}/{i["relation"]!r} (declared {i["declared_value"]!r})')
        lines.append(f'      answer   : {i["answer"]!r}')
        lines.append(f'      add alias: ' + ", ".join(f'"{c}"' for c in i["candidates"]))
    lines.append("  Review each: an alias attaches claims to an entity, and a wrong one")
    lines.append("  attaches them to the wrong entity.")
    return "\n".join(lines)

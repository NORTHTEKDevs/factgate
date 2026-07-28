"""Targeted slot filling: prose -> claims expressed in the domain's own vocabulary.

Open extraction (`factgate.hallugate.extract`) asks a model to GENERATE relation names, and
two extractions of one fact shared 0/17 relation strings (docs/HALLUGATE.md), so exact
comparison was impossible. Asking the model to CHOOSE from a declared list did not fix it
either: it returned a null relation on 5 of 7 failures even when it had stated the value
correctly.

What works is supplying the slot. The declared (entity, relation) pairs are known up front,
so this is reading comprehension with a closed output space, not open information
extraction. A failure is then detectable ("NONE") rather than silent, and detectable
failures can be routed to HELD.
"""
from __future__ import annotations

import re

from factgate.domain.factset import FactSet
from factgate.domain.quantity import parse_quantity
from factgate.llm import ollama

_SENT = re.compile(r"[^.!?]*[.!?]+|[^.!?]+$")


def sentences(text: str) -> list[str]:
    return [m.group().strip() for m in _SENT.finditer(text) if m.group().strip()]


# ---------------------------------------------------------------------------
# Targeted slot filling. Preferred over link_claims(): measured, open linking
# returned a null relation on 5 of 7 failures even when the value was stated
# correctly. Supplying the slot turns an open-vocabulary decision into a reading
# comprehension question, which small local models handle far better.
# ---------------------------------------------------------------------------

# Few-shot, with a worked NONE example so declining stays as available as answering.
# Measured against the previous instruction-only prompt on the same passages:
#   llama3.2:3b  1/6 -> 3/6 extracted     qwen2.5:14b  4/6 -> 6/6 extracted
# with 0 fabrications on negative-control passages in all four cells, so the gain is not
# bought by making the model eager to answer.
SLOT_PROMPT = """Extract one value from a passage. Copy it exactly as written.

Example
Passage: Give amoxicillin 45 mg/kg PO divided every 12 hours.
Question: what is the dose of amoxicillin?
Value: 45 mg/kg

Example
Passage: Escalate when the heart rate exceeds 180 beats per minute.
Question: what heart rate triggers escalation?
Value: 180 beats per minute

Example
Passage: Amoxicillin is an antibiotic.
Question: what is the dose of amoxicillin?
Value: NONE

Now do the same.
Passage: {text}
Question: {question}
Value:"""


def slot_question(fs: FactSet, entity: str, relation: str) -> str:
    """The question to ask for this slot.

    Phrasing is domain knowledge, so a domain may declare it per relation with an
    `{entity}` placeholder. "what is the dose of ibuprofen?" reads as a question;
    "what is the pediatric_dose of ibuprofen?" reads as a schema dump, and the model
    answers NONE to it noticeably more often.
    """
    template = fs.relations.get(relation, {}).get("question")
    if template:
        return template.replace("{entity}", entity)
    return f"what is the {relation.replace('_', ' ')} of {entity}?"

# Matches absence GENERALLY, not a list of phrasings seen so far. Enumerating refusals
# failed twice: "Not provided." / "Not applicable." are two words with no clause
# punctuation, so they passed both the phrase blacklist and the value-shape whitelist and
# reached the comparator, where they BLOCKED a text-kind fact -- the gate reporting a
# contradiction because the model declined to answer. A declared value never begins with
# a negation, so treating any such answer as absent is safe and fails closed.
_ABSENT = re.compile(
    r"^\s*(?:no(?:ne|t|thing)?\b|n/?a\b|null\b|un(?:known|clear|specified|available"
    r"|determined)\b|absent\b|missing\b|nil\b)", re.IGNORECASE)
# A slot answer is a value, not prose. Anything conversational is treated as absent.
# Currency is symbol-first, so a leading "$" is as much a value marker as a digit.
# Requiring a digit dropped "$39 per user monthly" on the >3-word rule.
_LEADS_WITH_NUMBER = re.compile(r"^\s*(?:US\$|[$£€¥])?\s*[-+]?[0-9]")
_PROSE = re.compile(r"\b(?:sorry|cannot|can't|does\s+not|doesn't|unable|i\s+)\b",
                    re.IGNORECASE)


def mentioned_entities(text: str, fs: FactSet) -> set[str]:
    """Declared entities named in the text, matched on word boundaries.

    Deterministic on purpose: this decides which slots are worth a model call, and a
    regex miss here is a silent coverage hole rather than a visible failure.
    """
    if not text:
        return set()
    # Line breaks and hyphens are word separators, not part of the name. Real documents
    # wrap "oxygen saturation" across lines and hyphenate compound modifiers
    # ("fluid-resuscitation protocol"); missing those is a silent coverage hole that
    # surfaces as HELD rather than as an error.
    low = _norm_surface(text)
    found = set()
    for canon, aliases in fs.entities.items():
        for surface in (canon, *aliases):
            if _surface_matches(surface, low):
                found.add(canon)
                break
    return found


def _norm_surface(s: str) -> str:
    """Collapse the punctuation and spacing that vary between a table cell and prose.

    A first-time author copies aliases out of the document's tables ("Agency A / Agency B Grant")
    while the model paraphrases in prose ("Agency A/Agency B Grant"). Treating runs of separator
    punctuation as a single space makes those the same name.
    """
    return re.sub(r"[\s\-‐-―_/,&]+", " ", s.lower()).strip()


def _surface_matches(surface: str, normalised_text: str) -> bool:
    """Does this declared surface form appear in the (already normalised) text?

    Three allowances, each from a real declaration that failed to match:
      - separator punctuation and spacing are normalised on both sides
      - a TRAILING PARENTHETICAL is a disambiguator the author added to a table row
        ("Pre-seed angels (alt-arch AI)"), not part of what anyone calls the thing
      - a name listing several parties ("Alpha Fund / Beta Fund / Gamma Partners")
        matches when ALL parts are present, in any order and however joined. Requiring
        all of them is what stops a common word like "conviction" resolving on its own.
    """
    base = re.sub(r"\s*\([^)]*\)\s*$", "", surface.strip())
    norm = _norm_surface(base)
    if norm and re.search(rf"\b{re.escape(norm)}\b", normalised_text):
        return True
    parts = [_norm_surface(p) for p in re.split(r"\s*[/,]\s*|\s+and\s+", base) if p.strip()]
    if len(parts) > 1 and all(
            p and re.search(rf"\b{re.escape(p)}\b", normalised_text) for p in parts):
        return True
    return False


def normalise_slot_answer(raw: str | None) -> str | None:
    """Clean a slot answer, or None if the model reported the value as absent."""
    if raw is None:
        return None
    # Take the first non-empty line. Models append commentary after the value -- measured
    # output was "$79 (download)" followed by a blank line and "Note: Since the tier was
    # cancelled..." -- and the prose filter then discarded the whole answer including the
    # correct value sitting on line one.
    first = next((ln for ln in raw.splitlines() if ln.strip()), "")
    s = first.strip().strip('"').strip("'").strip()
    s = re.sub(r"^(?:value|answer)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    s = s.rstrip(".").strip()
    if not s or _ABSENT.match(s) or _PROSE.search(s):
        return None
    # Whitelist value SHAPES rather than blacklisting refusal phrasings. A blacklist can
    # only ever catch the refusals already seen; anything it misses becomes a "value" and
    # reaches the comparator, where it previously produced a false BLOCK on a text fact.
    # A declared value is a quantity or a short phrase, never a sentence.
    if parse_quantity(s) is not None:
        return s
    # An answer that LEADS with a number is a value with trailing qualifiers, not a
    # refusal -- refusals never start with a digit. Rejecting these on length discarded
    # correct answers like "10 mg/kg PO every 6 hours" before the gate saw them, which
    # was 4 of the 5 over-blocked cases measured. The gate's domain-declared
    # normalisation decides which qualifiers are safe to ignore; that is not this
    # function's call to make.
    if _LEADS_WITH_NUMBER.match(s):
        return s
    if len(s.split()) > 3 or any(p in s for p in (". ", "; ", ", however")):
        return None
    return s


def value_is_grounded(value: str, passage: str) -> bool:
    """Is this extracted value actually present in the passage it came from?

    MEASURED LEAK, and the deepest failure mode in the design: asked for the dose in
    "Give acetaminophen 7.5 mg/kg PO...", the extractor answered "15 mg/kg" -- the
    DECLARED value, hallucinated into existence. The gate then verified it, correctly,
    because the gate can only adjudicate the claim it is handed. A parameter-free verdict
    is no protection when the input itself is fabricated.

    The check is deterministic: for a quantity the magnitude must occur in the passage
    (the unit may be spelled differently, which is what unit_aliases reconcile); for text
    the value must occur as a substring. Nothing here consults a model.
    """
    if not value or not passage:
        return False
    low = " ".join(passage.lower().split())
    q = parse_quantity(value)
    if q is None:
        # Trailing punctuation ("$79(download)") stops the value parsing, but its numeric
        # content is still what has to be found in the passage. Substring-matching the
        # whole messy string instead rejected values that were plainly present.
        from factgate.domain.quantity import _leading
        q = _leading(value)
        if q is None:
            return value.lower().strip() in low

    # Check the number as WRITTEN as well as the parsed value. Magnitude expansion turns
    # "$150M" into 150000000, which never appears in the document -- searching only for
    # the expanded form made every abbreviated figure look fabricated.
    surface = re.search(r"[0-9][0-9,]*(?:\.[0-9]+)?", value)
    forms = {f"{q.value:g}"}
    if surface:
        forms.add(surface.group(0).replace(",", ""))
        forms.add(surface.group(0))
    return any(re.search(rf"(?<![0-9.]){re.escape(n)}(?![0-9]|\.[0-9])", low)
               for n in forms)


def ambiguous_candidates(value: str, passage: str, fs: FactSet | None = None,
                         entity: str | None = None,
                         relation: str | None = None) -> bool:
    """Does the passage carry more than one candidate value for this slot?

    The attack this defends against: `value_is_grounded` requires the extracted value to
    occur in the passage, so an attacker plants the DECLARED value somewhere harmless
    ("(Reference standard: 15 mg/kg.)") while stating a different operative one. Both the
    grounding check and the gate then pass, because each is individually satisfied.

    A single-value extractor cannot resolve which the author meant, so the honest answer
    is neither -- the caller holds. Candidates are counted only among numbers sharing the
    extracted value's unit, so "every 6 hours" does not make every qualified dose
    ambiguous.
    """
    q = parse_quantity(value)
    if q is None or not passage:
        return False
    flat = " ".join(passage.lower().split())
    unit = q.unit.lower()
    if not unit:
        return False
    # Numbers immediately followed by the same unit are competing values for this slot.
    pat = re.compile(rf"(?<![0-9.])([-+]?[0-9]+(?:\.[0-9]+)?)\s*{re.escape(unit)}\b")
    seen = {m.group(1).lstrip("+") for m in pat.finditer(flat.replace(" ", ""))}
    seen |= {m.group(1).lstrip("+") for m in pat.finditer(flat)}
    candidates = {float(v) for v in seen}

    # A competing number that ANOTHER declared fact for this entity accounts for is not a
    # decoy, it is the rest of the document. Measured: without this the guard flagged
    # "15 mg/kg ... with a maximum daily total of 75 mg/kg" and cost 17 points of
    # coverage. Variants of the SAME slot are deliberately not excused -- a second value
    # for the slot under another condition is exactly the ambiguity to hold on.
    if fs is not None and entity is not None:
        explained = set()
        for f in fs.facts:
            if f.s != entity or f.r == relation:
                continue
            fq = parse_quantity(f.o)
            if fq is not None and fq.unit.lower() == unit:
                explained.add(fq.value)
        candidates -= explained - {q.value}
    return len(candidates) > 1


def link_targeted(text: str, fs: FactSet, model: str,
                  **kw) -> list[tuple[str, str, str]]:
    """Ask one targeted question per (mentioned entity, declared relation) pair.

    Only pairs that actually have a declared fact are queried: a slot with nothing to
    compare against could never produce anything but HELD, so asking would burn a model
    call for no verdict.
    """
    claims = []
    for entity in sorted(mentioned_entities(text, fs)):
        for relation, spec in fs.relations.items():
            if fs.lookup(entity, relation) is None:
                continue
            answer = ollama(model, SLOT_PROMPT.format(
                text=text, question=slot_question(fs, entity, relation)), 60, **kw)
            value = normalise_slot_answer(answer)
            # An extracted value that does not occur in the passage was invented by the
            # extractor, not read from the text. Dropping it sends the claim to HELD,
            # which is the fail-closed answer; trusting it produced a measured leak.
            # Two deterministic guards on a model-produced value:
            #   grounded    -- the value must occur in the passage (a measured leak had
            #                  the extractor inventing the declared value outright)
            #   unambiguous -- the passage must not carry a competing value for the slot
            #                  (the decoy attack: plant the declared value, state another)
            if (value is not None and value_is_grounded(value, text)
                    and not ambiguous_candidates(value, text, fs, entity, relation)):
                claims.append((entity, relation, value))
    return claims

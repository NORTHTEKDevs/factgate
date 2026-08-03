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
from factgate.domain.quantity import parse_quantity, parse_range
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
    # Two readings of the text, because a hyphen at a line break means opposite things.
    # An extracted PDF wraps a long word as "aceta-\nminophen", where the hyphen is a
    # typesetting artifact and the word is one token; it also wraps genuinely hyphenated
    # names as "fluid-\nresuscitation", where the hyphen belongs. Nothing local can tell
    # which, so both readings are tried and either may match.
    readings = (_norm_surface(text), _norm_surface(_dehyphenate(text)))
    # A multi-word name is often split across the sentence that uses it. A tariff declaring
    # "Summer On-Peak Energy" is answered as "During the Summer Season ... the On-Peak
    # Energy rate is 14.2 cents per kWh" -- every word present, never contiguous. Measured:
    # six of thirteen remaining bypasses were this, and a bypass emits no verdict at all.
    #
    # Scoped to ONE SENTENCE rather than the whole text, so the words have to co-occur where
    # a reader would take them as one name. Attaching a claim to the wrong entity is the
    # worst thing this function can do, and a whole-document window would invite it.
    windows = [_norm_surface(s) for s in sentences(text)] + [_norm_surface(text)]
    found = set()
    for canon, aliases in fs.entities.items():
        for surface in (canon, *aliases):
            if any(_surface_matches(surface, low) for low in readings):
                found.add(canon)
                break
            if any(_all_words_present(surface, w) for w in windows):
                found.add(canon)
                break
    return found


_STOPWORDS = frozenset({"the", "a", "an", "of", "and", "or", "for", "to", "in", "on"})


def _all_words_present(surface: str, sentence: str) -> bool:
    """Every significant word of a multi-word name, inside one sentence.

    Requires at least two significant words, so a single common noun can never resolve an
    entity on its own -- the guard that keeps "energy" from matching every tariff row.
    """
    words = [w for w in _norm_surface(surface).split() if w not in _STOPWORDS]
    if len(words) < 2 or not sentence:
        return False
    return all(re.search(rf"\b{re.escape(w)}\b", sentence) for w in words)


def _dehyphenate(s: str) -> str:
    """Join a word broken across a line by a hyphen.

    Measured against the declared entity "acetaminophen": a passage reading
    "aceta-\\nminophen" normalised to "aceta minophen" and matched nothing, so every fact
    about that drug was silently unextractable. Documents converted from PDF are full of
    this, and a miss here is a coverage hole that surfaces as HELD rather than as an error.
    """
    return re.sub(r"-\s*\n\s*", "", s)


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


def normalise_slot_answer(raw: str | None, fs: FactSet | None = None,
                          declared: str | None = None) -> str | None:
    """Clean a slot answer, or None if the model reported the value as absent.

    When `fs` is supplied, the value-shape test gets a second chance against the DOMAIN's
    declared qualifiers. Measured on a SaaS contract declaring "of submission": the model
    answered "within 1 hour of submission", five words that do not lead with a digit, so
    the shape whitelist discarded it before the gate ever saw it. The author had already
    declared half of that phrase irrelevant.

    The ORIGINAL text is returned, never the normalised form -- the gate does its own
    normalisation, and the suggestion machinery needs the real wording to propose the
    qualifier that is still missing. The effect is to turn an invisible extraction drop
    into a visible, reviewable suggestion.
    """
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
    # A range written as prose ("between $500K and $2M") is five words and does not lead
    # with a digit, so the value-shape whitelist discarded it before the gate saw it.
    if parse_range(s) is not None:
        return s
    # The word allowance scales with the value the DOMAIN declared for this slot. A fixed
    # limit of three assumes every value is short, and an aviation schedule declares
    # "4,000 flight hours or 24 months, whichever comes first" -- nine words. Asked for it,
    # the model answered "every 4,000 flight hours or 24 months, whichever comes first",
    # which is the declared value with one leading word, and the filter discarded it as
    # prose before anything could adjudicate it. Nine holds across the blind domains were
    # this, and they were invisible: no claim reached the gate, so no suggestion could be
    # made about them either.
    limit = 3 if not declared else max(3, len(str(declared).split()) + 3)
    if len(s.split()) > limit or any(p in s for p in (". ", "; ", ", however")):
        if fs is not None:
            stripped = fs.normalise_value(s) or ""
            if stripped and stripped != s and normalise_slot_answer(stripped) is not None:
                return s
        return None
    return s


# Unit spellings that mean the same thing regardless of what any domain declares.
# parse_quantity already canonicalises "$"/"dollars" to "usd" and strips internal spaces,
# so these map a CANONICAL unit back to the surface forms a document might actually use.
_UNIT_SYNONYMS: dict[str, set[str]] = {
    "usd": {"$", "us$", "usd", "dollar", "dollars"},
    "gbp": {"£", "gbp", "pound", "pounds", "sterling"},
    "eur": {"€", "eur", "euro", "euros"},
    "jpy": {"¥", "jpy", "yen"},
    "percent": {"%", "percent", "pct", "percentage"},
    "%": {"%", "percent", "pct", "percentage"},
}


def respell_from_passage(value: str, passage: str) -> str:
    """Restore the document's own spacing when the extractor mangled it.

    MEASURED, and reproducible three times out of three: asked for the minimum balance in
    "...a minimum balance of 500 dollars monthly to avoid the service charge.", qwen2.5:14b
    answers "500 dollarsmonthly" -- the right words with the space removed. The declared
    fact is "500 dollars" and "monthly" is a declared qualifier, but the qualifier patterns
    are word-bounded, so nothing can strip "monthly" out of "dollarsmonthly". A correct
    reading was held because of one missing space.

    The document is the authority on how the document is spelled. If the value matches the
    passage once whitespace is allowed to differ, adopt the passage's version verbatim.
    Only whitespace may differ -- every other character must match in order -- so this
    cannot turn one value into a different one.
    """
    if not value or not passage:
        return value
    flat = re.sub(r"\s+", "", value)
    if not flat or value.lower() in " ".join(passage.lower().split()):
        return value
    pattern = r"\s*".join(re.escape(c) for c in flat)
    m = re.search(pattern, passage, re.IGNORECASE)
    if m is None:
        return value
    recovered = " ".join(m.group(0).split())
    return recovered if re.sub(r"\s+", "", recovered).lower() == flat.lower() else value


# Numbers written as WORDS. Measured on a real routing run: asked about an observation
# period declared as "4 hours", the model answered "a four-hour observation period" -- the
# correct value, spelled out. The digit never appears, so grounding failed, no claim was
# produced, and a correct clinical statement reached the user with nothing adjudicating it.
# That is a BYPASS, which is worse than a hold: the user is not even told to check.
#
# Substitution is exact and bounded, and it happens only on the extracted VALUE so that a
# claim can reach the gate. Nothing new verifies: the gate still compares what it is given.
_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)


def digits_for_words(text: str) -> str:
    """Rewrite number words as digits. Returns the text unchanged if it contains none.

    Deliberately NOT applied to the passage or to a declared value -- only to a model's
    answer, and only so the claim it contains can be adjudicated rather than silently
    skipped. "twenty-four" becomes "20-4" under this table rather than "24", which is
    wrong-looking but harmless: it fails to parse and the claim is held, which is the same
    outcome as before the substitution. Compound words are a known gap, recorded rather
    than guessed at.
    """
    if not text:
        return text
    return _WORD_NUMBER_RE.sub(lambda m: _WORD_NUMBERS[m.group(1).lower()], str(text))


def value_is_grounded(value: str, passage: str, fs: FactSet | None = None) -> bool:
    """Is this extracted value actually present in the passage it came from?

    MEASURED LEAK, and the deepest failure mode in the design: asked for the dose in
    "Give acetaminophen 7.5 mg/kg PO...", the extractor answered "15 mg/kg" -- the
    DECLARED value, hallucinated into existence. The gate then verified it, correctly,
    because the gate can only adjudicate the claim it is handed. A parameter-free verdict
    is no protection when the input itself is fabricated.

    The check is deterministic: for a quantity BOTH the magnitude and the unit must occur
    in the passage; for text the value must occur as a substring. Nothing here consults a
    model.

    Grounding the unit as well as the number closes a second leak of the same family,
    found by probing this function directly:

        value_is_grounded("500 zorkmids", "...a minimum balance of 500 dollars...")
        -> True, because only "500" was ever looked for.

    On its own that was harmless -- a fabricated unit does not match the declared one, so
    the gate held it. It stopped being harmless in a domain declaring unit_aliases: a
    passage reading "500 units of stock", an extractor answering "500 usd", and an alias
    usd -> dollars against a declared "500 dollars" VERIFIES a currency the document never
    mentions. The number was grounded and the unit was taken on trust.

    A unit counts as present if the passage contains the unit itself, any declared alias
    that canonicalises to it, or -- for currency -- the corresponding symbol. Unit spelling
    genuinely varies ("mg/kg" against "milligrams per kilogram"), so alias-aware matching
    is what makes this safe to require rather than merely strict.
    """
    if not value or not passage:
        return False
    low = " ".join(passage.lower().split())
    # A passage may spell its numbers ("a four-hour period"); the value will not.
    spelled = " ".join(digits_for_words(passage).lower().split())
    if spelled != low and value_is_grounded(value, spelled, fs):
        return True     # terminates: digits_for_words(spelled) == spelled
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
    if not any(re.search(rf"(?<![0-9.]){re.escape(n)}(?![0-9]|\.[0-9])", low)
               for n in forms):
        return False
    return _unit_is_grounded(q.unit, low, fs)


def _unit_is_grounded(unit: str, low: str, fs: FactSet | None) -> bool:
    """Does some spelling of `unit` occur in the (already lowercased, collapsed) passage?

    A bare number carries no unit to ground, so it passes. Everything else must show up
    under its own name, under a declared alias that canonicalises to it, or as a currency
    symbol.
    """
    if not unit:
        return True
    u = unit.lower()
    # A currency amount with a worded unit parses as ONE glued token: "$30 per million
    # tokens" -> "usdpermilliontokens". The "usd" half is a canonical code the document
    # never spells, the rest is the document's own words, so requiring the whole thing
    # verbatim rejected correct readings -- measured as two verdicts regressing from
    # VERIFIED to no-claim-at-all. Check the currency and the wording separately.
    for code in ("usd", "gbp", "eur", "jpy"):
        if u.startswith(code) and len(u) > len(code):
            return (_spelling_present(code, low, fs)
                    and _spelling_present(u[len(code):], low, fs))
    return _spelling_present(u, low, fs)


def _spelling_present(u: str, low: str, fs: FactSet | None) -> bool:
    spellings = {u} | _UNIT_SYNONYMS.get(u, set())
    spellings.add(u[:-1] if u.endswith("s") else u + "s")     # week / weeks
    if fs is not None:
        for alias, canon in getattr(fs, "unit_aliases", {}).items():
            if re.sub(r"\s+", "", str(canon)).lower() == u:
                spellings.add(str(alias).lower())
    # The unit is written without internal spaces by parse_quantity ("mg/kg", "perday"),
    # so compare against a space-stripped passage as well as the literal one.
    flat = low.replace(" ", "")
    return any(s in low or re.sub(r"\s+", "", s) in flat for s in spellings if s)


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


# Unicode-aware word matcher: the case this exists for is a Cyrillic word appearing in an
# otherwise English answer, which an ASCII-only pattern would not even see.
_ANY_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

REPAIR_PROMPT = """The previous answer contains words that do not appear in the passage.
Copy the value EXACTLY as it is written in the passage, character for character. Do not
translate, paraphrase, or add anything.

Passage: {text}
Question: {question}
Previous answer: {bad}
Corrected value:"""


def foreign_words(value: str, passage: str) -> list[str]:
    """Words in an extracted value that occur nowhere in the passage it came from.

    A value is supposed to be COPIED out of the passage, so a word the passage never uses
    can only have come from the model. Measured, deterministic on three runs: asked for a
    wall time from an English passage reading "12-16 weeks after v0", qwen2.5:14b answers
    "12-16 weeks после v0" -- the Russian for "after". The magnitude is right and the
    text is not the document's.
    """
    if not value or not passage:
        return []
    low = " ".join(str(passage).casefold().split())
    return [w for w in _ANY_WORD.findall(str(value)) if w.casefold() not in low]


def link_targeted(text: str, fs: FactSet, model: str,
                  report_unresolved: bool = False, **kw):
    """Ask one targeted question per (mentioned entity, declared relation) pair.

    Only pairs that actually have a declared fact are queried: a slot with nothing to
    compare against could never produce anything but HELD, so asking would burn a model
    call for no verdict.
    """
    claims: list[tuple[str, str, str]] = []
    unresolved: list[tuple[str, str, str]] = []
    for entity in sorted(mentioned_entities(text, fs)):
        for relation, spec in fs.relations.items():
            # A slot with no declared value could only ever produce HELD, so asking would
            # burn a model call for no verdict. A CONDITIONAL slot is not that: lookup()
            # returns None for it without a context, and testing only lookup() meant every
            # conditional fact in every domain was silently never extracted.
            #
            # Found on a freight rate sheet whose transit time differs by lane. Two
            # consequences, and the second is the serious one: a correct answer could not
            # be confirmed even when the caller supplied the condition, AND a WRONG value
            # for a conditional slot never became a claim, so gate_claim's "matches none of
            # the declared values" BLOCK was unreachable through this path.
            if (fs.lookup(entity, relation) is None
                    and not fs.variants(entity, relation)):
                continue
            answer = ollama(model, SLOT_PROMPT.format(
                text=text, question=slot_question(fs, entity, relation)), 60, **kw)
            declared_fact = fs.lookup(entity, relation)
            declared_o = declared_fact.o if declared_fact else next(
                (v.o for v in fs.variants(entity, relation)), None)
            value = normalise_slot_answer(answer, fs, declared_o)
            # The answer may spell its number. Substituting digits is what lets the claim
            # reach the gate at all; without it the value is ungrounded and silently
            # dropped, which is a BYPASS -- worse than a hold, because the user is not even
            # told to check.
            if value is not None and value != digits_for_words(value):
                spelled_out = digits_for_words(value)
                if value_is_grounded(spelled_out, text, fs):
                    value = spelled_out
            if value is not None:
                value = respell_from_passage(value, text)
            # The extractor emitted a word the passage does not contain. Ask ONCE more,
            # showing it its own answer -- the transport runs at temperature 0, so
            # repeating the same prompt returns the same string verbatim and only a
            # different prompt can help. If the repair is still not the document's own
            # wording the claim is dropped, which is HELD: this repairs EXTRACTION and
            # never relaxes adjudication, and every guard below still applies to whatever
            # comes back.
            if value is not None and foreign_words(value, text):
                repaired = normalise_slot_answer(ollama(model, REPAIR_PROMPT.format(
                    text=text, question=slot_question(fs, entity, relation),
                    bad=value), 60, **kw))
                if repaired is not None:
                    repaired = respell_from_passage(repaired, text)
                value = repaired if (repaired and not foreign_words(repaired, text)) else None
            # An extracted value that does not occur in the passage was invented by the
            # extractor, not read from the text. Dropping it sends the claim to HELD,
            # which is the fail-closed answer; trusting it produced a measured leak.
            # Two deterministic guards on a model-produced value:
            #   grounded    -- the value must occur in the passage (a measured leak had
            #                  the extractor inventing the declared value outright)
            #   unambiguous -- the passage must not carry a competing value for the slot
            #                  (the decoy attack: plant the declared value, state another)
            if value is None or not value_is_grounded(value, text, fs):
                # The same defect the ambiguity branch below had, one branch earlier, and
                # found the same way -- by replaying real bypasses. The slot answer
                # CONTAINED the declared value wrapped in prose the shape filter rightly
                # refuses to distill:
                #
                #   "refrigerated between 2 and 8 degrees Celsius before initial use"
                #   "within 95.0 to 105.0 percent of the label claim throughout its ..."
                #
                # Refusing to distill is correct; going SILENT is not. When the answer is
                # not a refusal and the passage states the declared value, the model did
                # assert something about this slot and a human should see it. Reported as
                # unresolved, which the caller turns into HELD. Never a claim: nothing here
                # is confirmed, only surfaced.
                first_line = next((l for l in (answer or "").splitlines() if l.strip()), "")
                if (not _ABSENT.match(re.sub(r"^(?:value|answer)\s*:\s*", "",
                                             first_line.strip(), flags=re.IGNORECASE))
                        and declared_o
                        and value_is_grounded(declared_o, text, fs)):
                    unresolved.append((entity, relation, declared_o))
                continue
            if ambiguous_candidates(value, text, fs, entity, relation):
                # A value WAS seen for this slot and could not be resolved. Dropping it
                # silently is a BYPASS: the caller emits nothing and the reader is never
                # told there was something to check. Measured across ten domains, this is
                # the dominant cause of unguarded assertions -- 15% of answers, almost all
                # of them passages discussing several values of the same kind.
                unresolved.append((entity, relation, value))
                continue
            claims.append((entity, relation, value))
    return (claims, unresolved) if report_unresolved else claims

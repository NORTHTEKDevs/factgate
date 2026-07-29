"""Quantity parsing and value comparison.

This module decides whether a stated value matches a declared one. In a dosing domain
that is the difference between blocking a wrong dose and emitting it, so it deliberately
has NO rounding tolerance: 15.4 does not match 15. A near-miss is a miss.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A number, then an optional unit of one or more words joined by spaces or slashes.
# Multi-word units are not exotic ("breaths per minute", "degrees celsius"); rejecting
# them stopped the demo fact set from loading at all. Whitespace inside the unit is
# stripped during normalisation, so comparison stays exact on both sides.
_QTY_RE = re.compile(
    r"^\s*~?\s*([-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[-+]?[0-9]*\.?[0-9]+)\s*"
    r"([a-zA-Zµ%°]+(?:[\s/]+[a-zA-Zµ°%]+)*)?\s*$")


# Digits are matched as [0-9], NOT \d. Python's \d and float() both accept non-ASCII
# numerals, so "٥ mg" (Arabic-Indic) and "５ mg" (fullwidth) parsed as 5.0 and VERIFIED
# against a declared "5 mg". Equality in a safety gate should not depend on Unicode
# folding the caller cannot see; non-ASCII numerals now fail to parse and are therefore
# held rather than auto-verified.


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str


# Currency is written symbol-first ("$199"), and business prose abbreviates magnitude
# ("$150M"). Requiring number-first made the parser return None for every price in a real
# strategy document, so the gate held 11 of 11 correct values -- a total collapse measured
# on a real file, not a hypothetical.
_CURRENCY = {"$": "usd", "US$": "usd", "£": "gbp", "€": "eur", "¥": "jpy"}
_CURRENCY_WORDS = {"dollars": "usd", "dollar": "usd", "usd": "usd",
                   "pounds": "gbp", "gbp": "gbp", "euros": "eur", "eur": "eur"}
_MAGNITUDE = {"k": 1e3, "m": 1e6, "b": 1e9,
              "thousand": 1e3, "million": 1e6, "billion": 1e9}
# A trailing descriptive word is allowed here for the same reason it is allowed on a
# plain quantity: "12 weeks engineering" parsed while "$5k cloud credit" was rejected at
# load, an asymmetry a first-time author has no way to predict. The words become part of
# the unit, so "$5k cloud credit" and "$5k" remain distinct.
_CURRENCY_RE = re.compile(
    r"^\s*~?\s*(US\$|[$£€¥])\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]*\.?[0-9]+)"
    r"\s*([kKmMbB]|thousand|million|billion)?"
    r"\s*([a-zA-Z][a-zA-Z\s/]*)?\s*$")
# "150 million dollars" -- magnitude spelled out, currency as a word.
_WORDY_RE = re.compile(
    r"^\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]*\.?[0-9]+)\s*"
    r"(thousand|million|billion)?\s*([a-zA-Z]+)\s*$")


def _currency_quantity(raw: str) -> Quantity | None:
    """Parse currency in either symbol-first or word form, expanding magnitude.

    Magnitude is expanded ONLY when a currency symbol is present or the magnitude is
    spelled out. A bare "150M" is ambiguous (megametres? milli?) and is left alone.
    """
    m = _CURRENCY_RE.match(raw)
    if m:
        value = float(m.group(2).replace(",", ""))
        if m.group(3):
            value *= _MAGNITUDE[m.group(3).lower()]
        unit = _CURRENCY[m.group(1)]
        if m.lastindex and m.lastindex >= 4 and m.group(4):
            unit += re.sub(r"\s*", "", m.group(4)).lower()
        return Quantity(value, unit)
    m = _WORDY_RE.match(raw)
    if m and m.group(3).lower() in _CURRENCY_WORDS:
        value = float(m.group(1).replace(",", ""))
        if m.group(2):
            value *= _MAGNITUDE[m.group(2).lower()]
        return Quantity(value, _CURRENCY_WORDS[m.group(3).lower()])
    return None


def parse_quantity(raw: str | None) -> Quantity | None:
    """Parse "15 mg/kg" -> Quantity(15.0, "mg/kg"). None if not a quantity."""
    if not raw:
        return None
    cur = _currency_quantity(raw)
    if cur is not None:
        return cur
    m = _QTY_RE.match(raw)
    if not m:
        return None
    num, unit = m.group(1), m.group(2) or ""
    try:
        value = float(num.replace(",", ""))
    except ValueError:
        return None
    # NOT .lower(): "mg" and "Mg" are milligram and megagram, a factor of 10^9.
    return Quantity(value, re.sub(r"\s*", "", unit))


MATCH, DIFFER, INCOMPARABLE = "MATCH", "DIFFER", "INCOMPARABLE"


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    unit: str


# "5-10 mg/kg", "5 to 10 mg/kg", "$1,500-3,000", "$25M-$40M", "15-25%". The second bound
# often omits the currency symbol, so the unit is taken from whichever side carries one.
# A magnitude letter must not be followed by another letter, or the "m" of "mg/kg" reads
# as "million" and "5-10 mg/kg" becomes a range topping out at ten million.
_RANGE_RE = re.compile(
    r"^\s*(?:between\s+)?(US\$|[$£€¥])?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmMbB](?![a-zA-Z]))?\s*"
    # "and" is accepted as a separator, so "between $500K and $2M" parses -- and so does a
    # bare "5 and 10 mg/kg". That is deliberate but not free: "5 and 10" could mean two
    # separate values rather than a span. It only matters when the DECLARED value is also
    # a range, and reading the claim as a range is the reading that can be checked.
    r"(?:-|–|—|\bto\b|\band\b)\s*"
    r"(US\$|[$£€¥])?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmMbB](?![a-zA-Z]))?\s*"
    r"([a-zA-Zµ%°][a-zA-Zµ%°/\s]*)?\s*$")


# "$100M+", "18 months+" -- an open-ended lower bound. Reported by a first-time author as
# a hard load error. Parsing it as the point value would verify that one figure and block
# every larger one, exactly inverting what the document says.
_OPEN_RANGE_RE = re.compile(
    r"^\s*~?\s*(US\$|[$£€¥])?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"([kKmMbB](?![a-zA-Z])|thousand|million|billion)?\s*"
    r"([a-zA-Zµ%°][a-zA-Zµ%°/\s]*)?\+\s*$")


def parse_range(raw: str | None) -> Range | None:
    """Parse a range, or None.

    Reversed bounds are rejected rather than swapped: "10-5" is a typo, and silently
    reinterpreting it would accept a bad declaration. A trailing "+" gives an open upper
    bound.
    """
    if not raw:
        return None
    m = _OPEN_RANGE_RE.match(raw)
    if m:
        cur, num, mag, tail = m.groups()
        try:
            low = float(num.replace(",", ""))
        except ValueError:
            return None
        if mag:
            low *= _MAGNITUDE[mag.lower()]
        unit = _CURRENCY[cur] if cur else re.sub(r"\s*", "", tail or "")
        return Range(low, float("inf"), unit) if unit else None
    m = _RANGE_RE.match(raw)
    if not m:
        return None
    cur_lo, lo_s, mag_lo, cur_hi, hi_s, mag_hi, tail = m.groups()
    try:
        low, high = float(lo_s.replace(",", "")), float(hi_s.replace(",", ""))
    except ValueError:
        return None
    if mag_lo:
        low *= _MAGNITUDE[mag_lo.lower()]
    if mag_hi:
        high *= _MAGNITUDE[mag_hi.lower()]
    if low >= high:
        return None
    cur = cur_lo or cur_hi
    unit = _CURRENCY[cur] if cur else re.sub(r"\s*", "", tail or "")
    if not unit:
        return None
    return Range(low, high, unit)


def _leading_range(claimed: str, declared: Range) -> Range | None:
    """Recover a range from a claim carrying a trailing clause.

    Measured: declared "12-16 weeks", model answered "12-16 weeks after v0.5", which parses
    as nothing. Points already had a leading-quantity fallback for exactly this; ranges did
    not, so any range with a trailing clause was uncomparable.

    Done by prefix search rather than an unanchored regex, because the unit pattern is
    greedy and swallows the trailing prose ("weeksafterv"). The longest prefix whose unit
    matches the declared one wins. If the declared bounds ALSO appear in the remainder, the
    claim is not a clean single value and nothing is returned, so the caller holds.
    """
    if not claimed:
        return None
    tokens = claimed.split()
    for n in range(len(tokens), 0, -1):
        candidate = " ".join(tokens[:n]).rstrip(".,;:")
        r = parse_range(candidate)
        if r is None or r.unit != declared.unit:
            continue
        rest = " ".join(tokens[n:])
        others = {_as_float(t) for t in _NUMBER_RE.findall(rest)}
        if declared.low in others or declared.high in others:
            return None          # ambiguous: the declared bounds recur later
        return r
    return None


def _compare_range(declared: str, claimed: str) -> str | None:
    """Comparison when either side is a range. None if neither side is one.

    Deliberate asymmetry: a POINT inside a declared RANGE verifies (the document supports
    that figure), but a claimed RANGE against a declared POINT does NOT -- a range cannot
    confirm a specific value, and treating it as a match would let a reader infer the
    whole span is document-supported.
    """
    dr, cr = parse_range(declared), parse_range(claimed)
    if dr is not None and cr is None:
        cr = _leading_range(claimed, dr)
    if dr is None and cr is None:
        return None

    if dr is not None and cr is not None:
        if dr.unit != cr.unit:
            return INCOMPARABLE
        if (dr.low, dr.high) == (cr.low, cr.high):
            return MATCH
        if cr.high < dr.low or cr.low > dr.high:
            return DIFFER                      # disjoint -> provably different
        return INCOMPARABLE                    # overlapping but not equal

    if dr is not None:
        cq = parse_quantity(claimed) or _leading(claimed)
        if cq is None or cq.unit != dr.unit:
            return INCOMPARABLE
        return MATCH if dr.low <= cq.value <= dr.high else DIFFER

    # Declared is a point, claim is a range.
    return INCOMPARABLE

# A number followed by a unit run, ignoring any trailing annotation ("PO", "q6h").
_LEADING_QTY_RE = re.compile(
    r"^\s*([-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[-+]?[0-9]*\.?[0-9]+)\s*"
    r"([a-zA-Zµ%°]+(?:\s*/\s*[a-zA-Zµ°%]+)*)?")


# Counts distinct number-shaped tokens, to detect ambiguous multi-value claims.
_NUMBER_RE = re.compile(r"(?<![0-9.])[-+]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?(?![0-9])")


# "the model said nothing" markers, kept here so the comparator is safe even when called
# directly rather than through factgate.domain.link's slot-answer filter.
_ABSENCE_RE = re.compile(
    r"^\s*(?:no(?:ne|t|thing)?\b|n/?a\b|null\b|un(?:known|clear|specified|available"
    r"|determined)\b|absent\b|missing\b|nil\b)", re.IGNORECASE)


def _as_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


# Same as _CURRENCY_RE but unanchored at the end, so "$15/mo" yields its leading value
# instead of failing outright and leaving every rate-suffixed price INCOMPARABLE.
_LEADING_CURRENCY_RE = re.compile(
    r"^\s*(US\$|[$£€¥])\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]*\.?[0-9]+)"
    r"\s*([kKmMbB]|thousand|million|billion)?")


def _leading(raw: str) -> Quantity | None:
    m = _LEADING_CURRENCY_RE.match(raw or "")
    if m:
        value = float(m.group(2).replace(",", ""))
        if m.group(3):
            value *= _MAGNITUDE[m.group(3).lower()]
        return Quantity(value, _CURRENCY[m.group(1)])
    m = _LEADING_QTY_RE.match(raw or "")
    if not m:
        return None
    try:
        return Quantity(float(m.group(1).replace(",", "")),
                        re.sub(r"\s*", "", m.group(2) or ""))
    except ValueError:
        return None


_CURRENCY_UNITS = set(_CURRENCY.values()) | set(_CURRENCY_WORDS.values())


def _is_currency(raw: str | None) -> bool | None:
    """True/False if the side parses, None if it does not parse at all."""
    for parse in (parse_quantity, parse_range):
        v = parse(raw)
        if v is not None:
            return v.unit.split("/")[0][:3] in _CURRENCY_UNITS or v.unit[:3] in _CURRENCY_UNITS
    return None


def _currency_mismatch(declared: str, claimed: str | None) -> bool:
    d, c = _is_currency(declared), _is_currency(claimed)
    return d is not None and c is not None and d != c


def compare_values(declared: str, claimed: str | None) -> str:
    """Three-valued comparison. The third value is the safety-critical one.

    MATCH        -- provably the declared value
    DIFFER       -- provably a different value (the gate may BLOCK)
    INCOMPARABLE -- cannot decide (the gate must HOLD)

    Collapsing INCOMPARABLE into DIFFER caused a live false BLOCK: the model stated the
    correct dose as "10 mg/kg PO", the trailing route corrupted the parsed unit, and the
    gate reported that a correct dose contradicted the protocol. Collapsing it into MATCH
    would be worse: "10 mg/kg per day" prefixes identically and means something else.
    """
    if claimed is None:
        return INCOMPARABLE
    # A currency value and a non-currency one are not competing readings of the same
    # slot, they are a category error. Measured: asked "how much is the raise?" on a
    # passage about runway, the extractor answered "18 months". Comparing that as an
    # amount is meaningless, and DIFFER would wrongly imply a contradiction.
    if _currency_mismatch(declared, claimed):
        return INCOMPARABLE
    ranged = _compare_range(declared, claimed)
    if ranged is not None:
        return ranged
    dq, cq = parse_quantity(declared), parse_quantity(claimed)
    if dq is not None and cq is not None:
        if dq.value != cq.value:
            return DIFFER
        return MATCH if dq.unit == cq.unit else INCOMPARABLE

    if dq is not None:
        # Declared is a quantity but the claim did not parse cleanly: retry on its
        # leading quantity so a trailing annotation does not masquerade as a conflict.
        lead = _leading(claimed)
        if lead is None:
            return INCOMPARABLE
        # If the declared magnitude appears ANYWHERE in the claim, the claim is not a
        # contradiction of it. A range ("5 to 10 mg/kg") or a correction ("20 mg is
        # wrong, the correct dose is 10 mg/kg") both lead with a different number while
        # still containing the declared one; reading only the leading number reported a
        # correct dose as a conflict. Counting numbers instead would be too blunt --
        # "20 mg/kg every 6 hours" has two numbers but only one candidate value, and
        # must still block.
        if any(_as_float(t) == dq.value for t in _NUMBER_RE.findall(claimed)):
            return INCOMPARABLE
        if lead.value != dq.value:
            return DIFFER              # one unambiguous, different value -> blockable
        return INCOMPARABLE            # same number, unverified unit tail -> hold

    if cq is not None:
        return INCOMPARABLE            # declared is text, claim is a quantity

    d_norm = " ".join(declared.lower().split())
    c_norm = " ".join(claimed.lower().split())
    if d_norm == c_norm:
        return MATCH
    # Defence in depth: this is public API, so a caller may pass raw model text that
    # never went through the slot-answer filter. An absence marker is not a competing
    # value, and must never be reported as a contradiction.
    if _ABSENCE_RE.match(c_norm) and not _ABSENCE_RE.match(d_norm):
        return INCOMPARABLE
    # A sentence is not a competing value. Without this, a hedge that slipped the
    # slot-answer filter ("The passage never mentions a route") compared unequal to a
    # declared "oral" and BLOCKED -- the gate reporting a contradiction on the basis of
    # the model declining to answer. Short, value-shaped text still DIFFERs, so a real
    # conflict ("oral" vs "intravenous") is unaffected.
    if len(c_norm.split()) > max(4, len(d_norm.split()) + 2):
        return INCOMPARABLE
    return DIFFER


def values_match(declared: str, claimed: str | None) -> bool:
    """True only when the claim is provably the declared value."""
    return compare_values(declared, claimed) == MATCH

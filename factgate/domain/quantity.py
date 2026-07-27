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
    r"^\s*([-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[-+]?[0-9]*\.?[0-9]+)\s*"
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
_CURRENCY_RE = re.compile(
    r"^\s*(US\$|[$£€¥])\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]*\.?[0-9]+)"
    r"\s*([kKmMbB]|thousand|million|billion)?\s*$")
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
        return Quantity(value, _CURRENCY[m.group(1)])
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

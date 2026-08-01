"""Quantity parsing and value comparison.

This module decides whether a stated value matches a declared one. In a dosing domain
that is the difference between blocking a wrong dose and emitting it, so it deliberately
has NO rounding tolerance: 15.4 does not match 15. A near-miss is a miss.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# A number, then an optional unit of one or more words joined by spaces or slashes.
# Multi-word units are not exotic ("breaths per minute", "degrees celsius"); rejecting
# them stopped the demo fact set from loading at all. Whitespace inside the unit is
# stripped during normalisation, so comparison stays exact on both sides.
# A unit TOKEN may contain digits after its first letter: "CO2", "m2", "m3", "H2O",
# "kWh/m2/yr", "W/m2K". Excluding them meant a brewing specification could not declare
# "2.6 volumes CO2" and a building energy certificate could not declare a U-value at all --
# both rejected at LOAD, by the check that a kind=quantity value must parse.
#
# Each token must still START with a letter, so a bare number can never be read as a unit.
# That is what keeps "5 to 10" a RANGE rather than the quantity 5 in units of "to10".
_QTY_RE = re.compile(
    r"^\s*~?\s*([-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[-+]?[0-9]*\.?[0-9]+)\s*"
    r"([a-zA-Zµ%°][a-zA-Zµ%°0-9]*(?:[\s/]+[a-zA-Zµ°%][a-zA-Zµ°%0-9]*)*)?\s*$")


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
    if len(raw) > MAX_VALUE_CHARS:
        return None
    m = _WORDY_RE.match(raw)
    if m and m.group(3).lower() in _CURRENCY_WORDS:
        value = float(m.group(1).replace(",", ""))
        if m.group(2):
            value *= _MAGNITUDE[m.group(2).lower()]
        return Quantity(value, _CURRENCY_WORDS[m.group(3).lower()])
    return None


# Notations that are EXACT arithmetic, not interpretation. Each was measured producing a
# false BLOCK, because a value the parser cannot read is a value whose difference from
# another it cannot prove -- and the text fallback called that a contradiction:
#
#   declared "3:1"                claimed "3 to 1"                        -> BLOCK
#   declared "1.2e17 atoms/cm3"   claimed "120000000000000000 atoms/cm3"  -> BLOCK
#
# Both pairs are the same value. Fractions are exact rational arithmetic, scientific
# notation is exact float, and a ratio is a pair of integers -- no tolerance, no threshold,
# nothing learned, so the verdict layer stays parameter-free.
#
# A mixed number ("2 1/2 inch") is a whole part plus a fraction, which is how machining
# drawings write it.
_FRACTION_RE = re.compile(
    r"^\s*(?:([0-9]+)\s+)?([0-9]+)\s*/\s*([0-9]+)\s*"
    r"([a-zA-Zµ%°][a-zA-Zµ%°/\s]*)?\s*$")
# "1.2e17", "5 x 10^15", "5 * 10^15". The exponent form a document actually uses.
_SCI_RE = re.compile(
    r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*(?:[eE]\s*([-+]?[0-9]+)|"
    r"[x*×]\s*10\s*\^?\s*([-+]?[0-9]+))\s*"
    r"([a-zA-Z0-9µ%°][a-zA-Z0-9µ%°/\s]*)?\s*$")
# COLON FORM ONLY. "3 to 1" was accepted here too, and that was a LEAK I introduced and
# caught before it shipped: "5 to 10" and "1 to 2" are different RANGES that share a
# quotient, so both parsed as ratio 0.5 and compared MATCH. "N to M" is genuinely ambiguous
# between a range and a ratio and nothing local can tell which the author meant; a colon is
# not. "3:1" against "3 to 1" is therefore HELD rather than confirmed -- safe, and no longer
# the false BLOCK it used to be.
#
# A ratio's unit is the shape itself, so two ratios compare only with each other and never
# with a plain number.
_RATIO_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:ratio)?\s*$")


def _exact_notation(raw: str, numeric: bool = False) -> Quantity | None:
    """Fractions, scientific notation and ratios, or None.

    Only when the relation was DECLARED numeric. These notations were read out of every
    value regardless of kind, and that was a leak: an alphanumeric identifier shaped like
    scientific notation was silently reinterpreted as arithmetic.

        relation model_code, kind "text"
        declared "2E2"   claimed "20E1"   -> VERIFIED
        declared "2E2"   claimed "200E0"  -> VERIFIED

    Three different part numbers, all reducing to the float 200.0. Real catalogue codes,
    lot numbers and EIA component markings take exactly this shape. The author's declared
    kind is the statement that a value is a NUMBER, and nothing else can supply it.
    """
    if not numeric:
        return None
    m = _FRACTION_RE.match(raw)
    if m and int(m.group(3)) != 0:
        whole = int(m.group(1)) if m.group(1) else 0
        value = whole + int(m.group(2)) / int(m.group(3))
        return Quantity(value, re.sub(r"\s*", "", m.group(4) or ""))
    m = _SCI_RE.match(raw)
    if m:
        exponent = m.group(2) if m.group(2) is not None else m.group(3)
        try:
            mantissa = float(m.group(1))
            value = mantissa * (10.0 ** int(exponent))
        except (ValueError, OverflowError):
            return None
        # UNDERFLOW is as dangerous as overflow and does not raise. "1e-400" and "1e-500"
        # both became 0.0 and compared MATCH -- two different values confirmed as one.
        # Overflow already returned None via OverflowError; underflow has to be caught by
        # inspecting the result.
        if mantissa != 0 and value == 0:
            return None
        if value in (float("inf"), float("-inf")):
            return None
        return Quantity(value, re.sub(r"\s*", "", m.group(4) or ""))
    m = _RATIO_RE.match(raw)
    if m and float(m.group(2)) != 0:
        # The DENOMINATOR goes in the unit, so a colon value compares only against another
        # with the same denominator. Reducing to a quotient instead was a LEAK, caught the
        # same hour it was written:
        #
        #   declared "14:30" (a time)   claimed "7:15"   -> VERIFIED
        #
        # 14/30 and 7/15 are the same ratio and different times, and nothing local can tell
        # which a colon means -- exactly the ambiguity that already ruled out reading
        # "N to M" as a ratio. Same denominator, so "3:1" against "20:1" is still a
        # provable difference; different denominator is INCOMPARABLE, which is a hold.
        return Quantity(float(m.group(1)), f"ratio:{m.group(2)}")
    return None


def parse_quantity(raw: str | None, numeric: bool = False) -> Quantity | None:
    """Parse "15 mg/kg" -> Quantity(15.0, "mg/kg"). None if not a quantity.

    `numeric` says the relation was declared kind="quantity", which is what admits the
    exact notations (fractions, scientific notation, ratios). It defaults to False so that
    no caller reinterprets a text value arithmetically by accident.
    """
    if not raw or len(raw) > MAX_VALUE_CHARS:
        return None
    cur = _currency_quantity(raw)
    if cur is not None:
        return cur
    exact = _exact_notation(raw, numeric)
    if exact is not None:
        return exact
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


# A value is a short phrase. Anything longer is not a value, and the cap is what keeps the
# parser's cost bounded no matter what a model emits.
#
# MEASURED: a claim consisting of a run of digits cost O(n^2) in the quantity regex --
# 256 chars 5 ms, 512 chars 18.5 ms, 1024 chars 76.6 ms, 2048 chars 303.8 ms, and one
# reported 15.5 SECONDS for a single gate_claim. That is a denial of service reachable
# from a CLAIM, which arrives from the model at runtime, rather than from a fact set the
# operator installs once. Over-long input parses as nothing, which the gate turns into
# HELD.
MAX_VALUE_CHARS = 512

_DIGIT_ANY = re.compile(r"[0-9]|\d", re.UNICODE)

# Characters that render identically, or near enough that no reader distinguishes them, but
# compare unequal byte for byte. Text comparison had no canonicalisation at all, so values a
# human would call the same string were reported as CONTRADICTING each other:
#
#   "Café Ristretto" (NFC) against "Café Ristretto" (NFD)   -> BLOCK
#   "Tenant's option" against "Tenant’s option"             -> BLOCK
#
# A document copied out of a word processor carries curly quotes and en-dashes; a model
# retyping it produces the ASCII forms. Neither is asserting anything the other denies.
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "",
})


def _text_key(s: str) -> str:
    """Canonical form for comparing two TEXT values.

    Unicode composition, quote style and dash style are typography, not meaning. Applied to
    both sides symmetrically, so it can make two spellings of one value equal but can never
    make two different values equal: no character is dropped except zero-width marks, which
    render as nothing.
    """
    return " ".join(unicodedata.normalize("NFC", str(s))
                    .translate(_PUNCT_FOLD).lower().split())

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
# Bounds may be NEGATIVE. Without the sign, "-20 to -15 degrees C" did not parse at all, so
# a cold-chain specification declaring a freezer range under kind=quantity could not be
# LOADED -- the same class as unitless ranges, found the same way, by a genre nobody had
# written yet. Sub-zero ranges are ordinary in cold chain, audio loudness, depth and
# refrigeration.
_RANGE_RE = re.compile(
    r"^\s*(?:between\s+)?(US\$|[$£€¥])?\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmMbB](?![a-zA-Z]))?\s*"
    # "and" is accepted as a separator, so "between $500K and $2M" parses -- and so does a
    # bare "5 and 10 mg/kg". That is deliberate but not free: "5 and 10" could mean two
    # separate values rather than a span. It only matters when the DECLARED value is also
    # a range, and reading the claim as a range is the reading that can be checked.
    r"(?:-|–|—|\bto\b|\band\b)\s*"
    r"(US\$|[$£€¥])?\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmMbB](?![a-zA-Z]))?\s*"
    r"([a-zA-Zµ%°][a-zA-Zµ%°/\s]*)?\s*$")


# "$100M+", "18 months+" -- an open-ended lower bound. Reported by a first-time author as
# a hard load error. Parsing it as the point value would verify that one figure and block
# every larger one, exactly inverting what the document says.
_OPEN_RANGE_RE = re.compile(
    r"^\s*~?\s*(US\$|[$£€¥])?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"([kKmMbB](?![a-zA-Z])|thousand|million|billion)?\s*"
    r"([a-zA-Zµ%°][a-zA-Zµ%°/\s]*)?\+\s*$")


def parse_range(raw: str | None, allow_unitless: bool = False) -> Range | None:
    """Parse a range, or None.

    Reversed bounds are rejected rather than swapped: "10-5" is a typo, and silently
    reinterpreting it would accept a bad declaration. A trailing "+" gives an open upper
    bound.

    A range with NO unit -- "150-400", "4.6-5.2", "0.65-0.72" -- parses only when the
    caller asks for it, and the caller only asks when the domain declared the relation
    kind "quantity". Unitless ranges are ordinary in real documents (pH, water activity,
    platelet counts, ratios, index scores) and a food specification could not even be
    LOADED without this: kind=quantity rejected its own declared value.

    It is opt-in rather than automatic because plenty of TEXT looks like a range. A
    declared "2024-2025" academic year would otherwise acquire range semantics and confirm
    a claim of "2024" as a point inside it. The declared kind is the author saying which
    reading they meant, and that is the only thing that can distinguish the two.
    """
    if not raw or len(raw) > MAX_VALUE_CHARS:
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
        if not unit and not allow_unitless:
            return None
        return Range(low, float("inf"), unit)
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
    if not unit and not allow_unitless:
        return None
    return Range(low, high, unit)


def _leading_range(claimed: str, declared: Range,
                   allow_unitless: bool = False) -> Range | None:
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
        r = parse_range(candidate, allow_unitless)
        if r is None or r.unit != declared.unit:
            continue
        rest = " ".join(tokens[n:]).strip(" ,.;:")
        others = {_as_float(t) for t in _NUMBER_RE.findall(rest)}
        if declared.low in others or declared.high in others:
            return None          # ambiguous: the declared bounds recur later
        return r
    return None


def _compare_range(declared: str, claimed: str, numeric: bool = False) -> str | None:
    """Comparison when either side is a range. None if neither side is one.

    Deliberate asymmetry: a POINT inside a declared RANGE verifies (the document supports
    that figure), but a claimed RANGE against a declared POINT does NOT -- a range cannot
    confirm a specific value, and treating it as a match would let a reader infer the
    whole span is document-supported.
    """
    dr, cr = (parse_range(declared, numeric), parse_range(claimed, numeric))
    # Whether the claim needed a fallback, i.e. carries text the parser did not consume.
    # FOUND BY FUZZING: without this, unexplained trailing text was ignored on the range
    # paths while the point path rejected it, so "US$5,547M+ per query" VERIFIED against a
    # declared "US$5,547M+". Junk may still support a DIFFER (a provably other magnitude
    # is provably other however it is dressed) but it must never support a MATCH.
    inexact = False
    if dr is not None and cr is None:
        cr = _leading_range(claimed, dr, numeric)
        inexact = cr is not None
    if dr is None and cr is None:
        return None

    if dr is not None and cr is not None:
        if dr.unit != cr.unit:
            return INCOMPARABLE
        if cr.high < dr.low or cr.low > dr.high:
            return DIFFER                      # disjoint -> provably different
        if (dr.low, dr.high) == (cr.low, cr.high):
            return INCOMPARABLE if inexact else MATCH
        return INCOMPARABLE                    # overlapping but not equal

    if dr is not None:
        cq = parse_quantity(claimed, numeric)
        clean = cq is not None
        if cq is None:
            cq = _leading(claimed)
        if cq is None or cq.unit != dr.unit:
            return INCOMPARABLE
        if not (dr.low <= cq.value <= dr.high):
            return DIFFER          # outside the range however the claim is dressed
        return MATCH if clean else INCOMPARABLE

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


def compare_values(declared: str, claimed: str | None, numeric: bool = False) -> str:
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
    # Nothing longer than a value can be compared as one. Bounds every regex below.
    if len(declared) > MAX_VALUE_CHARS or len(claimed) > MAX_VALUE_CHARS:
        return INCOMPARABLE
    # A currency value and a non-currency one are not competing readings of the same
    # slot, they are a category error. Measured: asked "how much is the raise?" on a
    # passage about runway, the extractor answered "18 months". Comparing that as an
    # amount is meaningless, and DIFFER would wrongly imply a contradiction.
    if _currency_mismatch(declared, claimed):
        return INCOMPARABLE
    ranged = _compare_range(declared, claimed, numeric)
    if ranged is not None:
        return ranged
    dq, cq = parse_quantity(declared, numeric), parse_quantity(claimed, numeric)
    if dq is not None and cq is not None:
        # UNIT FIRST. This tested the number before the unit, so two quantities in
        # different units were reported as a provable contradiction on the strength of
        # their digits alone:
        #
        #   declared "5 g"  claimed "5000 mg"  -> DIFFER, and the gate BLOCKED
        #
        # 5000 mg IS 5 g. The gate told the user that a correct dose contradicted the
        # protocol. "5 g" against "5000 zz" went the same way, and nothing here knows what
        # zz is. Different units cannot prove a difference; they prove nothing, which is
        # INCOMPARABLE and a hold. No conversion is attempted deliberately -- a conversion
        # table needs a tolerance to survive float arithmetic (0.1 g x 1000 is not exactly
        # 100 mg), and a tolerance is precisely the tuned parameter this verdict layer
        # exists without.
        if dq.unit == cq.unit:
            return MATCH if dq.value == cq.value else DIFFER
        # Units differ. That is usually incomparable, but one case is not: the claim may
        # be the declared unit carrying an ANNOTATION, which parse_quantity glues on
        # ("20 mg/kg PO" parses with unit "mg/kgPO"). The token-leading quantity separates
        # the two -- it reads "mg/kg" from "20 mg/kg PO" and nothing at all from
        # "5000 mg" -- so a wrong dose with a route suffix still blocks while a different
        # scale of the same base unit does not.
        # DIFFER only, never MATCH. An unverified trailing annotation must not confirm a
        # value: "10 mg/kg PO" and "10 mg/kg per day" both lead-match a declared
        # "10 mg/kg" and only one of them means it. Returning MATCH here reintroduced the
        # exact leak this project already fixed once, and the suite caught it.
        lead = _leading(claimed)
        if lead is not None and lead.unit == dq.unit and lead.value != dq.value:
            return DIFFER
        return INCOMPARABLE

    if dq is not None:
        # The claim may be an exact notation we are simply not authorised to read here,
        # because the relation was not declared numeric. Found by the differential prover:
        #
        #   declared "0.5"   claimed "1/2"   ->  DIFFER, and the gate BLOCKED
        #
        # The leading-quantity fallback below read the "1" of "1/2" as the whole value and
        # compared it to 0.5. Recognising the shape without interpreting it is exactly the
        # state in which no difference can be proved.
        if not numeric and _exact_notation(claimed, True) is not None:
            return INCOMPARABLE
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
        # The leading unit must be the DECLARED one before its number can contradict.
        # Without this, a COMPOUND value was blocked against its own equal:
        #
        #   declared "90 minutes"   claimed "1 hr 30 min"   -> DIFFER, and the gate BLOCKED
        #
        # The claim leads with 1 hour, the declared value is 90 minutes, and comparing 1
        # against 90 is comparing hours to minutes. The both-parse branch above already
        # required matching units; this branch did not, so the same mistake survived in
        # the fallback.
        if lead.unit != dq.unit:
            return INCOMPARABLE
        if lead.value != dq.value:
            return DIFFER              # one unambiguous, different value -> blockable
        return INCOMPARABLE            # same number, unverified unit tail -> hold

    if cq is not None:
        return INCOMPARABLE            # declared is text, claim is a quantity

    d_norm = _text_key(declared)
    c_norm = _text_key(claimed)
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
    # One text value CONTAINING the other is not a contradiction, it is a difference in
    # completeness. Both of these BLOCKED, and both claims are faithful:
    #
    #   declared "450 inch-pounds (51 Nm)"  claimed "450 inch-pounds"
    #   declared "Board Certified"          claimed "is Board Certified"
    #
    # The first omits a parenthetical the document supplies, the second carries a leading
    # verb. Neither asserts anything the document denies. A genuinely competing value
    # ("oral" against "intravenous") shares no containment and still DIFFERs.
    if d_norm in c_norm or c_norm in d_norm:
        return INCOMPARABLE
    # Neither side parsed, and at least one carries a NUMBER. The gate cannot do arithmetic
    # on notation it could not read, so it cannot prove these differ -- and asserting a
    # contradiction it cannot prove is the failure this whole three-valued design exists to
    # avoid. Measured on shapes the parser does not cover, before fractions, scientific
    # notation and ratios were added:
    #
    #   declared "3:1"  claimed "3 to 1"  -> DIFFER, and the gate BLOCKED two equal values
    #
    # Purely alphabetic values are unaffected: "oral" against "intravenous" is still a
    # provable difference, because there is no unread arithmetic hiding in either.
    if _DIGIT_ANY.search(d_norm) or _DIGIT_ANY.search(c_norm):
        return INCOMPARABLE
    return DIFFER


def values_match(declared: str, claimed: str | None) -> bool:
    """True only when the claim is provably the declared value."""
    return compare_values(declared, claimed) == MATCH

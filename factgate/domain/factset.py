"""A bounded-domain fact set: the closed vocabulary the gate adjudicates against.

Declaring entities and relations up front is what makes the verdict parameter-free.
Free-text grounding fails because two extractions of one fact share no surface form
(measured: 0/17 relation overlap, docs/HALLUGATE.md). Here both sides are forced onto the
same declared identifiers, so comparison is exact and the gate cannot hallucinate.

Every fact carries a source quote, and `validate_sources` checks each quote really appears
in the corpus, so a wrong fact cannot enter the KB unnoticed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from factgate.domain.quantity import parse_quantity, parse_range


class ValidationError(ValueError):
    """Raised when a fact set is internally incoherent. Never warn-and-continue:
    an incoherent KB makes every downstream verdict meaningless."""


# Quantifiers that can match unboundedly many times. Nesting one inside a group that is
# itself unboundedly quantified is what makes a regex backtrack exponentially.
_OPEN_QUANT = re.compile(r"[*+]|\{\d+,\}|\{(\d+),(\d{2,})\}")


def _reject_catastrophic(pattern: str) -> None:
    """Refuse a value_qualifier whose shape can backtrack exponentially.

    MEASURED: a fact set declaring the qualifier "(a+)+b" takes 4.84 SECONDS to normalise
    a 26-character value, and the cost doubles per character -- 40 characters is hours.
    Fact sets are authored input, and in any deployment where the author is not the
    operator (a customer uploading their own domain, a fact set fetched from a registry)
    that is a denial of service against the gate, delivered as data.

    Qualifiers are deliberately regexes, not literals: real domains declare things like
    r"every \\d+ (?:to \\d+ )?hours?" and r"q\\d+h", and flattening those to literals would
    silently stop them matching. So the fix is not to remove the power but to reject the
    one shape that causes the blowup -- an open-ended quantifier applied to a group that
    already contains an open-ended quantifier. "(?:to \\d+ )?" is fine: `?` matches at most
    once, so it cannot compound.

    Linear scan, no execution of the pattern. Raises ValidationError at LOAD time, which
    is the only moment an operator can still do something about it.
    """
    depth, group_start = 0, []
    for i, ch in enumerate(pattern):
        if ch == "\\":
            continue
        if ch == "(":
            group_start.append(i)
            depth += 1
        elif ch == ")" and group_start:
            start = group_start.pop()
            depth -= 1
            body = pattern[start:i]
            follower = pattern[i + 1:i + 2]
            quantified = follower in "*+" or (
                follower == "{" and _OPEN_QUANT.match(pattern[i + 1:]) is not None)
            if quantified and _OPEN_QUANT.search(body):
                raise ValidationError(
                    f"value_qualifier {pattern!r} nests an unbounded quantifier inside a "
                    f"repeated group, which can backtrack exponentially and hang the gate. "
                    f"Rewrite it without the nesting (for example use \\d+ directly rather "
                    f"than (\\d+)+), or declare the alternatives as separate qualifiers.")


# SI prefixes, longest first so "mc" is tried before "m". Used only by lint(), never by a
# verdict -- this decides what to WARN an author about, not what a value means.
_SI_PREFIX = {"da": 1e1, "mc": 1e-6, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3,
              "c": 1e-2, "d": 1e-1, "h": 1e2, "k": 1e3, "M": 1e6, "G": 1e9}


def _scale_of(unit: str) -> tuple[str, float]:
    """Split a unit into (base, factor) if it carries an SI prefix, else (unit, 1)."""
    for p in sorted(_SI_PREFIX, key=len, reverse=True):
        if unit.startswith(p) and len(unit) > len(p):
            return unit[len(p):], _SI_PREFIX[p]
    return unit, 1.0


# Units whose PHYSICAL DIMENSION is known, for lint only -- never for a verdict, so the
# comparison layer stays parameter-free. Spelling variants of one unit share an entry;
# anything absent is simply unknown and draws no opinion.
#
# The SI-prefix check above only fires when both sides reduce to the SAME base string, so
# it caught mcg -> mg and nothing else. An alias between two UNRELATED units skipped it
# entirely and lint stayed silent while the gate VERIFIED:
#
#   unit_aliases {"mL": "mg"}   declared "15 mg"   claimed "15 mL"   -> VERIFIED
#   unit_aliases {"lbs": "kg"}  declared "150 kg"  claimed "150 lbs" -> VERIFIED
#
# A millilitre is not a milligram under any reading, and 150 lbs is not 150 kg.
# unit spelling -> (dimension, size relative to the dimension's base). Prefixed forms are
# listed explicitly rather than derived, because algorithmic prefix-stripping misreads
# ordinary words ("hrs" begins with the hecto prefix, "mi" with milli).
#
# "pound"/"pounds" are listed as MASS. They are ambiguous with sterling, but leaving them
# out made the alias {"lbs": "pounds"} -- an ordinary spelling variant -- trip the
# known-to-unknown check below. Flagging a money domain that writes {"pounds": "gbp"} is a
# rename; missing {"mcg": "iu"} is a dosing leak. The safe side is the one that errs.
_KNOWN_UNITS: dict[str, tuple[str, float]] = {}
for _dim, _entries in {
    "mass": [(1e-6, ["mcg", "ug", "µg", "microgram", "micrograms"]),
             (1e-3, ["mg", "milligram", "milligrams"]),
             (1.0, ["g", "gram", "grams", "gramme", "grammes"]),
             (1e3, ["kg", "kilogram", "kilograms", "kilo", "kilos"]),
             (453.592, ["lb", "lbs", "pound", "pounds"]),
             (28.3495, ["oz", "ounce", "ounces"])],
    "volume": [(1e-3, ["ml", "millilitre", "millilitres", "milliliter",
                       "milliliters", "cc"]),
               (1.0, ["l", "litre", "litres", "liter", "liters"]),
               (3.78541, ["gal", "gallon", "gallons"])],
    "length": [(1e-3, ["mm", "millimetre", "millimetres", "millimeter", "millimeters"]),
               (1e-2, ["cm", "centimetre", "centimetres", "centimeter", "centimeters"]),
               (1.0, ["m", "metre", "metres", "meter", "meters"]),
               (1e3, ["km", "kilometre", "kilometres", "kilometer", "kilometers"]),
               (0.0254, ["in", "inch", "inches"]),
               (0.3048, ["ft", "foot", "feet"]),
               (1609.34, ["mi", "mile", "miles"])],
    "time": [(1.0, ["s", "sec", "secs", "second", "seconds"]),
             (60.0, ["min", "mins", "minute", "minutes"]),
             (3600.0, ["h", "hr", "hrs", "hour", "hours"]),
             (86400.0, ["d", "day", "days"]),
             (604800.0, ["wk", "week", "weeks"])],
    "currency": [(1.0, ["usd", "dollar", "dollars", "$"]),
                 (2.0, ["gbp", "sterling", "£"]),
                 (3.0, ["eur", "euro", "euros", "€"])],
    "ratio": [(1.0, ["%", "percent", "pct", "percentage"])],
}.items():
    for _factor, _names in _entries:
        for _n in _names:
            _KNOWN_UNITS[_n] = (_dim, _factor)


@dataclass(frozen=True)
class Fact:
    s: str
    r: str
    o: str
    source: str
    # Conditions under which this value applies, e.g. {"indication": "otitis media"}.
    # Empty means unconditional. Real protocols are conditional (a dose depends on
    # indication, a rate on credit tier), and a schema that cannot say so excludes most
    # real source documents.
    when: tuple[tuple[str, str], ...] = ()

    @property
    def conditions(self) -> dict[str, str]:
        return dict(self.when)


def _check_shape(domain, entities, relations, facts, value_qualifiers, unit_aliases,
                 conditions) -> None:
    """Reject a structurally wrong fact set at LOAD, with a typed error naming the field.

    Three of these were previously accepted in SILENCE, and two of them are correctness
    bugs rather than ergonomics:

      entities={"acetaminophen": "paracetamol"}   -- a string where a list of aliases
        belongs. Python iterates the string, so the drug acquires the aliases
        'a','c','e','l','m','o','p','r','t'. Measured: gate_claim(fs, "a", "dose",
        "15 mg/kg") returned VERIFIED. A single letter anywhere in a response resolved to
        a specific drug.

      value_qualifiers="PO"                       -- likewise iterated, compiling the
        qualifiers \\bP\\b and \\bO\\b, which silently strip standalone letters out of every
        value in the domain.

    The remaining cases raised AttributeError or TypeError from somewhere inside the
    library, which a caller wrapping this in a service cannot reasonably catch or explain.
    Everything here raises ValidationError, which is the documented failure mode.
    """
    def need(cond, field, saw, want):
        if not cond:
            raise ValidationError(
                f"{field} must be {want}, got {type(saw).__name__}: {saw!r:.80}")

    # Not required to be non-empty: FactSet.from_dict({}) deliberately loads as an empty
    # gate that holds everything, which is a safe and tested default.
    need(isinstance(domain, str), "domain", domain, "a string")
    need(isinstance(entities, dict), "entities", entities, "a dict of name -> [aliases]")
    for name, aliases in entities.items():
        need(isinstance(name, str), "entity name", name, "a string")
        need(isinstance(aliases, (list, tuple, set)), f"entities[{name!r}]", aliases,
             "a LIST of aliases (a bare string is iterated character by character)")
        for al in aliases:
            need(isinstance(al, str), f"entities[{name!r}] alias", al, "a string")
    need(isinstance(relations, dict), "relations", relations, "a dict of name -> spec")
    for name, spec in relations.items():
        need(isinstance(name, str), "relation name", name, "a string")
        need(isinstance(spec, dict), f"relations[{name!r}]", spec, "a dict")
    need(isinstance(facts, (list, tuple)), "facts", facts, "a list")
    need(value_qualifiers is None or isinstance(value_qualifiers, (list, tuple)),
         "value_qualifiers", value_qualifiers,
         "a LIST of strings (a bare string is iterated character by character)")
    for q in value_qualifiers or []:
        need(isinstance(q, str), "value_qualifier", q, "a string")
    need(unit_aliases is None or isinstance(unit_aliases, dict), "unit_aliases",
         unit_aliases, "a dict of alias -> canonical unit")
    need(conditions is None or isinstance(conditions, (list, tuple)), "conditions",
         conditions, "a list of condition keys")


class FactSet:
    def __init__(self, domain: str, entities: dict[str, list[str]],
                 relations: dict[str, dict], facts: list[Fact],
                 value_qualifiers: list[str] | None = None,
                 unit_aliases: dict[str, str] | None = None,
                 conditions: list[str] | None = None):
        _check_shape(domain, entities, relations, facts, value_qualifiers,
                     unit_aliases, conditions)
        self.domain = domain
        self.entities = entities
        self.relations = relations
        self.facts = facts
        # Both default to empty: a domain that declares nothing gets no normalisation,
        # so adding this feature cannot change an existing fact set's verdicts.
        self.unit_aliases = unit_aliases or {}
        # Unit aliases were applied by blind substring replacement, which CORRUPTS values.
        # Found by a fresh domain, not by a test: a freight rate sheet declaring
        # {"mi": "miles"} turned the claim "$2.85 per mile" into "$2.85 per miles le",
        # because "mile" contains "mi". The claim was character-for-character identical to
        # the declared value and was HELD. The same sheet's {"hr": "hour"} would turn
        # "threshold" into "thoursesold".
        #
        # Word-bounded, exactly as value_qualifiers already are, and longest alias first so
        # "hrs" is consumed before "hr" can bite into it rather than by dict luck.
        self._unit_alias_patterns = []
        for alias in sorted(self.unit_aliases, key=len, reverse=True):
            a = str(alias)
            lead = r"\b" if a[:1].isalnum() else ""
            tail = r"\b" if a[-1:].isalnum() else ""
            self._unit_alias_patterns.append(
                (re.compile(rf"{lead}{re.escape(a)}{tail}"), self.unit_aliases[alias]))
        self.conditions = conditions or []
        self._fingerprint: str | None = None
        self._qualifiers = []
        self.qualifier_warnings: list[str] = []
        for q in value_qualifiers or []:
            _reject_catastrophic(q)
            # \b only asserts between a word and a non-word character, so a boundary is
            # applied only where the pattern edge is actually alphanumeric -- otherwise a
            # qualifier like "/mo" could never match.
            lead = r"\b" if q[:1].isalnum() else ""
            tail = r"\b" if q[-1:].isalnum() else ""
            try:
                self._qualifiers.append(
                    re.compile(rf"{lead}(?:{q}){tail}", re.IGNORECASE))
                # A pattern that compiles but contains metacharacters may not mean what
                # the author intended: "C++" is a valid regex ("C" then one-or-more "+")
                # and strips only "C+" from "$5 C++". The field cannot tell intent, so it
                # says so rather than guessing.
                if any(ch in q for ch in "+*?[](){}|^$\\") and q not in ("/mo",):
                    self.qualifier_warnings.append(
                        f"qualifier {q!r} contains regex metacharacters and is being "
                        f"treated as a PATTERN; if you meant it literally, escape it")
            except re.error:
                # Not a valid regex -- treat it as a literal. "+ benefits" and "C++" are
                # ordinary English that happen to contain metacharacters, and refusing to
                # load with a regex parser error makes the author debug a field they did
                # not know was a regex. A literal reading can never be wrong, only narrow.
                self._qualifiers.append(
                    re.compile(rf"{lead}(?:{re.escape(q)}){tail}", re.IGNORECASE))
        # An alias claimed by two entities would resolve arbitrarily and silently attach
        # a claim to the wrong entity -- in a dosing domain, the wrong drug. Refuse it.
        self._alias: dict[str, str] = {}
        for canon, aliases in entities.items():
            for surface in (canon, *aliases):
                key = surface.lower()
                owner = self._alias.get(key)
                if owner is not None and owner != canon:
                    raise ValidationError(
                        f"ambiguous entity name {surface!r}: claimed by both "
                        f"{owner!r} and {canon!r}")
                self._alias[key] = canon
        # All variants for a slot, so an unconditioned query can see that the slot is
        # conditional and hold rather than silently pick one.
        self._variants: dict[tuple[str, str], list[Fact]] = {}
        for f in facts:
            self._variants.setdefault((f.s, f.r), []).append(f)

    # ------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, d: dict) -> "FactSet":
        if not isinstance(d, dict):
            raise ValidationError(f"fact set must be an object, got {type(d).__name__}")
        entities = d.get("entities", {})
        relations = d.get("relations", {})
        # Shape-check BEFORE walking the facts: the fact loop dereferences
        # relations[r].get(...), so a malformed relation spec raised AttributeError from
        # inside the loop rather than the typed error the constructor would have given.
        _check_shape(d.get("domain", ""), entities, relations, d.get("facts", []),
                     d.get("value_qualifiers"), d.get("unit_aliases"),
                     d.get("conditions"))
        declared_conditions = list(d.get("conditions") or [])
        facts, seen = [], {}
        for raw in d.get("facts", []):
            # Report malformed input as ValidationError, not KeyError, so a caller can
            # tell "your fact file is wrong" apart from "the library crashed".
            if not isinstance(raw, dict) or not {"s", "r", "o"} <= set(raw):
                raise ValidationError(
                    f"fact must be an object with keys s, r, o -- got {raw!r}")
            s, r, o = raw["s"], raw["r"], raw["o"]
            if s not in entities:
                raise ValidationError(f"fact declares undeclared entity {s!r}")
            if r not in relations:
                raise ValidationError(f"fact declares undeclared relation {r!r}")
            if (relations[r].get("kind") == "quantity"
                    and parse_quantity(o, numeric=True) is None
                    and parse_range(o, allow_unitless=True) is None):
                raise ValidationError(
                    f"relation {r!r} is kind=quantity but value {o!r} is not a quantity")
            when = raw.get("when") or {}
            if not isinstance(when, dict):
                raise ValidationError(f"`when` must be an object -- got {when!r}")
            for key in when:
                if key not in declared_conditions:
                    raise ValidationError(
                        f"fact uses undeclared condition {key!r}; declare it in "
                        f"`conditions` (a typo here would silently never match)")
            # Uniqueness is per (subject, relation, condition-set): two values under the
            # SAME condition is still incoherent, two under different conditions is the
            # whole point.
            key = (s, r, tuple(sorted(when.items())))
            if key in seen and seen[key] != o:
                raise ValidationError(
                    f"conflict: {s!r}/{r!r} under {when or 'no condition'} declared as "
                    f"{seen[key]!r} and {o!r}")
            seen[key] = o
            facts.append(Fact(s, r, o, raw.get("source", ""),
                              tuple(sorted(when.items()))))
        return cls(d.get("domain", "unnamed"), entities, relations, facts,
                   value_qualifiers=d.get("value_qualifiers"),
                   unit_aliases=d.get("unit_aliases"),
                   conditions=declared_conditions)

    @classmethod
    def from_json(cls, path: str | Path) -> "FactSet":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------- queries
    def resolve_entity(self, mention: str | None) -> str | None:
        """Map a surface mention onto a declared entity, or None if it is not ours.

        None is the important case: it means the claim is outside this domain, which the
        gate must HOLD rather than pass.
        """
        if not mention:
            return None
        return self._alias.get(mention.strip().lower())

    def variants(self, entity: str, relation: str) -> list[Fact]:
        """Every declared value for this slot, across conditions."""
        return self._variants.get((entity, relation), [])

    def lookup(self, entity: str, relation: str,
               context: dict[str, str] | None = None) -> Fact | None:
        """The single applicable fact, or None if the slot is unknown OR ambiguous.

        Returning None for "ambiguous" is deliberate: the gate turns None into HELD, so a
        conditional slot queried without its condition can never verify. Picking a variant
        would confirm an otitis-media dose in a standard-indication context.
        """
        vs = self._variants.get((entity, relation), [])
        if not vs:
            return None
        if len(vs) == 1 and not vs[0].when:
            return vs[0]
        ctx = {k.lower(): str(v).lower() for k, v in (context or {}).items()}
        # A CONDITIONAL variant applies only when every one of its conditions is met.
        conditional = [f for f in vs if f.when and
                       all(ctx.get(k.lower()) == str(v).lower() for k, v in f.when)]
        if len(conditional) == 1:
            return conditional[0]          # most specific rule wins
        if conditional:
            return None                    # several apply at once: ambiguous
        # No conditional variant applies. An UNCONDITIONAL default is not automatically
        # the answer: `all()` over an empty condition set is True, so the default matched
        # every query and this method returned it even when the caller had supplied no
        # context at all -- silently picking a variant, which the docstring above says it
        # must never do.
        #
        # MEASURED on a payroll sheet declaring a threshold of $200,000 unconditionally
        # and $250,000 for married filing jointly. Queried with no filing status, the gate
        # VERIFIED $200,000. Harmless there; in the dosing case the docstring warns about
        # it confirms the standard dose for a patient who may qualify for the conditional
        # one. Without context the slot is ambiguous, which is HELD.
        if not ctx:
            return None
        unconditional = [f for f in vs if not f.when]
        return unconditional[0] if len(unconditional) == 1 else None

    # A model-produced value is a short phrase. Anything longer is not a value, and a
    # length cap keeps qualifier matching bounded no matter what a pattern does.
    MAX_VALUE_CHARS = 512

    def normalise_value(self, value: str | None) -> str | None:
        """Strip qualifiers and apply unit aliases the DOMAIN declared irrelevant.

        Measured: a model answered "10 mg/kg PO every 6 hours" -- the correct dose -- and
        the gate held it, because route and frequency corrupted the parsed unit. Stripping
        that is safe only because the domain author said those tokens do not change what
        the value means. Nothing is stripped by inference: an undeclared qualifier such as
        "per day" survives and the claim stays HELD, which is the right answer because it
        genuinely means something else.
        """
        if value is None:
            return None
        if len(value) > self.MAX_VALUE_CHARS:
            return value              # too long to be a value; leave it to fail closed
        out = value
        for pat in self._qualifiers:
            out = pat.sub(" ", out)
        for pat, canon in self._unit_alias_patterns:
            out = pat.sub(f" {canon} ", out)
        # Remove brackets this stripping just emptied: "$79(download)" -> "$79( )".
        # Tidying the strip's own residue, not inferring meaning.
        # Repeat until stable: one pass removes one level, so "(( ))" became "( )" and a
        # second call changed it again. normalise_value must be idempotent -- a value's
        # meaning cannot depend on how many times it has been normalised, and both the
        # gate and the suggestion machinery call it more than once on the same string.
        for _ in range(8):
            reduced = re.sub(r"[(\[{]\s*[)\]}]", " ", out)
            if reduced == out:
                break
            out = reduced
        # Same idea for separators the stripping orphaned. MEASURED: a domain declaring
        # "of the amount advanced" and "deducted at closing" turned the claim "2 percent of
        # the amount advanced, deducted at closing" into "2 percent ," -- a dangling comma
        # that stops the value parsing, so a correct reading was HELD. Only punctuation
        # left stranded by the strip is touched; "5,000 mg" keeps its comma because that
        # one is between digits, not floating at an edge.
        out = re.sub(r"\s+([,;])", r"\1", out)
        out = out.strip(" ,;")
        return " ".join(out.split()) or value

    @property
    def fingerprint(self) -> str:
        """Stable digest of everything that can change a verdict.

        A verdict is only auditable if you can prove which fact set produced it. Covers
        facts (including conditions), qualifiers and unit aliases -- qualifiers change
        verdicts, so leaving them out would let a fact set be altered without the
        fingerprint moving. Order-independent: reordering facts is not a change.

        CACHED, because every Verdict carries it and this recomputed a sha256 over the
        whole fact set on EVERY adjudication. Measured at 5,000 facts: gate_claim took
        2.809 ms, of which 2.7 ms was this property -- the gate was 100x slower than its
        own logic, and linear in fact count where the lookups it performs are O(1). A fact
        set is immutable after construction, so there is nothing to invalidate.
        """
        if self._fingerprint is not None:
            return self._fingerprint
        payload = {
            "facts": sorted((f.s, f.r, f.o, list(f.when)) for f in self.facts),
            "qualifiers": sorted(p.pattern for p in self._qualifiers),
            "unit_aliases": sorted(self.unit_aliases.items()),
            "conditions": sorted(self.conditions),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._fingerprint = hashlib.sha256(blob.encode()).hexdigest()[:16]
        return self._fingerprint

    # ---------------------------------------------------------- validation
    # Time words almost always change what a value means per unit time, so declaring one
    # irrelevant is the classic way to break a fact set silently.
    _RATE_WORDS = ("day", "daily", "week", "weekly", "month", "monthly", "year",
                   "yearly", "annual", "annually", "hour", "hourly", "dose", "per")

    def lint(self) -> list[dict]:
        """Static checks on the declarations themselves.

        `value_qualifiers` is trusted as declared, which makes it the sharpest footgun in
        the schema: declaring "per day" irrelevant would let a daily total verify against
        a per-dose fact. This catches the provable form of that (two declared values that
        normalise to the same string) as an error, and flags rate-bearing qualifiers as a
        warning even when nothing collides yet.
        """
        problems: list[dict] = []

        # Error: a qualifier that makes two distinct declared values indistinguishable.
        by_norm: dict[str, list[Fact]] = {}
        for f in self.facts:
            by_norm.setdefault(self.normalise_value(f.o) or f.o, []).append(f)
        for norm, group in by_norm.items():
            originals = {f.o for f in group}
            if len(originals) > 1:
                culprits = [p.pattern for p in self._qualifiers
                            if any(p.search(o) for o in originals)]
                problems.append({
                    "level": "error",
                    "message": (f"value_qualifiers {culprits or '(unknown)'} collapse "
                                f"distinct declared values {sorted(originals)} to the "
                                f"same normalised form {norm!r}; the gate would verify "
                                f"one against the other"),
                })

        # ERROR: a unit alias that RESCALES. An author writing {"mcg": "mg"} makes every
        # microgram claim normalise onto a milligram fact, and the gate then VERIFIES a
        # 1000-fold dosing error with no way to know. Measured: declared "15 mg", claimed
        # "15 mcg", verdict VERIFIED, lint clean. Aliases exist to reconcile SPELLINGS of
        # one unit, never to equate two.
        for alias, canon in self.unit_aliases.items():
            a_known = _KNOWN_UNITS.get(str(alias).strip().lower())
            c_known = _KNOWN_UNITS.get(str(canon).strip().lower())
            if a_known and c_known and a_known != c_known:
                same_dim = a_known[0] == c_known[0]
                detail = (f"the same {a_known[0]} in different units, a factor of "
                          f"{max(a_known[1], c_known[1]) / min(a_known[1], c_known[1]):g}"
                          if same_dim else
                          f"{a_known[0]} and {c_known[0]}, which are not even the same "
                          f"kind of quantity")
                problems.append({
                    "level": "error",
                    "message": (f"unit_alias {alias!r} -> {canon!r} equates {detail}. "
                                f"Every claim written in {alias!r} would verify against a "
                                f"fact declared in {canon!r}, unconverted. Aliases "
                                f"reconcile spellings of one unit, never two."),
                })
                continue
            # ONE side known, the other not, and they are not the same word. The dimension
            # check abstains on unknown units, and abstaining was itself a leak:
            #
            #   unit_aliases {"mcg": "iu"}   declared "15 iu"   claimed "15 mcg"
            #   -> VERIFIED, lint clean
            #
            # IU is not in the table and never can be: it is a biological activity unit
            # whose conversion depends on the SUBSTANCE (1 mcg of vitamin D is 40 IU, and
            # vitamin E differs again). The same is true of mEq, mmol, tsp, drops and
            # puffs. An alias joining a unit we understand to one we do not is exactly the
            # case that cannot be checked, so it must be refused rather than waved through.
            a_txt, c_txt = str(alias).strip().lower(), str(canon).strip().lower()
            # A multi-word spelling of a known unit is still that unit: "US gallons" is
            # gallons. Look inside the unknown side for a word naming the SAME unit before
            # complaining, or every unlisted spelling becomes a false alarm.
            known_side = a_known or c_known
            other = c_txt if a_known else a_txt
            names_same_unit = any(_KNOWN_UNITS.get(w) == known_side
                                  for w in re.findall(r"[a-zµ%°]+", other))
            if (bool(a_known) != bool(c_known) and not names_same_unit
                    and a_txt.rstrip("s") != c_txt.rstrip("s")):
                known, unknown = (a_txt, c_txt) if a_known else (c_txt, a_known and "" or a_txt)
                problems.append({
                    "level": "error",
                    "message": (f"unit_alias {alias!r} -> {canon!r} joins {known!r}, a unit "
                                f"with a known physical size, to one that is not a "
                                f"recognised unit. That cannot be checked, and units like "
                                f"IU, mEq and mmol convert differently per substance. "
                                f"Declare both sides as spellings of the SAME unit, or "
                                f"drop the alias and let the values compare as written."),
                })
                continue
            a_base, a_f = _scale_of(str(alias))
            c_base, c_f = _scale_of(str(canon))
            if a_base == c_base and a_f != c_f:
                problems.append({
                    "level": "error",
                    "message": (f"unit_alias {alias!r} -> {canon!r} maps one scale of "
                                f"{a_base!r} onto another (a factor of "
                                f"{max(a_f, c_f) / min(a_f, c_f):g}). Every claim using "
                                f"{alias!r} would verify against a fact declared in "
                                f"{canon!r}. Aliases reconcile spellings of one unit, "
                                f"never two different units."),
                })

        # ERROR: a slot with exactly ONE declared value that still carries a condition.
        # The condition cannot select between alternatives because there are none, but it
        # makes lookup() return None -- so the slot can never verify unless the caller
        # supplies context, and a caller who knew the context would not need the gate.
        #
        # MEASURED on a utility tariff declaring six rates as entities that already encode
        # the condition ("Summer On-Peak Energy", when={season: summer, period: on-peak}).
        # Every one was unverifiable, which was the whole of that domain's 46% hold rate.
        by_slot: dict[tuple[str, str], list[Fact]] = {}
        for f in self.facts:
            by_slot.setdefault((f.s, f.r), []).append(f)
        for (s_, r_), group in by_slot.items():
            if len(group) == 1 and group[0].when:
                keys = sorted(k for k, _ in group[0].when)
                problems.append({
                    "level": "error",
                    "message": (f"{s_!r}/{r_!r} declares ONE value conditional on {keys}, "
                                f"with no alternative for the condition to select. The "
                                f"slot can never verify unless the caller supplies "
                                f"{keys}. Either drop `when`, or declare the alternative "
                                f"values on the same entity so the condition discriminates."),
                })

        # Warning: the fact's own source states a MORE SPECIFIC value than was declared.
        # Real, from the shipped bench data: declared "$100M" against the source "GPT-4
        # cost ~$100M+.". The document says at least $100M; the declaration says exactly
        # $100M. A model that reads the document correctly and answers "$100M+" is then
        # held, because a range cannot confirm a point -- the gate is right and the fact
        # set is wrong, which is the hardest kind of over-block for an author to diagnose.
        for f in self.facts:
            flat = " ".join(str(f.source).split())
            if re.search(rf"{re.escape(str(f.o))}\s*\+", flat):
                problems.append({
                    "level": "warning",
                    "message": (f"fact {f.s!r}/{f.r!r} declares {f.o!r} but its own source "
                                f"writes {f.o + '+'!r} (an open-ended range). A correct "
                                f"reading of the document will be HELD, because a range "
                                f"never confirms a declared point. Declare {f.o + '+'!r}."),
                })

        # Warning: rate-bearing qualifiers, which usually do change meaning.
        for pat in self._qualifiers:
            words = re.findall(r"[a-z]+", pat.pattern.lower())
            hit = [w for w in words if w in self._RATE_WORDS]
            if hit:
                problems.append({
                    "level": "warning",
                    "message": (f"qualifier {pat.pattern!r} contains time/rate wording "
                                f"{hit}; stripping it may change what the value means "
                                f"per unit time"),
                })
        return problems

    def validate_sources(self, corpus: str) -> tuple[int, list[Fact]]:
        """Check every fact's source quote actually appears in the corpus.

        Returns (n_validated, unquoted_facts). Whitespace is normalised so a quote
        spanning a line break still matches.
        """
        flat = " ".join(corpus.split())
        ok, missing = 0, []
        for f in self.facts:
            quote = " ".join(f.source.split())
            if quote and quote in flat:
                ok += 1
            else:
                missing.append(f)
        return ok, missing

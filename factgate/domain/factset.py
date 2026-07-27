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


class FactSet:
    def __init__(self, domain: str, entities: dict[str, list[str]],
                 relations: dict[str, dict], facts: list[Fact],
                 value_qualifiers: list[str] | None = None,
                 unit_aliases: dict[str, str] | None = None,
                 conditions: list[str] | None = None):
        self.domain = domain
        self.entities = entities
        self.relations = relations
        self.facts = facts
        # Both default to empty: a domain that declares nothing gets no normalisation,
        # so adding this feature cannot change an existing fact set's verdicts.
        self.unit_aliases = unit_aliases or {}
        self.conditions = conditions or []
        self._qualifiers = []
        self.qualifier_warnings: list[str] = []
        for q in value_qualifiers or []:
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
        entities = d.get("entities", {})
        relations = d.get("relations", {})
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
                    and parse_quantity(o) is None and parse_range(o) is None):
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
        matches = [f for f in vs
                   if all(ctx.get(k.lower()) == str(v).lower() for k, v in f.when)]
        return matches[0] if len(matches) == 1 else None

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
        out = value
        for pat in self._qualifiers:
            out = pat.sub(" ", out)
        for alias, canon in self.unit_aliases.items():
            out = out.replace(alias, f" {canon} ")
        # Remove brackets this stripping just emptied: "$79(download)" -> "$79( )".
        # Tidying the strip's own residue, not inferring meaning.
        out = re.sub(r"[(\[{]\s*[)\]}]", " ", out)
        return " ".join(out.split()) or value

    @property
    def fingerprint(self) -> str:
        """Stable digest of everything that can change a verdict.

        A verdict is only auditable if you can prove which fact set produced it. Covers
        facts (including conditions), qualifiers and unit aliases -- qualifiers change
        verdicts, so leaving them out would let a fact set be altered without the
        fingerprint moving. Order-independent: reordering facts is not a change.
        """
        payload = {
            "facts": sorted((f.s, f.r, f.o, list(f.when)) for f in self.facts),
            "qualifiers": sorted(p.pattern for p in self._qualifiers),
            "unit_aliases": sorted(self.unit_aliases.items()),
            "conditions": sorted(self.conditions),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

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

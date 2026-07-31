"""Soak the whole pipeline and assert the safety invariants on every LIVE verdict.

The benchmark measures rates. The property tests check invariants on generated strings.
Neither checks invariants on verdicts produced by a real model reading real prose, which is
the only configuration that will actually run in production.

This walks every available domain, drives live extraction, and asserts on each verdict:

  S1  VERIFIED is re-derivable without the gate's own comparison
  S2  VERIFIED implies the value occurs in the passage it came from
  S3  BLOCK implies a provable difference
  S4  an out-of-domain entity is never anything but HELD
  S5  no verdict is ever absent, and no call raises
  S6  every verdict carries the fact set's fingerprint, for audit

    python scripts/soak.py --model qwen2.5:14b --minutes 40

Exit code is non-zero if any invariant fails. Rates are reported but are NOT the pass
criterion -- a slow gate is a tuning problem, an unsound one is a defect.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgate.domain.factset import FactSet, ValidationError
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim
from factgate.domain.link import link_targeted, value_is_grounded
from factgate.domain.quantity import (DIFFER, MATCH, compare_values, parse_quantity,
                                      parse_range)
from factgate.llm import ExtractionUnavailable, ollama

REPO = Path(__file__).resolve().parents[1]
ASK = ("Using ONLY the document below, answer in one sentence.\n\n"
       "DOCUMENT: {corpus}\n\nQUESTION: {q}\nANSWER:")


def equal_by_reparse(a: str, b: str) -> bool:
    """Independent equality, deliberately not the code under test."""
    qa, qb = parse_quantity(a), parse_quantity(b)
    if qa is not None and qb is not None:
        return qa.value == qb.value and qa.unit == qb.unit
    ra, rb = parse_range(a), parse_range(b)
    if ra is not None and rb is not None:
        return (ra.low, ra.high, ra.unit) == (rb.low, rb.high, rb.unit)
    if ra is not None and qb is not None:
        return ra.unit == qb.unit and ra.low <= qb.value <= ra.high
    if all(x is None for x in (qa, qb, ra, rb)):
        return " ".join(a.lower().split()) == " ".join(b.lower().split())
    return False


def question_for(fs: FactSet, s: str, r: str) -> str:
    tpl = fs.relations.get(r, {}).get("question")
    return (tpl.replace("{entity}", s) if tpl
            else f"what is the {r.replace('_', ' ')} of {s}?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--domains", default="")
    a = ap.parse_args()

    files = ([REPO / "data/domains" / f"{n}.json" for n in a.domains.split(",") if n]
             or sorted((REPO / "data/domains").glob("*.json")))
    deadline = time.time() + a.minutes * 60
    violations: list[str] = []
    verdicts: Counter = Counter()
    n_claims = 0
    transport_failures = 0

    for f in files:
        if time.time() > deadline:
            print(f"\ndeadline reached; {f.name} onward not exercised")
            break
        try:
            spec = json.loads(f.read_text(encoding="utf-8"))
            fs = FactSet.from_dict(spec)
        except (ValidationError, ValueError) as e:
            violations.append(f"{f.name}: domain failed to load: {e}")
            continue
        print(f"\n{f.name}: {len(fs.facts)} facts", flush=True)

        for fact in fs.facts:
            if time.time() > deadline:
                break
            try:
                answer = ollama(a.model, ASK.format(
                    corpus=spec["corpus"], q=question_for(fs, fact.s, fact.r)), 120)
            except ExtractionUnavailable:
                transport_failures += 1
                continue

            try:
                claims = link_targeted(answer, fs, a.model)
            except ExtractionUnavailable:
                transport_failures += 1
                continue

            for entity, relation, value in claims:
                n_claims += 1
                try:
                    v = gate_claim(fs, entity, relation, value)
                except Exception as e:                      # S5
                    violations.append(f"{f.name} {fact.s}/{fact.r}: raised {e!r}")
                    continue
                verdicts[v.status] += 1

                if v.status not in (VERIFIED, BLOCK, HELD):                    # S5
                    violations.append(f"{f.name}: bad verdict {v.status!r}")
                if not v.factset_fingerprint:                                  # S6
                    violations.append(f"{f.name}: verdict without fingerprint")
                if v.status == VERIFIED:
                    declared = fs.lookup(entity if fs.resolve_entity(entity) else "",
                                         relation)
                    target = v.declared or (declared.o if declared else "")
                    # S1 admits a second route since the residue rule landed: a claim may
                    # be VERIFIED because the fact's own source sentence states it in
                    # full. Re-derived here independently of residue.py -- plain
                    # containment in the source, not the rule's clause logic.
                    #
                    # BOTH sides are normalised, and this check itself is why. It compared
                    # the RAW declared value against the NORMALISED claim and reported
                    # three violations on a lab sheet where declared and claimed were
                    # BYTE-IDENTICAL ("20 K/uL" against "20 K/uL"), because the domain's
                    # unit aliases rewrote one side only. That is the same asymmetry that
                    # was a real bug in the gate; the harness simply had not been updated
                    # with it. A check that has to be corrected after the code is a check
                    # that was doing its job -- it failed loudly rather than agreeing.
                    norm = fs.normalise_value(value) or ""
                    target_norm = fs.normalise_value(target) or ""
                    src = " ".join((declared.source if declared else "").casefold().split())
                    quoted = bool(src) and any(
                        " ".join(str(c).casefold().split()) in src for c in (norm, value))
                    if not equal_by_reparse(target_norm, norm) and not quoted:
                        violations.append(                                     # S1
                            f"{f.name} {entity}/{relation}: VERIFIED neither re-derivable "
                            f"nor quoted from source declared={target!r} claimed={value!r}")
                    if not value_is_grounded(value, answer):                   # S2
                        violations.append(
                            f"{f.name} {entity}/{relation}: VERIFIED value not in the "
                            f"passage: {value!r}")
                if v.status == BLOCK and v.declared:                           # S3
                    if equal_by_reparse(v.declared, fs.normalise_value(value) or ""):
                        violations.append(
                            f"{f.name} {entity}/{relation}: BLOCK on equal values "
                            f"{v.declared!r} vs {value!r}")

            # S4: an entity outside the domain must always hold
            for bogus in ("__not_a_declared_entity__", "", None):
                out = gate_claim(fs, bogus, fact.r, fact.o)
                if out.status != HELD:
                    violations.append(f"{f.name}: out-of-domain {bogus!r} -> {out.status}")

    print("\n" + "=" * 70)
    print(f"SOAK  model={a.model}  claims adjudicated={n_claims}")
    print(f"  verdicts: {dict(verdicts)}")
    if transport_failures:
        print(f"  transport failures (not invariant violations): {transport_failures}")
    if violations:
        print(f"\n  INVARIANT VIOLATIONS: {len(violations)}")
        for v in violations[:20]:
            print(f"    - {v}")
        if len(violations) > 20:
            print(f"    ... and {len(violations) - 20} more")
        print("=" * 70)
        return 1
    print("  INVARIANTS HELD on every live verdict")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

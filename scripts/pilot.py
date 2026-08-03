"""A supervised pilot: run the gate as if deployed, on one document, and record every
verdict for a human to confirm.

Every prior harness measures a rate. A pilot is different: it simulates ACTUAL USE and
produces a per-verdict record a human reviews, because the question a deployment answers is
not "what is the leak rate" but "can a reviewer trust this enough to stop reading every
answer, and what does it cost them when they can't."

The flow, once per natural question:

    a person asks a natural question about the document
      -> a model answers in prose
        -> the extractor turns the answer into a claim in the declared vocabulary
          -> the gate returns VERIFIED / BLOCK / HELD
            -> the SUPERVISOR reads the verdict against the document and confirms it

The report separates the two things a pilot must not conflate:

  REVIEWER LOAD   of the questions asked, how many did the gate settle (VERIFIED) versus
                  hand to a human (HELD or BLOCK). This is the efficiency story: a gate that
                  holds everything is safe and useless.
  TRUST BREACH    a claim the gate VERIFIED that does NOT match the document. This is the
                  only outcome that breaks a deployment, and it must be zero. Computed
                  decidably against the corpus, not judged.

Ground truth for a pilot is THE DOCUMENT, exactly as it is for a real reviewer: their job is
"does the gate's verdict match what this document says", not "is the document true about the
world". validate_sources already guarantees every declared value is quoted from the corpus.

    python scripts/pilot.py --domain data/domains/pilot_cold_chain.json --model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgate.domain.factset import FactSet
from factgate.domain.gate import BLOCK, HELD, VERIFIED, gate_claim
from factgate.domain.link import link_targeted, value_is_grounded
from factgate.domain.quantity import DIFFER, compare_values

REPO = Path(__file__).resolve().parents[1]

# The framings a person actually types -- none names the relation as a field.
FRAMINGS = [
    "A colleague asked me about the {relation} for {entity}. What should I tell them?",
    "Summarise what this document says about the {relation} of {entity}.",
    "What do I need to know about the {relation} of {entity} before I act on it?",
]
PROMPT = ("Answer using ONLY the document below, in two or three sentences of normal "
          "prose.\n\nDOCUMENT: {corpus}\n\nQUESTION: {question}\nANSWER:")

_NUM = re.compile(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?")


def contradicts_document(answer: str, fact, fs: FactSet) -> bool:
    """Does the answer state a value for THIS slot that PROVABLY differs from the declared
    one? Decidable, conservative: fires only on a number that differs under the declared
    unit and is not a figure belonging to another declared fact."""
    unit = re.sub(r"^[-+]?[0-9][0-9,.]*\s*", "", fact.o).strip()
    declared_nums = {m.group(0).replace(",", "") for m in _NUM.finditer(fact.o)}
    other = {n for f in fs.facts if (f.s, f.r) != (fact.s, fact.r)
             for n in _NUM.findall(f.o)}
    for m in _NUM.finditer(answer):
        if m.group(0).replace(",", "") in declared_nums:
            return False
    for m in _NUM.finditer(answer):
        if m.group(0).replace(",", "") in other:
            continue
        trailing = answer[m.end():m.end() + 24].strip().lower()
        if unit and not trailing.startswith(unit.lower()[:3]):
            continue
        if compare_values(fact.o, f"{m.group(0)} {unit}".strip()) == DIFFER:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--framings", type=int, default=2)
    a = ap.parse_args()

    from factgate.llm import ExtractionUnavailable, ollama

    spec = json.loads(Path(a.domain).read_text(encoding="utf-8"))
    fs = FactSet.from_dict(spec)
    ok, missing = fs.validate_sources(spec["corpus"])
    if missing:
        raise SystemExit(f"REFUSING: {len(missing)} facts are not quoted from the corpus")
    errs = [p for p in fs.lint() if p["level"] == "error"]
    if errs:
        raise SystemExit(f"REFUSING: fact set has {len(errs)} lint errors: {errs[0]['message']}")

    print(f"PILOT  domain={fs.domain}  facts={len(fs.facts)}  quotes validated={ok}/{len(fs.facts)}")
    print(f"       {len(fs.facts)} facts x {a.framings} framings = "
          f"{len(fs.facts) * a.framings} questions\n", flush=True)

    records = []
    counts = {VERIFIED: 0, BLOCK: 0, HELD: 0, "no-answer": 0}
    breaches = []
    t0 = time.time()

    for fact in fs.facts:
        for j in range(a.framings):
            q = FRAMINGS[j % len(FRAMINGS)].format(
                entity=fact.s, relation=fact.r.replace("_", " "))
            try:
                answer = ollama(a.model, PROMPT.format(corpus=spec["corpus"],
                                                       question=q), 150)
            except ExtractionUnavailable:
                counts["no-answer"] += 1
                continue

            wrong = contradicts_document(answer, fact, fs)
            try:
                claims, unresolved = link_targeted(answer, fs, a.model,
                                                   report_unresolved=True)
            except ExtractionUnavailable:
                counts["no-answer"] += 1
                continue
            mine = [gate_claim(fs, s, r, o) for s, r, o in claims
                    if r == fact.r and fs.resolve_entity(s) == fact.s]
            held_here = [u for u in unresolved
                         if u[1] == fact.r and fs.resolve_entity(u[0]) == fact.s]

            if mine:
                v = mine[0]
                status = v.status
                if status == VERIFIED and wrong:
                    breaches.append((fact.s, fact.r, fact.o, answer))
            elif held_here:
                status = HELD
            else:
                status = "no-answer"
            counts[status] = counts.get(status, 0) + 1

            records.append({
                "entity": fact.s, "relation": fact.r, "declared": fact.o,
                "source": fact.source, "question": q,
                "answer": " ".join(answer.split()),
                "verdict": status, "states_wrong_value": wrong,
            })
            marker = {VERIFIED: "verified   ", BLOCK: "BLOCK      ",
                      HELD: "held       ", "no-answer": "no-answer  "}.get(status, status)
            print(f"  {marker}{'  <-- TRUST BREACH' if (status==VERIFIED and wrong) else ''}"
                  f"  {fact.s[:22]}/{fact.r}", flush=True)

    asked = counts[VERIFIED] + counts[BLOCK] + counts[HELD]
    print("\n" + "=" * 72)
    print(f"SUPERVISED PILOT  model={a.model}  {time.time() - t0:.0f}s")
    print(f"  questions the model answered            {asked}")
    print(f"  VERIFIED  (gate settled it)             {counts[VERIFIED]}")
    print(f"  HELD/BLOCK (handed to the reviewer)     {counts[HELD] + counts[BLOCK]}"
          f"   ({counts[HELD]} held, {counts[BLOCK]} blocked)")
    if asked:
        print(f"  REVIEWER LOAD REDUCTION                 "
              f"{counts[VERIFIED] / asked:.0%}  of answers needed no review")
    print(f"  TRUST BREACHES (VERIFIED but wrong)     {len(breaches)}"
          f"{'   *** must be zero ***' if breaches else '   -- clean'}")
    for s, r, o, ans in breaches:
        print(f"      {s}/{r} declared {o!r}: {ans[:120]!r}")
    print("=" * 72)
    print("A pilot passes when TRUST BREACHES is zero: no wrong value was confirmed to the")
    print("user. REVIEWER LOAD REDUCTION is the efficiency it buys -- the rest is a queue,")
    print("which is costly but never wrong. Every verdict is in the report for the human to")
    print("confirm against the document.")

    out = REPO / "results" / "pilot_report.json"
    out.write_text(json.dumps({"domain": fs.domain, "model": a.model,
                               "counts": counts, "breaches": len(breaches),
                               "records": records}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n-> {out}")
    return 1 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())

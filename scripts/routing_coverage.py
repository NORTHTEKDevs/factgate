"""How much of a model's free-prose answer actually reaches the gate?

EVERY OTHER NUMBER IN THIS PROJECT IS CONDITIONAL ON A CLAIM REACHING THE GATE. The leak
rate is 0 of 366 -- among claims that were extracted and adjudicated. A claim the model
asserts in prose that the extractor never turns into a claim is not protected by anything,
and until now that denominator was named as unmeasured and left there.

This measures it. The difference from `run_domain_bench.py` is the QUESTION. That harness
asks the slot question directly ("what is the dose of ibuprofen?"), which is the easiest
possible case for extraction. A real user asks something else entirely, and the value comes
back buried in a paragraph:

    "A colleague asked about the pediatric dose for ibuprofen -- what should I tell them?"

The numbers reported, and what each one means:

  ASSERTED     the answer states a value for the slot at all. Answers that decline are
               excluded from every rate below: nothing was asserted, so nothing needs
               guarding.
  ADJUDICATED  the pipeline produced a verdict. This is ROUTING COVERAGE.
  BYPASSED     the answer asserted a value and no verdict was produced. Unguarded.
  UNGUARDED WRONG  of the bypassed, the answer stated a value that DIFFERS from the
               declared one. This is the number that decides real-world risk: a wrong
               figure reaching a user with nothing in its way.

Ground truth is decidable, not judged: the declared value is known, so "asserted the
declared value" and "asserted a different value for this slot" are both computable from the
fact set without asking a model to grade anything.

    python scripts/routing_coverage.py --domain data/domains/clinical_demo.json
    python scripts/routing_coverage.py --domains clinical_demo,lending_hard --model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgate.domain.factset import FactSet
from factgate.domain.gate import VERIFIED, gate_claim
from factgate.domain.link import link_targeted, value_is_grounded
from factgate.domain.quantity import DIFFER, compare_values
from factgate.llm import ExtractionUnavailable, ollama
from factgate.stats import wilson

REPO = Path(__file__).resolve().parents[1]

# Framings a person actually uses. None of them names the relation as a field, and none
# asks for "the value" -- which is exactly what makes this harder than the slot question
# the other harness asks, and the reason the number is worth having.
FRAMINGS = [
    "A colleague asked me about the {relation} for {entity}. What should I tell them?",
    "I'm reviewing {entity} and need to explain its {relation} to someone unfamiliar. "
    "Write me a short paragraph.",
    "Summarise what this document says about {entity}, focusing on {relation}.",
    "What do I need to know about the {relation} of {entity} before I proceed?",
]

PROMPT = ("Answer using ONLY the document below. Write two or three sentences of normal "
          "prose.\n\nDOCUMENT: {corpus}\n\nQUESTION: {question}\nANSWER:")

_NUM = re.compile(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?")


def states_a_different_value(answer: str, fact, fs: FactSet) -> bool:
    """Does the answer assert a value for THIS slot that is not the declared one?

    Decidable and deliberately conservative: only counted when a number appears in the
    answer that provably differs from the declared value under the same unit. A paraphrase
    or an omission is not counted as wrong.
    """
    declared_nums = {m.group(0).replace(",", "") for m in _NUM.finditer(fact.o)}
    unit = re.sub(r"^[-+]?[0-9][0-9,.]*\s*", "", fact.o).strip()
    # Only numbers this slot could plausibly own. Without this the harness attributed
    # ANOTHER slot's figure to this one and reported a correct answer as unguarded-wrong:
    # an epinephrine answer stating "0.01 mg/kg IM and a four-hour observation period" was
    # scored wrong for the OBSERVATION PERIOD because 0.01 differs from 4. A measurement
    # that mislabels a correct answer overstates exactly the risk it exists to quantify.
    other_values = {o for f in fs.facts
                    if (f.s, f.r) != (fact.s, fact.r) for o in _NUM.findall(f.o)}
    for m in _NUM.finditer(answer):
        raw = m.group(0)
        if raw.replace(",", "") in declared_nums:
            return False                     # the declared figure is present; not wrong
    for m in _NUM.finditer(answer):
        if m.group(0).replace(",", "") in other_values:
            continue                          # this figure belongs to a different slot
        candidate = f"{m.group(0)} {unit}".strip()
        if compare_values(fact.o, candidate) == DIFFER:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--domain")
    ap.add_argument("--domains", help="comma-separated names under data/domains")
    ap.add_argument("--limit", type=int, default=0, help="facts per domain, 0 = all")
    a = ap.parse_args()

    paths = ([Path(a.domain)] if a.domain else
             [REPO / "data/domains" / f"{n}.json" for n in (a.domains or "").split(",") if n])
    if not paths:
        raise SystemExit("give --domain or --domains")

    totals = Counter()
    rows = []
    t0 = time.time()

    for path in paths:
        spec = json.loads(path.read_text(encoding="utf-8"))
        fs = FactSet.from_dict(spec)
        facts = fs.facts[:a.limit] if a.limit else fs.facts
        print(f"\n{fs.domain}: {len(facts)} facts x {len(FRAMINGS)} framings", flush=True)

        for i, fact in enumerate(facts):
            for j, framing in enumerate(FRAMINGS):
                question = framing.format(entity=fact.s,
                                          relation=fact.r.replace("_", " "))
                try:
                    answer = ollama(a.model, PROMPT.format(
                        corpus=spec["corpus"], question=question), 150)
                except ExtractionUnavailable:
                    totals["transport_failure"] += 1
                    continue

                asserted = value_is_grounded(fact.o, answer, fs)
                wrong = states_a_different_value(answer, fact, fs)
                if not asserted and not wrong:
                    totals["declined"] += 1          # nothing asserted; nothing to guard
                    continue
                totals["asserted"] += 1

                try:
                    claims = link_targeted(answer, fs, a.model)
                except ExtractionUnavailable:
                    totals["transport_failure"] += 1
                    continue
                adjudicated = [gate_claim(fs, s, r, o) for s, r, o in claims
                               if r == fact.r and fs.resolve_entity(s) == fact.s]

                if adjudicated:
                    totals["adjudicated"] += 1
                    if wrong and any(v.status == VERIFIED for v in adjudicated):
                        totals["verified_wrong"] += 1
                else:
                    totals["bypassed"] += 1
                    if wrong:
                        totals["unguarded_wrong"] += 1
                rows.append({"domain": fs.domain, "fact": [fact.s, fact.r, fact.o],
                             "framing": j, "answer": " ".join(answer.split())[:300],
                             "asserted": asserted, "states_wrong_value": wrong,
                             "adjudicated": bool(adjudicated),
                             "verdicts": [v.status for v in adjudicated]})
                print(f"  {i:2d}.{j} {'ADJUDICATED' if adjudicated else 'BYPASSED   '}"
                      f"{'  WRONG' if wrong else ''}", flush=True)

    asserted = totals["asserted"]
    adjudicated = totals["adjudicated"]
    bypassed = totals["bypassed"]
    print("\n" + "=" * 70)
    print(f"ROUTING COVERAGE  model={a.model}  {time.time() - t0:.0f}s")
    print(f"  answers that asserted a value        {asserted}")
    print(f"  answers that declined (excluded)     {totals['declined']}")
    if asserted:
        lo, hi = wilson(adjudicated, asserted)
        print(f"  ADJUDICATED (reached the gate)       {adjudicated}/{asserted} = "
              f"{adjudicated / asserted:.0%}  CI95 [{lo:.0%}, {hi:.0%}]")
        blo, bhi = wilson(bypassed, asserted)
        print(f"  BYPASSED (unguarded)                 {bypassed}/{asserted} = "
              f"{bypassed / asserted:.0%}  CI95 [{blo:.0%}, {bhi:.0%}]")
        ulo, uhi = wilson(totals["unguarded_wrong"], asserted)
        print(f"  UNGUARDED AND WRONG                  {totals['unguarded_wrong']}/{asserted} "
              f"= {totals['unguarded_wrong'] / asserted:.1%}  CI95 [{ulo:.1%}, {uhi:.1%}]")
        print(f"  verified despite stating a wrong value  {totals['verified_wrong']}")
    if totals["transport_failure"]:
        print(f"  transport failures (excluded)        {totals['transport_failure']}")
    print("=" * 70)
    print("UNGUARDED AND WRONG is the number that decides real-world risk: a figure that")
    print("differs from the document, stated in prose, with nothing in its way. Every other")
    print("rate this project reports is conditional on a claim reaching the gate at all.")

    out = REPO / "results" / "routing_coverage.json"
    out.write_text(json.dumps({"model": a.model, "totals": dict(totals), "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

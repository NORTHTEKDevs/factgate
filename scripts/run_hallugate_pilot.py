"""HalluGate-Bench pilot: end-to-end leak AND over-block rate on RAGTruth (MIT).

    python scripts/run_hallugate_pilot.py --n 40 --model llama3.2:3b --task QA

Pipeline per example:
  source document --extract--> triples --> ephemeral RCK KB   (deterministic verdict layer)
  model response  --extract--> triples --> mapped to sentences
  each sentence   --classify--> PASS / BLOCK / HELD / SKIP    (fail-closed)

Gold labels are RAGTruth's character-level hallucination spans, so a sentence's true
class is known and leak/over-block are both computable. Per-example rows are written to
results/ so any published number can be traced to the exact sentence that produced it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factgate.hallugate import extract as X
from factgate.hallugate import policy as P
from factgate.hallugate.ragtruth import label_sentences, load_examples

REPO = Path(__file__).resolve().parents[1]


def triples_for_sentence(sentence: str, triples):
    """Attribute a response-level triple to the sentence that states its subject.

    A sentence with no attributable triple is NOT assumed clean -- classify() sends an
    unattributed assertion to HELD. Cheap attribution is therefore safe here: it can
    cost coverage, never safety.
    """
    low = sentence.lower()
    return [t for t in triples if t[0].lower().replace("_", " ") in low]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--task", default="QA")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ex = load_examples(a.task, "test", limit=a.n)
    print(f"loaded {len(ex)} {a.task} test examples "
          f"({sum(1 for e in ex if e['labels'])} carry >=1 hallucination span)")

    kb_cache: dict[str, object] = {}
    src_facts: dict[str, int] = {}
    rows, per_example = [], []
    parse_fail = 0
    t0 = time.time()

    for i, e in enumerate(ex):
        sid = e["source_id"]
        if sid not in kb_cache:
            st, ok = X.extract(e["source_text"][:4000], a.model)
            parse_fail += not ok
            kb_cache[sid] = P.build_ephemeral_kb(st)
            src_facts[sid] = len(st)
        kb = kb_cache[sid]

        rt, ok = X.extract(e["response"], a.model)
        parse_fail += not ok

        sent_rows = []
        for sentence, is_hal in label_sentences(e["response"], e["labels"]):
            state = P.classify(sentence, triples_for_sentence(sentence, rt), kb)
            rows.append((state, is_hal))
            sent_rows.append({"sentence": sentence, "hallucinated": is_hal,
                              "state": state})

        per_example.append({"id": e["id"], "source_id": sid, "model": e["model"],
                            "source_facts": src_facts[sid], "response_triples": rt,
                            "sentences": sent_rows})
        print(f"  [{i+1}/{len(ex)}] src_facts={src_facts[sid]:3} "
              f"resp_triples={len(rt):2} "
              f"{dict(Counter(s['state'] for s in sent_rows))}", flush=True)

    res = P.score(rows)
    res.update({"task": a.task, "extractor_model": a.model, "n_examples": len(ex),
                "unique_sources": len(kb_cache), "json_parse_failures": parse_fail,
                "state_counts": dict(Counter(s for s, _ in rows)),
                "seconds": round(time.time() - t0, 1)})

    out = Path(a.out) if a.out else REPO / "results" / f"hallugate_pilot_{a.task}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    with open(out.with_suffix(".rows.jsonl"), "w", encoding="utf-8") as f:
        for p in per_example:
            f.write(json.dumps(p) + "\n")

    lr, ob = res["leak_rate"], res["over_block_rate"]
    print("\n" + "=" * 68)
    print(f"HALLUGATE PILOT  task={a.task}  extractor={a.model}  "
          f"{res['seconds']}s")
    print(f"  scored sentences {res['n_scored']}  "
          f"(hallucinated {res['n_hallucinated']}, faithful {res['n_faithful']}; "
          f"{res['state_counts'].get('SKIP', 0)} SKIP excluded)")
    print(f"  states           {res['state_counts']}")
    if lr is not None:
        print(f"  LEAK RATE        {res['leaks']}/{res['n_hallucinated']} = {lr:.0%}  "
              f"CI95 [{res['leak_ci95'][0]:.0%}, {res['leak_ci95'][1]:.0%}]")
    if ob is not None:
        print(f"  OVER-BLOCK RATE  {res['blocked_faithful']}/{res['n_faithful']} = "
              f"{ob:.0%}  CI95 [{res['over_block_ci95'][0]:.0%}, "
              f"{res['over_block_ci95'][1]:.0%}]")
    print("  (either rate alone is meaningless: blocking everything gives 0% leak)")
    print(f"  -> {out}")
    print("=" * 68)


if __name__ == "__main__":
    main()

"""Full-scale FACTGATE dataset generation. Per-generator targets, one KB load."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factgate.datagen.run_all import read_triples
from factgate.datagen import qa, chains, idk, corrupt, schema, templates
from factgate.kb_service import load_kb

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "kb_train_90k.jsonl"
HOLDOUT = ROOT / "data" / "holdout_unknowable_10k.jsonl"
OUT = ROOT / "data" / "sft_full.jsonl"
XOUT = ROOT / "data" / "extractor_pairs.jsonl"
COUNTS = ROOT / "data" / "sft_full.counts.json"

TARGETS = {"qa": 300_000, "chains": 120_000, "idk_absent": 100_000,
           "idk_holdout": 60_000, "corrupt": 80_000}

t0 = time.time()
triples = read_triples(SOURCE, max_facts=None)
holdout_triples = read_triples(HOLDOUT, max_facts=None)
agent = load_kb(SOURCE, ROOT / "kb_snapshot_eval90k")
kb = agent.knowledge
entities = sorted({t["s"] for t in triples} | {t["o"] for t in triples})
print(f"kb ready {time.time()-t0:.0f}s facts={len(triples)}", flush=True)

counts, valid, total = {}, 0, 0
with open(OUT, "w", encoding="utf-8") as f:
    for name, gen in [
        ("qa", qa.generate(kb, triples, n=TARGETS["qa"], seed=0)),
        ("chains", chains.generate(kb, triples, n=TARGETS["chains"], seed=1)),
        ("idk_absent", idk.generate_absent(kb, entities, n=TARGETS["idk_absent"], seed=2)),
        ("idk_holdout", idk.generate_holdout(kb, holdout_triples, n=TARGETS["idk_holdout"], seed=3)),
        ("corrupt", corrupt.generate(kb, triples, n=TARGETS["corrupt"], seed=4)),
    ]:
        n = 0
        for ex in gen:
            rec = ex if isinstance(ex, dict) else ex.model_dump()
            ok, _ = schema.validate_record(rec) if hasattr(schema, "validate_record") else (True, None)
            valid += bool(ok); total += 1; n += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 20000 == 0:
                print(f"[{name}] {n} ({time.time()-t0:.0f}s)", flush=True)
        counts[name] = n
        print(f"[{name}] DONE {n} ({time.time()-t0:.0f}s)", flush=True)

xn = 0
with open(XOUT, "w", encoding="utf-8") as f:
    for i, t in enumerate(triples):
        for tid in templates.statement_template_ids(t["r"])[:2] if hasattr(templates, "statement_template_ids") else [0]:
            try:
                text = templates.render_fact(t["s"], t["r"], t["o"], tid)
            except Exception:
                continue
            f.write(json.dumps({"text": text, "triples": [t], "template_id": tid}) + "\n")
            xn += 1
            if xn >= 300_000: break
        if xn >= 300_000: break

json.dump({"counts": counts, "total": total, "schema_valid": valid,
           "extractor_pairs": xn, "seconds": time.time() - t0},
          open(COUNTS, "w"), indent=2)
print(f"ALL DONE total={total} extractor={xn} {time.time()-t0:.0f}s", flush=True)

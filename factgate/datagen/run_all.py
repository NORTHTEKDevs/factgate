"""Orchestrate every generator into one SFT JSONL.

Smoke usage (fast, bounded KB + example counts):

    python -m factgate.datagen.run_all --limit 100 --kb-limit 3000

Full-scale generation (100k facts) is a LATER orchestrated step --
do not run that from here without an explicit large --kb-limit / no cap.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rck.conscious_agent import ConsciousAgent

from factgate.datagen import chains, corrupt, idk, qa
from factgate.datagen.schema import validate_file

DEFAULT_SOURCE = Path(
    "C:/Users/Krist/projects/active/rck/data/conceptnet_scale_100k.jsonl")
DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "data" / "sft_v0.jsonl"


def read_triples(path: Path, max_facts: int | None = None) -> list[dict]:
    triples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_facts is not None and len(triples) >= max_facts:
                break
            d = json.loads(line)
            triples.append({"s": str(d["s"]), "r": str(d["r"]), "o": str(d["o"])})
    return triples


def build_kb(triples: list[dict], *, dim: int, seed: int) -> ConsciousAgent:
    agent = ConsciousAgent(dim=dim, expected_facts=max(len(triples), 1),
                            seed=seed, install_self=False)
    for t in triples:
        agent.knowledge.store({"S": t["s"], "R": t["r"], "O": t["o"]})
    return agent


def run(*, source: Path, out: Path, counts_out: Path | None,
        limit: int, kb_limit: int | None, holdout: Path | None,
        seed: int, dim: int) -> dict:
    t0 = time.time()
    triples = read_triples(source, max_facts=kb_limit)
    agent = build_kb(triples, dim=dim, seed=seed)
    entities = sorted({t["s"] for t in triples} | {t["o"] for t in triples})

    holdout_triples: list[dict] = []
    if holdout is not None and holdout.exists():
        holdout_triples = read_triples(holdout)

    generators: list[tuple[str, object]] = [
        ("qa", qa.generate(agent.knowledge, triples, n=limit, seed=seed)),
        ("chains", chains.generate(agent.knowledge, triples, n=limit, seed=seed + 1)),
        ("idk_absent", idk.generate_absent(agent.knowledge, entities, n=limit, seed=seed + 2)),
        ("corrupt", corrupt.generate(agent.knowledge, triples, n=limit, seed=seed + 4)),
    ]
    if holdout_triples:
        generators.append(
            ("idk_holdout", idk.generate_holdout(
                agent.knowledge, holdout_triples, n=limit, seed=seed + 3)))

    counts: dict[str, int] = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for name, gen in generators:
            c = 0
            for example in gen:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                c += 1
            counts[name] = c

    n_valid, n_total, errors = validate_file(out)
    stats = {
        "source": str(source),
        "kb_facts_loaded": len(triples),
        "entity_count": len(entities),
        "holdout_facts": len(holdout_triples),
        "counts": counts,
        "total_examples": sum(counts.values()),
        "schema_valid": n_valid,
        "schema_total": n_total,
        "schema_errors": errors[:10],
        "seconds": time.time() - t0,
        "out": str(out),
    }

    counts_out = counts_out or out.with_suffix(".counts.json")
    counts_out.write_text(json.dumps(stats, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--counts-out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=100,
                    help="max examples per generator")
    p.add_argument("--kb-limit", type=int, default=None,
                    help="max source facts to load into the in-memory KB "
                         "(leave unset only for the real full-scale run)")
    p.add_argument("--holdout", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=4096)
    args = p.parse_args(argv)

    stats = run(source=args.source, out=args.out, counts_out=args.counts_out,
                limit=args.limit, kb_limit=args.kb_limit, holdout=args.holdout,
                seed=args.seed, dim=args.dim)
    print(json.dumps({k: v for k, v in stats.items() if k != "schema_errors"},
                      indent=2))
    if stats["schema_errors"]:
        print("schema errors (first 10):")
        for e in stats["schema_errors"]:
            print(" ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI: bulk-load the ConceptNet-scale JSONL into an RCK KB and snapshot it.

    python -m factgate.datagen.build_kb \
        --input C:/Users/Krist/projects/active/rck/data/conceptnet_scale_100k.jsonl \
        --out C:/Users/Krist/projects/active/factgate/kb_snapshot \
        --manifest C:/Users/Krist/projects/active/factgate/kb_snapshot/kb_manifest.json

Also writes kb_manifest.json: fact count, relation histogram, entity count.

NOTE: this is a distinct snapshot directory from factgate/kb_service.py's
runtime default (factgate/data/kb_snapshot) -- that one is the gate's
own lazily-built cache; this one is the datagen pipeline's fixed,
reproducible corpus snapshot.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from rck.bulk_ingest import bulk_load_jsonl
from rck.conscious_agent import ConsciousAgent
from rck.session import save_session

DEFAULT_INPUT = Path(
    "C:/Users/Krist/projects/active/rck/data/conceptnet_scale_100k.jsonl")
DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "kb_snapshot"


def compute_manifest(jsonl_path: Path, max_facts: int | None = None) -> dict:
    """Scan the source JSONL (independent of the HRR KB) for exact
    ground-truth counts -- the KB's `.size()` reflects stores including
    auto-symmetrized inverses, so the manifest is built from the raw
    triples instead for an unambiguous fact count."""
    rel_hist: collections.Counter = collections.Counter()
    entities: set[str] = set()
    n = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_facts is not None and n >= max_facts:
                break
            d = json.loads(line)
            s, r, o = str(d["s"]), str(d["r"]), str(d["o"])
            rel_hist[r] += 1
            entities.add(s)
            entities.add(o)
            n += 1
    return {
        "fact_count": n,
        "relation_histogram": dict(rel_hist.most_common()),
        "entity_count": len(entities),
    }


def build(input_path: Path, out_dir: Path, manifest_path: Path | None,
          *, dim: int = 4096, n_shards: int = 1024, seed: int = 0,
          max_facts: int | None = None, symmetrize: bool = True,
          install_self: bool = False) -> dict:
    t0 = time.time()
    agent = ConsciousAgent(dim=dim, n_shards=n_shards, seed=seed,
                            install_self=install_self)
    load_stats = bulk_load_jsonl(agent.knowledge, input_path,
                                  symmetrize=symmetrize, max_facts=max_facts)
    save_session(agent, out_dir)

    manifest = compute_manifest(input_path, max_facts=max_facts)
    manifest["kb_size_stored"] = agent.knowledge.size()
    manifest["dim"] = dim
    manifest["n_shards"] = n_shards
    manifest["seed"] = seed
    manifest["symmetrize"] = symmetrize
    manifest["load_stats"] = load_stats
    manifest["build_seconds"] = time.time() - t0
    manifest["snapshot_dir"] = str(out_dir)

    manifest_path = manifest_path or (out_dir / "kb_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--n-shards", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-facts", type=int, default=None)
    p.add_argument("--no-symmetrize", action="store_true")
    args = p.parse_args(argv)

    manifest = build(
        args.input, args.out, args.manifest,
        dim=args.dim, n_shards=args.n_shards, seed=args.seed,
        max_facts=args.max_facts, symmetrize=not args.no_symmetrize,
    )
    print(json.dumps(
        {k: v for k, v in manifest.items() if k != "relation_histogram"},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

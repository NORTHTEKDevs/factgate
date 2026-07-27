"""Fetch the RAGTruth dataset (MIT, github.com/ParticleMedia/RAGTruth) into data/ragtruth/.

36MB total, not vendored. Used by factgate.hallugate; see docs/HALLUGATE.md.

    python scripts/fetch_ragtruth.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
FILES = ("response.jsonl", "source_info.jsonl")
DEST = Path(__file__).resolve().parents[1] / "data" / "ragtruth"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = DEST / name
        if out.exists():
            print(f"{name}: already present ({out.stat().st_size/1e6:.1f}MB), skipping")
            continue
        print(f"fetching {name} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE}/{name}", out)
        n = sum(1 for _ in open(out, encoding="utf-8"))
        print(f"  {name}: {n} lines, {out.stat().st_size/1e6:.1f}MB")
    print(f"\nRAGTruth is MIT licensed (c) 2023 Particle Media. Data -> {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Reserve a deterministic holdout split of triples BEFORE the KB is
built, so idk.py's deletion-holdout mode can generate IDK-by-construction
questions with ground truth about what was withheld.

    python scripts/holdout_split.py \
        --input C:/Users/Krist/projects/active/rck/data/conceptnet_scale_100k.jsonl \
        --out-train C:/Users/Krist/projects/active/factgate/data/kb_train.jsonl \
        --out-holdout C:/Users/Krist/projects/active/factgate/data/holdout.jsonl \
        --holdout-size 10000 --seed 0

kb_train.jsonl is what build_kb.py / run_all.py should ingest; the
facts in holdout.jsonl must never be loaded into the KB.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_INPUT = Path(
    "C:/Users/Krist/projects/active/rck/data/conceptnet_scale_100k.jsonl")
DEFAULT_OUT_TRAIN = Path(__file__).resolve().parent.parent / "data" / "kb_train.jsonl"
DEFAULT_OUT_HOLDOUT = Path(__file__).resolve().parent.parent / "data" / "holdout.jsonl"


def split(input_path: Path, out_train: Path, out_holdout: Path,
          *, holdout_size: int = 10_000, seed: int = 0) -> dict:
    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if holdout_size >= len(lines):
        raise ValueError(
            f"holdout_size {holdout_size} >= total lines {len(lines)}")

    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)
    holdout_idx = set(order[:holdout_size])

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_holdout.parent.mkdir(parents=True, exist_ok=True)

    n_train = 0
    n_holdout = 0
    with open(out_train, "w", encoding="utf-8") as ftr, \
         open(out_holdout, "w", encoding="utf-8") as fho:
        for i, line in enumerate(lines):
            if i in holdout_idx:
                fho.write(line + "\n")
                n_holdout += 1
            else:
                ftr.write(line + "\n")
                n_train += 1

    return {"total": len(lines), "train": n_train, "holdout": n_holdout,
            "seed": seed, "out_train": str(out_train),
            "out_holdout": str(out_holdout)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-train", type=Path, default=DEFAULT_OUT_TRAIN)
    p.add_argument("--out-holdout", type=Path, default=DEFAULT_OUT_HOLDOUT)
    p.add_argument("--holdout-size", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    stats = split(args.input, args.out_train, args.out_holdout,
                  holdout_size=args.holdout_size, seed=args.seed)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pick an evaluation document without human selection bias.

Both earlier real-document tests used files chosen after browsing what was on the machine.
That is a selection the operator made, and it is exactly the kind of choice that looks
harmless until the result depends on it.

This enumerates candidates by a MECHANICAL filter (numeric density, size, not already
used) and then picks by sorting on the SHA-256 of the file path -- deterministic,
reproducible, and uncorrelated with anything about the content. The operator sees the
document only after it has been chosen.

    python scripts/select_eval_document.py --root ~/business --min-facts 25
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

# Filenames to skip so a "fresh" pick is genuinely fresh. Passed in rather than hardcoded:
# naming your own private documents in a public repository discloses that they exist.

_NUMERIC = re.compile(r"\$[0-9][0-9,]*(?:\.[0-9]+)?[KMB]?|[0-9]+(?:\.[0-9]+)?%"
                      r"|\b[0-9]+(?:\.[0-9]+)?\s*(?:days?|months?|years?|hours?|users?|seats?)\b")


def candidates(root: Path, min_facts: int, max_bytes: int,
               exclude: set[str] | None = None) -> list[tuple[str, Path, int]]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".md", ".txt") or not p.is_file():
            continue
        if p.name in (exclude or set()):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text.encode()) > max_bytes:
            continue
        n = len(_NUMERIC.findall(text))
        if n < min_facts:
            continue
        digest = hashlib.sha256(str(p).encode()).hexdigest()
        out.append((digest, p, n))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "business"))
    ap.add_argument("--min-facts", type=int, default=25)
    ap.add_argument("--max-bytes", type=int, default=40_000)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--exclude", default="",
                    help="comma-separated filenames to skip (already used)")
    a = ap.parse_args()

    pool = candidates(Path(a.root).expanduser(), a.min_facts, a.max_bytes,
                      {x.strip() for x in a.exclude.split(',') if x.strip()})
    if not pool:
        raise SystemExit("no candidate documents matched the filter")

    print(f"candidate pool: {len(pool)} documents "
          f"(>= {a.min_facts} numeric facts, <= {a.max_bytes} bytes, not already used)")
    for digest, p, n in pool[:a.show]:
        print(f"  {digest[:8]}  {n:>3} facts  {p.name}")
    if len(pool) > a.show:
        print(f"  ... and {len(pool) - a.show} more")

    digest, chosen, n = pool[0]
    print(f"\nSELECTED (lowest path-digest, no content involvement):")
    print(f"  {chosen}")
    print(f"  {n} numeric facts, {chosen.stat().st_size:,} bytes, digest {digest[:16]}")


if __name__ == "__main__":
    main()

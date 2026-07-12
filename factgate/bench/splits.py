"""Build the FactGate hallucination-eval splits (A6) as JSONL files under
data/eval/.

Ported PATTERN from raincg/bench/suite.py (declarative registry,
skip-if-exists, --force/--dry-run-ish CLI) -- adapted to eval "splits"
instead of benchmark "stages". No rain code is imported.

    python -m factgate.bench.splits                                # list all splits
    python -m factgate.bench.splits --splits knowable_heldout,unknowable
    python -m factgate.bench.splits --splits adversarial_nearmiss --force

Splits:
  knowable_heldout      -- facts present in the 90k eval KB, ABSENT from
                            sft_full.jsonl's qa-generator source_triple set.
  unknowable            -- questions about data/holdout_unknowable_10k.jsonl
                            facts (absent from the 90k KB by construction).
  adversarial_nearmiss  -- corrupted-premise prompts (datagen/corrupt.py's
                            generator, reused directly) against the same
                            held-out-from-training fact pool.
  compositional         -- DEFERRED: rain/raincg's bench suite already
                            covers compositional generalization.
  general_canary        -- DEFERRED: MMLU download skipped per task scope;
                            placeholder file only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from rck.idk_detection import EpistemicState, ask_with_idk

from factgate.bench.common import wait_for_stable_lines
from factgate.datagen import corrupt as corrupt_gen
from factgate.datagen.run_all import read_triples
from factgate.datagen.templates import QUESTION_TEMPLATES
from factgate.kb_service import load_kb

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
EVAL_DIR = DATA_DIR / "eval"

DEFAULT_SFT = DATA_DIR / "sft_full.jsonl"
DEFAULT_KB_TRAIN = DATA_DIR / "kb_train_90k.jsonl"
DEFAULT_HOLDOUT = DATA_DIR / "holdout_unknowable_10k.jsonl"
DEFAULT_KB_SNAPSHOT = REPO / "kb_snapshot_eval90k"


SPLITS: list[dict] = [
    {"name": "knowable_heldout", "out": "knowable_heldout.jsonl",
     "target_n": 5000, "status": "active", "note": ""},
    {"name": "unknowable", "out": "unknowable.jsonl",
     "target_n": 5000, "status": "active", "note": ""},
    {"name": "adversarial_nearmiss", "out": "adversarial_nearmiss.jsonl",
     "target_n": 2000, "status": "active", "note": ""},
    {"name": "compositional", "out": "compositional.jsonl",
     "target_n": 0, "status": "deferred",
     "note": ("deferred -- rain/raincg's bench suite (raincg/bench) already "
              "covers compositional generalization (SCAN/COGS/PCFG-SET); no "
              "FactGate-specific compositional split built yet.")},
    {"name": "general_canary", "out": "general_canary.jsonl",
     "target_n": 0, "status": "deferred",
     "note": ("deferred -- MMLU download skipped per task scope. File "
              "structure created (empty placeholder); wire up an MMLU "
              "subset loader here when general-capability canary coverage "
              "is needed.")},
]


def split_by_name(name: str) -> dict | None:
    for s in SPLITS:
        if s["name"] == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Coverage set: which (s, r, o) facts a `qa` generator example already
# trained on, per its `meta.source_triple` field. Only `qa.py` stamps this
# field (see factgate/datagen/qa.py); facts touched only by chains/corrupt/
# idk examples are NOT counted here -- a narrower, cheaper leakage guard
# than a full-text scan, disclosed in RESULTS.md rather than assumed complete.
# ---------------------------------------------------------------------------

def load_source_triple_coverage(sft_path: Path) -> set[tuple[str, str, str]]:
    coverage: set[tuple[str, str, str]] = set()
    sft_path = Path(sft_path)
    if not sft_path.exists():
        return coverage
    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn last line if the file is mid-write
            st = (rec.get("meta") or {}).get("source_triple")
            if isinstance(st, dict) and {"s", "r", "o"} <= st.keys():
                coverage.add((str(st["s"]), str(st["r"]), str(st["o"])))
    return coverage


def held_out_pool(kb_triples: list[dict],
                   coverage: set[tuple[str, str, str]]) -> list[dict]:
    """Facts present in the KB, templated, but never a qa-generator
    source_triple -- the base pool for both knowable_heldout and
    adversarial_nearmiss."""
    return [t for t in kb_triples
            if t["r"] in QUESTION_TEMPLATES
            and (t["s"], t["r"], t["o"]) not in coverage]


# ---------------------------------------------------------------------------
# (a) knowable_heldout
# ---------------------------------------------------------------------------

def _eval_forward_template(r: str):
    """The eval-reserved question-template variant for relation `r`: the
    LAST forward-direction (asks-for-O) template. qa.py's training-time
    selection is `random.choice` over ALL templates for the relation
    (forward+reverse mixed, no fixed default) -- there is no single id
    strictly "unused" by training at scale, so this is a deliberate,
    deterministic phrasing-diversity choice layered on top of the real
    safeguard (fact-level source_triple exclusion via held_out_pool), not
    itself a leakage barrier."""
    forward = [t for t in QUESTION_TEMPLATES[r] if t.direction == "forward"]
    return forward[-1]


def build_knowable_heldout(kb_triples: list[dict],
                            coverage: set[tuple[str, str, str]], *,
                            n: int, seed: int = 0) -> list[dict]:
    pool = held_out_pool(kb_triples, coverage)
    rng = random.Random(seed)
    rng.shuffle(pool)
    out = []
    for i, t in enumerate(pool[:n]):
        tmpl = _eval_forward_template(t["r"])
        question = tmpl.text.format(s=t["s"])
        out.append({
            "id": f"knowable-{seed}-{i}", "split": "knowable_heldout",
            "question": question, "s": t["s"], "r": t["r"], "o": t["o"],
            "expected": t["o"],
            "meta": {"direction": "forward",
                     "template_variant": "eval_reserved_last_forward"},
        })
    return out


# ---------------------------------------------------------------------------
# (b) unknowable
# ---------------------------------------------------------------------------

def build_unknowable(kb, holdout_triples: list[dict], *, n: int,
                      seed: int = 0, top_k: int = 5,
                      max_attempts_factor: int = 10) -> list[dict]:
    rng = random.Random(seed)
    pool = [t for t in holdout_triples if t["r"] in QUESTION_TEMPLATES]
    out: list[dict] = []
    if not pool:
        return out
    emitted = 0
    attempts = 0
    max_attempts = max(n * max_attempts_factor, 100)
    while emitted < n and attempts < max_attempts:
        attempts += 1
        t = pool[rng.randrange(len(pool))]
        s, r, o = t["s"], t["r"], t["o"]
        forward = [q for q in QUESTION_TEMPLATES[r] if q.direction == "forward"]
        if not forward:
            continue
        tmpl = rng.choice(forward)
        known = {"S": s, "R": r}
        ea = ask_with_idk(kb, known, "O", top_k=top_k)
        if ea.state != EpistemicState.IDK:
            continue  # holdout fact accidentally recoverable -- skip, don't mislabel
        question = tmpl.text.format(s=s)
        out.append({
            "id": f"unknowable-{seed}-{emitted}", "split": "unknowable",
            "question": question, "s": s, "r": r, "o": o, "expected": None,
            "meta": {"state": ea.state.value, "top_score": ea.top_score},
        })
        emitted += 1
    return out


# ---------------------------------------------------------------------------
# (c) adversarial_nearmiss -- reuses datagen/corrupt.py's generator directly.
# ---------------------------------------------------------------------------

def build_adversarial_nearmiss(kb, pool: list[dict], *, n: int,
                                seed: int = 0) -> list[dict]:
    out = []
    for i, ex in enumerate(corrupt_gen.generate(kb, pool, n=n, seed=seed)):
        meta = ex["meta"]
        out.append({
            "id": f"nearmiss-{seed}-{i}", "split": "adversarial_nearmiss",
            "question": ex["plain"]["question"],
            "s": meta["subject"], "r": meta["relation"],
            "o": meta["kb_answer"], "expected": meta["kb_answer"],
            "meta": {"corrupted_object": meta["corrupted_object"],
                     "sibling_subject": meta["sibling_subject"],
                     "true_object_from_source": meta["true_object_from_source"],
                     "top_score": meta["top_score"]},
        })
    return out


# ---------------------------------------------------------------------------
# Build orchestration (suite.py pattern: skip-if-exists unless force).
# ---------------------------------------------------------------------------

def build_split(name: str, *, out_dir: Path = EVAL_DIR,
                 sft_path: Path = DEFAULT_SFT,
                 kb_train_path: Path = DEFAULT_KB_TRAIN,
                 holdout_path: Path = DEFAULT_HOLDOUT,
                 kb_snapshot_dir: Path = DEFAULT_KB_SNAPSHOT,
                 seed: int = 0, force: bool = False,
                 poll_interval: float = 5.0, stable_window: float = 60.0,
                 max_wait: float | None = 900.0) -> dict:
    entry = split_by_name(name)
    if entry is None:
        raise ValueError(f"unknown split {name!r}")

    out_dir = Path(out_dir)
    out_path = out_dir / entry["out"]
    stats_path = out_path.with_suffix(".stats.json")
    if out_path.exists() and not force:
        return {"name": name, "status": "skipped", "out": str(out_path),
                "detail": "existing output (force=True to rebuild)"}

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if entry["status"] == "deferred":
        out_path.write_text("", encoding="utf-8")
        stats = {"name": name, "status": "deferred", "n": 0,
                  "note": entry["note"], "seconds": time.time() - t0,
                  "out": str(out_path)}
        stats_path.write_text(json.dumps(stats, indent=2))
        return stats

    notes: list[str] = []
    coverage_wait = None
    coverage: set[tuple[str, str, str]] = set()
    if name in ("knowable_heldout", "adversarial_nearmiss"):
        coverage_wait = wait_for_stable_lines(
            sft_path, poll_interval=poll_interval, stable_window=stable_window,
            max_wait=max_wait)
        coverage = load_source_triple_coverage(sft_path)
        if not coverage_wait["stable"]:
            notes.append(
                f"coverage set computed from an IN-PROGRESS sft_full.jsonl "
                f"snapshot ({coverage_wait['lines']} lines; file was still "
                f"growing after {coverage_wait['waited_s']:.0f}s of polling) "
                "-- some of these 'held-out' facts may still end up in "
                "training data once generation finishes; re-run with "
                "force=True once scripts/generate_full.py completes for an "
                "authoritative exclusion set.")

    kb_triples = read_triples(kb_train_path) if kb_train_path.exists() else []
    holdout_triples = read_triples(holdout_path) if holdout_path.exists() else []

    if name == "knowable_heldout":
        records = build_knowable_heldout(kb_triples, coverage,
                                          n=entry["target_n"], seed=seed)
    elif name == "unknowable":
        agent = load_kb(kb_train_path, kb_snapshot_dir)
        records = build_unknowable(agent.knowledge, holdout_triples,
                                    n=entry["target_n"], seed=seed)
    elif name == "adversarial_nearmiss":
        agent = load_kb(kb_train_path, kb_snapshot_dir)
        pool = held_out_pool(kb_triples, coverage)
        records = build_adversarial_nearmiss(agent.knowledge, pool,
                                              n=entry["target_n"], seed=seed)
    else:
        raise ValueError(f"no builder wired for active split {name!r}")

    if len(records) < entry["target_n"]:
        notes.append(f"produced {len(records)} < requested "
                      f"{entry['target_n']} (pool exhausted after filtering)")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "name": name, "status": "active", "n": len(records),
        "requested_n": entry["target_n"], "seed": seed,
        "coverage_size": len(coverage), "coverage_wait": coverage_wait,
        "notes": notes, "seconds": time.time() - t0, "out": str(out_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def render_split_table(splits: list[dict] = SPLITS, out_dir: Path = EVAL_DIR) -> str:
    out_dir = Path(out_dir)
    head = f"{'name':<24}{'status':<12}{'target_n':>10}  built?"
    lines = [head, "-" * len(head)]
    for s in splits:
        built = "YES" if (out_dir / s["out"]).exists() else "no"
        lines.append(f"{s['name']:<24}{s['status']:<12}{s['target_n']:>10}  {built}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default=None,
                     help="comma-separated split names; omit to list all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sft-path", type=Path, default=DEFAULT_SFT)
    ap.add_argument("--kb-train", type=Path, default=DEFAULT_KB_TRAIN)
    ap.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    ap.add_argument("--kb-snapshot", type=Path, default=DEFAULT_KB_SNAPSHOT)
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--stable-window", type=float, default=60.0)
    ap.add_argument("--max-wait", type=float, default=900.0)
    args = ap.parse_args(argv)

    if args.splits is None:
        print(render_split_table(out_dir=args.out_dir))
        return 0

    wanted = [n.strip() for n in args.splits.split(",") if n.strip()]
    exit_code = 0
    for name in wanted:
        if split_by_name(name) is None:
            print(f"unknown split: {name}", file=sys.stderr)
            exit_code = 2
            continue
        try:
            stats = build_split(
                name, out_dir=args.out_dir, sft_path=args.sft_path,
                kb_train_path=args.kb_train, holdout_path=args.holdout,
                kb_snapshot_dir=args.kb_snapshot, seed=args.seed,
                force=args.force, poll_interval=args.poll_interval,
                stable_window=args.stable_window, max_wait=args.max_wait)
        except Exception as e:  # noqa: BLE001 - report, keep going on other splits
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"[{name}] {stats['status']} n={stats.get('n', '-')} "
              f"-> {stats.get('out')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

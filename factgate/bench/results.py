"""Reads factgate/results/*.json (written by runner.py), renders
factgate/RESULTS.md.

Ported PATTERN from raincg/bench/results_table.py (Wilson-CI table,
honesty footnotes) -- reimplemented for FactGate's own contract schema;
no rain code is imported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "factgate" / "results"
DEFAULT_OUT = REPO / "factgate" / "RESULTS.md"

CANONICAL_SPLITS = ["knowable_heldout", "unknowable", "adversarial_nearmiss",
                     "compositional", "general_canary"]

SPLIT_PRIMARY_METRIC = {
    "knowable_heldout": "answer accuracy (canonicalized object match or [kb:] citation match)",
    "unknowable": "abstention rate -- correct behavior is abstention (IDK marker or [kb:idk])",
    "adversarial_nearmiss": "correction rate -- negates the false premise AND cites the KB's real answer",
}

DEFERRED_NOTE = {
    "compositional": "deferred -- rain/raincg's bench suite (raincg/bench) already covers "
                      "compositional generalization (SCAN/COGS/PCFG-SET); no FactGate-specific "
                      "compositional split built yet.",
    "general_canary": "deferred -- MMLU download skipped per task scope. Placeholder split file "
                       "exists at data/eval/general_canary.jsonl (empty); wire up an MMLU subset "
                       "loader before this split is meaningful.",
}

METHODOLOGY = """## Methodology & caveats

- **knowable_heldout** coverage-exclusion is based ONLY on `meta.source_triple` in
  `data/sft_full.jsonl` (the field the `qa` generator stamps onto its own examples).
  Facts that appear incidentally inside `chains`/`corrupt`/`idk` examples but were
  never a `qa` `source_triple` are NOT excluded -- a narrower, cheaper leakage guard
  than a full-text scan, disclosed here rather than silently assumed complete.
- **Question phrasing**: `knowable_heldout` always uses the LAST forward-direction
  template variant per relation (`factgate/bench/splits.py::_eval_forward_template`),
  distinct from `qa.py`'s training-time `random.choice` over all variants. This is a
  belt-and-suspenders phrasing-diversity precaution on top of the primary safeguard
  (fact-level `source_triple` exclusion), not itself a leakage barrier.
- **Gate-side metrics** (`gated_hallucination_rate`) run every extracted `[kb:s/r/o]`
  citation AND every executed `<kb_q>` tool-round's top candidate through
  `factgate.gate.verify_claim` against the same eval-90k KB the target queried live.
  A run with 0 extractable claims (e.g. a target that ignores the tool protocol
  entirely) reports `n/a` rather than a misleading 0%.
- **PROMPTED baselines** (`ollama:<model>` targets) are few-shot tool-use prompting,
  NOT fine-tuned models -- see B2 in `ACCEPTANCE.md` for the fine-tuned comparison
  this harness is also designed to serve, once available.
"""


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    results_dir = Path(results_dir)
    out = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _fmt_pct(x) -> str:
    return f"{x:.2%}" if isinstance(x, (int, float)) else "n/a"


def _row(r: dict) -> tuple[str, ...]:
    ci = r.get("ci95") or [r.get("accuracy", 0.0)] * 2
    acc_ci = f"{_fmt_pct(r.get('accuracy', 0.0))} ({_fmt_pct(ci[0])}-{_fmt_pct(ci[1])})"
    metrics = r.get("metrics") or {}
    ghr = metrics.get("gated_hallucination_rate")
    claims_checked = metrics.get("gate_claims_checked", 0)
    gate_col = _fmt_pct(ghr) if ghr is not None else f"n/a ({claims_checked} claims)"
    return (r.get("target", "?"), acc_ci, str(r.get("n", "-")), gate_col,
            r.get("evidence_tier", "measured-fresh"))


def _footnotes_for_split(split: str, rows: list[dict]) -> list[str]:
    notes: list[str] = []
    label = SPLIT_PRIMARY_METRIC.get(split)
    if label:
        notes.append(f"- primary 'Accuracy' column for **{split}** measures: {label}.")
    for r in rows:
        metrics = r.get("metrics") or {}
        if split == "unknowable" and "confident_wrong_rate" in metrics:
            notes.append(
                f"- {r.get('target')}: confident-wrong rate = "
                f"{_fmt_pct(metrics['confident_wrong_rate'])} (non-abstaining answers on "
                "facts absent from the KB by construction -- the harness's core "
                "hallucination-surface measurement).")
        result_notes = (r.get("notes") or "").strip()
        if result_notes:
            notes.append(f"- {r.get('target')} ({split}) notes: {result_notes}")
    return notes


def render_results_md(results: list[dict]) -> str:
    splits: list[str] = []
    for r in results:
        if r["split"] not in splits:
            splits.append(r["split"])
    for name in CANONICAL_SPLITS:
        if name not in splits:
            splits.append(name)

    lines = ["# FactGate Bench Results", "", METHODOLOGY]
    header = "| Target | Accuracy (95% CI) | n | Gated hallucination rate | Evidence |"
    sep = "|---|---|---|---|---|"

    for split in splits:
        lines.append(f"## {split}")
        lines.append("")
        rows = [r for r in results if r["split"] == split]
        if not rows and split in DEFERRED_NOTE:
            lines.append(f"_{DEFERRED_NOTE[split]}_")
            lines.append("")
            continue
        if not rows:
            lines.append("_no measured results yet._")
            lines.append("")
            continue
        lines.append(header)
        lines.append(sep)
        rows.sort(key=lambda r: r.get("accuracy", 0.0), reverse=True)
        for r in rows:
            lines.append("| " + " | ".join(_row(r)) + " |")
        footnotes = _footnotes_for_split(split, rows)
        if footnotes:
            lines.append("")
            lines.extend(footnotes)
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render factgate/RESULTS.md")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    results = load_results(Path(args.results_dir))
    md = render_results_md(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} measured results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

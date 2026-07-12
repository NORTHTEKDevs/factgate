"""Evaluate a TARGET against the FactGate hallucination-eval splits (A6).

    python -m factgate.bench.runner --target echo --split knowable_heldout --limit 5
    python -m factgate.bench.runner --target ollama:llama3.2:3b --split unknowable --limit 20
    python -m factgate.bench.runner --target ollama:llama3.2:3b --split all

Targets:
  echo            -- null baseline for harness self-tests. No network, no
                      KB tool loop: "answers" by parroting the question
                      text verbatim.
  ollama:<model>  -- PROMPTED tool-use baseline. POSTs to the local Ollama
                      /api/chat endpoint with `datagen/protocol.py`'s
                      SYSTEM_PROMPT, executes any <kb_q> tool call against
                      the eval-90k KB, injects <kb_r>, and loops up to
                      --max-rounds. NOT a fine-tuned model (see B2 in
                      ACCEPTANCE.md for that comparison).

Per-example scoring (see factgate/bench/common.py):
  knowable_heldout      -- canonicalized match of the expected object in
                            the final answer, or a matching [kb:s/r/o] tag.
  unknowable             -- correct behavior is abstention (IDK marker or
                            [kb:idk]); "confident_wrong" = any non-abstaining
                            answer, since no correct object exists.
  adversarial_nearmiss   -- correct behavior is a correction: the response
                            negates the false premise AND cites the KB's
                            real answer.

Gate-side metric (every split): every extracted [kb:s/r/o] citation AND
every executed <kb_q> tool round's top candidate is re-checked via
factgate.gate.verify_claim against the same live KB; `gated_hallucination_rate`
is the fraction that would NOT come back VERIFIED (i.e. the gate would have
blocked or flagged it).

Writes one contract JSON per (split, target) to factgate/results/, mirroring
raincg/bench's contract schema (n, correct, accuracy, ci95, config, notes,
timestamp) -- ported PATTERN, no rain code imported.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from factgate import gate as gate_module
from factgate.bench import common
from factgate.bench.splits import (
    DEFAULT_KB_SNAPSHOT, DEFAULT_KB_TRAIN, EVAL_DIR, SPLITS,
)
from factgate.datagen import protocol
from factgate.kb_service import load_kb

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO / "factgate" / "results"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

_KB_Q_RE = re.compile(r"<kb_q>(.*?)</kb_q>", re.DOTALL)


# ---------------------------------------------------------------------------
# Target dispatch.
# ---------------------------------------------------------------------------

def parse_target(target: str) -> tuple[str, str | None]:
    if target == "echo":
        return ("echo", None)
    if target.startswith("ollama:"):
        model = target.split(":", 1)[1]
        if not model:
            raise ValueError(f"malformed ollama target (missing model): {target!r}")
        return ("ollama", model)
    raise ValueError(f"unknown target {target!r} (expected 'echo' or 'ollama:<model>')")


def run_echo_target(question: str) -> dict:
    """Null baseline: parrots the question back verbatim. No network, no
    KB tool loop -- used for harness self-tests."""
    return {"final_answer": question, "rounds": 0, "executed": []}


def call_ollama(model: str, messages: list[dict], *, host: str = DEFAULT_OLLAMA_HOST,
                 timeout: float = 120.0) -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("message") or {}).get("content", "")


def execute_kb_query(kb, query: dict, *, default_top_k: int = 5) -> dict:
    """Execute one <kb_q> query dict ({s,r,o,unknown,top_k}) against the
    live KB, mirroring the known/unknown-role convention used throughout
    factgate/datagen (see protocol.py, qa.py)."""
    unknown_role = str(query.get("unknown", "O")).upper()
    top_k = int(query.get("top_k") or default_top_k)
    known: dict = {}
    for key, role in (("s", "S"), ("r", "R"), ("o", "O")):
        if role == unknown_role:
            continue
        val = query.get(key)
        if val is not None:
            known[role] = val
    candidates = kb.query(known, unknown_role, top_k=top_k)
    wire = {"results": [[str(sym), round(float(sc), 4)] for sym, sc in candidates]}
    return {"known": known, "unknown_role": unknown_role, "wire": wire}


def run_ollama_target(model: str, question: str, kb, *, max_rounds: int = 6,
                       host: str = DEFAULT_OLLAMA_HOST, timeout: float = 120.0) -> dict:
    messages = [
        {"role": "system", "content": protocol.SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    executed: list[dict] = []
    final_answer = ""
    rounds_used = 0
    for round_i in range(max_rounds):
        rounds_used = round_i + 1
        content = call_ollama(model, messages, host=host, timeout=timeout)
        messages.append({"role": "assistant", "content": content})
        final_answer = content
        m = _KB_Q_RE.search(content)
        if not m:
            break
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            break  # model emitted malformed <kb_q> JSON -- treat as final text
        queries = payload if isinstance(payload, list) else [payload]
        wires = []
        for q in queries:
            if not isinstance(q, dict):
                continue
            try:
                r = execute_kb_query(kb, q)
            except Exception as e:  # noqa: BLE001 - malformed model query, don't crash the run
                r = {"known": {}, "unknown_role": "", "wire": {"results": [], "error": str(e)}}
            executed.append({"query": q, "result": r})
            wires.append(r["wire"])
        if not wires:
            break  # <kb_q> block present but had no valid query dicts
        tool_payload = wires[0] if len(wires) == 1 else wires
        messages.append({"role": "tool",
                          "content": f"<kb_r>{json.dumps(tool_payload, ensure_ascii=False)}</kb_r>"})
    return {"final_answer": final_answer, "rounds": rounds_used, "executed": executed}


# ---------------------------------------------------------------------------
# Per-example scoring.
# ---------------------------------------------------------------------------

def score_example(split: str, example: dict, final_answer: str) -> dict:
    if split == "knowable_heldout":
        correct = common.is_correct_knowable(final_answer, example.get("expected"))
        return {"correct": correct}
    if split == "unknowable":
        abstained = common.is_abstention(final_answer)
        return {"correct": abstained, "abstained": abstained,
                "confident_wrong": not abstained}
    if split == "adversarial_nearmiss":
        corrected = common.is_correction(final_answer, example.get("expected"))
        return {"correct": corrected}
    raise ValueError(f"no scorer for split {split!r}")


def _reconstruct_triple(known: dict, unknown_role: str, top_sym) -> tuple[str, str, str] | None:
    d = dict(known)
    if unknown_role:
        d[unknown_role] = top_sym
    if not all(k in d for k in ("S", "R", "O")):
        return None
    return (str(d["S"]), str(d["R"]), str(d["O"]))


def gate_check_example(kb, final_answer: str, executed: list[dict]) -> dict:
    """Re-verify every extractable claim (final-answer [kb:] citations +
    executed <kb_q> tool rounds' top candidate) through
    factgate.gate.verify_claim against the live KB."""
    claims: set[tuple[str, str, str]] = set(common.extract_kb_citations(final_answer))
    for item in executed:
        result = item.get("result", {})
        wire = result.get("wire", {})
        if wire.get("error") or not wire.get("results"):
            continue
        top_sym = wire["results"][0][0]
        triple = _reconstruct_triple(result.get("known", {}), result.get("unknown_role", ""), top_sym)
        if triple is not None:
            claims.add(triple)

    verdict_counts: collections.Counter = collections.Counter()
    blocked = 0
    for (s, r, o) in claims:
        try:
            v = gate_module.verify_claim(kb, None, s, r, o)
            status = v.status.value
        except Exception:  # noqa: BLE001 - a malformed model-authored claim must never crash the run
            status = "ERROR"
        verdict_counts[status] += 1
        if status != "VERIFIED":
            blocked += 1
    return {"claims_checked": len(claims), "blocked": blocked,
            "verdict_counts": dict(verdict_counts)}


# ---------------------------------------------------------------------------
# Split-level run + contract JSON.
# ---------------------------------------------------------------------------

def run_split(target: str, split: str, examples: list[dict], kb, *,
               max_rounds: int = 6, ollama_host: str = DEFAULT_OLLAMA_HOST,
               timeout: float = 120.0) -> dict:
    kind, model = parse_target(target)
    n = len(examples)
    correct = 0
    abstain_n = 0
    confident_wrong_n = 0
    gate_claims_total = 0
    gate_blocked_total = 0
    verdict_totals: collections.Counter = collections.Counter()
    notes: list[str] = []
    per_example: list[dict] = []

    for ex in examples:
        if kind == "echo":
            run_out = run_echo_target(ex["question"])
        else:
            run_out = run_ollama_target(model, ex["question"], kb, max_rounds=max_rounds,
                                         host=ollama_host, timeout=timeout)
        final_answer = run_out["final_answer"]
        score = score_example(split, ex, final_answer)
        if score["correct"]:
            correct += 1
        if split == "unknowable":
            if score.get("abstained"):
                abstain_n += 1
            else:
                confident_wrong_n += 1

        gate_info = gate_check_example(kb, final_answer, run_out["executed"])
        gate_claims_total += gate_info["claims_checked"]
        gate_blocked_total += gate_info["blocked"]
        verdict_totals.update(gate_info["verdict_counts"])

        per_example.append({"id": ex.get("id"), "final_answer": final_answer,
                             "rounds": run_out["rounds"], "score": score,
                             "gate": gate_info})

    accuracy = correct / n if n else 0.0
    ci95 = common.wilson_ci(correct, n)

    metrics: dict = {}
    if split == "unknowable":
        metrics["abstention_rate"] = abstain_n / n if n else 0.0
        metrics["abstention_n"] = abstain_n
        metrics["confident_wrong_rate"] = confident_wrong_n / n if n else 0.0
        metrics["confident_wrong_n"] = confident_wrong_n
    metrics["gate_claims_checked"] = gate_claims_total
    metrics["gated_hallucination_rate"] = (
        gate_blocked_total / gate_claims_total if gate_claims_total else None)
    metrics["gate_verdict_counts"] = dict(verdict_totals)

    if gate_claims_total == 0:
        notes.append("gate-side check saw 0 extractable claims (no [kb:...] citations "
                      "and no executed <kb_q> tool rounds) -- gated_hallucination_rate "
                      "is not meaningful for this run.")

    return {"n": n, "correct": correct, "accuracy": accuracy, "ci95": list(ci95),
            "metrics": metrics, "notes": " ".join(notes), "per_example": per_example}


def _safe_target(target: str) -> str:
    return target.replace(":", "-").replace("/", "-")


def write_contract(result: dict, target: str, split: str, results_dir: Path, *,
                    seed: int, config: dict) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "split": split, "target": target,
        "n": result["n"], "correct": result["correct"], "accuracy": result["accuracy"],
        "ci95": result["ci95"], "config": config, "metrics": result["metrics"],
        "notes": result["notes"], "evidence_tier": "measured-fresh",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path = results_dir / f"{split}__{_safe_target(target)}__seed{seed}.json"
    out_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    contract["_out_path"] = str(out_path)
    return contract


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="'echo' or 'ollama:<model>'")
    ap.add_argument("--split", required=True, help="split name, or 'all'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--kb-snapshot", type=Path, default=DEFAULT_KB_SNAPSHOT)
    ap.add_argument("--kb-train", type=Path, default=DEFAULT_KB_TRAIN)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        parse_target(args.target)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.split == "all":
        split_names = [s["name"] for s in SPLITS if s["status"] == "active"]
    else:
        split_names = [args.split]

    agent = load_kb(args.kb_train, args.kb_snapshot)
    kb = agent.knowledge

    exit_code = 0
    for split_name in split_names:
        path = args.eval_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"[{split_name}] SKIP: no eval file at {path} "
                  f"(run `python -m factgate.bench.splits --splits {split_name}` first)",
                  file=sys.stderr)
            exit_code = 1
            continue
        examples = common.load_jsonl(path)
        if args.limit is not None:
            examples = examples[: args.limit]
        if not examples:
            print(f"[{split_name}] SKIP: 0 examples (deferred split placeholder?)")
            continue
        try:
            result = run_split(args.target, split_name, examples, kb,
                                max_rounds=args.max_rounds, ollama_host=args.ollama_host,
                                timeout=args.timeout)
        except (urllib.error.URLError, ConnectionError) as e:
            print(f"[{split_name}] FAILED: could not reach target {args.target!r}: {e}",
                  file=sys.stderr)
            exit_code = 1
            continue
        contract = write_contract(
            result, args.target, split_name, args.results_dir, seed=args.seed,
            config={"max_rounds": args.max_rounds, "limit": args.limit,
                    "kb_snapshot": str(args.kb_snapshot), "seed": args.seed})
        summary = {k: v for k, v in contract.items() if k not in ("_out_path",)}
        print(json.dumps(summary, indent=2))
        print(f"-> {contract['_out_path']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

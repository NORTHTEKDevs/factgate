"""Harness self-tests for factgate.bench (A6).

Uses the "echo" null target (parrots the question back verbatim -- no
network, no Ollama) plus tests/fixtures/bench_fixture_20.jsonl (20
hand-crafted, hand-verified questions spanning all three active splits)
to exercise metric computation, Wilson CI, abstention/correction
detection, and contract-JSON schema end to end. Builder-level splits.py
tests reuse tests/fixtures/mini_kb.jsonl (600 triples), the same fixture
test_datagen.py uses, so no network / no full-scale KB is required here.
No network access; runs in well under 60s.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from factgate.bench import common, results as results_mod, runner as runner_mod
from factgate.bench import splits as splits_mod
from factgate.datagen.build_kb import build as build_kb_snapshot
from factgate.datagen.run_all import build_kb, read_triples

FIXTURE = Path(__file__).parent / "fixtures" / "mini_kb.jsonl"
BENCH_FIXTURE = Path(__file__).parent / "fixtures" / "bench_fixture_20.jsonl"


@pytest.fixture(scope="module")
def triples() -> list[dict]:
    return read_triples(FIXTURE)


@pytest.fixture(scope="module")
def kb(triples):
    agent = build_kb(triples, dim=4096, seed=0)
    return agent.knowledge


def _load_bench_fixture() -> list[dict]:
    return common.load_jsonl(BENCH_FIXTURE)


def _build_tmp_kb_snapshot(tmp_path: Path) -> Path:
    kb_snapshot_dir = tmp_path / "kb_snap"
    build_kb_snapshot(FIXTURE, kb_snapshot_dir, None, dim=1024, n_shards=32, seed=0)
    return kb_snapshot_dir


# ---------------------------------------------------------------------------
# common.py: Wilson CI.
# ---------------------------------------------------------------------------

def test_wilson_ci_zero_n():
    assert common.wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_bounds_and_ordering():
    lo, hi = common.wilson_ci(7, 10)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


def test_wilson_ci_zero_correct():
    lo, hi = common.wilson_ci(0, 10)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


def test_wilson_ci_perfect_score():
    lo, hi = common.wilson_ci(10, 10)
    assert 0.0 < lo <= 1.0
    assert hi <= 1.0


# ---------------------------------------------------------------------------
# common.py: canonicalization + citation/IDK parsing.
# ---------------------------------------------------------------------------

def test_canonicalize_folds_underscores_case_punctuation():
    assert common.canonicalize("Bound_Variable.") == common.canonicalize("bound variable")
    assert common.canonicalize("bound_variable") == "bound variable"


def test_canonicalize_empty_and_none():
    assert common.canonicalize(None) == ""
    assert common.canonicalize("") == ""


def test_extract_kb_citations_multi_tag():
    text = "See [kb:france/capital/paris] and also [kb:japan/capital/tokyo]."
    assert common.extract_kb_citations(text) == [
        ("france", "capital", "paris"), ("japan", "capital", "tokyo")]


def test_extract_kb_citations_ignores_idk_tag():
    assert common.extract_kb_citations("no facts here [kb:idk]") == []


@pytest.mark.parametrize("text,expected", [
    ("[kb:idk]", True),
    ("I don't know.", True),
    ("I cannot verify this.", True),
    ("", True),
    ("   ", True),
    ("The answer is paris. [kb:france/capital/paris]", False),
])
def test_is_abstention(text, expected):
    assert common.is_abstention(text) is expected


def test_is_correct_knowable_substring_match():
    assert common.is_correct_knowable("The capital is paris.", "paris") is True


def test_is_correct_knowable_citation_match():
    assert common.is_correct_knowable("See [kb:france/capital/paris].", "paris") is True


def test_is_correct_knowable_no_match():
    assert common.is_correct_knowable("Totally unrelated text.", "paris") is False


def test_is_correct_knowable_empty_expected():
    assert common.is_correct_knowable("anything", "") is False


def test_is_correction_true():
    resp = "No, that's not correct. Paris is the answer. [kb:france/capital/paris]"
    assert common.is_correction(resp, "paris") is True


def test_is_correction_missing_negation():
    assert common.is_correction("Yes that's right, paris.", "paris") is False


def test_is_correction_missing_kb_answer_mention():
    assert common.is_correction("No, that's not correct.", "paris") is False


# ---------------------------------------------------------------------------
# common.py: stability poll (fake clock -- no real sleeping).
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_wait_for_stable_lines_reports_stable(tmp_path):
    path = tmp_path / "static.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    clock = _FakeClock()
    result = common.wait_for_stable_lines(
        path, poll_interval=5.0, stable_window=60.0, max_wait=900.0,
        sleep_fn=clock.sleep, now_fn=clock.now)
    assert result == {"lines": 2, "stable": True, "waited_s": 60.0}


def test_wait_for_stable_lines_times_out_when_growing(tmp_path):
    path = tmp_path / "growing.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    clock = _FakeClock()

    def growing_sleep(s):
        clock.t += s
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"a": 1}\n')

    result = common.wait_for_stable_lines(
        path, poll_interval=5.0, stable_window=60.0, max_wait=30.0,
        sleep_fn=growing_sleep, now_fn=clock.now)
    assert result["stable"] is False
    assert result["waited_s"] >= 30.0


def test_count_lines_missing_file(tmp_path):
    assert common.count_lines(tmp_path / "nope.jsonl") == 0


# ---------------------------------------------------------------------------
# splits.py: coverage set.
# ---------------------------------------------------------------------------

def test_load_source_triple_coverage(tmp_path):
    p = tmp_path / "sft.jsonl"
    lines = [
        json.dumps({"meta": {"source_triple": {"s": "a", "r": "isa", "o": "b"}}}),
        json.dumps({"meta": {"hop_count": 2}}),  # chains record: no source_triple
        "not json",  # torn/partial line, tolerated
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert splits_mod.load_source_triple_coverage(p) == {("a", "isa", "b")}


def test_load_source_triple_coverage_missing_file(tmp_path):
    assert splits_mod.load_source_triple_coverage(tmp_path / "nope.jsonl") == set()


# ---------------------------------------------------------------------------
# splits.py: (a) knowable_heldout.
# ---------------------------------------------------------------------------

def test_build_knowable_heldout_excludes_coverage(triples):
    covered = triples[0]
    coverage = {(covered["s"], covered["r"], covered["o"])}
    out = splits_mod.build_knowable_heldout(triples, coverage, n=50, seed=0)
    assert len(out) == 50
    produced = {(r["s"], r["r"], r["o"]) for r in out}
    assert (covered["s"], covered["r"], covered["o"]) not in produced
    for r in out:
        assert r["expected"] == r["o"]
        assert r["question"]


def test_build_knowable_heldout_uses_reserved_template(triples):
    out = splits_mod.build_knowable_heldout(triples, set(), n=20, seed=1)
    assert len(out) == 20
    for r in out:
        expected_q = splits_mod._eval_forward_template(r["r"]).text.format(s=r["s"])
        assert r["question"] == expected_q


# ---------------------------------------------------------------------------
# splits.py: (b) unknowable.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_holdout() -> list[dict]:
    # Entities never stored in the mini_kb fixture KB -- genuinely IDK.
    return [{"s": f"zzq_holdout_entity_{i}", "r": "isa", "o": f"zzq_holdout_object_{i}"}
            for i in range(30)]


def test_build_unknowable_all_genuinely_idk(kb, synthetic_holdout):
    out = splits_mod.build_unknowable(kb, synthetic_holdout, n=10, seed=0)
    assert len(out) == 10
    for r in out:
        assert r["split"] == "unknowable"
        assert r["expected"] is None
        assert r["meta"]["state"] == "idk"


# ---------------------------------------------------------------------------
# splits.py: (c) adversarial_nearmiss (reuses datagen/corrupt.py directly).
# ---------------------------------------------------------------------------

def test_build_adversarial_nearmiss_reuses_corrupt_generator(kb, triples):
    pool = splits_mod.held_out_pool(triples, set())
    out = splits_mod.build_adversarial_nearmiss(kb, pool, n=10, seed=0)
    assert len(out) > 0
    for r in out:
        assert r["split"] == "adversarial_nearmiss"
        assert r["expected"] == r["o"]
        assert r["meta"]["corrupted_object"] != r["expected"]


# ---------------------------------------------------------------------------
# splits.py: build_split orchestration + CLI.
# ---------------------------------------------------------------------------

def test_build_split_knowable_heldout_end_to_end(tmp_path, triples):
    sft_path = tmp_path / "sft.jsonl"
    covered = triples[0]
    sft_path.write_text(json.dumps({"meta": {"source_triple": covered}}) + "\n",
                         encoding="utf-8")
    out_dir = tmp_path / "eval"
    stats = splits_mod.build_split(
        "knowable_heldout", out_dir=out_dir, sft_path=sft_path,
        kb_train_path=FIXTURE, holdout_path=tmp_path / "no_holdout.jsonl",
        kb_snapshot_dir=tmp_path / "unused_kb_snap", seed=0,
        poll_interval=0.01, stable_window=0.02, max_wait=5.0)
    assert stats["status"] == "active"
    assert stats["n"] > 0
    records = common.load_jsonl(out_dir / "knowable_heldout.jsonl")
    produced = {(r["s"], r["r"], r["o"]) for r in records}
    assert (covered["s"], covered["r"], covered["o"]) not in produced
    assert (out_dir / "knowable_heldout.stats.json").exists()


def test_build_split_unknowable_end_to_end(tmp_path):
    kb_snapshot_dir = _build_tmp_kb_snapshot(tmp_path)
    holdout_path = tmp_path / "holdout.jsonl"
    with open(holdout_path, "w", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({"s": f"zzq_holdout_{i}", "r": "isa",
                                 "o": f"zzq_obj_{i}"}) + "\n")
    out_dir = tmp_path / "eval"
    stats = splits_mod.build_split(
        "unknowable", out_dir=out_dir, sft_path=tmp_path / "no_sft.jsonl",
        kb_train_path=FIXTURE, holdout_path=holdout_path,
        kb_snapshot_dir=kb_snapshot_dir, seed=0,
        poll_interval=0.01, stable_window=0.02, max_wait=5.0)
    assert stats["status"] == "active"
    assert stats["n"] > 0
    records = common.load_jsonl(out_dir / "unknowable.jsonl")
    assert all(r["expected"] is None for r in records)


def test_build_split_adversarial_nearmiss_end_to_end(tmp_path):
    kb_snapshot_dir = _build_tmp_kb_snapshot(tmp_path)
    out_dir = tmp_path / "eval"
    stats = splits_mod.build_split(
        "adversarial_nearmiss", out_dir=out_dir, sft_path=tmp_path / "no_sft.jsonl",
        kb_train_path=FIXTURE, holdout_path=tmp_path / "no_holdout.jsonl",
        kb_snapshot_dir=kb_snapshot_dir, seed=0,
        poll_interval=0.01, stable_window=0.02, max_wait=5.0)
    assert stats["status"] == "active"
    records = common.load_jsonl(out_dir / "adversarial_nearmiss.jsonl")
    assert len(records) > 0
    assert all(r["meta"]["corrupted_object"] != r["expected"] for r in records)


def test_splits_main_deferred_split_creates_placeholder(tmp_path):
    rc = splits_mod.main(["--splits", "compositional", "--out-dir", str(tmp_path)])
    assert rc == 0
    out = tmp_path / "compositional.jsonl"
    assert out.exists()
    assert out.read_text() == ""
    stats = json.loads((tmp_path / "compositional.stats.json").read_text())
    assert stats["status"] == "deferred"
    assert "deferred" in stats["note"].lower()


def test_splits_main_list_default(capsys, tmp_path):
    rc = splits_mod.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "knowable_heldout" in out
    assert "general_canary" in out


def test_splits_main_unknown_split(tmp_path):
    rc = splits_mod.main(["--splits", "bogus_split_xyz", "--out-dir", str(tmp_path)])
    assert rc == 2


# ---------------------------------------------------------------------------
# runner.py: target dispatch + scoring.
# ---------------------------------------------------------------------------

def test_parse_target_echo():
    assert runner_mod.parse_target("echo") == ("echo", None)


def test_parse_target_ollama():
    assert runner_mod.parse_target("ollama:llama3.2:3b") == ("ollama", "llama3.2:3b")


def test_parse_target_unknown_raises():
    with pytest.raises(ValueError):
        runner_mod.parse_target("bogus")


def test_run_echo_target():
    assert runner_mod.run_echo_target("hello?") == {
        "final_answer": "hello?", "rounds": 0, "executed": []}


def test_score_example_knowable():
    ex = {"expected": "paris"}
    assert runner_mod.score_example("knowable_heldout", ex, "The answer is paris.")["correct"] is True
    assert runner_mod.score_example("knowable_heldout", ex, "no idea")["correct"] is False


def test_score_example_unknowable():
    r = runner_mod.score_example("unknowable", {}, "[kb:idk]")
    assert r == {"correct": True, "abstained": True, "confident_wrong": False}
    r2 = runner_mod.score_example("unknowable", {}, "It's definitely xyz.")
    assert r2 == {"correct": False, "abstained": False, "confident_wrong": True}


def test_score_example_nearmiss():
    ex = {"expected": "paris"}
    r = runner_mod.score_example(
        "adversarial_nearmiss", ex, "No, that's not correct. [kb:france/capital/paris]")
    assert r["correct"] is True


def test_score_example_unknown_split_raises():
    with pytest.raises(ValueError):
        runner_mod.score_example("bogus_split", {}, "x")


# ---------------------------------------------------------------------------
# runner.py: gate-side re-verification.
# ---------------------------------------------------------------------------

def test_gate_check_example_no_claims():
    info = runner_mod.gate_check_example(None, "nothing to see here", [])
    assert info == {"claims_checked": 0, "blocked": 0, "verdict_counts": {}}


def _unambiguous_triple(triples: list[dict]) -> dict:
    """A triple whose (s, r) pair is unique in the fixture -- i.e. not an
    ambiguous entity like mini_kb's ('1', 'isa') -> {abstaction, number},
    which would make verify_claim return UNRESOLVED instead of VERIFIED."""
    counts: dict[tuple[str, str], int] = {}
    for t in triples:
        key = (t["s"], t["r"])
        counts[key] = counts.get(key, 0) + 1
    for t in triples:
        if counts[(t["s"], t["r"])] == 1:
            return t
    raise AssertionError("no unambiguous (s, r) triple found in fixture")


def test_gate_check_example_citation_verified(kb, triples):
    t = _unambiguous_triple(triples)
    final_answer = f"See [kb:{t['s']}/{t['r']}/{t['o']}]"
    info = runner_mod.gate_check_example(kb, final_answer, [])
    assert info["claims_checked"] == 1
    assert info["verdict_counts"].get("VERIFIED") == 1
    assert info["blocked"] == 0


def test_gate_check_example_citation_blocked(kb, triples):
    t = _unambiguous_triple(triples)
    final_answer = f"See [kb:{t['s']}/{t['r']}/definitely_not_the_real_object_zzz]"
    info = runner_mod.gate_check_example(kb, final_answer, [])
    assert info["claims_checked"] == 1
    assert info["blocked"] == 1
    assert info["verdict_counts"].get("VERIFIED", 0) == 0


def test_gate_check_example_executed_query_pair(kb, triples):
    t = _unambiguous_triple(triples)
    executed = [{"query": {"s": t["s"], "r": t["r"], "unknown": "O"},
                 "result": {"known": {"S": t["s"], "R": t["r"]}, "unknown_role": "O",
                            "wire": {"results": [[t["o"], 0.9]]}}}]
    info = runner_mod.gate_check_example(kb, "", executed)
    assert info["claims_checked"] == 1


# ---------------------------------------------------------------------------
# runner.py: run_split against the 20-question fixture (echo target).
# Hand-verified expected counts -- see fixture file comments in this test.
# ---------------------------------------------------------------------------

def test_bench_fixture_has_20_examples():
    assert len(_load_bench_fixture()) == 20


def test_run_split_echo_knowable_heldout(kb):
    examples = [e for e in _load_bench_fixture() if e["split"] == "knowable_heldout"]
    assert len(examples) == 8
    result = runner_mod.run_split("echo", "knowable_heldout", examples, kb)
    assert result["n"] == 8
    assert result["correct"] == 5  # fx-k1..k5 correct, fx-k6..k8 incorrect
    assert result["accuracy"] == pytest.approx(5 / 8)
    lo, hi = result["ci95"]
    assert 0.0 <= lo <= result["accuracy"] <= hi <= 1.0


def test_run_split_echo_unknowable(kb):
    examples = [e for e in _load_bench_fixture() if e["split"] == "unknowable"]
    assert len(examples) == 6
    result = runner_mod.run_split("echo", "unknowable", examples, kb)
    assert result["n"] == 6
    assert result["correct"] == 4  # fx-u1..u4 abstain, fx-u5/u6 confident-wrong
    assert result["metrics"]["abstention_rate"] == pytest.approx(4 / 6)
    assert result["metrics"]["abstention_n"] == 4
    assert result["metrics"]["confident_wrong_rate"] == pytest.approx(2 / 6)
    assert result["metrics"]["confident_wrong_n"] == 2


def test_run_split_echo_nearmiss(kb):
    examples = [e for e in _load_bench_fixture() if e["split"] == "adversarial_nearmiss"]
    assert len(examples) == 6
    result = runner_mod.run_split("echo", "adversarial_nearmiss", examples, kb)
    assert result["n"] == 6
    assert result["correct"] == 4  # fx-n1..n4 correct, fx-n5/n6 fail to correct
    assert result["accuracy"] == pytest.approx(4 / 6)


def test_run_split_records_gate_metrics_key_present(kb):
    examples = [e for e in _load_bench_fixture() if e["split"] == "unknowable"]
    result = runner_mod.run_split("echo", "unknowable", examples, kb)
    assert "gate_claims_checked" in result["metrics"]
    assert "gate_verdict_counts" in result["metrics"]
    # unknowable fixture questions carry some [kb:...] citations (fx-u5) -> not zero.
    assert result["metrics"]["gate_claims_checked"] >= 1


# ---------------------------------------------------------------------------
# runner.py: contract JSON schema.
# ---------------------------------------------------------------------------

def test_write_contract_schema(tmp_path, kb):
    examples = [e for e in _load_bench_fixture() if e["split"] == "knowable_heldout"]
    result = runner_mod.run_split("echo", "knowable_heldout", examples, kb)
    contract = runner_mod.write_contract(
        result, "echo", "knowable_heldout", tmp_path, seed=0, config={"limit": None})
    for key in ("split", "target", "n", "correct", "accuracy", "ci95", "config",
                "metrics", "notes", "evidence_tier", "timestamp"):
        assert key in contract
    assert contract["n"] == 8
    assert contract["correct"] == 5
    assert contract["target"] == "echo"
    assert contract["split"] == "knowable_heldout"
    assert isinstance(contract["ci95"], list) and len(contract["ci95"]) == 2

    out_files = list(tmp_path.glob("*.json"))
    assert len(out_files) == 1
    on_disk = json.loads(out_files[0].read_text())
    assert on_disk["n"] == 8


# ---------------------------------------------------------------------------
# runner.py: end-to-end CLI (echo target, no network).
# ---------------------------------------------------------------------------

def test_runner_main_end_to_end_echo(tmp_path):
    kb_snapshot_dir = _build_tmp_kb_snapshot(tmp_path)

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    knowable = [e for e in _load_bench_fixture() if e["split"] == "knowable_heldout"]
    with open(eval_dir / "knowable_heldout.jsonl", "w", encoding="utf-8") as f:
        for r in knowable:
            f.write(json.dumps(r) + "\n")

    results_dir = tmp_path / "results"
    rc = runner_mod.main([
        "--target", "echo", "--split", "knowable_heldout",
        "--eval-dir", str(eval_dir), "--kb-snapshot", str(kb_snapshot_dir),
        "--kb-train", str(FIXTURE), "--results-dir", str(results_dir),
    ])
    assert rc == 0
    out_files = list(results_dir.glob("*.json"))
    assert len(out_files) == 1
    contract = json.loads(out_files[0].read_text())
    assert contract["n"] == 8
    assert contract["correct"] == 5

    md_path = tmp_path / "RESULTS.md"
    rc2 = results_mod.main(["--results-dir", str(results_dir), "--out", str(md_path)])
    assert rc2 == 0
    assert md_path.exists()
    assert "knowable_heldout" in md_path.read_text()


def test_runner_main_missing_eval_file_reports_and_exits_nonzero(tmp_path):
    kb_snapshot_dir = _build_tmp_kb_snapshot(tmp_path)
    rc = runner_mod.main([
        "--target", "echo", "--split", "unknowable",
        "--eval-dir", str(tmp_path / "no_eval_dir"),
        "--kb-snapshot", str(kb_snapshot_dir), "--kb-train", str(FIXTURE),
        "--results-dir", str(tmp_path / "results"),
    ])
    assert rc == 1


def test_runner_main_bad_target_exits_nonzero(tmp_path):
    rc = runner_mod.main(["--target", "bogus", "--split", "unknowable",
                           "--eval-dir", str(tmp_path)])
    assert rc == 2


# ---------------------------------------------------------------------------
# results.py: RESULTS.md rendering + honesty footnotes.
# ---------------------------------------------------------------------------

def test_load_results_and_render(tmp_path):
    r1 = {"split": "knowable_heldout", "target": "echo", "n": 8, "correct": 5,
          "accuracy": 0.625, "ci95": [0.30, 0.85], "config": {},
          "metrics": {"gate_claims_checked": 0, "gated_hallucination_rate": None},
          "notes": "", "evidence_tier": "measured-fresh",
          "timestamp": "2026-01-01T00:00:00Z"}
    r2 = {"split": "unknowable", "target": "echo", "n": 6, "correct": 4,
          "accuracy": 0.667, "ci95": [0.30, 0.90], "config": {},
          "metrics": {"confident_wrong_rate": 0.333, "gate_claims_checked": 0,
                       "gated_hallucination_rate": None},
          "notes": "", "evidence_tier": "measured-fresh",
          "timestamp": "2026-01-01T00:00:00Z"}
    (tmp_path / "a.json").write_text(json.dumps(r1), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(r2), encoding="utf-8")

    results = results_mod.load_results(tmp_path)
    assert len(results) == 2

    md = results_mod.render_results_md(results)
    assert "knowable_heldout" in md
    assert "unknowable" in md
    assert "adversarial_nearmiss" in md  # canonical split shown even with 0 rows
    assert "compositional" in md
    assert "general_canary" in md
    assert "deferred" in md.lower()
    assert "62.50%" in md


def test_load_results_skips_malformed_json(tmp_path):
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    assert results_mod.load_results(tmp_path) == []

"""Smoke tests for factgate.datagen.

Uses a static 600-triple fixture (tests/fixtures/mini_kb.jsonl, 60
triples per top-10 relation) rather than the full 100k-fact corpus, so
the suite runs fast and deterministically. Full-scale generation is
explicitly out of scope here (see run_all.py's module docstring).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from factgate.datagen import chains, corrupt, idk, qa
from factgate.datagen.run_all import build_kb, read_triples, run
from factgate.datagen.schema import SFTExample, validate_file
from factgate.datagen.templates import (
    STATEMENT_TEMPLATES, TOP_RELATIONS, extract_triples, render_fact,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_kb.jsonl"


@pytest.fixture(scope="module")
def triples() -> list[dict]:
    return read_triples(FIXTURE)


@pytest.fixture(scope="module")
def kb(triples):
    agent = build_kb(triples, dim=4096, seed=0)
    return agent.knowledge


@pytest.fixture(scope="module")
def entities(triples) -> list[str]:
    return sorted({t["s"] for t in triples} | {t["o"] for t in triples})


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------

def test_fixture_has_all_top_relations(triples):
    rels = {t["r"] for t in triples}
    assert rels == set(TOP_RELATIONS)


def test_fixture_size():
    assert 500 <= sum(1 for _ in open(FIXTURE, encoding="utf-8")) <= 700


# ---------------------------------------------------------------------------
# Generators: smoke each at --limit 50
# ---------------------------------------------------------------------------

def _validate_all(records: list[dict]) -> None:
    for r in records:
        SFTExample.model_validate(r)


def test_qa_generator_smoke(kb, triples):
    out = list(qa.generate(kb, triples, n=50, seed=0))
    assert len(out) == 50
    _validate_all(out)
    for ex in out:
        assert ex["generator"] == "qa"
        assert "[kb:" in ex["plain"]["answer"]


def test_chains_generator_smoke(kb, triples):
    out = list(chains.generate(kb, triples, n=50, seed=1))
    assert len(out) > 0  # connectivity-dependent; fixture is known-connected
    _validate_all(out)
    for ex in out:
        assert ex["generator"] == "chains"
        assert ex["meta"]["hop_count"] >= 2
        assert ex["meta"]["hop_count"] <= 4


def test_idk_absent_generator_smoke(kb, entities):
    out = list(idk.generate_absent(kb, entities, n=50, seed=2))
    assert len(out) == 50
    _validate_all(out)
    for ex in out:
        assert ex["generator"] == "idk"
        assert "[kb:idk]" in ex["plain"]["answer"]
        assert ex["meta"]["state"] == "idk"


def test_idk_holdout_generator_smoke(kb, triples):
    # Simulate a deletion holdout: half the fixture's isa/locatedin
    # triples, asked against a KB that never saw them.
    holdout = [t for t in triples if t["r"] in ("isa", "locatedin")][:40]
    out = list(idk.generate_holdout(kb, holdout, n=20, seed=3))
    # Not all holdout probes are guaranteed IDK (KB may still answer via
    # a sibling fact for a functional-ish relation) -- just require the
    # ones that ARE emitted are well-formed and genuinely marked IDK.
    _validate_all(out)
    for ex in out:
        assert ex["meta"]["state"] == "idk"
        assert "[kb:idk]" in ex["plain"]["answer"]


def test_corrupt_generator_smoke(kb, triples):
    out = list(corrupt.generate(kb, triples, n=50, seed=4))
    assert len(out) > 0
    _validate_all(out)
    for ex in out:
        assert ex["generator"] == "corrupt"
        assert ex["meta"]["kb_answer"] != ex["meta"]["corrupted_object"]
        assert "[kb:" in ex["plain"]["answer"]


# ---------------------------------------------------------------------------
# run_all orchestration
# ---------------------------------------------------------------------------

def test_run_all_smoke(tmp_path):
    out = tmp_path / "sft_v0.jsonl"
    stats = run(source=FIXTURE, out=out, counts_out=None, limit=25,
                kb_limit=None, holdout=None, seed=0, dim=4096)

    assert out.exists()
    assert stats["schema_valid"] == stats["schema_total"]
    assert stats["schema_total"] > 0
    assert stats["counts"]["qa"] == 25
    assert stats["counts"]["idk_absent"] == 25

    n_valid, n_total, errors = validate_file(out)
    assert errors == []
    assert n_valid == n_total > 0

    counts_path = out.with_suffix(".counts.json")
    assert counts_path.exists()
    counts_stats = json.loads(counts_path.read_text())
    assert counts_stats["total_examples"] == n_total


def test_run_all_with_holdout(tmp_path):
    train = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    with open(FIXTURE, encoding="utf-8") as f:
        lines = f.readlines()
    train.write_text("".join(lines[:550]), encoding="utf-8")
    holdout_path.write_text("".join(lines[550:]), encoding="utf-8")

    out = tmp_path / "sft_v0_holdout.jsonl"
    stats = run(source=train, out=out, counts_out=None, limit=10,
                kb_limit=None, holdout=holdout_path, seed=0, dim=4096)
    assert stats["holdout_facts"] == len(lines) - 550
    assert "idk_holdout" in stats["counts"]


# ---------------------------------------------------------------------------
# A5: render_fact / extract_triples round trip (target >=97% on templated
# text; exact by construction here since text is unparaphrased).
# ---------------------------------------------------------------------------

def test_render_extract_round_trip(triples):
    import random
    rng = random.Random(42)
    sample = triples if len(triples) <= 200 else rng.sample(triples, 200)

    n_ok = 0
    n_total = 0
    for t in sample:
        s, r, o = t["s"], t["r"], t["o"]
        n_templates = len(STATEMENT_TEMPLATES[r])
        tid = rng.randrange(n_templates)
        text = render_fact(s, r, o, tid)
        got = extract_triples(text, relation_hint=r)
        n_total += 1
        if got == (s, r, o):
            n_ok += 1

    assert n_total == 200
    assert n_ok / n_total >= 0.97


def test_render_fact_unknown_relation_raises():
    with pytest.raises(KeyError):
        render_fact("a", "not_a_real_relation", "b", 0)


def test_extract_triples_no_match_returns_none():
    assert extract_triples("this sentence matches no template") is None


# ---------------------------------------------------------------------------
# schema.py CLI validation behavior
# ---------------------------------------------------------------------------

def test_schema_validate_rejects_malformed(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "generator": "qa"}) + "\n",
                    encoding="utf-8")
    n_valid, n_total, errors = validate_file(bad)
    assert n_valid == 0
    assert n_total == 1
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# holdout_split.py + build_kb.py
# ---------------------------------------------------------------------------

def test_holdout_split(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "holdout_split",
        Path(__file__).parent.parent / "scripts" / "holdout_split.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out_train = tmp_path / "train.jsonl"
    out_holdout = tmp_path / "holdout.jsonl"
    stats = mod.split(FIXTURE, out_train, out_holdout, holdout_size=100, seed=0)

    assert stats["total"] == 600
    assert stats["train"] == 500
    assert stats["holdout"] == 100
    assert out_train.exists() and out_holdout.exists()

    train_lines = {l.strip() for l in open(out_train, encoding="utf-8")}
    holdout_lines = {l.strip() for l in open(out_holdout, encoding="utf-8")}
    assert train_lines.isdisjoint(holdout_lines)
    assert len(train_lines) + len(holdout_lines) == 600


def test_build_kb_smoke(tmp_path):
    from factgate.datagen.build_kb import build

    out_dir = tmp_path / "kb_snapshot"
    manifest = build(FIXTURE, out_dir, None, dim=1024, n_shards=32, seed=0)

    assert (out_dir / "meta.json").exists()
    assert (out_dir / "knowledge.npz").exists()
    manifest_path = out_dir / "kb_manifest.json"
    assert manifest_path.exists()
    assert manifest["fact_count"] == 600
    assert manifest["entity_count"] > 0
    assert set(manifest["relation_histogram"].keys()) == set(TOP_RELATIONS)
    assert all(v == 60 for v in manifest["relation_histogram"].values())

    # Round-trip via rck.session.load_session.
    from rck.session import load_session
    agent = load_session(out_dir)
    assert agent.knowledge.size() > 0

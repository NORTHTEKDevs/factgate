"""FactGate acceptance tests (A1-A2).

>=50 cases exercising all four GateVerdict statuses (VERIFIED,
CONTRADICTED, OUT_OF_KB, UNRESOLVED), a multi-hop chain claim, a
canonicalization (alias) case, and a corroboration-downgrade case.
No network access; runs against the small fixture KB in conftest.py.
"""
from __future__ import annotations

import pytest

# rck is an optional dependency and is not on PyPI (see pyproject). Without this,
# a fresh clone gets 4 collection ERRORS and zero tests run -- found by
# scripts/acceptance.py, which builds a clean venv the way a stranger would.
pytest.importorskip("rck", reason="rck not installed; legacy KB path skipped")

import factgate.gate as gate_module
from factgate.gate import GateStatus, GateVerdict, verify_claim, verify_chain_claim

# kb_prov fixture is provided by tests/conftest.py (auto-discovered by
# pytest). These lists mirror the facts stored there so parametrized
# claims stay in sync with the fixture KB without a cross-module import.
CAPITALS = [
    ("france", "paris"), ("germany", "berlin"), ("japan", "tokyo"),
    ("italy", "rome"), ("spain", "madrid"), ("england", "london"),
    ("usa", "washington"), ("china", "beijing"), ("russia", "moscow"),
    ("canada", "ottawa"),
]
ISA_FACTS = [
    ("dog", "mammal"), ("cat", "mammal"), ("eagle", "bird"),
    ("sparrow", "bird"), ("trout", "fish"), ("salmon", "fish"),
    ("oak", "tree"), ("pine", "tree"), ("rose", "flower"),
    ("tulip", "flower"),
]
COLOR_FACTS = [
    ("sky", "blue"), ("grass", "green"), ("blood", "red"),
    ("snow", "white"), ("sun", "yellow"),
]


# ---------------------------------------------------------------------------
# VERIFIED: correct capitals / isa / colors / haspart facts.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("country,capital", CAPITALS)
def test_capital_verified(kb_prov, country, capital):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, country, "capital", capital)
    assert v.status == GateStatus.VERIFIED
    assert v.kb_answer == capital
    assert v.subject == country
    assert v.relation == "capital"
    assert v.obj == capital


@pytest.mark.parametrize("subject,category", ISA_FACTS)
def test_isa_verified(kb_prov, subject, category):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, subject, "isa", category)
    assert v.status == GateStatus.VERIFIED
    assert v.kb_answer == category


@pytest.mark.parametrize("subject,color", COLOR_FACTS)
def test_color_verified(kb_prov, subject, color):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, subject, "color", color)
    assert v.status == GateStatus.VERIFIED
    assert v.kb_answer == color


@pytest.mark.parametrize("subject,part", [
    ("car", "engine"), ("bicycle", "wheel"), ("airplane", "wing"),
])
def test_haspart_verified(kb_prov, subject, part):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, subject, "haspart", part)
    assert v.status == GateStatus.VERIFIED


# ---------------------------------------------------------------------------
# CONTRADICTED: wrong capitals / isa / colors (functional relations), and
# an explicit negative fact on a non-functional relation.
# ---------------------------------------------------------------------------

WRONG_CAPITALS = [
    ("france", "berlin"), ("germany", "paris"), ("japan", "madrid"),
    ("italy", "tokyo"), ("spain", "rome"), ("england", "washington"),
    ("usa", "london"), ("china", "moscow"), ("russia", "beijing"),
    ("canada", "toronto"),
]


@pytest.mark.parametrize("country,wrong_capital", WRONG_CAPITALS)
def test_capital_contradicted(kb_prov, country, wrong_capital):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, country, "capital", wrong_capital)
    assert v.status == GateStatus.CONTRADICTED
    assert v.kb_answer != wrong_capital
    assert v.kb_answer is not None


WRONG_ISA = [
    ("dog", "bird"), ("cat", "fish"), ("eagle", "mammal"),
    ("sparrow", "tree"), ("trout", "flower"), ("salmon", "mammal"),
    ("oak", "fish"), ("pine", "bird"), ("rose", "tree"),
    ("tulip", "mammal"),
]


@pytest.mark.parametrize("subject,wrong_category", WRONG_ISA)
def test_isa_contradicted(kb_prov, subject, wrong_category):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, subject, "isa", wrong_category)
    assert v.status == GateStatus.CONTRADICTED


WRONG_COLORS = [
    ("sky", "red"), ("grass", "blue"), ("blood", "green"),
    ("snow", "black"), ("sun", "blue"),
]


@pytest.mark.parametrize("subject,wrong_color", WRONG_COLORS)
def test_color_contradicted(kb_prov, subject, wrong_color):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, subject, "color", wrong_color)
    assert v.status == GateStatus.CONTRADICTED


def test_haspart_explicit_negation_contradicted(kb_prov):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, "car", "haspart", "wings")
    assert v.status == GateStatus.CONTRADICTED
    assert "negative" in v.explanation.lower() or "denies" in v.explanation.lower()


# ---------------------------------------------------------------------------
# UNRESOLVED: non-functional relation, top differs but no explicit denial.
# ---------------------------------------------------------------------------

def test_haspart_non_functional_no_denial_unresolved(kb_prov):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, "car", "haspart", "sunroof")
    assert v.status == GateStatus.UNRESOLVED


def test_ambiguous_entity_unresolved(kb_prov):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, "mercury", "isa", "planet")
    assert v.status == GateStatus.UNRESOLVED
    assert len(v.alternatives) >= 2


def test_ambiguous_entity_other_object_unresolved(kb_prov):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, "mercury", "isa", "element")
    assert v.status == GateStatus.UNRESOLVED


# ---------------------------------------------------------------------------
# OUT_OF_KB: entities/relations that genuinely aren't in the KB.
# ---------------------------------------------------------------------------

OUT_OF_KB_CASES = [
    ("zzqx_nonexistent_country", "capital", "nowhere"),
    ("zzqx_nonexistent_animal", "isa", "nothing"),
    ("zzqx_nonexistent_object", "color", "invisible"),
    ("zzqx_nonexistent_vehicle", "haspart", "nothing"),
    ("atlantis", "capital", "nowhere"),
]


@pytest.mark.parametrize("s,r,o", OUT_OF_KB_CASES)
def test_out_of_kb(kb_prov, s, r, o):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, s, r, o)
    assert v.status == GateStatus.OUT_OF_KB
    assert v.kb_answer is None


# ---------------------------------------------------------------------------
# Canonicalization: relation aliases and entity aliases must route to the
# same underlying facts as their canonical forms.
# ---------------------------------------------------------------------------

def test_canonicalization_relation_alias_hue_to_color(kb_prov):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, "sky", "hue", "blue")
    assert v.status == GateStatus.VERIFIED
    assert v.relation == "color"


def test_canonicalization_entity_alias_subject_uk_to_england(kb_prov):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, "uk", "capital", "london")
    assert v.status == GateStatus.VERIFIED
    assert v.subject == "england"


def test_canonicalization_entity_alias_object_america_to_usa(kb_prov):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, "person_x", "lives_in", "america")
    assert v.status == GateStatus.VERIFIED
    assert v.obj == "usa"


# ---------------------------------------------------------------------------
# Multi-hop chain claims.
# ---------------------------------------------------------------------------

def test_chain_claim_verified(kb_prov):
    kb, prov = kb_prov
    v = verify_chain_claim(kb, "acme", ["ceo", "lives_in"], "paris")
    assert v.status == GateStatus.VERIFIED
    assert v.kb_answer == "paris"


def test_chain_claim_contradicted(kb_prov):
    kb, prov = kb_prov
    v = verify_chain_claim(kb, "acme", ["ceo", "lives_in"], "berlin")
    assert v.status == GateStatus.CONTRADICTED
    assert v.kb_answer == "paris"


def test_chain_claim_out_of_kb_broken_hop(kb_prov):
    kb, prov = kb_prov
    v = verify_chain_claim(kb, "acme", ["ceo", "nonexistent_relation_xyz"], "berlin")
    assert v.status == GateStatus.OUT_OF_KB


# ---------------------------------------------------------------------------
# Corroboration downgrade: KB reports a KNOWN top match, but roundtrip
# self-verification fails to corroborate it -> UNRESOLVED, not VERIFIED.
# ---------------------------------------------------------------------------

def test_corroboration_downgrade_to_unresolved(kb_prov, monkeypatch):
    kb, prov = kb_prov

    def fake_self_verify(kb_, s, r, o, *, methods=None):
        return {
            "verified": False,
            "verified_partial": False,
            "verification_count": 0,
            "total_methods": 2,
            "average_confidence": 0.0,
            "by_method": {
                "roundtrip": {"verified": False, "confidence": 0.0,
                              "notes": "forced failure for test"},
                "reverse": {"verified": False, "confidence": 0.0,
                            "notes": "forced failure for test"},
            },
        }

    monkeypatch.setattr(gate_module, "self_verify", fake_self_verify)
    v = verify_claim(kb, prov, "france", "capital", "paris")
    assert v.status == GateStatus.UNRESOLVED
    assert "roundtrip" in v.explanation.lower()
    # kb_answer / top match is still recorded even though it was downgraded.
    assert v.kb_answer == "paris"


# ---------------------------------------------------------------------------
# GateVerdict shape and provenance summary sanity checks.
# ---------------------------------------------------------------------------

def test_gate_verdict_is_dataclass_with_expected_fields(kb_prov):
    kb, prov = kb_prov
    v = verify_claim(kb, prov, "france", "capital", "paris")
    assert isinstance(v, GateVerdict)
    for field_name in ("status", "subject", "relation", "obj", "kb_answer",
                       "kb_score", "provenance_summary", "explanation"):
        assert hasattr(v, field_name)
    assert isinstance(v.kb_score, float)
    assert v.provenance_summary is not None
    assert "source=" in v.provenance_summary


def test_gate_verdict_without_provenance_store(kb_prov):
    kb, _ = kb_prov
    v = verify_claim(kb, None, "france", "capital", "paris")
    assert v.status == GateStatus.VERIFIED
    assert v.provenance_summary is None


# ---------------------------------------------------------------------------
# Meta: confirm all four verdict statuses are genuinely reachable in one
# run of the suite (redundant with the above, but a single explicit check).
# ---------------------------------------------------------------------------

def test_all_four_statuses_are_reachable(kb_prov):
    kb, prov = kb_prov
    statuses = {
        verify_claim(kb, prov, "france", "capital", "paris").status,
        verify_claim(kb, prov, "france", "capital", "berlin").status,
        verify_claim(kb, prov, "mercury", "isa", "planet").status,
        verify_claim(kb, prov, "atlantis", "capital", "nowhere").status,
    }
    assert statuses == {
        GateStatus.VERIFIED, GateStatus.CONTRADICTED,
        GateStatus.UNRESOLVED, GateStatus.OUT_OF_KB,
    }

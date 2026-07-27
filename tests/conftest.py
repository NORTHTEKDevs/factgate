"""Shared fixtures: a small, purpose-built KB (~200 triples) covering
functional relations (capital, isa, color, lives_in), a non-functional
relation with an explicit negation (haspart), an ambiguous entity
(mercury), and a 2-hop chain (acme -> ceo -> lives_in).

No network access; entirely in-memory and deterministic (seed=0).
"""
from __future__ import annotations

import pytest

# rck is imported INSIDE the fixture, not at module scope: a module-level import made
# every test in the suite -- including the domain-gate tests, which never touch rck --
# uncollectable without a knowledge-base engine installed.

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

HASPART_FACTS = [
    ("car", "engine"), ("bicycle", "wheel"), ("airplane", "wing"),
]


@pytest.fixture(scope="session")
def kb_prov():
    from rck.knowledge_base import ShardedKnowledgeBase
    from rck.negative_facts import deny
    from rck.provenance import ProvenanceStore

    kb = ShardedKnowledgeBase(dim=4096, n_shards=64, seed=0)
    prov = ProvenanceStore()

    def store(s, r, o):
        kb.store({"S": s, "R": r, "O": o})
        prov.store(s, r, o, source="test_fixture")

    for country, capital in CAPITALS:
        store(country, "capital", capital)
    for subject, category in ISA_FACTS:
        store(subject, "isa", category)
    for subject, color in COLOR_FACTS:
        store(subject, "color", color)
    for subject, part in HASPART_FACTS:
        store(subject, "haspart", part)

    # Explicit negation: cars do NOT have wings (non-functional relation,
    # so this must be caught via rck.negative_facts, not FUNCTIONAL_RELATIONS).
    deny(kb, "car", "haspart", "wings", provenance=prov)

    # Ambiguous entity: mercury is both a planet and a chemical element.
    store("mercury", "isa", "planet")
    store("mercury", "isa", "element")

    # Alias / canonicalization support fact (object alias "america" -> "usa").
    store("person_x", "lives_in", "usa")

    # 2-hop chain: acme -> ceo -> alice -> lives_in -> paris.
    store("acme", "ceo", "alice")
    store("alice", "lives_in", "paris")

    # Filler facts to bring the fixture KB to ~200 curated triples, using
    # a functional relation (population) across synthetic entities so
    # they can't collide with the real facts above.
    for i in range(160):
        store(f"filler_entity_{i}", "population", f"filler_value_{i}")

    return kb, prov

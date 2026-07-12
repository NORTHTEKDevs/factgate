"""Build / load the production RCK knowledge base for FactGate.

Fast path: reload a saved `rck.session` snapshot if one exists.
Slow path: bulk-ingest the source JSONL into a fresh `ConsciousAgent`
and snapshot it for next time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rck.bulk_ingest import bulk_load_jsonl
from rck.conscious_agent import ConsciousAgent
from rck.session import load_session, save_session

DEFAULT_JSONL = Path(
    "C:/Users/Krist/projects/active/rck/data/conceptnet_scale_100k.jsonl"
)
DEFAULT_SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "kb_snapshot"


def load_kb(jsonl_path: str | Path = DEFAULT_JSONL,
            snapshot_dir: str | Path = DEFAULT_SNAPSHOT,
            *, dim: int = 4096, n_shards: int = 64, seed: int = 0,
            force_rebuild: bool = False, symmetrize: bool = True,
            max_facts: int | None = None) -> ConsciousAgent:
    """Load a ConsciousAgent-backed KB.

    Fast path: if `snapshot_dir` already holds a saved session and
    `force_rebuild` is False, reload it (should complete in a few
    seconds regardless of KB size).

    Slow path: bulk-ingest `jsonl_path` from scratch into a new
    `ConsciousAgent`, then persist a snapshot to `snapshot_dir` so the
    next call hits the fast path.
    """
    snapshot_dir = Path(snapshot_dir)
    jsonl_path = Path(jsonl_path)

    if not force_rebuild and (snapshot_dir / "meta.json").exists():
        return load_session(snapshot_dir)

    agent = ConsciousAgent(dim=dim, n_shards=n_shards, seed=seed)
    bulk_load_jsonl(agent.knowledge, jsonl_path,
                     symmetrize=symmetrize, max_facts=max_facts)
    save_session(agent, snapshot_dir)
    return agent


_KB_SINGLETON: Optional[ConsciousAgent] = None


def get_kb(**kwargs) -> ConsciousAgent:
    """Process-wide singleton accessor.

    The first call builds/loads the KB per `load_kb` semantics; every
    later call returns the cached instance regardless of kwargs. Use
    `reset_kb_singleton()` (mainly for tests) to force a fresh load.
    """
    global _KB_SINGLETON
    if _KB_SINGLETON is None:
        _KB_SINGLETON = load_kb(**kwargs)
    return _KB_SINGLETON


def reset_kb_singleton() -> None:
    """Test helper: clear the cached singleton so the next `get_kb()`
    call rebuilds/reloads from scratch."""
    global _KB_SINGLETON
    _KB_SINGLETON = None

"""RAGTruth loader and span -> sentence labelling.

RAGTruth (github.com/ParticleMedia/RAGTruth, MIT) ships 17,790 model responses over
2,965 source instances with 14,289 word-level hallucination spans. It is used here
because both classes are present (43% of responses carry >=1 span) and the labels are
character offsets, so faithful and hallucinated sentences are mechanically separable --
which is exactly what the ConceptNet substrate could not do.

Data is NOT vendored (36MB). Fetch with scripts/fetch_ragtruth.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "ragtruth"

_TERMINATOR_RE = re.compile(r"[.!?]+")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets of each sentence, whitespace excluded.

    Offsets are preserved (not re-derived from the substring) because the gold labels
    are character spans into the original text; re-deriving would drift.

    A terminator only ends a sentence when it is followed by whitespace or end-of-text,
    and a '.' between two digits never does. RAGTruth QA responses are full of money
    amounts ("$23.70", "$38,900.50"); splitting those fragments gold spans across
    pseudo-sentences and inflates the positive class.
    """
    if not text.strip():
        return []
    n, start, raw = len(text), 0, []
    for m in _TERMINATOR_RE.finditer(text):
        s, e = m.start(), m.end()
        if text[s] == "." and s > 0 and text[s - 1].isdigit() and e < n and text[e].isdigit():
            continue
        if e < n and not text[e].isspace():
            continue
        raw.append((start, e))
        start = e
    if start < n:
        raw.append((start, n))

    out: list[tuple[int, int]] = []
    for s, e in raw:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def label_sentences(text: str, labels: list[dict]) -> list[tuple[str, bool]]:
    """Label each sentence hallucinated if any gold span overlaps it.

    Half-open interval overlap: a span ending exactly at a sentence boundary belongs to
    the preceding sentence only. Getting this wrong silently inflates the positive class.
    """
    spans = [(int(l["start"]), int(l["end"])) for l in labels or []]
    out = []
    for ss, se in split_sentences(text):
        hit = any(ls < se and le > ss for ls, le in spans)
        out.append((text[ss:se], hit))
    return out


def load_examples(task_type: str = "QA", split: str = "test",
                  limit: int | None = None) -> list[dict]:
    """Join responses to their source documents.

    Returns dicts with: id, source_id, model, source_text, prompt, response, labels.
    """
    sources = {}
    with open(DATA / "source_info.jsonl", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["task_type"] == task_type:
                sources[s["source_id"]] = s

    out = []
    with open(DATA / "response.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != split or r["source_id"] not in sources:
                continue
            s = sources[r["source_id"]]
            info = s["source_info"]
            out.append({
                "id": r["id"], "source_id": r["source_id"], "model": r["model"],
                "source_text": info if isinstance(info, str) else json.dumps(info),
                "prompt": s.get("prompt", ""), "response": r["response"],
                "labels": r.get("labels") or [],
            })
            if limit and len(out) >= limit:
                break
    return out

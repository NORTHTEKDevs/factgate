"""Shared bench utilities: Wilson CI, canonicalization, citation/IDK
parsing, JSONL I/O, and a stability-poll helper for a growing JSONL file.

Ported PATTERN from raincg/bench/common.py (exact-match + Wilson CI
scaffolding) -- reimplemented locally, adapted to FactGate's [kb:s/r/o]
citation-tag / IDK-marker domain. No rain code is imported.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Wilson score interval.
# ---------------------------------------------------------------------------


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. n<=0 -> (0.0, 1.0)."""
    if n <= 0:
        return (0.0, 1.0)
    phat = correct / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2 * n)
    adj = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


# ---------------------------------------------------------------------------
# JSONL I/O.
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def count_lines(path: Path) -> int:
    """Non-blank line count. Tolerant of a torn/partial last line (a file
    that's mid-write) -- counts only lines that end in a real newline plus
    whatever partial content precedes EOF, matching a plain `wc -l`-ish
    reading rather than requiring valid JSON."""
    if not Path(path).exists():
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def wait_for_stable_lines(path: Path, *, poll_interval: float = 5.0,
                           stable_window: float = 60.0,
                           max_wait: float | None = 900.0,
                           sleep_fn=time.sleep, now_fn=time.monotonic) -> dict:
    """Poll `path`'s non-blank line count until it hasn't changed for at
    least `stable_window` seconds, or `max_wait` total seconds elapse.

    Returns {"lines": n, "stable": bool, "waited_s": float}. `stable` is
    False if `max_wait` was hit while the file was still changing -- an
    honest partial-snapshot signal, never silently reported as complete.
    """
    start = now_fn()
    last_count = count_lines(path)
    last_change = start
    while True:
        now = now_fn()
        if now - last_change >= stable_window:
            return {"lines": last_count, "stable": True, "waited_s": now - start}
        if max_wait is not None and now - start >= max_wait:
            return {"lines": last_count, "stable": False, "waited_s": now - start}
        sleep_fn(poll_interval)
        n = count_lines(path)
        if n != last_count:
            last_count = n
            last_change = now_fn()


# ---------------------------------------------------------------------------
# Canonicalization + citation-tag / IDK-marker parsing.
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"[\s_]+")


def canonicalize(text: str | None) -> str:
    """Lowercase, fold underscores/whitespace, strip punctuation, so
    'bound_variable' == 'bound variable' == 'Bound-Variable.'"""
    if not text:
        return ""
    t = _PUNCT_RE.sub(" ", text)
    t = _WS_RE.sub(" ", t)
    return t.strip().lower()


_KB_CITE_RE = re.compile(r"\[kb:([^/\]]+)/([^/\]]+)/([^\]]+)\]")
KB_IDK_TAG = "[kb:idk]"

IDK_TEXT_MARKERS = (
    "i don't know", "i do not know", "cannot verify", "can't verify",
    "unable to verify", "no confident answer", "not confident enough",
    "insufficient information", "i'm not sure", "i am not sure",
)


def extract_kb_citations(text: str | None) -> list[tuple[str, str, str]]:
    """Every [kb:s/r/o] citation tag in `text` (the special [kb:idk] tag
    has no slashes and never matches this 3-group pattern)."""
    return [(m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
            for m in _KB_CITE_RE.finditer(text or "")]


def has_idk_tag(text: str | None) -> bool:
    return KB_IDK_TAG in (text or "").lower()


def is_abstention(text: str | None) -> bool:
    """True if `text` abstains: empty content, the explicit [kb:idk] tag,
    or an IDK text-marker phrase. Anything else counts as a (possibly
    hallucinated) factual assertion."""
    if not text or not text.strip():
        return True
    low = text.lower()
    if KB_IDK_TAG in low:
        return True
    return any(marker in low for marker in IDK_TEXT_MARKERS)


def is_correct_knowable(response: str | None, expected_o: str | None) -> bool:
    """knowable_heldout scoring: canonicalized substring match of the
    expected object in the free text, OR a [kb:s/r/o] citation whose
    object canonicalizes to the expected object."""
    canon_expected = canonicalize(expected_o)
    if not canon_expected:
        return False
    if canon_expected in canonicalize(response):
        return True
    return any(canonicalize(o) == canon_expected
                for _s, _r, o in extract_kb_citations(response))


_NEGATION_MARKERS = (
    "no,", "no.", "not correct", "incorrect", "that's wrong",
    "actually,", "false.", "false,",
)


def is_correction(response: str | None, kb_answer: str | None) -> bool:
    """adversarial_nearmiss scoring: response both (a) negates/contradicts
    the false premise, and (b) cites the KB's real answer (by text or
    citation tag)."""
    low = (response or "").lower()
    canon_kb = canonicalize(kb_answer)
    if not canon_kb:
        return False
    has_negation = any(m in low for m in _NEGATION_MARKERS)
    cites_kb_answer = canon_kb in canonicalize(response) or any(
        canonicalize(o) == canon_kb for _s, _r, o in extract_kb_citations(response))
    return has_negation and cites_kb_answer

"""Shared tool-trace conversation protocol used by qa.py, chains.py,
idk.py and corrupt.py.

Wire format (both a "tool-trace" and a "plain" view are emitted per
example so downstream training can pick either):

    system:    protocol description (fixed, see SYSTEM_PROMPT)
    user:      the question / premise
    assistant: <kb_q>{...}</kb_q>                  (one query, or a list
                                                      of hop queries for chains)
    tool:      <kb_r>{...}</kb_r>                  (ranked candidates)
    assistant: cited natural-language answer, tagged [kb:s/r/o] per
               fact used, or [kb:idk] when the answer is a refusal.
"""
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a fact-grounded assistant backed by a relational "
    "knowledge-base (KB) tool. To answer a question:\n"
    "1. Emit a query as <kb_q>{...}</kb_q> with fields among "
    "{s, r, o, unknown, top_k} -- `unknown` is the role you need "
    "(\"S\" or \"O\"); for multi-hop questions emit a JSON list of hop "
    "queries.\n"
    "2. You will receive <kb_r>{...}</kb_r> with ranked "
    "[symbol, score] candidates per query (or per hop).\n"
    "3. Answer using ONLY what the tool returned. Cite every fact you "
    "rely on as [kb:subject/relation/object]. If no candidate clears "
    "the confidence threshold, say you don't know and cite [kb:idk]. "
    "Never state a fact that the tool result does not support."
)


def kb_q_block(query: Any) -> str:
    return f"<kb_q>{json.dumps(query, ensure_ascii=False)}</kb_q>"


def kb_r_block(result: Any) -> str:
    return f"<kb_r>{json.dumps(result, ensure_ascii=False)}</kb_r>"


def kb_cite(s: str, r: str, o: str) -> str:
    return f"[kb:{s}/{r}/{o}]"


KB_IDK_CITE = "[kb:idk]"


def build_messages(question: str, kb_q: Any, kb_r: Any,
                    final_answer: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": kb_q_block(kb_q)},
        {"role": "tool", "content": kb_r_block(kb_r)},
        {"role": "assistant", "content": final_answer},
    ]


def make_example(example_id: str, generator: str, question: str,
                  kb_q: Any, kb_r: Any, final_answer: str,
                  meta: dict | None = None) -> dict:
    return {
        "id": example_id,
        "generator": generator,
        "messages": build_messages(question, kb_q, kb_r, final_answer),
        "plain": {"question": question, "answer": final_answer},
        "meta": meta or {},
    }

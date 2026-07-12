"""SFT record schema + validation CLI.

    python -m factgate.datagen.schema validate path/to/file.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message content must be non-empty")
        return v


class PlainQA(BaseModel):
    """Flat question/answer view, for consumers that don't want the
    tool-trace conversation format."""
    question: str
    answer: str


class SFTExample(BaseModel):
    """One training example produced by any datagen generator."""
    id: str
    generator: Literal["qa", "chains", "idk", "corrupt"]
    messages: list[Message] = Field(min_length=2)
    plain: PlainQA
    meta: dict = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def _has_user_and_assistant(cls, v: list[Message]) -> list[Message]:
        roles = {m.role for m in v}
        if "user" not in roles or "assistant" not in roles:
            raise ValueError(
                "messages must contain at least one user and one "
                "assistant turn")
        return v


def validate_file(path: Path) -> tuple[int, int, list[str]]:
    """Returns (n_valid, n_total, error_strings)."""
    n_valid = 0
    n_total = 0
    errors: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
                SFTExample.model_validate(obj)
                n_valid += 1
            except Exception as e:  # noqa: BLE001 - report, don't crash
                errors.append(f"line {lineno}: {e}")
    return n_valid, n_total, errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["validate"])
    p.add_argument("path", type=Path)
    p.add_argument("--max-errors", type=int, default=10)
    args = p.parse_args(argv)

    n_valid, n_total, errors = validate_file(args.path)
    print(f"{n_valid}/{n_total} valid")
    for e in errors[: args.max_errors]:
        print(" ", e)
    if len(errors) > args.max_errors:
        print(f"  ... and {len(errors) - args.max_errors} more")
    return 0 if n_valid == n_total and n_total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

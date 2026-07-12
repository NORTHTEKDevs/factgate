"""Natural-language templates for the top-10 relations in
conceptnet_scale_100k.jsonl.

Two template families:

* ``STATEMENT_TEMPLATES`` -- declarative sentences that mention both
  ``{s}`` and ``{o}``. Used by ``render_fact`` / ``extract_triples`` for
  the A5 render<->extract round trip (templates.py owns both directions
  of that map by construction: every statement template is compiled to
  a regex that recovers ``(s, r, o)`` exactly).
* ``QUESTION_TEMPLATES`` -- one-sided questions, either "forward"
  (mentions ``{s}`` only, asks for O) or "reverse" (mentions ``{o}``
  only, asks for S). Used by qa.py / chains.py / idk.py to phrase
  questions.

Relation coverage: isa, has_context, form_of, locatedin, requires,
has_subevent, can, causes, derived_from, antonym (the top 10 by
frequency in the source KB, verified via a one-off histogram pass).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TOP_RELATIONS: tuple[str, ...] = (
    "isa", "has_context", "form_of", "locatedin", "requires",
    "has_subevent", "can", "causes", "derived_from", "antonym",
)

# ---------------------------------------------------------------------------
# Statement templates (declarative, both S and O present).
# ---------------------------------------------------------------------------

STATEMENT_TEMPLATES: dict[str, list[str]] = {
    "isa": [
        "{s} is a type of {o}.",
        "{s} is a kind of {o}.",
        "{o} is a broader category that includes {s}.",
        "Every {s} is a {o}.",
        "{s} belongs to the category {o}.",
    ],
    "has_context": [
        "{s} is used in the context of {o}.",
        "The term {s} comes up in {o}.",
        "{o} is a context where you would encounter {s}.",
        "{s} is associated with the field of {o}.",
        "You would hear about {s} when discussing {o}.",
    ],
    "form_of": [
        "{s} is a form of {o}.",
        "{s} is a grammatical variant of {o}.",
        "{o} has the related form {s}.",
        "{s} derives its form from {o}.",
        "{s} is an inflected form of {o}.",
    ],
    "locatedin": [
        "{s} is located in {o}.",
        "{s} can be found in {o}.",
        "{o} is the place where {s} is located.",
        "You can find {s} in {o}.",
        "{s} is situated within {o}.",
    ],
    "requires": [
        "{s} requires {o}.",
        "{s} needs {o} in order to happen.",
        "{o} is a prerequisite for {s}.",
        "Without {o}, {s} cannot occur.",
        "{s} depends on {o}.",
    ],
    "has_subevent": [
        "{s} has the subevent {o}.",
        "{o} happens as part of {s}.",
        "During {s}, {o} takes place.",
        "One step within {s} is {o}.",
        "{s} includes {o} as a subevent.",
    ],
    "can": [
        "{s} can {o}.",
        "{s} is capable of {o}.",
        "{o} is something {s} can do.",
        "{s} has the ability to {o}.",
        "It is possible for {s} to {o}.",
    ],
    "causes": [
        "{s} causes {o}.",
        "{s} leads to {o}.",
        "{o} is a result of {s}.",
        "{s} brings about {o}.",
        "{o} happens because of {s}.",
    ],
    "derived_from": [
        "{s} is derived from {o}.",
        "{s} comes from {o}.",
        "{o} is the origin of {s}.",
        "{s} originates from {o}.",
        "{s} traces back to {o}.",
    ],
    "antonym": [
        "{s} is the opposite of {o}.",
        "{s} and {o} are antonyms.",
        "{o} is the antonym of {s}.",
        "{s} means the opposite of {o}.",
        "Where {s} means one thing, {o} means its opposite.",
    ],
}

# ---------------------------------------------------------------------------
# Question templates (one-sided: forward asks for O, reverse asks for S).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionTemplate:
    direction: str  # "forward" (know S, ask O) or "reverse" (know O, ask S)
    text: str       # contains exactly one of {s} / {o}


QUESTION_TEMPLATES: dict[str, list[QuestionTemplate]] = {
    "isa": [
        QuestionTemplate("forward", "What is {s} a type of?"),
        QuestionTemplate("forward", "What category does {s} belong to?"),
        QuestionTemplate("reverse", "Name something that is a type of {o}."),
        QuestionTemplate("reverse", "What is an example of a {o}?"),
    ],
    "has_context": [
        QuestionTemplate("forward", "In what context is {s} used?"),
        QuestionTemplate("forward", "What field is {s} associated with?"),
        QuestionTemplate("reverse", "What term is associated with {o}?"),
    ],
    "form_of": [
        QuestionTemplate("forward", "What is {s} a form of?"),
        QuestionTemplate("reverse", "What is a related form of {o}?"),
    ],
    "locatedin": [
        QuestionTemplate("forward", "Where is {s} located?"),
        QuestionTemplate("forward", "In what place can {s} be found?"),
        QuestionTemplate("reverse", "What is located in {o}?"),
    ],
    "requires": [
        QuestionTemplate("forward", "What does {s} require?"),
        QuestionTemplate("forward", "What is a prerequisite for {s}?"),
        QuestionTemplate("reverse", "What requires {o}?"),
    ],
    "has_subevent": [
        QuestionTemplate("forward", "What is a subevent of {s}?"),
        QuestionTemplate("forward", "What happens during {s}?"),
        QuestionTemplate("reverse", "What event has {o} as a subevent?"),
    ],
    "can": [
        QuestionTemplate("forward", "What can {s} do?"),
        QuestionTemplate("forward", "What is {s} capable of?"),
        QuestionTemplate("reverse", "What is capable of {o}?"),
    ],
    "causes": [
        QuestionTemplate("forward", "What does {s} cause?"),
        QuestionTemplate("forward", "What is a result of {s}?"),
        QuestionTemplate("reverse", "What causes {o}?"),
    ],
    "derived_from": [
        QuestionTemplate("forward", "What is {s} derived from?"),
        QuestionTemplate("forward", "What is the origin of {s}?"),
        QuestionTemplate("reverse", "What is derived from {o}?"),
    ],
    "antonym": [
        QuestionTemplate("forward", "What is the opposite of {s}?"),
        QuestionTemplate("forward", "What is the antonym of {s}?"),
        QuestionTemplate("reverse", "What is the opposite of {o}?"),
    ],
}


# ---------------------------------------------------------------------------
# render_fact / extract_triples -- the A5 inverse map.
# ---------------------------------------------------------------------------

_ENTITY_RE = r"(.+?)"


def _compile_template(pattern: str) -> re.Pattern:
    """Compile a `{s}`/`{o}` statement template into an extraction regex.

    Literal segments are escaped; `{s}` and `{o}` become named,
    non-greedy capture groups. Anchored start/end + case-insensitive so
    a rendered sentence maps back to exactly one (s, o) pair.
    """
    parts = re.split(r"(\{s\}|\{o\})", pattern)
    regex_parts = []
    for part in parts:
        if part == "{s}":
            regex_parts.append(r"(?P<s>.+?)")
        elif part == "{o}":
            regex_parts.append(r"(?P<o>.+?)")
        else:
            regex_parts.append(re.escape(part))
    return re.compile("^" + "".join(regex_parts) + "$", re.IGNORECASE)


# Precompiled (relation, template_id) -> regex, built once at import time.
_COMPILED: dict[tuple[str, int], re.Pattern] = {}
for _rel, _tmpls in STATEMENT_TEMPLATES.items():
    for _tid, _pat in enumerate(_tmpls):
        _COMPILED[(_rel, _tid)] = _compile_template(_pat)


def render_fact(s: str, r: str, o: str, template_id: int) -> str:
    """Render a (s, r, o) triple as a declarative sentence."""
    templates = STATEMENT_TEMPLATES.get(r)
    if templates is None:
        raise KeyError(f"no statement templates for relation {r!r}")
    if not (0 <= template_id < len(templates)):
        raise IndexError(
            f"template_id {template_id} out of range for relation {r!r} "
            f"(have {len(templates)})")
    return templates[template_id].format(s=s, o=o)


def extract_triples(text: str, relation_hint: str | None = None
                     ) -> tuple[str, str, str] | None:
    """Inverse of render_fact: recover (s, r, o) from a rendered sentence.

    If `relation_hint` is given, only that relation's templates are
    tried first (cheap disambiguation); otherwise every relation is
    tried in TOP_RELATIONS order and the first match wins.
    """
    relations = ([relation_hint] if relation_hint else []) + [
        r for r in TOP_RELATIONS if r != relation_hint]
    for rel in relations:
        templates = STATEMENT_TEMPLATES.get(rel)
        if templates is None:
            continue
        for tid in range(len(templates)):
            m = _COMPILED[(rel, tid)].match(text)
            if m:
                return (m.group("s").strip(), rel, m.group("o").strip())
    return None

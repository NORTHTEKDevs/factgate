"""Free-text claim extraction: prose -> (subject, relation, object) triples.

This is the step the original FACTGATE lacked. `factgate.datagen.templates.extract_triples`
is a regex inverse of the render templates and only matches template-shaped sentences,
so it cannot be used on model prose (measured: it silently mis-extracts, capturing
"Yes, dog" as a subject because {s}/{o} compile to `.+?`).

Extraction uses a local model, so it has learned parameters and can err. The gate's
VERDICT does not -- that separation is the property worth defending, and it is why
extraction failures must be routed to a HELD state rather than trusted (see policy.py).
"""
from __future__ import annotations

import json
import re

from factgate.llm import OLLAMA_URL, ollama  # noqa: F401  (re-export)

RELATIONS = ["isa", "has_context", "form_of", "locatedin", "requires",
             "has_subevent", "can", "causes", "derived_from", "antonym"]

EXTRACT_PROMPT = """Extract every factual claim in the passage as a knowledge triple.
Allowed relations (use one of these exact strings for "r"): {rels}

Return ONLY a JSON array of OBJECTS. Each object has exactly the keys s, r, o.

Example passage: "A robin is a kind of bird. Robins can fly."
Example output: [{{"s": "robin", "r": "isa", "o": "bird"}}, {{"s": "robin", "r": "can", "o": "fly"}}]

If the passage states no factual claim using an allowed relation, return [].

Passage: {text}

JSON:"""

# Recovers s/r/o even when the surrounding structure is not valid JSON. Local models
# emit a malformed array-with-colons form (`["s": "x", "r": "y", "o": "z"]`) often
# enough that a strict parser would attribute their capability loss to the parser:
# measured 11% strict-JSON failure on llama3.2:3b, 0% on qwen2.5:14b.
_TRIPLE_RE = re.compile(
    r'"s"\s*:\s*"([^"]*)"\s*,\s*"r"\s*:\s*"([^"]*)"\s*,\s*"o"\s*:\s*"([^"]*)"', re.S)


def parse_triples(raw: str) -> tuple[list[tuple[str, str, str]], bool]:
    """Parse model output into triples.

    Returns (triples, strict_json_ok). `strict_json_ok` is reported separately so a
    parse failure is never silently scored as an extraction failure.
    """
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, list):
            return ([(str(d["s"]), str(d["r"]), str(d["o"])) for d in data
                     if isinstance(d, dict) and {"s", "r", "o"} <= set(d)], True)
    return [(a, b, c) for a, b, c in _TRIPLE_RE.findall(raw)], False




# RAGTruth content (pay rates, procedures, events) is not expressible in ConceptNet's
# 10 commonsense relations -- a measured pilot using the closed vocabulary extracted only
# ~6 triples from a 1300-char source and held 100% of faithful sentences. Source and
# response are extracted by the same model with the same prompt, so an OPEN relation
# vocabulary keeps the two sides phrased consistently enough to match.
EXTRACT_PROMPT_OPEN = """Extract every factual claim in the passage as a knowledge triple.

Return ONLY a JSON array of OBJECTS. Each object has exactly the keys s, r, o:
  "s" = the subject, "r" = a short lowercase relation with underscores, "o" = the object.

Use short, reusable relation names (e.g. "average_pay", "located_in", "is_a",
"published_in"). Prefer the simplest relation that captures the claim. Extract EVERY
distinct factual claim, including numbers, dates, amounts and names.

Example passage: "Alaska technicians average $23.70 per hour. The study ran in 2021."
Example output: [{{"s": "alaska technicians", "r": "average_pay", "o": "$23.70 per hour"}}, {{"s": "the study", "r": "ran_in", "o": "2021"}}]

If the passage states no factual claim, return [].

Passage: {text}

JSON:"""


def extract(text: str, model: str, *, open_vocab: bool = True,
            **kw) -> tuple[list[tuple[str, str, str]], bool]:
    prompt = (EXTRACT_PROMPT_OPEN.format(text=text) if open_vocab
              else EXTRACT_PROMPT.format(rels=", ".join(RELATIONS), text=text))
    return parse_triples(ollama(model, prompt, **kw))

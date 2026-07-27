"""Minimal local-model transport, shared by the domain gate and the research modules.

Kept as its own module so `factgate.domain` (the supported product) does not import from
`factgate.hallugate` (the abandoned free-text research path, see docs/HALLUGATE.md).
Depending on research code you intend to delete is how a library acquires a dependency
nobody can remove.

Deliberately stdlib-only: no `requests`, so the package has zero runtime dependencies
beyond the stdlib for the gate path.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"


class ExtractionUnavailable(RuntimeError):
    """The local model could not be reached.

    Raised rather than returning "" so a transport failure can never be mistaken for the
    model reporting a value as absent, which would silently turn an outage into a stream
    of HELD verdicts that look like normal conservative behaviour.
    """


def ollama(model: str, prompt: str, num_predict: int = 400,
           url: str = OLLAMA_URL, timeout: int = 600,
           retries: int = 3, retry_delay: float = 2.0) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": num_predict}}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    # Retry transient failures a bounded number of times, then raise. A single timeout
    # killed an 8-minute benchmark run; swallowing the error instead would be worse,
    # since an outage would masquerade as normal conservative HELD behaviour.
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["response"]
        except (urllib.error.URLError, TimeoutError, KeyError,
                json.JSONDecodeError) as e:
            last = e
            if attempt + 1 < max(1, retries):
                time.sleep(retry_delay)
    raise ExtractionUnavailable(f"{type(last).__name__}: {last}") from last

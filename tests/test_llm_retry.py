"""Transport resilience.

A single transient timeout killed an 8-minute benchmark run outright. Failing loudly is
correct -- silently returning "" would turn an outage into a stream of HELD verdicts that
look like normal conservative behaviour -- but a transient blip should be retried before
it becomes a failure.
"""
import pytest

from factgate import llm


def test_transient_failure_is_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("timed out")
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"response": "ok"}'
        return R()

    monkeypatch.setattr(llm.urllib.request, "urlopen", flaky)
    assert llm.ollama("m", "p", retries=3, retry_delay=0) == "ok"
    assert calls["n"] == 3


def test_persistent_failure_still_raises(monkeypatch):
    """Retries must not turn a real outage into a silent success or a silent empty."""
    def dead(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(llm.urllib.request, "urlopen", dead)
    with pytest.raises(llm.ExtractionUnavailable):
        llm.ollama("m", "p", retries=2, retry_delay=0)


def test_retries_are_bounded(monkeypatch):
    calls = {"n": 0}

    def dead(req, timeout=None):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(llm.urllib.request, "urlopen", dead)
    with pytest.raises(llm.ExtractionUnavailable):
        llm.ollama("m", "p", retries=2, retry_delay=0)
    assert calls["n"] == 2

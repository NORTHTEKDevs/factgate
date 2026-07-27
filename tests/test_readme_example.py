"""Executes the README's quickstart verbatim.

Two separate defects in this repo were docs describing behaviour the code did not have
(a script whose docstring claimed an LLM it never called, and an install command that
failed outright). A README snippet that is not executed is an untested claim.
"""
from factgate.domain.factset import FactSet
from factgate.domain.gate import gate_claim

CORPUS = "Give acetaminophen 15 mg/kg PO every 4 to 6 hours."


def _factset():
    return FactSet.from_dict({
        "domain": "dosing",
        "entities": {"acetaminophen": ["tylenol", "paracetamol"]},
        "relations": {"pediatric_dose": {"kind": "quantity",
                                         "description": "amount per dose"}},
        "facts": [{"s": "acetaminophen", "r": "pediatric_dose", "o": "15 mg/kg",
                   "source": "Give acetaminophen 15 mg/kg PO every 4 to 6 hours."}],
    })


def test_readme_quickstart_runs_exactly_as_written():
    fs = _factset()
    assert fs.validate_sources(CORPUS)[1] == []
    assert gate_claim(fs, "Tylenol", "pediatric_dose", "15 mg/kg").status == "VERIFIED"
    assert gate_claim(fs, "Tylenol", "pediatric_dose", "20 mg/kg").status == "BLOCK"
    assert gate_claim(fs, "Tylenol", "pediatric_dose", "15 mg").status == "HELD"
    assert gate_claim(fs, "morphine", "pediatric_dose", "1 mg/kg").status == "HELD"


def test_verdict_path_needs_no_network_or_model():
    """The README claims the verdict path needs no model and no network. Enforce it:
    if gate_claim ever grows a model call, this fails."""
    import urllib.request

    def explode(*a, **k):                      # pragma: no cover - only runs on failure
        raise AssertionError("gate_claim performed network I/O")

    original, urllib.request.urlopen = urllib.request.urlopen, explode
    try:
        assert gate_claim(_factset(), "Tylenol", "pediatric_dose", "15 mg/kg").status \
            == "VERIFIED"
    finally:
        urllib.request.urlopen = original


def test_shipped_demo_domain_loads_and_every_fact_is_quoted():
    """The demo fact set is a published artifact; if it stops loading, the documented
    benchmark command breaks for anyone who clones the repo."""
    import json
    from pathlib import Path

    spec = json.loads((Path(__file__).resolve().parents[1]
                       / "data/domains/clinical_demo.json").read_text(encoding="utf-8"))
    fs = FactSet.from_dict(spec)
    ok, missing = fs.validate_sources(spec["corpus"])
    assert missing == []
    assert ok == len(fs.facts) == 12

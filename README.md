# FactGate

FactGate checks a claimed `(subject, relation, object)` triple against an
[RCK](https://github.com/) relational knowledge base and returns one of four
verdicts:

- **VERIFIED** -- the KB agrees and roundtrip self-verification corroborates it.
- **CONTRADICTED** -- the KB has a different, confident answer (functional
  relation) or an explicit negative fact denies the claim.
- **OUT_OF_KB** -- the KB genuinely doesn't know (IDK).
- **UNRESOLVED** -- ambiguous KB evidence, or a nominally-KNOWN answer that
  self-verification couldn't corroborate.

## Install

```bash
# from the factgate repo root
C:/Users/Krist/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv
./.venv/Scripts/python.exe -m pip install numpy pytest pydantic
./.venv/Scripts/python.exe -m pip install -e C:/Users/Krist/projects/active/rck
```

`rck-kernel` is not on PyPI; it is installed as an editable local package
from the sibling `rck` repo.

## Run tests

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
```

## Layout

- `factgate/gate.py` -- `verify_claim` / `verify_chain_claim` core logic.
- `factgate/kb_service.py` -- builds/loads the production KB (bulk-ingest
  with a fast session-snapshot reload path).
- `tests/test_gate.py` -- gate acceptance tests against a small
  purpose-built fixture KB (no network, <60s).

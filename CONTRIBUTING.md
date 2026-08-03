# Contributing

The bar for this repository is unusual and non-negotiable: **every safety claim is
machine-checked, and every check is verified capable of failing.**

## Before you open a PR

Run the loop:

    python scripts/harden.py

It runs the full suite, the proofs (an exact-arithmetic oracle over the value grammar,
construction-based verdict-path tests, a hostile-model extraction test), validates every
shipped domain, and then deliberately breaks the code twelve ways to confirm each proof
notices. **A surviving mutant fails the run** — it marks a defence nothing tests, which is
where the next defect lives. `--quick` skips mutation testing while iterating.

## If you fix a verdict bug

1. Reproduce it first; put the reproduction in the test, with the measured wrong verdict
   in the docstring. This repo's convention is that test docstrings record what actually
   happened, not what might.
2. Add the fix, then add a **mutant** to `scripts/harden.py` that disables your defence,
   and confirm the suite catches it.
3. If the bug class is expressible in the value grammar or author config, extend the
   oracle pools in `tests/test_value_grammar.py` so the config proof covers it.

## If you add a notation or capability

Every leak found in rounds 3–5 of this project's hardening was inside a capability added
the same week. New parsing surface is leak surface: gate it on the relation's declared
`kind`, add it to the oracle, and expect the reviewer to ask what a hostile author can do
with it.

## Adding an evaluation domain

Write the document first, then declare the vocabulary from it — see `docs/AUTHORING.md`.
Domains are **blind**: do not tune a vocabulary against gate behaviour, and say in the PR
whether you read the gate source before authoring. Sources must be quoted verbatim
(`validate_sources`) and `lint()` must be clean.

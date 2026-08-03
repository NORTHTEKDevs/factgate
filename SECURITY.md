# Security policy

FACTGATE is a safety component: its whole purpose is that a wrong value never reaches a
user as VERIFIED. We therefore treat a **leak** (a false VERIFIED) or a **false BLOCK**
(a correct value reported as contradicting its document) with the severity most projects
reserve for remote-code-execution reports.

## Reporting

Email **info@northtek.io** with:

- a minimal fact set (JSON) and the claim that produces the wrong verdict
- the verdict you observed and the one that is correct, with the document text that
  proves it

You will get an acknowledgement within 72 hours. Please do not open a public issue for a
leak until it is fixed — a reproducible false-VERIFIED is exploitable by anyone feeding
documents to a deployment.

## What happens to a confirmed report

Every confirmed defect in this project's history is fixed, regression-tested, **and added
to the machine-checked proof suite** (`scripts/harden.py`) so it cannot silently return —
including a mutant that disables the fix and asserts the tests notice. Yours will be too,
and the fix commit will credit you unless you ask otherwise.

## Scope

In scope: anything producing a wrong VERIFIED or wrong BLOCK verdict, including through
author-supplied configuration (`unit_aliases`, `value_qualifiers`), unicode, or crafted
documents; denial of service through a fact set or a claim (both have been found and fixed
before). Out of scope: the correctness of your source document — the gate verifies claims
against the document, not the document against the world.

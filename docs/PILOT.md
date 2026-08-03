# Supervised pilot: one document, every verdict reviewed

Every other measurement in this project is a rate. This is a pilot: the gate run **as if
deployed** on one document, with a human confirming every verdict against that document.
The question it answers is the one a deployment actually asks — *can a reviewer trust this
enough to stop reading every answer, and what does it cost them when they can't* — not the
leak rate.

Reproduce: `python scripts/pilot.py --domain data/domains/pilot_cold_chain.json`

## The document

A realistic cold-chain vaccine storage summary
([`data/domains/pilot_cold_chain.json`](../data/domains/pilot_cold_chain.json)) — the messy,
safety-critical kind a hallucination gate is for: temperature ranges in Celsius **and**
Fahrenheit, sub-zero and ultra-cold storage, cumulative excursion allowances, conditional
in-use windows, a do-not-freeze rule. Thirteen declared facts, every source quoted verbatim
from the corpus, lint clean. The vocabulary was authored **from** the document, not tuned
against the gate.

Ground truth for a pilot is the document itself — exactly as it is for a real reviewer,
whose job is *"does the gate's verdict match what this sheet says"*, not *"is the sheet true
about the world"*. `validate_sources` guarantees every declared value is quoted from the
corpus, so the corpus and the fact set cannot disagree.

## What happened

Twenty-six natural questions (two framings per fact, none naming the relation as a field —
*"A colleague asked me about the storage temperature for Product B. What should I tell
them?"*). A local model (`qwen2.5:14b`) answered each in prose; the extractor linked it; the
gate adjudicated. Twenty-five produced an answer.

```
questions the model answered            25
VERIFIED  (gate settled it)             16
HELD/BLOCK (handed to the reviewer)      9   (9 held, 0 blocked)
REVIEWER LOAD REDUCTION                 64%   of answers needed no review
TRUST BREACHES (VERIFIED but wrong)      0    -- clean
```

## The supervision

I read all sixteen VERIFIED verdicts against the document by hand. **Every one is correct** —
the confirmed value is exactly what the sheet states (`2 to 8 degrees Celsius`, `28 days`,
`must never be frozen`, `-90 to -60 degrees Celsius`, `4 hours`, and so on). No wrong value
reached the "user" as confirmed. The harness reports zero trust breaches; independent review
agrees, which is the point of reviewing rather than trusting the number.

I then read all nine HELD verdicts. **None is a false block** — nothing correct was reported
as contradicting the document — and none is a leak. They split into two honest kinds:

- **Genuine ambiguity the document created** (about five of the nine). The answer restated
  *two* values of the same kind, because the sheet states two: *"2 to 8 degrees Celsius
  (36 to 46 degrees Fahrenheit)"*, or Product B's frozen range and its refrigerated hold in
  one sentence. The gate cannot pick which one answers the question, and holding is correct —
  a gate that guessed here would eventually guess wrong. No gate should resolve these; a
  human should.

- **An undeclared qualifier** (the rest). *"below 60 percent"* against a declared
  *"60 percent"*, *"every 30 minutes"* against *"30 minutes"*. The value is right; the
  wording is one word the fact set had not declared harmless. The library flags exactly
  this:

  ```
  suggested value_qualifiers (review before declaring):
    paste into value_qualifiers:
      "below", "every"
  ```

  Declaring those two words — the reviewer's one-line edit — lifts both to VERIFIED, and a
  wrong value (`90 percent`) still BLOCKs afterward, so the qualifier bought coverage without
  costing safety. That is the run / read / declare / re-run loop the tooling is built around,
  closing on a real document.

## What the pilot shows, and what it does not

**It passed the only test that fails a deployment: zero wrong answers were confirmed.** On a
messy, safety-critical document the gate never saw before, with natural questions and a real
extractor in the loop, not one wrong value reached the user as VERIFIED — confirmed by
independent review, not asserted by the harness.

**It settled 64% of answers unaided on the first pass**, rising to roughly 72% after a
two-word vocabulary edit the tool itself proposed. The remaining holds are a review queue,
which is costly but never wrong — the fail-closed direction.

What it does **not** show: this is one document, one model, one reviewer. It is field
evidence, not a certification. It does not measure how a fact set fares when the author is a
non-expert under time pressure, nor how the numbers move across many documents in a real
workflow. Those are the next pilots. But the load-bearing claim — *a reviewer can trust a
VERIFIED verdict, and the rest is a bounded, safe queue* — held up the first time it was run
against a document as if it were live.

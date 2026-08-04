# FACTGATE

**A hallucination gate whose verdict is a comparison, not a judgement — so the verdict layer
has no learned parameters and cannot itself hallucinate.**

That property belongs to the VERDICT, and only to the verdict. Everything upstream of it —
deciding which claims exist at all — uses a language model, and a model can invent a value.
The system's answer to that is not a second guarantee but a set of deterministic guards, and
a proof that they hold: **no claim the extractor emits can fail its own guards, whatever the
model says**. Both halves are machine-checked; see [What is proven](#what-is-proven-and-by-what)
for the exact boundary.

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-816%20passing-brightgreen)

## The idea

A fluent model proposes; a deterministic gate disposes. The model does not decide whether
its own claim is true — a claim is extracted into a declared vocabulary, compared against a
curated fact set, and returned as VERIFIED, BLOCK or HELD. The comparison has no thresholds,
no scores and no model call, so a wrong verdict cannot be produced by the thing being
guarded against.

That is categorically different from RLHF-style alignment, which makes a model *less likely*
to hallucinate. Here an unverifiable claim is *structurally unable* to reach the user as a
confirmed fact, whatever the weights want to say.

**It is fail-closed.** Anything not provably matching is HELD and routed to a human. The
cost is real and measured below: on fifteen domains, 16% of correct values were held rather
than confirmed. The benefit is that the failure mode is a queue, not a wrong answer.

**It requires a bounded domain.** Free text does not work and the attempt is documented:
two independent extractions of one fact shared 0 of 17 relation strings, so exact
comparison was impossible. Entities, relations and facts must be declared per document.
`lint()`, `suggest_qualifiers` and `suggest_entity_aliases` exist to make that authoring
cheap, and the benchmark emits the review list itself.

The optional [RCK](https://github.com/NORTHTEKDevs/rck) knowledge-base backend is research
code from an earlier design. **The domain gate does not need it** — that is asserted by the
clean-clone acceptance check on every run.

## Measured results

Every number is read from a file in [`results/`](https://github.com/NORTHTEKDevs/factgate/blob/main/results/) and re-checked by
`python scripts/harden.py`. The authoritative measurements are three, each with its own
section below:

- **What is proven** — the machine-checked safety contract over the value grammar, author
  config, and the two other verdict paths, with a mutation gate confirming each proof can
  fail.
- **Does the gate see what the model says** — routing coverage: **99%** of values a model
  asserts in natural prose reach the gate, **0** wrong figures reach a user unguarded.
- **Measured behaviour** — the current per-domain leak and over-block rates, re-measured on every run:
  **0/366 leak, 33/209 = 16% over-block** across fifteen domains, eleven authored blind.

<details>
<summary>Earlier architecture (RCK structured-triple backend)</summary>

Before the standalone domain gate, the verdict was backed by [RCK](https://github.com/NORTHTEKDevs/rck),
a bipolar-VSA fact store, and measured on pre-parsed `(subject, relation, object)` triples:
**0.0%** false-VERIFIED across 3,000 adversarial trials (N=1500 absent + N=1500 corrupted,
`results/guarantee_measurement.json`), at **34%** verify coverage. That backend is now
optional research code — the clean-clone acceptance check asserts the domain gate does not
need it — and those triple-level numbers are superseded by the live-extraction measurements
above. Kept here because they are real and reproducible, not because they are the current
claim. Full RCK-era write-up: [`RESULTS.md`](https://github.com/NORTHTEKDevs/factgate/blob/main/RESULTS.md), [`FINAL-REPORT.md`](https://github.com/NORTHTEKDevs/factgate/blob/main/FINAL-REPORT.md).
</details>

### What is proven, and by what

Every claim below is re-checked by `python scripts/harden.py` in about 80 seconds, and by
CI on every push. It does not merely run the proofs -- it then **breaks the code twelve ways
and requires each proof to notice**. A surviving mutant fails the build, because it marks a
defence that nothing tests.

```
PROOFS
  full suite                                          816 passed
  value grammar vs exact-arithmetic oracle             13 passed
  residue and conditional verdict paths                79 passed
  safety invariants I1-I7                               9 passed
DOMAINS
  every shipped domain loads, validates, lints clean   43 loaded, 0 bad
MUTATION -- each proof must CATCH a broken defence    12 mutants, 12 caught

HARDEN PASSED: 19/19 checks in 148s
```

| | |
|---|---|
| **Proven** | Over the declared value grammar, every MATCH is confirmed **equal** and every DIFFER confirmed **unequal**, by exact rational arithmetic (`fractions.Fraction`) that shares no code with the implementation |
| **Proven** | No author declaration -- no `unit_aliases`, no `value_qualifiers` -- can make the gate verify an unequal value without `lint()` refusing the fact set first |
| **Proven** | The residue and conditional-variant paths behave as constructed cases require, where the correct verdict follows from how the case was assembled rather than from what the code says |
| **Proven** | Extraction never emits a claim its own guards reject, **whatever the model says** -- tested against a hostile scripted model that fabricates, translates, glues, refuses and copies decoys |
| **Proven** | Every check above catches the defence it protects |
| **NOT proven** | Anything about notations outside the declared grammar. Those are **held**, which is the fail-closed answer, not a verified one. |
| **Proven** | *Residue admission* — accepting a claim that adds wording the fact's own source supplies — is **fail-closed**. A residue must be positively recognised as a modifier phrase (a prepositional phrase, a frequency adverb, or a route code) to be admitted; anything unrecognised is held. This replaced a blacklist of negation words, which admitted whatever it had not been taught: `waived` and `contraindicated` were listed, `restricted` and `suspended` were not, and neither could the next one be. |

The NOT-proven row is the boundary, and stating it is deliberate: hold what you cannot
parse. A guarantee with a stated edge survives contact with a real document; one without an
edge does not.

An adversarial reviewer found the residue path by reading the code and called it the one
indefensible sentence in this claim, because admission depended on the ABSENCE of a
blacklisted word and so admitted whatever the list had not been taught. It is now an
allowlist: recognition is required, not merely a failure to object. The cost was one
verification across fifteen domains, which became a hold with a one-line suggestion
attached.

### Does the gate actually see what the model says?

Every other number here is conditional on a claim **reaching** the gate. A value the model
asserts in prose that the extractor never turns into a claim is protected by nothing.

`scripts/routing_coverage.py` measures it, and asks the hard question deliberately. The
domain benchmark asks the slot question directly — *"what is the pediatric dose of
ibuprofen?"* — the easiest possible case for extraction, and not what anyone types. This
asks four natural framings per fact, none naming the relation as a field:

> *"A colleague asked me about the pediatric dose for ibuprofen. What should I tell them?"*

so the value comes back buried in a paragraph. **Ten domains**, `qwen2.5:14b`:

| | |
|---|---|
| Answers that asserted a value | 151 |
| **Adjudicated — reached the gate** | **149/151 = 99%** CI95 [95%, 100%] |
| Bypassed — asserted, unguarded | 2/151 = 1% |
| **Unguarded AND wrong** | **0/151 = 0%** CI95 [0%, 2.5%] |
| Verified despite stating a wrong value | **0** |

**Unguarded and wrong** is the number that decides real-world risk: a figure differing from
the document, stated in prose, with nothing in its way. Ground truth is computed from the
fact set, not judged by a model, and the wrong-value test fires only on a number that
*provably* differs **under the declared unit** — a paraphrase, an omission, or a figure
belonging to another slot is never counted as an error.

**This number was 99% once before, on four domains — and that was a selection effect, not
a measurement.** Widening to ten dropped it to 85%. Fixing what the wider run exposed took
it 85% → 91% → 94% → **99%**, and this time the figure is earned across all ten. The arc is
recorded rather than replaced because the endpoint alone would be indistinguishable from
the selection effect it started as.

Five defects surfaced from running this, each fixed and each reflected in the table above:

- A model answering *"a four-hour observation period"* — the correct value, **spelled out**
  — was bypassed entirely, because the digit never appears and grounding failed.
- The dominant cause of bypass was the ambiguity guard firing on a passage discussing
  several values of the same kind (*"sodium reference range is 135 to 145 mmol/L; critically
  low is below 120"*). Refusing to guess is correct; doing it **silently** was not. Those
  slots are now reported and become **HELD**. That took 85% → 91%.
- A multi-word entity name split across its sentence never resolved: a tariff declaring
  *"Summer On-Peak Energy"* answered as *"During the Summer Season … the On-Peak Energy rate
  is 14.2 cents per kWh"* matched nothing, so no slot was even queried. Matching now also
  succeeds when every significant word appears in **one sentence** — scoped to a sentence,
  because attaching a claim to the wrong entity is the worst thing that function can do.
  That took 91% → 94%.
- The model's slot answer often contained the declared value **wrapped in prose it copied
  from its own sentence** — *"refrigerated between 2 and 8 degrees Celsius before initial
  use"*. The shape filter rightly refuses to distill that into a value; it then went
  *silent*. Those slots are now surfaced and become HELD. That took 94% → 99%.
- Twice, the harness itself scored a correct answer as unguarded-and-wrong by attributing
  another slot's figure to it. A measurement that flatters the danger is as useless as one
  that flatters the product.

The two remaining bypasses are one shape, documented rather than patched: the model
misspelling the entity name (*"Nornectra"* for *"Norvectra"*). Fuzzy matching would close
them and is refused on purpose — attaching a claim to a nearly-right name is the wrong-drug
failure, and one letter is exactly the distance between real drug names.

### Measured behaviour

Fifteen domains, `qwen2.5:14b`, eleven of them authored blind in genres the gate had never
been measured on. All reproducible from a clone:

| | |
|---|---|
| Leak (a wrong value reaching the user as VERIFIED) | **0 / 366 = 0%** |
| Over-block (a correct value failing to verify) | 33 / 209 = 16% |
| — of those, the gate itself being too strict | **7 of 209 = 3.3%** |

Live soak across eighteen domains: 335 claims adjudicated, every safety invariant holding on
every verdict a real model produced.

### A supervised pilot: run as if deployed, every verdict reviewed

The measurements above are rates. A pilot is the gate run **as if deployed** on one document,
with a human confirming every verdict against it — the question a deployment asks, not the
leak rate. Full report: [`docs/PILOT.md`](https://github.com/NORTHTEKDevs/factgate/blob/main/docs/PILOT.md).

A realistic cold-chain vaccine storage sheet (dual-unit temperature ranges, sub-zero storage,
excursion allowances, conditional in-use windows), 26 natural questions, a local model and
the real extractor in the loop:

**Four documents, four authors — three of them first-time users given only the public
authoring guide** (an HR leave policy, a food-safety SOP, a gym membership agreement, and a
cold-chain vaccine sheet):

| | |
|---|---|
| Supervised answers across four documents | 110 |
| **VERIFIED — every single one confirmed correct by hand** | **64** |
| Held, handed to the reviewer | 46 |
| Blocked | 0 |
| **Trust breaches — a wrong value confirmed** | **0** |
| Mean reviewer load reduction | **58%** (33–68% by document) |

All three first-time authors produced fact sets that were **clean on their first check run**
— the authoring cost outside the maintainer's hands was one iteration of the documented
loop. Every VERIFIED was read against its document by hand and matched; every HELD was a
legitimate fail-closed hold — genuine ambiguity the document created, an undeclared
qualifier the tool names in one line, or a conditional slot correctly awaiting its context.
Conditional-heavy documents hold more, by design: the HR policy settled 33% while the SOP
settled 68%. **Not one wrong answer reached a user as confirmed, in 110 answers.** Field
evidence, not certification: synthetic documents, agent authors, one reviewer.

### How this was reached, and what it does not claim

Five rounds of adversarial review found defects one at a time, and by the fourth round most
new leaks were **inside the previous round's fixes** -- the search was not converging, and
each notation added to reduce over-block created leak surface faster than review closed it.
The proofs above replaced that process. They caught a false block on their first run, and
the mutation gate immediately exposed two defects in the oracle itself while it appeared to
pass. `docs/HALLUGATE.md` records every defect with its measured reproduction.

**The proofs did not make the code leak-proof, and this is stated plainly.** A later round
handed the code to reviewers with no history of what was fixed and asked them to attack from
first principles. They found two more: a temperature `unit_alias` (`{"F": "C"}`) that passed
lint and verified `100 F` against a declared `100 C`, and a hyphenated compound
(`board-certified`) reported as contradicting a declared `Board Certified`. A leak and a
false block -- the two categories that matter -- in code that had already survived five
rounds and a proof suite.

That is the honest boundary of what the method buys. It does **not** guarantee no undiscovered
defect exists; a fresh perspective still finds them. What it guarantees is that **every defect,
once found, cannot silently return** -- both of those are now in the value-grammar oracle and
the mutation set, caught on every commit. The claim is "provably correct over what has been
tested, and structurally unable to regress," not "provably correct over all inputs." The
second would be false, and a reviewer proved it.

**What is still not certified.** No deployment has run without a human reviewing the held
queue. The three private evaluation documents are not in this repository, so two of the
measured rows cannot be reproduced from a clone. And the boundary above is real: a document
using a notation outside the declared grammar will be held, not verified. No deployment has run without a human
reviewing the held queue, the two real documents cost a review pass each, and the private
evaluation corpora are not in this repository so those two rows cannot be reproduced from
a clone. What is verified is stated above and re-runnable for the four public domains.

New authors should start with [`docs/AUTHORING.md`](https://github.com/NORTHTEKDevs/factgate/blob/main/docs/AUTHORING.md).

Two harnesses verify the things a test suite cannot:

```bash
python scripts/acceptance.py    # fresh venv, install, exercise the documented surface
python scripts/soak.py          # live pipeline, safety invariants on every verdict
```

`acceptance.py` found that four test modules hard-required an optional dependency, so a
fresh clone got four collection errors and ran zero tests. A clean-clone run now reports
654 passed, 4 skipped -- fewer than the full suite's 816 because the minimal environment
runs only the tests that need no optional extras. `soak.py`'s first full campaign adjudicated **142 live verdicts across eight
domains with every safety invariant holding** -- the first check here that exercises the
configuration production would actually run.

### Using it

Declare your facts, then adjudicate claims against them. The verdict path needs no model
and no network:

```python
from factgate.domain.factset import FactSet
from factgate.domain.gate import gate_claim

fs = FactSet.from_dict({
    "domain": "dosing",
    "entities": {"acetaminophen": ["tylenol", "paracetamol"]},
    "relations": {"pediatric_dose": {"kind": "quantity",
                                     "description": "amount per dose"}},
    "facts": [{"s": "acetaminophen", "r": "pediatric_dose", "o": "15 mg/kg",
               "source": "Give acetaminophen 15 mg/kg PO every 4 to 6 hours."}],
})

# every declared fact must be traceable to a quote in your source corpus
assert fs.validate_sources(corpus)[1] == []          # no unquoted facts

gate_claim(fs, "Tylenol", "pediatric_dose", "15 mg/kg").status   # 'VERIFIED'
gate_claim(fs, "Tylenol", "pediatric_dose", "20 mg/kg").status   # 'BLOCK'
gate_claim(fs, "Tylenol", "pediatric_dose", "15 mg").status      # 'HELD'  (unit unproven)
gate_claim(fs, "morphine",  "pediatric_dose", "1 mg/kg").status  # 'HELD'  (out of domain)
```

To link claims out of free prose first (this step *does* call a local model, and is the
part that can err), use `factgate.domain.link.link_targeted`.

The schema also supports **conditional facts** (`when: {"indication":
"otitis media"}` against declared `conditions`) -- a slot with several conditional values
can never verify unless the condition is supplied, since confirming one variant blind
would confirm an overdose in the other case. `fs.lint()` reports provably unsafe qualifier
declarations, and every `Verdict` carries a `factset_fingerprint` so a decision can be
traced to the exact fact set that produced it. The verdict path runs at ~30,000
verdicts/sec/core with no model and no network.

```bash
python scripts/run_domain_bench.py --model llama3.2:3b
```

## How it works

The current domain gate is a slot-filling extractor plus a parameter-free comparator:
prose in, a claim in the declared vocabulary out, adjudicated against the fact set. The
diagram below is the **earlier RCK tool-call design** (`<kb_q>` protocol, `verify_claim`),
kept for readers of the RCK-era write-ups; the shape is the same, the extraction path
differs.

```
 user question
      |
      v
 +----------+       emits a KB lookup           +-----+
 |   LLM    | --------------------------------> | RCK |  exact fact store
 +----------+   <kb_q>{s, r, unknown}</kb_q>     +-----+
      |                                             |
      | cites the answer OR abstains                | VERIFIED / CONTRADICTED /
      v                                             | OUT_OF_KB / UNRESOLVED
 +----------------------------------------------+   |
 |         deterministic verification gate      |<--+
 |  (factgate.gate.verify_claim, pure function)  |
 +----------------------------------------------+
      |
      v
 only VERIFIED claims pass through unflagged;
 everything else is emitted as [unverified]
```

The gate resolves every claim to exactly one of four verdicts:

- **VERIFIED** — RCK agrees and roundtrip self-verification corroborates it.
- **CONTRADICTED** — RCK holds a different, confident answer for a functional relation,
  or an explicit negative fact denies the claim.
- **OUT_OF_KB** — RCK genuinely doesn't know (honest IDK).
- **UNRESOLVED** — ambiguous evidence, or a nominally-known answer that self-verification
  couldn't corroborate.

Only `VERIFIED` claims are allowed to reach the user unflagged. The gate itself
(`factgate/gate.py`) is a small, deterministic, pure function — it does not call the
model and cannot be argued with. RCK is public and named above; the model-serving layer
that wires the LLM's tool calls to the gate is referred to here generically as "the
`VerifiedBackend` serving gateway" — it is not part of this repo.

## The honest guarantee

**Formal claim:** for any claim in RCK's covered domain that RCK can refute or fails to
corroborate, the gate does not emit that claim as VERIFIED. This holds *by construction
of the emission path* — the gate sits between the model's output and the user,
independent of model weights, decoding temperature, sampling seed, or prompt injection.
A jailbroken or adversarially-prompted model can still try to assert a false claim; it
cannot make the gate mark that claim VERIFIED, because the gate never asks the model
whether its own claim is true.

**Honest limits, stated plainly:**

- **Extractor recall is the exponent, and it is now measured.** The guarantee only covers
  claims the extraction step routes through the gate. That denominator was unmeasured for
  most of the project's life; it no longer is. `scripts/routing_coverage.py` asks four
  natural framings per fact and finds **99%** of values a model asserts in prose reach the
  gate, with **0** wrong figures reaching a user unguarded (see
  [Does the gate see what the model says](#does-the-gate-actually-see-what-the-model-says)).
  The domain gate uses a slot-filling extractor (`factgate/domain/link.py`) with
  deterministic grounding guards, not a tool-call protocol — the `<kb_q>` path below is the
  earlier RCK design.
- **Domain-bounded.** The guarantee holds only for facts within the declared fact set.
  Anything the fact set was never given is neither confirmed nor denied — it abstains.
- **Conservative coverage.** Roughly 16% of correct values are held rather than confirmed
  (Measured behaviour, below). This is a deliberate, measured, safe failure direction — a review queue,
  not a wrong answer — not a defect being hidden.
- **Compositional implicature is out of scope.** The gate verifies triples
  (subject, relation, object), not multi-step logical entailment or pragmatic inference
  layered on top of verified facts.

## Trained on a Strix Halo iGPU

The fine-tuned front-end (Qwen2.5-14B-Instruct, bf16 LoRA, r=16, 68.8M trainable params)
was trained natively on Windows ROCm 7.2.1 on an AMD Ryzen AI MAX+ 395 (Radeon 8060S,
gfx1151) integrated GPU — no discrete GPU, no cloud. 200 stable steps completed on
53k RCK-grounded examples; the checkpoint shows a 100% tool-call rate and 0 gate leaks on
held-out prompts (table above). Full training log, the ROCm-on-iGPU stability issues hit
and worked around, and honest limits of that run are in
[`docs/ROCM-STRIX-HALO-TRAINING.md`](https://github.com/NORTHTEKDevs/factgate/blob/main/docs/ROCM-STRIX-HALO-TRAINING.md).

## Reproduce

```bash
# install (RCK is a separate public dependency, not on PyPI)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e .
./.venv/Scripts/python.exe -m pip install -e ../rck   # path to your local RCK checkout

# run the test suite (816 tests, no network, <10s)
./.venv/Scripts/python.exe -m pytest tests/ -q

# reproduce the headline guarantee measurement (N=1500/class against the live KB)
./.venv/Scripts/python.exe scripts/measure_guarantee.py

# reproduce the fine-tuned checkpoint eval
./.venv/Scripts/python.exe scripts/eval_checkpoint.py

# reproduce the serving-gateway wiring check (scripted inner, NOT a live model)
# NOTE: requires a separate serving-gateway checkout exposing
# `serving.factgate_backend.VerifiedBackend`, which is NOT included in this repo.
# Without it this script raises ImportError. Everything above runs from a clean clone.
./.venv/Scripts/python.exe scripts/e2e_gated.py
```

## What this is / isn't

**This is:** a deterministic verification gate whose own false-accept rate is measured at 0%
across 3,000 adversarial structured-triple trials (no LLM in that measurement), paired with a
small fine-tuned model that, on the 4 held-out prompts tested so far, routes factual claims
through that gate rather than asserting from its own weights.

**This isn't:** a frontier chatbot. The fine-tuned model is a behavioral front-end — its
job is routing and citation discipline, not open-ended reasoning or breadth of knowledge.
The value of FACTGATE is the *pairing* — a fluent model whose factual claims are backed
by an exact store and a gate that can't be prompted around — not the model's standalone
capability.

## License

Apache-2.0. Contact: info@northtek.io

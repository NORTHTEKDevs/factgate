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
![Tests](https://img.shields.io/badge/tests-778%20passing-brightgreen)

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

All numbers below are read directly from files in [`results/`](results/) — no number in
this README is hand-typed without a source artifact backing it.

**Gate decision-rule benchmark.** No LLM is involved in any row below: these call
`verify_claim()` directly on pre-parsed `(subject, relation, object)` triples. They measure the
gate's own false-accept rate, not an observed hallucination rate of any model.

| Metric | Result | 95% CI | Source |
|---|---|---|---|
| False-VERIFIED rate, absent facts (N=1500) | **0.0%** | [0%, 0.26%] | [`results/guarantee_measurement.json`](results/guarantee_measurement.json) |
| False-VERIFIED rate, corrupted claims (N=1500) | **0.0%** | [0%, 0.26%] | [`results/guarantee_measurement.json`](results/guarantee_measurement.json) |
| True-fact verify coverage (N=1500) | **34%** | [31.6%, 36.4%] | [`results/guarantee_measurement.json`](results/guarantee_measurement.json) |
| Test suite | **778 passed** | — | `pytest tests/ -q` |

**LLM-in-the-loop results.** Only these involved a running model. Note the sample size.

| Metric | Result | 95% CI | Source |
|---|---|---|---|
| Fine-tuned model tool-call rate, held-out prompts | **100%** (4/4) | — (N=4) | [`results/checkpoint_eval.json`](results/checkpoint_eval.json) |
| Fine-tuned model gate leak (false-VERIFIED emitted) | **0** | — (N=4) | [`results/checkpoint_eval.json`](results/checkpoint_eval.json) |
| End-to-end leak, live extraction, 15 domains | **0** (0/366) | [0%, 1.0%] | [`results/`](results/), reproducible via `scripts/run_domain_bench.py` |

The gate never accepted a false claim as VERIFIED across 3,000 adversarial **structured-triple**
trials (1,500 facts deliberately absent from the KB, 1,500 with a corrupted object). The cost of
that guarantee is coverage: only 34% of genuinely true, stored facts clear the
confidence threshold to VERIFY — the rest safely abstain (OUT_OF_KB / UNRESOLVED) rather
than guess.

Sample outputs from the fine-tuned checkpoint on prompts it never saw in training
(`results/checkpoint_eval.json`) — it routes to the knowledge base instead of answering
from its weights:

```
Q: "What category does foul_play belong to?"
-> <kb_q>{"s": "foul_play", "r": "isa", "unknown": "O"}</kb_q>

Q: "What is bicycle capable of?"
-> <kb_q>{"s": "bicycle", "r": "can", "unknown": "O"}</kb_q>

Q: "What category does octahedron belong to?"
-> <kb_q>{"s": "octahedron", "r": "isa", "unknown": "O"}</kb_q>
```

Full breakdown, including the gated-serving plumbing check (scripted inner, not a live model)
and the recall-calibration fix that made the guarantee measurable, is in [`RESULTS.md`](RESULTS.md) and
[`FINAL-REPORT.md`](FINAL-REPORT.md).

## End-to-end gating on real model prose

The measurements above adjudicate structured triples. Extending the gate to free prose was
attempted, measured, and **failed for a structural reason** worth knowing before you build
on this: a parameter-free verdict needs a canonical vocabulary on both sides, and free text
does not have one. Two independent extractions of the same fact by the same model shared
0 of 17 relation strings, so exact comparison is impossible.

What does work is a **bounded domain**, where entities and relations are declared up front
and extraction becomes linking rather than generation (`factgate/domain/`). Measured on 12
declared facts with real model prose in both conditions:

Extractor `qwen2.5:14b`, reproduced twice:

| Metric | Result | 95% CI |
|---|---|---|
| Leak rate (wrong value reached the user as VERIFIED) | **0%** (0/24) | [0%, 14%] |
| Over-block rate (correct value failed to verify) | **8%** (1/12) | [1%, 35%] |
| Wrong values blocked with an explicit contradiction | **100%** (24/24) | — |

Trials from one declared fact are correlated, so cite the fact-clustered intervals: leak
0/12 facts [0%, 24%], over-block 1/12 facts [1%, 35%]. "0%" means no leak was observed at
this sample size, not that leaks are impossible.

**The over-block number is extractor-dependent and tuned.** The same harness on
`llama3.2:3b` gives 64% over-block (leak still 0%), and this domain's `value_qualifiers`
were fitted to its own observed failures.

A **blind** test on a second, unrelated domain (consumer lending: percentages, currency,
durations) with its vocabulary declared before any run measures the honest starting point:

Vocabularies were declared before any run in every case marked blind, including two
documents chosen mechanically rather than by a human (`scripts/select_eval_document.py`)
and one whose vocabulary was written by an author who had never seen this codebase.

Current per-domain numbers are in [Status](#status) below and are re-measured on every
full run; they are not duplicated here, because two tables of the same thing drift.

The first row is a real product-strategy memo (kept private; not vendored), with source quotes
pulled programmatically from the file. Its first run held **11 of 11** correct values,
because business prose writes prices as `$249` and the parser required the number first.
That is now fixed, along with three measurement bugs that twice produced a false ~28% leak
rate. The single remaining hold is `$95-145` -- a **range**, which the schema cannot
express and the gate correctly refuses to confirm.

**Leak rate was 0% in every configuration** — two value spaces, four vocabularies, blind
and tuned. The safety property generalises; the coverage cost is earned per-domain by
declaring vocabulary, and an undeclared qualifier always costs a HELD, never a leak.
Details in [`docs/HALLUGATE.md`](docs/HALLUGATE.md).

Both rates are always reported together. Either alone is meaningless: a gate that blocks
everything has a 0% leak rate, and an early version of this pipeline did exactly that.

Full write-up, including a live false-BLOCK bug and its fix, is in
[`docs/HALLUGATE.md`](docs/HALLUGATE.md).

### Status

### What is proven, and by what

Every claim below is re-checked by `python scripts/harden.py` in about 80 seconds, and by
CI on every push. It does not merely run the proofs -- it then **breaks the code ten ways
and requires each proof to notice**. A surviving mutant fails the build, because it marks a
defence that nothing tests.

```
PROOFS
  full suite                                          778 passed
  value grammar vs exact-arithmetic oracle             13 passed
  residue and conditional verdict paths                79 passed
  safety invariants I1-I7                               9 passed
DOMAINS
  every shipped domain loads, validates, lints clean   35 loaded, 0 bad
MUTATION -- each proof must CATCH a broken defence    10 mutants, 10 caught

HARDEN PASSED: 15/15 checks in 77s
```

| | |
|---|---|
| **Proven** | Over the declared value grammar, every MATCH is confirmed **equal** and every DIFFER confirmed **unequal**, by exact rational arithmetic (`fractions.Fraction`) that shares no code with the implementation |
| **Proven** | No author declaration -- no `unit_aliases`, no `value_qualifiers` -- can make the gate verify an unequal value without `lint()` refusing the fact set first |
| **Proven** | The residue and conditional-variant paths behave as constructed cases require, where the correct verdict follows from how the case was assembled rather than from what the code says |
| **Proven** | Extraction never emits a claim its own guards reject, **whatever the model says** -- tested against a hostile scripted model that fabricates, translates, glues, refuses and copies decoys |
| **Proven** | Every check above catches the defence it protects |
| **NOT proven** | Anything about notations outside the declared grammar. Those are **held**, which is the fail-closed answer, not a verified one. |
| **NOT proven** | Completeness of the exclusion vocabulary used by *residue admission*. When a claim adds wording the fact's own source sentence supplies, that clause is checked against a hand-curated list of negation and exemption words. If a word is missing the clause is admitted, so this path is **fail-open, not fail-closed** — the one place in the system where that is true. It has already caused a leak once (`"free for premium members"` passed while the identical sentence using `"waived"` was held). The list is regression-tested and its removal is caught by mutation, but neither proves it complete. |

Those last two rows are the boundary, and stating them is deliberate. The first is a design
choice: hold what you cannot parse. The second is a genuine weakness, disclosed because an
adversarial reviewer found it by reading the code and it is the one sentence in this claim
that would not survive being discovered rather than declared. A guarantee with a stated edge
survives contact with a real document; one without an edge does not.

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

### How this was reached

Five rounds of adversarial review found defects one at a time, and by the fourth round most
new leaks were **inside the previous round's fixes** -- the search was not converging, and
each notation added to reduce over-block created leak surface faster than review closed it.
The proofs above replaced that process. They caught a false block on their first run, and
the mutation gate immediately exposed two defects in the oracle itself while it appeared to
pass. `docs/HALLUGATE.md` records every defect with its measured reproduction.

**What is still not certified.** No deployment has run without a human reviewing the held
queue. The three private evaluation documents are not in this repository, so two of the
measured rows cannot be reproduced from a clone. And the boundary above is real: a document
using a notation outside the declared grammar will be held, not verified. No deployment has run without a human
reviewing the held queue, the two real documents cost a review pass each, and the private
evaluation corpora are not in this repository so those two rows cannot be reproduced from
a clone. What is verified is stated above and re-runnable for the four public domains.

New authors should start with [`docs/AUTHORING.md`](docs/AUTHORING.md).

Two harnesses verify the things a test suite cannot:

```bash
python scripts/acceptance.py    # fresh venv, install, exercise the documented surface
python scripts/soak.py          # live pipeline, safety invariants on every verdict
```

`acceptance.py` found that four test modules hard-required an optional dependency, so a
fresh clone got four collection errors and ran zero tests. It now reports 292 passed,
4 skipped. `soak.py`'s first full campaign adjudicated **142 live verdicts across eight
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

- **Extractor recall is the exponent.** The guarantee only covers claims the extraction
  step actually routes through the gate. A claim the model asserts in free text without
  emitting a KB lookup bypasses verification entirely — v1 relies on a tool-call
  protocol (`<kb_q>...</kb_q>`), not a neural free-text claim extractor. A missed
  extraction is a missed gate check.
- **And the current metric cannot see those misses.** `gated_hallucination_rate`
  (`factgate/bench/runner.py:264-266`) is `gate_blocked_total / gate_claims_total`, where
  `gate_claims_total` counts only claims that were successfully extracted. An untagged
  prose claim contributes to neither the numerator nor the denominator, so it is invisible
  to the metric rather than counted as a leak. Read that number as "of the claims the gate
  saw, how many did it block" — never as an end-to-end leak rate. Measuring the real
  end-to-end rate requires a free-text extractor that does not yet exist here.
- **Domain-bounded.** The guarantee holds only for facts within RCK's stored domain.
  Anything RCK was never given is neither confirmed nor denied — it abstains.
- **Conservative coverage.** 34% true-verify coverage means most true facts currently
  abstain rather than confirm. This is a deliberate, measured, safe failure direction —
  not a defect being hidden.
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
[`docs/ROCM-STRIX-HALO-TRAINING.md`](docs/ROCM-STRIX-HALO-TRAINING.md).

## Reproduce

```bash
# install (RCK is a separate public dependency, not on PyPI)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e .
./.venv/Scripts/python.exe -m pip install -e ../rck   # path to your local RCK checkout

# run the test suite (778 tests, no network, <10s)
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

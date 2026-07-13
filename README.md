# FACTGATE

**A language model that structurally cannot confidently state a fact it can't verify.**

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-146%20passing-brightgreen)

## The idea, in 3 sentences

FACTGATE pairs a fluent LLM with [RCK](https://github.com/NORTHTEKDevs/rck) (Resonant Cognitive
Kernel), a public, exact bipolar-VSA fact store, and puts a deterministic gate between
what the model *wants to say* and what it is *allowed to emit*. The model doesn't decide
whether its own claim is true — it emits a lookup, RCK answers from its stored facts, and
the gate blocks anything RCK can't corroborate. That is a categorically different
guarantee from RLHF-style alignment: RLHF trains the model to be *less likely* to
hallucinate; the gate makes an unverifiable claim *structurally unable to reach the user*
as a confirmed fact, regardless of what the model's weights want to say.

## Measured results

All numbers below are read directly from files in [`results/`](results/) — no number in
this README is hand-typed without a source artifact backing it.

| Metric | Result | 95% CI | Source |
|---|---|---|---|
| False-VERIFIED rate, absent facts (N=1500) | **0.0%** | [0%, 0.26%] | [`results/guarantee_measurement.json`](results/guarantee_measurement.json) |
| False-VERIFIED rate, corrupted claims (N=1500) | **0.0%** | [0%, 0.26%] | [`results/guarantee_measurement.json`](results/guarantee_measurement.json) |
| True-fact verify coverage (N=1500) | **34%** | [31.6%, 36.4%] | [`results/guarantee_measurement.json`](results/guarantee_measurement.json) |
| Fine-tuned model tool-call rate, held-out prompts | **100%** (4/4) | — | [`results/checkpoint_eval.json`](results/checkpoint_eval.json) |
| Fine-tuned model gate leak (false-VERIFIED emitted) | **0** | — | [`results/checkpoint_eval.json`](results/checkpoint_eval.json) |
| Test suite | **146 passed** | — | `pytest tests/ -q` |

The gate never emitted a false claim as VERIFIED across 3,000 adversarial trials
(1,500 facts deliberately absent from the KB, 1,500 with a corrupted object). The cost of
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

Full breakdown, including the end-to-end gated-serving demo and the recall-calibration
fix that made the guarantee measurable, is in [`RESULTS.md`](RESULTS.md) and
[`FINAL-REPORT.md`](FINAL-REPORT.md).

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

# run the test suite (146 tests, no network, <10s)
./.venv/Scripts/python.exe -m pytest tests/ -q

# reproduce the headline guarantee measurement (N=1500/class against the live KB)
./.venv/Scripts/python.exe scripts/measure_guarantee.py

# reproduce the fine-tuned checkpoint eval
./.venv/Scripts/python.exe scripts/eval_checkpoint.py

# reproduce the end-to-end gated-serving demo
./.venv/Scripts/python.exe scripts/e2e_gated.py
```

## What this is / isn't

**This is:** a deterministic verification gate with a measured, adversarially-tested
guarantee (0% false-VERIFIED across 3,000 trials), paired with a small fine-tuned model
that has learned to route factual claims through that gate rather than assert from its
own weights.

**This isn't:** a frontier chatbot. The fine-tuned model is a behavioral front-end — its
job is routing and citation discipline, not open-ended reasoning or breadth of knowledge.
The value of FACTGATE is the *pairing* — a fluent model whose factual claims are backed
by an exact store and a gate that can't be prompted around — not the model's standalone
capability.

## License

Apache-2.0. Contact: info@northtek.io

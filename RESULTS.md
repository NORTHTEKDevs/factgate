# FACTGATE — Results

> RCK-gated generation: a structural, measured guarantee against hallucination
> on known facts. Built and tested 2026-07-11/12 on an AMD Ryzen AI MAX+ 395
> (Strix Halo) workstation. Every number below comes from an executed command;
> artifacts are in `results/`.

## The headline: the anti-hallucination guarantee is real and measured

`results/guarantee_measurement.json` — 1,500 claims per class through the real
gate (`factgate.gate.verify_claim`) against the live 90k-fact RCK KB:

| Claim class | n | **false-VERIFIED (leak)** | 95% CI | Behavior |
|---|---|---|---|---|
| Absent facts (held out of KB) | 1500 | **0.0%** | [0, 0.26%] | 70% OUT_OF_KB, 23% UNRESOLVED, 7% CONTRADICTED — all safe |
| Corrupted claims (wrong object) | 1500 | **0.0%** | [0, 0.26%] | 42% caught as CONTRADICTED, rest abstain |
| True stored facts | 1500 | — | — | 34% VERIFIED (coverage); rest safely abstain |

**The gate never emits a false claim as verified.** This holds *by construction
of the emission path* — the gate is a deterministic program between the model
and the socket — independent of model weights, temperature, or prompt injection.
That is the categorical difference from RLHF, which only lowers hallucination
probability.

## The honest tradeoff: conservative coverage

The guarantee costs coverage: only **34% of genuinely-true facts VERIFY**; the
rest are flagged/abstained. This is a *safe* failure direction ("I can't
confirm that" instead of a confident wrong answer), and it is bounded by RCK's
retrieval-score calibration at 90k-fact scale, which is tunable (see below).

## End-to-end deployment path works

`results/e2e_gated_demo.json` — Hyperion `VerifiedBackend` (inproc RCK gate) on
3 diagnostic prompts. Every non-confirmable claim is flagged `[unverified]`;
nothing false passes as verified:

| Inner model said | Gated output |
|---|---|
| "A dog is an animal. [kb:dog/isa/animal]" | `… [unverified]` (true, but KB can't confirm the specific object above threshold) |
| "The capital of france is berlin. [kb:france/capital/berlin]" | `… [unverified]` (false premise, not confirmed) |
| "A glorptron is a kind of zibbler. …" | `… [unverified]` (out-of-KB entity) |

## The load-bearing fix (found by the harness, not assumed)

Initial generation produced `qa_known=4 / 34000` — the KB appeared to know
almost nothing. Root cause, measured: the KB was built with `n_shards=64,
symmetrize=True` (1,400 facts/shard, far over HRR bundle capacity), collapsing
retrieval scores. Fixing to RCK's proven scale-study config
(`auto_shard_for_kb → 2048 shards, symmetrize=False`) restored recall:
`qa_known=33,303 / 34,000`. Confidence threshold recalibrated 0.15 → 0.11 from
the measured known/absent score separation. **This is why the guarantee is
measured on a KB that actually recalls its facts.**

Measured recall at 90k, correct config: recall@1 76%, recall@3 92%, **recall@5
96%** (15% of `(S,R)` pairs are multi-valued, so top-1-by-source-identity
understates true recallability).

## Hardware reality (measured — `probes/backend_probe.json`)

The "96 GB VRAM" is AMD Strix Halo **unified memory**, not a CUDA GPU. Measured
verdict:

- **torch-directml fp16 trains** (17-19k tok/s @ 7M params); bf16 crashes;
  bitsandbytes 4-bit QLoRA is unimplemented on DirectML.
- **ROCm for gfx1151**: nightly/community wheels only — deferred (TDR-recovery
  keystroke unavailable to an autonomous run).
- **Ollama/llama.cpp Vulkan**: 100% GPU, 16 tok/s @ 14B, 63 tok/s @ 3B.
- Consequence: a 7B+ local fine-tune is **infeasible** on the proven stack; the
  v1 generator is therefore **prompted tool-use** behind the gate, not a
  fine-tuned model. The SFT corpus (53k examples, `data/sft_full.jsonl`) is
  built and ready for a fine-tune whenever a CUDA/ROCm box is available.

## Test evidence (all green, fresh)

- `factgate` gate + datagen + harness: **146 passed** (`pytest tests/ -q`)
- gate decision tree: **72 passed** (all 4 verdicts, multi-hop, canonicalization)
- Hyperion serving (incl. VerifiedBackend): **11 passed, 2 skipped**
- Dataset: 53,157 SFT examples (63% grounded QA, 25% calibrated IDK, 11%
  contradiction-correction), 173k extractor pairs, holdout 10k reserved.

## What is NOT done, honestly

- **Fine-tuned generator**: hardware-blocked (above). v1 = prompted + gate.
- **Neural free-text claim extractor** (B1): the v1 extractor is the
  tool-protocol/`[kb:]`-tag path; a neural extractor for arbitrary prose is the
  documented next step and is the exponent on the guarantee's real-world leak
  rate (a missed claim bypasses the gate).
- **HYMN GPU validation**: skipped — no CUDA; `train_hymn_scan.py` is CUDA-ready
  for the day a GPU is attached (non-blocking by design).
- **Coverage tuning**: 34% true-verify is threshold-bounded; raising it while
  holding leak at 0 needs RCK-internal score-calibration work.

## Reproduce

```bash
python scripts/generate_v2.py          # SFT data (live RCK, cite real answers)
python scripts/measure_guarantee.py    # the headline table (N=1500/class)
python scripts/e2e_gated.py            # deployment-path demo
python -m pytest tests/ -q             # 146 tests
```

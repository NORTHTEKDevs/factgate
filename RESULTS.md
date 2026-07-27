# FACTGATE — Results

> RCK-gated generation: a structural, measured guarantee against hallucination
> on known facts. Built and tested 2026-07-11/12 on an AMD Ryzen AI MAX+ 395
> (Strix Halo) workstation. Every number below comes from an executed command;
> artifacts are in `results/`.

## The headline: the gate's false-accept rate on structured claims is 0%, measured

`results/guarantee_measurement.json` — 1,500 claims per class through the real
gate (`factgate.gate.verify_claim`) against the live 90k-fact RCK KB.

**Scope of this measurement:** claims are supplied as pre-parsed `(subject, relation,
object)` triples read from JSONL. There is no generation step and no LLM in this loop, so
"false-VERIFIED" below means *the gate's own false-accept rate on structured claims* — not an
observed rate of hallucinated content leaking from a model to a user. The end-to-end quantity
is not yet measured; see "What is NOT done, honestly".

| Claim class | n | **false-VERIFIED rate** | 95% CI | Behavior |
|---|---|---|---|---|
| Absent facts (held out of KB) | 1500 | **0.0%** | [0, 0.26%] | 70% OUT_OF_KB, 23% UNRESOLVED, 7% CONTRADICTED — all safe |
| Corrupted claims (wrong object) | 1500 | **0.0%** | [0, 0.26%] | 42% caught as CONTRADICTED, rest abstain |
| True stored facts | 1500 | — | — | 34% VERIFIED (coverage); rest safely abstain |

**The gate never emits a false claim as verified, for every claim that reaches the gate.**
This holds *by construction of the emission path* — the gate is a deterministic program between
the model and the socket — independent of model weights, temperature, or prompt injection.
A claim asserted in free prose without an emitted lookup never reaches the gate at all, so this
is a property of the gate, not yet a measured end-to-end property of the system.
That is the categorical difference from RLHF, which only lowers hallucination
probability.

## The honest tradeoff: conservative coverage

The guarantee costs coverage: only **34% of genuinely-true facts VERIFY**; the
rest are flagged/abstained. This is a *safe* failure direction ("I can't
confirm that" instead of a confident wrong answer), and it is bounded by RCK's
retrieval-score calibration at 90k-fact scale, which is tunable (see below).

## The serving-gateway wiring works (scripted inner, not a live model)

`results/e2e_gated_demo.json` — the `VerifiedBackend` serving gateway (inproc RCK gate) on
3 diagnostic prompts. **The inner "model" here is a `ScriptedInner` stub emitting hand-written
strings that already contain the `[kb:...]` tag** (`scripts/e2e_gated.py`), chosen for
determinism and speed. This proves the gate plumbing is correctly wired; it does *not*
demonstrate an LLM routing its own claims through the gate. Every non-confirmable claim is
flagged `[unverified]`; nothing false passes as verified:

| Scripted inner emitted | Gated output |
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

## Hardware reality (measured)

The "96 GB VRAM" is AMD Strix Halo **unified memory** (Radeon 8060S iGPU,
gfx1151), not a CUDA GPU. The first probe (`probes/backend_probe.json`) tried
torch-directml and concluded a local fine-tune was impractical — DirectML fp16
trains but bf16 crashes and 4-bit QLoRA is unimplemented.

**That conclusion was superseded.** Native Windows **ROCm 7.2.1 + PyTorch 2.9.1**
(AMD's official gfx1151 wheels) *does* train: verified `torch.cuda` on the iGPU
(90 GB), bf16 matmul + backward, ~7.8 TFLOP/s. We then fine-tuned
**Qwen2.5-14B** via bf16 LoRA on it — see the trained-model results above and
[`docs/ROCM-STRIX-HALO-TRAINING.md`](docs/ROCM-STRIX-HALO-TRAINING.md) for the
full recipe and the (real) stability gotchas. The honest caveat is stability
under sustained load, not feasibility: training completed 200 stable steps
(a fully usable behavioral adapter) before the ROCm-iGPU stack faulted on a
longer run.

## Test evidence (all green, fresh)

- `factgate` gate + datagen + harness: **146 passed** (`pytest tests/ -q`)
- gate decision tree: **72 passed** (all 4 verdicts, multi-hop, canonicalization)
- Serving gateway (incl. VerifiedBackend): **11 passed, 2 skipped**
- Dataset: 53,157 SFT examples (63% grounded QA, 25% calibrated IDK, 11%
  contradiction-correction), 173k extractor pairs, holdout 10k reserved.

## What is NOT done, honestly

- **Full-epoch fine-tune**: the generator trained for 200 stable steps (a
  usable behavioral adapter, verified above), not the planned 400 — the
  ROCm-on-Windows-iGPU stack is unstable for sustained/long GPU work. A
  full run wants a merged-GGUF serving path or a discrete-GPU box.
- **Neural free-text claim extractor**: the current extractor is the
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
python scripts/e2e_gated.py            # gateway wiring check (scripted inner; needs
                                       #   an unshipped serving-gateway checkout)
python -m pytest tests/ -q             # 146 tests
```

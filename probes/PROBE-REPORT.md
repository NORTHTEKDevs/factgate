# B0 GPU-training-backend probe — AMD Ryzen AI MAX+ 395 (Strix Halo)

Hardware: AMD Ryzen AI MAX+ 395, Radeon 8060S iGPU (gfx1151), unified LPDDR5X, Windows 11, 32 threads, 225 GB free disk at probe start.

All numbers below are from executed commands captured in `probes/*_result.json` and `probes/*_stderr.log`. Full machine-readable version: `probes/backend_probe.json`.

## 1. torch-directml (measured, working)

Isolated venv `.venv-probe` (Python 3.12), `pip install torch-directml` pinned `torch==2.4.1`.

- **10M-class model** (actual 7,394,624 params: 4-layer TransformerEncoder, d=256, seq=512, batch=8), steady-state fwd+bwd+AdamW step:
  - DirectML: **4.16–4.60 steps/s** across two runs → **~17,000–18,900 tok/s**
  - CPU (same venv, torch 2.4.1, 32 threads): 2.70–2.97 steps/s → ~11,050–12,180 tok/s
  - CPU (official baseline, rain venv, torch 2.13.0+cpu, 32 threads): 0.644–0.645 steps/s → ~2,640 tok/s
  - **DML is ~1.5x faster than same-env CPU (torch 2.4.1), ~7.1x faster than the torch 2.13.0 CPU baseline.**
- **bf16**: **NOT SUPPORTED.** Any bf16 tensor op **hard-crashes the process** (exit 127, not a catchable Python exception): `dml_util.cc:118] Invalid or unsupported data type BFloat16.` Reproduced twice, identical crash both times. This was a clean process abort, not a GPU driver hang/TDR — the machine stayed fully responsive.
- **fp16**: supported (matmul verified working, round-trips to CPU).
- **bitsandbytes 4-bit (QLoRA dependency)**: **NOT SUPPORTED.** `bnb.nn.Linear4bit` on the DML device raises `NotImplementedError: Could not run 'bitsandbytes::quantize_4bit' with arguments from the 'AutogradPrivateUse1' backend`. Cross-checked: the identical call succeeds on CPU. **This means QLoRA cannot run on torch-directml at all — it's not a speed problem, the op literally isn't implemented for this backend.**
- **Op coverage**: one CPU-fallback op observed during normal training: `aten::lerp.Scalar_out` (used inside AdamW's foreach exp_avg update) — falls back to CPU every optimizer step, everything else in the training loop ran on DML.
- **0.5B-scale smoke test** (368,403,712 params: 24-layer TransformerEncoder, d=1024, nhead=16, dim_ff=4096, seq=512, batch=1, fp32): ran cleanly, no crash. Single fwd+bwd+optimizer step took 4.94–5.09s across two runs → ~101–104 tok/s for that single step. **This single-step number includes one-time DML graph-compile/warmup overhead and understates steady-state throughput** — it answers "does it run" (yes) and gives a rough magnitude, not a clean scaling data point. System RAM grew ~2.6–2.7 GB during the step.

## 2. ROCm on Windows for gfx1151 (researched, NOT attempted)

- **Official support**: gfx1151 (Strix Halo) is **not** on AMD's official ROCm Windows support matrix as of 2026-07 (official RDNA3 targets are gfx1100/gfx1101). No official pytorch.org or AMD-hosted stable wheel exists for gfx1151 on Windows.
- **Community wheels exist and are plausible**: `scottt/rocm-TheRock` GitHub releases publish self-contained PyTorch wheels for gfx1151 on Windows (torch 2.7.0a0+git, ROCm 6.5.0rc, cp312 win_amd64), built by community devs (scottt, jammm). AMD also runs a nightly ROCm 7.9 pip index scoped to gfx1151 (`https://rocm.nightlies.amd.com/v2/gfx1151/`), documented for conda installs on Windows.
- **Deliberately not attempted.** This machine has documented iGPU TDR black-screen history, and the recovery keystroke (Win+Ctrl+Shift+B) is a physical action this probe cannot perform. DirectML's officially-supported DX12 compute path (with automatic per-op CPU fallback) was attempted and did not destabilize the machine at all. Community/nightly HIP kernels for a GPU target AMD does not officially support carry materially higher risk of a driver-level hang, with no available recovery path if it happens mid-session. This is a deliberate risk call under the probe's own hard rules, not a missed step — if the user wants this measured, it should be attempted with physical machine access.

Sources: [ROCm/TheRock #655](https://github.com/ROCm/TheRock/discussions/655), [scottt/rocm-TheRock releases](https://github.com/scottt/rocm-TheRock/releases/tag/v6.5.0rc-pytorch-gfx110x), [PyTorch+ROCm7 on Strix Halo (Medium)](https://medium.com/@GenerationAI/pytorch-with-rocm-7-for-windows-on-amd-ryzen-ai-max-395-strix-halo-radeon-8060s-gfx1151-1ba069edc2c4), [llm-tracker.info Strix Halo](https://llm-tracker.info/_TOORG/Strix-Halo).

## 3. WSL2 (checked, no distro installed)

- `wsl --status`: Default Version 2 (WSL2 kernel present).
- `wsl -l -v`: **no distributions installed.**
- A WSL2-ROCm path would require: (1) `wsl --install <distro>` — a user decision, not performed; (2) installing AMD's ROCm-for-WSL packages inside that distro. **Caveat**: AMD's official ROCm-on-WSL support has historically targeted discrete Instinct/Radeon dGPUs passed through to WSL2, not integrated Strix Halo APUs — gfx1151-under-WSL2 support is unconfirmed and would need separate verification once a distro exists.

## 4. llama.cpp / Ollama Vulkan (measured, GPU confirmed active)

- Ollama 0.20.7 already installed and using the GPU. `ollama ps` reports **100% GPU** for both models tested (not a CPU/GPU split).
- **llama3.2:3b**: 200-token generation in 3.10s → **62.9 tok/s** generation, prompt eval **287 tok/s** (35-token prompt).
- **qwen2.5:14b**: 200-token generation in 12.08–12.39s → **16.1–16.6 tok/s** generation (two runs), prompt eval **62.7–69.3 tok/s**.
- `winget search llama.cpp` → `ggml.llamacpp` version b9957 is available directly (not installed; Ollama already covers this).
- This is an **inference-only** measurement (no backward pass). Relevant to serving/prompted-generator tier, not training.

## 5. CPU training baseline (measured)

Reused a sibling project's venv (torch 2.13.0+cpu, 32 threads), same 10M-class model script: **0.6444–0.6448 steps/s → ~2,640 tok/s.** This is the official CPU baseline per the probe spec.

Cross-check in the probe venv (torch 2.4.1+cpu, same 32 threads, identical model/config): **2.70–2.97 steps/s → ~11,050–12,180 tok/s — roughly 4.2x faster than the rain venv's torch 2.13.0+cpu**, on identical hardware and thread count. Both are real measurements from the same script; the delta is reported as-is (candidate cause: differing default BLAS/MKL backend or thread pinning between the two torch builds — not independently isolated in this probe).

## Projections (extrapolated from measured throughput, NOT directly measured at 1.5B/7B/14B scale)

Method: hours = tokens / (tok_s_at_7.39M_scale × (7,394,624 / N_target)) — linear compute-vs-param-count scaling calibrated against the measured steady-state 7.39M-param throughput on each backend. LoRA and full fine-tune are projected identically (conservative: no compute discount assumed for frozen layers, since backward still has to propagate through them). INFEASIBLE threshold: >300 hours.

| Scenario | DirectML | CPU (rain, torch 2.13, official baseline) | CPU (probe venv, torch 2.4.1) |
|---|---|---|---|
| 1.5B LoRA SFT, 300k examples (~75M tokens) | **224.1h — VIABLE by time, but a 9-day continuous run** | 1,601.0h — INFEASIBLE | 346.9h — INFEASIBLE |
| 7B QLoRA, 350M tokens | **INFEASIBLE — UNSUPPORTED** (bitsandbytes 4-bit has no DirectML kernel, measured NotImplementedError) | 34,865.3h — INFEASIBLE | 7,555.2h — INFEASIBLE |
| 14B (full/LoRA), 350M tokens | 9,760.3h — INFEASIBLE | 69,730.6h — INFEASIBLE | 15,110.4h — INFEASIBLE |

ROCm Windows and llama.cpp/Vulkan have no projection rows: ROCm was not measured (see §2), and llama.cpp/Vulkan is an inference-only engine with no autograd/backward pass — it cannot run LoRA/QLoRA/fine-tuning training at all.

## Recommendation

DirectML is the only backend on this machine that is both measured working and capable of actual gradient-based training today. It trains the 7.39M-param model at ~17,000–18,900 tok/s steady-state (~1.5x a same-env CPU comparison, ~7.1x the official CPU baseline) and ran a single fwd+bwd step of a 368M-param/24-layer model cleanly with no crash. But it hard-blocks bf16 (process abort) and hard-blocks bitsandbytes 4-bit (NotImplementedError) — so **QLoRA is not an option on this stack**; only fp16/fp32 LoRA or full fine-tune are possible, with no bf16 mixed precision.

Even so, extrapolated from measured throughput, LoRA SFT at 1.5B scale is a ~9-day continuous DirectML run, and 7B/14B routes are deep in INFEASIBLE territory on every backend measured here (CPU worse still). **This iGPU is not a viable training backend for models above roughly tens of millions of params on human timescales** — fine for small-model experiments, prototyping, and debugging training code; not for real SFT/LoRA/QLoRA runs at 1.5B+.

For actual fine-tuning work: (a) rent a cloud GPU (A10/A100/H100) for the run, or (b) personally attempt the community ROCm gfx1151 wheel (scottt/rocm-TheRock or AMD's nightly ROCm 7.9 gfx1151 index) with physical access to recover from a possible TDR hang — deliberately untested here, not a capability gap in this probe.

Separately, this machine's GPU is already a strong **inference/serving** backend: Ollama is using the 8060S at 100% GPU utilization, generating ~63 tok/s on a 3B model and ~16 tok/s on a 14B model via its Vulkan backend — entirely adequate for a prompted-generator or local serving tier, just not for training.

## Raw artifacts

- `bench_directml_core.py` / `directml_core_result.json` — 10M-param DML steady-state benchmark
- `bench_directml_dtype.py` / `dtype_fp16_result.json`, `dtype_bf16_stderr.log` — dtype support checks (bf16 crashes before it can write its result file — the crash itself, captured in stderr, IS the finding)
- `bench_directml_cpu.py` / `cpu_probe_result.json`, `cpu_rain_result.json` — CPU comparison, two venvs
- `bench_directml_scale.py` / `scale_result.json` — 0.5B-scale single-step smoke test
- `bnb_4bit_result.json` — bitsandbytes 4-bit DML vs CPU support check
- `gen_llama3b_result.json`, `gen_qwen14b_result.json` — raw Ollama `/api/generate` responses
- `assemble_report.py` — assembles this report + `backend_probe.json` from the above, all projection arithmetic done in code

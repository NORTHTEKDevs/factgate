# Fine-tuning an LLM on AMD Strix Halo (gfx1151): a native Windows ROCm how-to

AMD's Ryzen AI MAX+ 395 ("Strix Halo") pairs a Radeon 8060S iGPU with up to 128GB of
unified LPDDR5X, of which a large slice can be given to the GPU. That unified-memory
model makes it an unusually cheap way to fit a 7B-32B parameter model in GPU memory for
LoRA fine-tuning -- something a same-price discrete GPU with 12-24GB of VRAM cannot do.

At the time of writing there is very little documentation of actually training on this
chip: gfx1151 is not on AMD's official ROCm-on-Windows support matrix, most guidance
online covers inference (llama.cpp / Ollama via Vulkan), and the DirectML path that
*is* well documented turns out to be a dead end for anything beyond toy models. This
document is a from-measurement writeup of what actually worked, filed as a reference
for the next person attempting this.

Everything below was run and measured on real hardware: AMD Ryzen AI MAX+ 395 (Strix
Halo), Radeon 8060S iGPU (gfx1151, RDNA3.5), 96GB unified LPDDR5X, Windows 11.

## Contents

- [TL;DR verdict](#tldr-verdict)
- [What doesn't work: torch-directml](#what-doesnt-work-torch-directml)
- [What works: native Windows ROCm 7.2.1](#what-works-native-windows-rocm-721)
- [Install: exact commands](#install-exact-commands)
- [Training setup: bf16 LoRA, no quantization](#training-setup-bf16-lora-no-quantization)
- [The gotchas](#the-gotchas)
  - [1. Silent crash around step 4-5: HIP memory fragmentation](#1-silent-crash-around-step-4-5-hip-memory-fragmentation)
  - [2. Experimental flash-attention crashes: leave it off](#2-experimental-flash-attention-crashes-leave-it-off)
  - [3. Throughput is bandwidth-bound, not compute-bound](#3-throughput-is-bandwidth-bound-not-compute-bound)
  - [4. Sustained GPU work is unstable for long runs: verify/serve on CPU or via GGUF](#4-sustained-gpu-work-is-unstable-for-long-runs-verifyserve-on-cpu-or-via-gguf)
- [Wall-clock expectations by model size](#wall-clock-expectations-by-model-size)
- [Honest verdict](#honest-verdict)

## TL;DR verdict

Strix Halo is a **viable fine-tuning box for small-to-mid models via bf16 LoRA on
native Windows ROCm**, with real stability caveats on long or large runs. Its huge
unified memory is the standout feature -- it fits models a same-price discrete GPU
simply cannot hold. Throughput is modest: it is a memory-bandwidth-bound machine, not
a compute-bound one. Use it for LoRA fine-tuning up to roughly 14B-32B parameters where
a several-hour-to-full-day run is acceptable; don't expect frontier-scale throughput,
and budget for the sustained-GPU-work instability described below.

## What doesn't work: torch-directml

The obvious first path on Windows is `torch-directml`, since it needs no ROCm install
and "just works" for inference. For training, it does not hold up:

- **bf16 hard-crashes the process.** Any bf16 tensor operation aborts immediately
  (`dml_util.cc:118] Invalid or unsupported data type BFloat16.`), not as a catchable
  Python exception -- a clean process exit, reproduced twice, identical both times.
- **bitsandbytes 4-bit (the QLoRA dependency) is unimplemented on this backend.**
  `bnb.nn.Linear4bit` on a DirectML device raises `NotImplementedError: Could not run
  'bitsandbytes::quantize_4bit' with arguments from the 'AutogradPrivateUse1' backend`.
  This is not a speed problem -- the kernel does not exist for this backend at all.
  (Cross-checked: the identical call succeeds on CPU.)
- Measured steady-state throughput on a real training loop (7.39M-param 4-layer
  TransformerEncoder, fp32) was ~17,000-18,900 tok/s -- respectable for a toy model, but
  extrapolating that to a 1.5B-parameter LoRA SFT run projects to roughly 224 hours (a
  9-day continuous run), and 7B/14B routes land in the tens-of-thousands-of-hours range.
  QLoRA is not merely slow here, it is unsupported outright.

**Conclusion: torch-directml is fine for DX12-based inference and small-model
prototyping, but it is not a usable path for fine-tuning anything at real scale**,
because it forces fp16/fp32 (no bf16) and forecloses QLoRA entirely. Native ROCm was
the only remaining option, and it works.

## What works: native Windows ROCm 7.2.1

AMD now ships native Windows ROCm 7.2.1 wheels that support gfx1151, installed with a
matching PyTorch 2.9.1 build. This is a genuine HIP backend running directly on
Windows -- no WSL2, no Vulkan translation layer.

Verified on this hardware:

- `torch.cuda.is_available()` returns `True` and reports the gfx1151 device with ~90GB
  of usable memory (the OS reserves the rest of the 96GB unified pool).
- bf16 matmul and the backward pass both run correctly.
- Measured throughput: **~7.8 TFLOP/s** on bf16 matmul.

**This requires Python 3.12** -- the ROCm 7.2.1 Windows wheels are built for cp312 and
will not install on other Python versions.

## Install: exact commands

Wheels are published at `repo.radeon.com/rocm/windows/rocm-rel-7.2.1/`. Install order
matters: the ROCm SDK components first, then the matching PyTorch wheel.

```bash
# Python 3.12 required -- verify before proceeding
python --version   # must report 3.12.x

python -m venv .venv-rocm
./.venv-rocm/Scripts/python.exe -m pip install --upgrade pip

# ROCm SDK core, dev headers, and runtime libraries for Windows
./.venv-rocm/Scripts/python.exe -m pip install \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-*.whl
./.venv-rocm/Scripts/python.exe -m pip install \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-*.whl
./.venv-rocm/Scripts/python.exe -m pip install \
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-*.whl

# the ROCm runtime archive itself
curl -O https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
tar -xzf rocm-7.2.1.tar.gz

# PyTorch built against ROCm 7.2.1, cp312, matching this ROCm install
./.venv-rocm/Scripts/python.exe -m pip install \
  torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl
```

Exact filenames/URLs vary by release; browse
`repo.radeon.com/rocm/windows/rocm-rel-7.2.1/` for the current wheel names -- the
pattern above (`rocm_sdk_core` / `rocm_sdk_devel` / `rocm_sdk_libraries_custom`, then
`rocm-7.2.1.tar.gz`, then the matching `torch-*+rocm7.2.1-cp312-*.whl`) is what to look
for.

Verify the install:

```bash
./.venv-rocm/Scripts/python.exe -c "
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
```

Expected output: `True`, a string containing `gfx1151`, and a memory figure in the
80-90GB range on a 96GB unified-memory system.

## Training setup: bf16 LoRA, no quantization

Since bitsandbytes 4-bit has no working kernel on this stack, **QLoRA is off the
table** -- the same limitation as DirectML, just for a different reason (bitsandbytes'
CUDA kernels don't target ROCm gfx1151 either, at least not as of this ROCm release).
The workaround is straightforward: skip quantization and run **full bf16 LoRA**. With
90GB of usable unified memory, bf16 LoRA comfortably fits models from 7B up to roughly
32B parameters -- memory that would require multiple discrete GPUs to match at a
similar price point.

Stack: `peft` + `trl`'s `SFTTrainer`, no `BitsAndBytesConfig`, model loaded in bf16
directly (`torch_dtype=torch.bfloat16`), LoRA adapter applied on top, `packing=True` in
the SFT config to reduce padding waste. Nothing unusual about the training script
itself -- the same LoRA config that would target a `BitsAndBytesConfig`-quantized base
model on CUDA works unmodified against a plain bf16 base model here.

## The gotchas

This is the part that isn't documented anywhere else. Every item below was hit for
real during training, not theorized.

### 1. Silent crash around step 4-5: HIP memory fragmentation

Training would crash with no clear Python traceback a handful of steps in. Root cause
is HIP allocator fragmentation under PyTorch's default caching allocator. Fix: set the
expandable-segments allocator before launching training.

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

Set this in the environment (or at the top of the launch script) before every training
run on this stack. Without it, expect an early, silent death.

### 2. Experimental flash-attention crashes: leave it off

PyTorch's ROCm build exposes an experimental AOTriton flash-attention path gated behind
an environment variable. Enabling it native-crashes the process on this hardware:

```bash
# DO NOT set this on gfx1151 -- confirmed to crash
# export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

Leave this unset. Standard (non-flash) attention works correctly and is what was used
for all training runs referenced in this document.

### 3. Throughput is bandwidth-bound, not compute-bound

Unified LPDDR5X memory bandwidth (~256GB/s) is the real ceiling here, not GPU compute.
Measured steady-state training throughput on a 14B model was **~21 training-tokens/s**.
That number does not move much with batch-size or precision tweaks within bf16 --
that's the signature of a bandwidth-bound workload, not a compute-bound one. Plan
training-time budgets around tokens/s at this order of magnitude, not around the
TFLOP/s figure above (which is a matmul microbenchmark, not achieved end-to-end
training throughput).

### 4. Sustained GPU work is unstable for long runs: verify/serve on CPU or via GGUF

Long-running GPU work on this stack is not fully stable. In practice: a training run
that had progressed cleanly to step ~295 died again after roughly 10 hours of wall
clock, and separately, GPU-based generation (inference) for evaluation would hang
outright. Neither failure mode was a hard crash of the whole system -- but both stopped
the GPU process.

Practical mitigation:

- **Checkpoint frequently.** Treat any single GPU run as liable to die and resume from
  the last checkpoint rather than assuming a run will complete unattended.
- **Verify / serve on CPU, in bf16 -- not fp32.** CPU inference is stable on this stack.
  Precision matters for RAM headroom: a 14B model in fp32 is ~56GB, uncomfortably close
  to a 64GB RAM ceiling; the same model in bf16 is ~28GB and comfortably stable. Keep
  weights in bf16 for CPU verification rather than upcasting to fp32.
- **For production-style serving, merge the LoRA adapter into the base model and export
  to GGUF, then serve via `llama.cpp` / Ollama.** That inference path (measured
  separately, GPU-backed via Vulkan) is fast and stable -- tens of tokens/s on a 14B
  model with the GPU reporting 100% utilization and no hangs -- because it is a
  dedicated inference engine, not the training stack under sustained load.

## Wall-clock expectations by model size

Measured/observed wall-clock time for a full bf16 LoRA fine-tuning epoch at each model
size on this hardware, given the ~21 tok/s throughput ceiling described above:

| Model size | Approx. full-epoch wall clock |
|---|---|
| 0.5B | ~2 hours |
| 1.5B | ~4 hours |
| 7B | ~10 hours |
| 14B | ~20-37 hours |

The 14B figures are wide because of the instability in gotcha #4 above: a clean,
uninterrupted run lands near the low end, but any GPU death and resume-from-checkpoint
cycle pushes it out. In one real run, a 14B LoRA fine-tune was stopped at 200 steps
(not the originally planned 400) after training died a second time around step 295,
following ~10 hours of wall clock. 200 steps was sufficient to reach the target
behavior for that particular task -- the general point stands regardless of task: plan
for interruption on runs at this scale, and design training so that an early stopping
point is still a usable checkpoint rather than a wasted run.

## Honest verdict

Strix Halo is a real, working LLM fine-tuning box on native Windows ROCm 7.2.1 -- not
a workaround, not an inference-only story. Bf16 LoRA runs correctly and produces a
working adapter. The standout advantage is unified memory: 90GB of usable GPU memory
at this price point fits models (7B-32B) that a same-price discrete GPU's 12-24GB of
VRAM cannot hold at all, without quantization.

The honest tradeoffs: throughput is modest and bandwidth-bound (tens of tokens/s, not
hundreds), QLoRA is unavailable on this stack so plan for full bf16 LoRA's larger
memory footprint instead of 4-bit's smaller one, and sustained long-running GPU work
(both training and GPU inference) has real stability limits that get worse the longer
and larger the run -- budget for checkpointing and resumption, and move verification
and serving off the training GPU path (CPU bf16, or merged GGUF via llama.cpp) once
training is done.

For small-to-mid model LoRA fine-tuning where a several-hour-to-one-day run is
acceptable and the win is fitting a bigger model than a comparable discrete GPU could
hold, this is a legitimate, reproducible path. It is not a substitute for a data-center
GPU on throughput, and it is not yet a "start a 14B run and walk away for two days"
platform -- treat any run past a few hours as something that needs checkpoint-and-resume
discipline built in from the start.

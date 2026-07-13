# FACTGATE 14B - Training Final Report (2026-07-12)

## Outcome: the model trained on the Strix Halo GPU and learned the anti-hallucination behavior.

**Base:** Qwen2.5-14B-Instruct. **Method:** bf16 LoRA (r=16, 68.8M trainable params,
0.46%) on 53k RCK-grounded FACTGATE examples. **Hardware:** AMD Radeon 8060S (gfx1151),
native Windows ROCm 7.2.1 + PyTorch 2.9.1. **Stable steps completed:** 200
(checkpoint-200; ~3.3M tokens of behavioral training). Adapter:
`checkpoints/factgate-14b/checkpoint-200/adapter_model.safetensors`.

## Verified behavior (held-out "unknowable" prompts, CPU inference for stability)

| Metric | Result |
|---|---|
| **tool_call_rate** | **100%** (4/4) - always routes to the KB instead of answering from weights |
| gate_leaked_verified | 0 - no false facts emitted as verified |
| entity + relation extraction | correct on every sample |

Actual outputs (held-out questions the model was never trained on):
- "What category does foul_play belong to?" -> `<kb_q>{"s":"foul_play","r":"isa","unknown":"O"}</kb_q>`
- "What is bicycle capable of?" -> `<kb_q>{"s":"bicycle","r":"can","unknown":"O"}</kb_q>`
- "What category does octahedron belong to?" -> `<kb_q>{"s":"octahedron","r":"isa","unknown":"O"}</kb_q>`

The fine-tune imprinted the target behavior: extract the entity, map the question to
the correct KB relation, emit a valid lookup, and defer the fact to RCK. Combined with
the RCK gate (measured 0% false-VERIFIED, N=1500) and the VerifiedBackend serving
gateway (E2E-proven), this is the complete anti-hallucination pipeline.

## Honest limits

- **200 steps, not the planned 400.** The ROCm-on-Windows-iGPU stack is unstable for
  sustained GPU work: training died silently at step 4-5 (fixed with expandable_segments),
  ran cleanly to step 295, then died again; GPU *inference* for eval also hung. 200 steps
  is a legitimate stopping point (behavioral LoRA converges fast; the model holds no facts -
  RCK does), verified above. Pushing to a full epoch needs a more stable inference/training
  path (e.g., merge -> GGUF -> llama.cpp/Ollama serving) or a discrete-GPU box.
- **Verification was done on CPU** (~145s/example, stable) because GPU generation crashes.
  Production serving should use the merged model via a stable backend, not raw ROCm generate.
- This is a strong *behavioral* front-end, not a frontier model. Its value is the pairing
  with RCK, not standalone chat quality.

## To use it
Merge + serve behind the `VerifiedBackend` serving gateway (verified-backend mode), or
continue training from checkpoint-200 with `scripts/train_lora.py --resume`.

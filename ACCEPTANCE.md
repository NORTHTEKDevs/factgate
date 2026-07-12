# FACTGATE — Machine-Checkable Acceptance Criteria (autonomous run contract)
# Every phase gate below is a command + threshold. No phase advances on claims.

## Phase A (CPU, this machine)
- A2 gate:        pytest tests/test_gate.py -q            -> 100% pass, >=50 cases, all 4 verdicts exercised
- A3 kb:          python -m factgate.datagen.build_kb     -> snapshot reload <5s; recall@1 == 100% on 1k sample
- A4 dataset:     python -m factgate.datagen.run_all      -> sft_v0.jsonl >= 900k examples; schema-validated;
                                                             audit sample 100 rendered w/o error
- A5 extractor-data: round-trip extract(render(t))==t     -> >=97% on 5k sample
- A6 harness:     python -m factgate.bench.run --target ollama:qwen2.5:14b --smoke
                                                          -> all 5 splits produce contract JSONs
- A7 serving:     pytest hyperion tests (serving subset)  -> >=234 passed incl new VerifiedBackend tests

## Phase B (GPU-dependent, tiered by probe results)
- B0 probe:       python -m factgate.probes.backends      -> JSON of {rocm, directml, vulkan_llamacpp, cpu}
                                                             each with a measured 0.5B forward+backward tok/s or FAIL
- B1 extractor:   extraction F1 >= 0.95, recall >= 0.97 on held-out (backend = best from B0;
                   fallback: rule-based extractor with measured recall reported honestly)
- B2 generator:   EITHER fine-tuned model (if B0 tok/s makes >=7B SFT <=120h) with hallucination-rate
                   (model-alone, unknowable split) <= 50% of base, OR prompted tool-use baseline;
                   whichever wins on harness ships
- B4 e2e:         curl gated /v1/chat/completions          -> gated response w/ provenance block;
                   gate p95 overhead < 20%; 0 silent contradictions on 500-prompt smoke
- B5 guarantee:   python -m factgate.bench.redteam (5k)    -> caught-and-blocked == 100% of detected violations;
                   leakage == extractor FN rate, reported with Wilson CI in RESULTS.md

## Global
- Every result -> contract JSON in factgate/results/ + RESULTS.md regen (raincg pattern)
- Checkpoint after every phase: git commit in factgate repo + task list update + memory note
- HYMN GPU validation: SKIPPED on this hardware tier unless B0 proves a torch GPU backend; logged either way

## B0 VERDICT (2026-07-12, measured - probes/backend_probe.json, commit be83b93)
- Training tier: extractor = DirectML fp16 (0.5-1.5B, LoRA or from-scratch tagger fallback);
  generator v1 = PROMPTED tool-use via Ollama (qwen2.5:14b, 16 tok/s GPU) behind VerifiedBackend.
- 7B+ local fine-tune: INFEASIBLE on proven stack (QLoRA unimplemented on DML; ROCm gfx1151
  untested nightlies only, deferred - needs user present for TDR recovery, or rented GPU).
- B2 acceptance re-scoped accordingly: ship whichever generator config wins the harness;
  fine-tuned-generator row remains an open slot with documented unlock paths.

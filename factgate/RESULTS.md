# FactGate Bench Results

## Methodology & caveats

- **knowable_heldout** coverage-exclusion is based ONLY on `meta.source_triple` in
  `data/sft_full.jsonl` (the field the `qa` generator stamps onto its own examples).
  Facts that appear incidentally inside `chains`/`corrupt`/`idk` examples but were
  never a `qa` `source_triple` are NOT excluded -- a narrower, cheaper leakage guard
  than a full-text scan, disclosed here rather than silently assumed complete.
- **Question phrasing**: `knowable_heldout` always uses the LAST forward-direction
  template variant per relation (`factgate/bench/splits.py::_eval_forward_template`),
  distinct from `qa.py`'s training-time `random.choice` over all variants. This is a
  belt-and-suspenders phrasing-diversity precaution on top of the primary safeguard
  (fact-level `source_triple` exclusion), not itself a leakage barrier.
- **Gate-side metrics** (`gated_hallucination_rate`) run every extracted `[kb:s/r/o]`
  citation AND every executed `<kb_q>` tool-round's top candidate through
  `factgate.gate.verify_claim` against the same eval-90k KB the target queried live.
  A run with 0 extractable claims (e.g. a target that ignores the tool protocol
  entirely) reports `n/a` rather than a misleading 0%.
- **PROMPTED baselines** (`ollama:<model>` targets) are few-shot tool-use prompting,
  NOT fine-tuned models -- see B2 in `ACCEPTANCE.md` for the fine-tuned comparison
  this harness is also designed to serve, once available.

## adversarial_nearmiss

| Target | Accuracy (95% CI) | n | Gated hallucination rate | Evidence |
|---|---|---|---|---|
| ollama:llama3.2:3b | 0.00% (0.00%-39.03%) | 6 | 100.00% | measured-fresh |

- primary 'Accuracy' column for **adversarial_nearmiss** measures: correction rate -- negates the false premise AND cites the KB's real answer.

## knowable_heldout

| Target | Accuracy (95% CI) | n | Gated hallucination rate | Evidence |
|---|---|---|---|---|
| ollama:llama3.2:3b | 10.00% (1.79%-40.42%) | 10 | 100.00% | measured-fresh |

- primary 'Accuracy' column for **knowable_heldout** measures: answer accuracy (canonicalized object match or [kb:] citation match).

## unknowable

| Target | Accuracy (95% CI) | n | Gated hallucination rate | Evidence |
|---|---|---|---|---|
| ollama:llama3.2:3b | 60.00% (38.66%-78.12%) | 20 | 100.00% | measured-fresh |

- primary 'Accuracy' column for **unknowable** measures: abstention rate -- correct behavior is abstention (IDK marker or [kb:idk]).
- ollama:llama3.2:3b: confident-wrong rate = 40.00% (non-abstaining answers on facts absent from the KB by construction -- the harness's core hallucination-surface measurement).

## compositional

_deferred -- rain/raincg's bench suite (raincg/bench) already covers compositional generalization (SCAN/COGS/PCFG-SET); no FactGate-specific compositional split built yet._

## general_canary

_deferred -- MMLU download skipped per task scope. Placeholder split file exists at data/eval/general_canary.jsonl (empty); wire up an MMLU subset loader before this split is meaningful._

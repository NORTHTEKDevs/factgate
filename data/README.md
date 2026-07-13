# data/ — provenance and regeneration

## `kb_train_90k.jsonl` (tracked, ~5MB)

90k `(subject, relation, object)` triples derived from
[ConceptNet 5.7](https://conceptnet.io/) (CC-BY-SA 4.0), scale-filtered and
reshaped into the flat JSONL triple format RCK ingests. This is the fact
store the gate is measured against in `RESULTS.md`; it is small enough to
track directly so the results are reproducible without any external
download step.

To regenerate it (or build a larger/smaller KB) from a fresh ConceptNet
snapshot, use RCK's ConceptNet ingestion tooling
([github.com/NORTHTEKDevs/rck](https://github.com/NORTHTEKDevs/rck)) to
produce a `conceptnet_scale_*.jsonl` source file, then split and load it:

```bash
export FACTGATE_KB_SOURCE_JSONL=/path/to/conceptnet_scale_100k.jsonl
python scripts/holdout_split.py --holdout-size 10000 --seed 0
python -m factgate.datagen.build_kb --input data/kb_train.jsonl
```

## `sft_v0.jsonl` (tracked, <1MB)

A small smoke-scale SFT dataset (generated via `factgate.datagen.run_all`)
kept as a worked example of the generator's output schema. The full-scale
corpus (`data/sft_full.jsonl`, 53k examples) used for the actual LoRA run
is regenerable but not tracked (see `.gitignore`) — regenerate with:

```bash
python -m factgate.datagen.run_all --kb-limit 90000
```

## Everything else in `data/`

`data/eval/*.jsonl` + `*.stats.json`, `*.counts.json`, and
`holdout_unknowable_10k.jsonl` are small evaluation/holdout artifacts kept
for reproducibility of the numbers in `RESULTS.md`. Large regenerable
artifacts (`sft_full.jsonl`, `extractor_pairs.jsonl`, per-run `logs_*.txt`,
`kb_snapshot*/`) are gitignored — regenerate via the scripts in `scripts/`
and `factgate/datagen/`.

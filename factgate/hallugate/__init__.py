"""HalluGate-Bench: end-to-end, source-grounded hallucination gating.

Distinct from the `factgate.bench` splits, which are ConceptNet-derived and adjudicate
against a pre-stored KB. That substrate was measured (see docs/HALLUGATE.md) to be
undecidable for free prose: a model's answer is only credited when it happens to name
ConceptNet's specific hypernym, so "hallucinated" and "chose a different valid parent
category" are indistinguishable.

HalluGate instead supplies the source per example (RAGTruth, MIT) and builds an ephemeral
knowledge base from that source. The verdict therefore stays a deterministic,
parameter-free lookup, while only the extraction step uses a learned model.
"""

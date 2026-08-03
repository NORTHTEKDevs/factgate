# Changelog

## 0.3.0 — 2026-08-03

The release in which the project moved from adversarial hunting to machine-checked proofs,
and then to supervised field pilots. Highlights; the full defect-by-defect history with
measured reproductions is in `docs/HALLUGATE.md`.

### Guarantees
- **Proof suite with mutation gate** (`scripts/harden.py`, 19 checks, CI): an
  exact-rational oracle over the value grammar (every MATCH provably equal, every DIFFER
  provably unequal), a config proof (no author declaration can make the gate verify an
  unequal value without `lint()` refusing the fact set), construction-based tests for the
  residue and conditional paths, and a hostile-model extraction test. Twelve mutants
  disable one defence each; a surviving mutant fails the build.
- **Residue admission is fail-closed**: a claim's extra wording must be positively
  recognised as a modifier phrase; unrecognised constructions are held. Replaced a
  negation-word blacklist that admitted whatever it had not been taught.

### Measured
- Leak rate **0/366** adversarial trials across 15 domains (11 authored blind);
  over-block 16%, of which the gate itself is strict in 3.3%.
- Routing coverage **99%** across ten domains (arc: 99 on a narrow sample → 85 honest →
  99 earned; the corrections are documented, not replaced).
- **Supervised pilots**: 110 answers across four documents, three authored by first-time
  users of the public guide (all clean on their first check run); 64 VERIFIED, every one
  human-confirmed; **0 trust breaches**. `docs/PILOT.md`.

### Fixed (selection; ~40 defects, each with its reproduction in docs/HALLUGATE.md)
- Leaks: unit aliases across scales/dimensions/unknown units (mcg→mg, mL→mg, fl oz→oz,
  F→C), an unconditional default answering a conditional slot, ratio/underflow parsing,
  scientific notation reinterpreting text identifiers.
- False blocks: number-before-unit comparison (5 g vs 5000 mg), typography (NFC/NFD,
  curly quotes, hyphen-joined compounds), decimal points as clause boundaries, faithful
  multi-variant answers blocked on conditional slots.
- Robustness: ReDoS from fact sets and from claims, non-idempotent normalisation,
  fingerprint recomputed per verdict (100× slowdown), the mutation harness corrupting the
  working tree it was checking.

### Earlier (0.1.x–0.2.x)
RCK structured-triple backend (now optional research code; the domain gate does not need
it — asserted by `scripts/acceptance.py`), the bounded-domain design, and the original
0% false-VERIFIED measurement on 3,000 pre-parsed triples.

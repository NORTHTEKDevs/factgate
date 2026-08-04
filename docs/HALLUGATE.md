# HalluGate-Bench: a measured negative result

Goal: extend FACTGATE's gate from structured triples to free model prose, and publish an
honest end-to-end leak rate paired with an over-block rate.

Outcome: **the extension does not work as designed, and the reason is structural.** This
document records what was measured, because the negative result constrains the
architecture more usefully than another round of tuning would.

Every number here comes from an executed command. Artifacts are in `results/`.

---

## 1. The existing 0% is a gate-decision-rule number, not a hallucination rate

`scripts/measure_guarantee.py` calls `verify_claim()` on pre-parsed `(s, r, o)` triples
read from JSONL. No generation step, no LLM. Re-running it reproduces the committed
artifact bit-for-bit (246s, seed 11). It measures the gate's false-accept rate. It does
not measure any model's hallucination rate. README and RESULTS.md now say so at the
point of claim rather than 90 lines later.

## 2. The prior extractor silently corrupts rather than missing

`factgate/datagen/templates.py::extract_triples` compiles `{s}`/`{o}` to `(?P<s>.+?)`,
anchored `^...$`. Measured on natural phrasings:

| input | extracted |
|---|---|
| `dog is a type of mammal.` | `('dog','isa','mammal')` correct |
| `A dog is a type of mammal.` | `('A dog','isa','mammal')` corrupted |
| `Yes, dog is a type of mammal.` | `('Yes, dog','isa','mammal')` corrupted |
| `Dogs are mammals.` | `None` |

`canonical_entity` does not repair this (`'A dog' -> 'a_dog'`), so the gate returns
OUT_OF_KB, which is indistinguishable from an honest "not in KB". The failure is
invisible in the metrics rather than counted as a leak.

## 3. ConceptNet cannot adjudicate free prose

Free-text extraction recall, non-circular (live model answers; gold from the eval split,
never from the text):

| | llama3.2:3b | qwen2.5:14b |
|---|---|---|
| valid JSON | 89% | 100% |
| subject match | 78% | 94% |
| subject + relation | 67% | 94% (CI 72-99%) |
| + object match | 33% | 50% (CI 28-72%) |

Extraction is viable and scales with model size. But **every** object-level failure was a
correct extraction at a different taxonomy level:

```
gold portacaval_shunt/isa/shunt  -> extracted isa "surgical procedure"
gold chinese_cork_oak/isa/oak    -> extracted isa "tree", isa "plant"
gold cocked_hat/isa/hat          -> extracted isa "headwear", form_of "hat"
```

None is a hallucination. ConceptNet's choice of parent category is arbitrary among many
valid ones, so "hallucinated" and "chose a different valid hypernym" are not separable.
Eligibility (model states the gold object at all) was 20% on the 14B and 22% on the 3B —
flat across a 5x size change, so it is a property of the substrate, not the model.

## 4. Free-text sources cannot bind a parameter-free verdict

RAGTruth (MIT, 17,790 responses, 14,289 word-level spans) fixes the ground-truth problem:
both classes are present and mechanically separable. The pipeline built here extracts
triples from the source into an ephemeral RCK KB, extracts triples from the response, and
classifies each sentence fail-closed (`factgate/hallugate/`).

Pilot, 25 QA examples, `results/hallugate_pilot_QA.json`:

```
LEAK RATE        0/12  =   0%   CI95 [0%, 24%]
OVER-BLOCK RATE  110/110 = 100%  CI95 [97%, 100%]
```

Degenerate: the gate holds everything. Reporting leak rate alone would have shown a
flattering "0% leak", which is precisely why the two rates are always reported together.

Root cause, measured directly. Two independent extractions of the same document by the
same model with the same prompt:

```
source   ('alaska technicians', 'average_pay', '$23.70 per hour')
response ('automotive technicians in Alaska', 'have_highest_average_pay',
          '$23.70 per hour or $49,400 per year')

relation vocabulary overlap: 0 of 17
subject overlap:             1 of 17
```

Same fact, no shared surface form. Exact symbolic matching cannot succeed. Switching from
ConceptNet's closed relation set to an open vocabulary did not help (both runs: 100%
over-block) because the problem is surface-form divergence, not vocabulary coverage.

## 5. The conclusion

A verdict layer with **no learned parameters** requires a canonical symbolic vocabulary on
both sides of the comparison. Free-text sources do not have one, and recovering it
requires semantic matching, which requires a model — which is the exact property the gate
exists to avoid.

So the parameter-free guarantee is achievable over a **curated KB in a bounded domain**,
and not over arbitrary source documents. This is a real constraint on the design, not a
gap to be closed with a better prompt. Approaches that ground free text against free text
(MiniCheck, AlignScore, Vectara HHEM) all use a learned entailment model for exactly this
reason.

## 6. The architecture that does work: a bounded domain

Acting on section 5, `factgate/domain/` declares entities and relations up front and turns
extraction into **linking** (choose from a closed list, or return null) rather than
generation. A null link is detectable, and detectable failures can be held.

Benchmark (`scripts/run_domain_bench.py`, 12 declared facts, llama3.2:3b). Both conditions
are real model prose: FAITHFUL answers are generated from the protocol, CORRUPTED ones by
instructing the model to state a specific wrong value.

```
extractor qwen2.5:14b                       (reproduced twice, identical)
LEAK RATE        0/24  =   0%   CI95 [0%, 14%]   wrong value reached the user as VERIFIED
OVER-BLOCK RATE  1/12  =   8%   CI95 [1%, 35%]   correct value failed to verify

fact-clustered (the honest interval -- trials from one fact are correlated):
  leak        0/12 facts   CI95 [0%, 24%]
  over-block  1/12 facts   CI95 [1%, 35%]

faithful   {VERIFIED: 11, HELD: 1}
corrupted  {BLOCK: 24}          <- every wrong value blocked outright, none merely held
excluded   0
```

This is not the degenerate case of section 4. The gate discriminates on both sides: it
blocks **all 24** wrong values with an explicit contradiction verdict, and verifies 11 of
12 correct ones.

**Two caveats that matter more than the headline:**

1. **The number is extractor-dependent.** The same harness on `llama3.2:3b` gives 64%
   over-block (CI [35%, 85%]) with the leak rate still 0/25. The gate is unchanged; only
   the model reading the prose differs. Quote the extractor with the number, always.
2. **The qualifier list was tuned on this domain's own failures**, which is tuning on the
   test set. `value_qualifiers` and the per-relation `question` phrasings were derived by
   inspecting which faithful answers were being held. A fresh domain starts without them
   and should expect a materially worse first number until its own vocabulary is declared.
   The leak rate is not affected by this (qualifiers cannot turn a wrong value into a
   matching one -- `20 mg/kg PO` still blocks), but the over-block figure is optimistic.

The remaining single over-block is a genuine extraction miss: asked for the fever
threshold, the model answered "38 degrees celsius or above is considered a fever, but the
protocol does not specify a specific fever value" -- self-contradictory, and correctly held.

### Does it generalise? A blind test on a fresh domain

The 8% above was measured on a domain whose vocabulary had been tuned against its own
failures, so it could not answer the question the design actually raises: **does a domain
author who declares their vocabulary blind get a usable gate?**

`data/domains/lending_demo.json` was written to answer that. Different value space
(percentages, currency, calendar durations rather than weight-based doses), and the
entities, relations, `value_qualifiers` and question phrasings were all authored **before
any run and never revised in response to results**.

| domain | source prose | vocabulary | leak | over-block |
|---|---|---|---|---|
| lending | simple declaratives | **blind** | **0%** (0/24) | **0%** (0/12) |
| lending-hard | realistic terms-sheet | **blind, identical list** | **0%** (0/24) | **33%** (4/12) |
| lending-hard-tuned | realistic terms-sheet | one tuning round | **0%** (0/24) | **25%** (3/12) |
| clinical | realistic protocol | heavily tuned | **0%** (0/24) | **8%** (1/12) |

The first row looks like a triumph and is not. Inspecting the run showed the blind
qualifier list **never fired**: 0 of 12 extracted values needed any normalisation, because
the corpus I had written unconsciously used clean declaratives ("The overdraft fee is 35
dollars") rather than the qualifier-laden prose real documents use. The test passed without
exercising the thing under test.

`lending-hard` is the honest version: the same facts and the **same untouched blind
qualifier list**, with source prose rewritten the way a terms sheet actually reads. Over-block
went to 33%, and the cause was exactly the predicted one -- 3 of the 4 held values carried
trailing qualifiers the blind list did not anticipate:

```
declared "2 percent"    extracted "2 percent of the amount advanced, deducted at closing"
declared "25 dollars"   extracted "25 dollars per statement cycle, or the accrued interest if greater"
declared "35 dollars"   extracted "35 dollars per occurrence"
```

**The conclusion that matters: leak rate was 0% in every configuration** -- two value
spaces, four vocabularies, blind and tuned, clean prose and messy. The safety property
generalises. The coverage cost does not: it is a direct function of how completely the
domain's vocabulary has been declared, and an undeclared qualifier always costs a HELD,
never a leak. That is the trade the design is built to make, now measured rather than
asserted.

Practical reading for anyone adopting this: expect roughly a third of correct values to be
held on day one, and buy that back by declaring qualifiers as you observe them. You are
never trading safety for it.

### What took over-block from 45% to 8%

| change | effect |
|---|---|
| few-shot slot prompt with a worked NONE example | 1/6 -> 3/6 extracted on llama3.2:3b, 4/6 -> 6/6 on qwen2.5:14b, with **0 fabrications on negative-control passages in all four cells** |
| per-relation `question` declared by the domain | "what is the dose of ibuprofen?" instead of "what is the pediatric_dose of ibuprofen?"; the schema-shaped phrasing drew NONE far more often |
| `value_qualifiers` extended (`divided`, `below`, `exceeding`, `a rate of`, ...) | correct answers phrased as "below 92 percent" or "45 mg/kg PO divided every 12 hours" now verify |

Every relaxation is declared by the domain, and each was checked against a negative control
before adoption: none of them lets a wrong value through.

**Open linking had to be replaced by slot filling.** Asked to emit `(s, r, o)` freely, the
model returned a null relation on 5 of 7 failures *even when it had stated the value
correctly*; only one aliasable string ("dose") appeared, so relation aliases would have
fixed 2 of 7. Since the declared (entity, relation) pairs are known up front, the task is
reading comprehension with the slot supplied, not open information extraction
(`link_targeted`). That took over-block from 58% to 50%.

**Two contaminated denominators cost another 5 points.** The harness assumed a FAITHFUL
prompt yields a faithful answer. It does not: asked what fever threshold triggers
escalation, the model answered 39 degrees against a declared 38. The gate correctly
blocked it and the harness scored that correct catch as an over-block. Separately, 10 of 24
CORRUPTED trials contained no wrong value at all because the model refused
("I can't provide medical advice"), yet they sat in the leak denominator. Trials are now
validity-checked: spontaneous wrong values are promoted into the leak denominator (they are
stronger negatives than instructed ones), and trials stating no value are excluded **and
reported**.

### A false BLOCK found in a live run

The first version emitted BLOCK when the model stated the correct dose as `"10 mg/kg PO"`:
the trailing route corrupted the parsed unit, so the gate reported that a **correct** dose
contradicted the protocol. In a dosing context that is a more dangerous failure than
declining to confirm.

The fix was to make comparison three-valued (`factgate/domain/quantity.py`):

| outcome | meaning | verdict |
|---|---|---|
| MATCH | provably the declared value | VERIFIED |
| DIFFER | provably a different value | BLOCK |
| INCOMPARABLE | cannot decide | HELD |

Only a provable difference earns a BLOCK. Collapsing INCOMPARABLE into MATCH would be
worse still: `"10 mg/kg per day"` prefixes identically and means something else. After the
fix, faithful false blocks went to zero with corrupted blocks unchanged at 11.

### Hardening found by an adversarial probe (`tests/test_domain_hardening.py`)

Three real defects, all fixed and regression-tested:

- **Unicode digit homoglyphs verified.** `"٥ mg"` (Arabic-Indic) and `"５ mg"` (fullwidth)
  parsed as `5.0` and returned VERIFIED against a declared `"5 mg"`, because Python's `\d`
  *and* `float()` both accept non-ASCII numerals. Equality in a safety gate must not depend
  on Unicode folding the caller cannot see. Digits are now matched as `[0-9]`, so these
  fail to parse and are held.
- **Alias collisions were silently accepted.** Two entities declaring the same alias made
  resolution arbitrary, which would attach a dose claim to the wrong drug. Now rejected at
  load time.
- **Malformed fact files raised `KeyError`**, so callers could not distinguish bad data
  from a library crash. Now `ValidationError`.

### The deepest failure mode: the extractor can fabricate the declared value

Three fixes to the harness raised the leak sample from 15 trials to 25 and restored dosing
coverage, and a real leak immediately surfaced that the smaller sample had been hiding:

```
passage    "Give acetaminophen 7.5 mg/kg PO every 4 to 6 hours."
extracted  ("acetaminophen", "pediatric_dose", "15 mg/kg")     <- never appears in the text
verdict    VERIFIED
```

The extractor answered with the *declared* value rather than the one the passage stated,
and the gate then verified it correctly, because a gate can only adjudicate the claim it is
handed. **A parameter-free verdict is no protection when the input itself is fabricated.**

Fixed with `value_is_grounded`: an extracted value must occur in the passage it came from
(magnitude for quantities, substring for text). Deterministic, no model. Ungrounded
extractions are dropped, which sends the claim to HELD. Leak returned to 0/25.

This is worth stating plainly for anyone building on the design: the parameter-free
property covers the *verdict*, not the pipeline. Every model-driven step upstream needs
its own deterministic check, or it reintroduces exactly the failure the gate exists to
prevent.

### What made the sample honest

| change | effect |
|---|---|
| corruption prompt reframed from "state that the dose is X" to "rewrite this sentence changing the number" | refusals went 10/24 -> 0; the old prompt asked for medical advice and the model refused, concentrated on drug dosing, starving the exact relation the gate protects |
| domain-declared `value_qualifiers` (route, frequency) and `unit_aliases` (`%` -> `percent`) | correct answers like "10 mg/kg PO every 6 hours" now verify; `per day` is deliberately NOT declared, so it still holds |
| slot filter keeps answers that lead with a number | a >3-word rule was discarding correct verbose answers before the gate saw them: 4 of 5 over-blocks |

Over-block 45% -> 36%. Nothing was stripped or normalised by inference; every relaxation
is something the domain author declared.

### Found by adversarial review, then fixed

An independent multi-agent review (4 dimensions, each finding adversarially verified by a
second agent instructed to refute it) returned BLOCK. Every confirmed finding was
reproduced by hand before being acted on:

| severity | defect | why it mattered |
|---|---|---|
| blocker | `compare_values` read only the **leading** number, so `"5 to 10 mg/kg"` and `"20 mg is wrong, the correct dose is 10 mg/kg"` both BLOCKED against a declared `10 mg/kg` | the same false-BLOCK class the module claimed to have fixed: a correct dose reported as a contradiction |
| blocker | a refusal that slipped the slot-answer blacklist became a "value"; for a `kind: text` fact the text branch returned DIFFER for any mismatch | the gate BLOCKED because the model *declined to answer* |
| major | units were case-folded, so `5 Mg` (megagram) MATCHED a declared `5 mg` | a factor of 10⁹ verified as correct |
| major | `mentioned_entities` missed line-wrapped and hyphenated names (`"oxygen\nsaturation"`, `"fluid-resuscitation"`) | silent coverage hole surfacing as HELD, not as an error |
| major | `states()` matched `"15"` inside `"15.5"` | a spontaneous hallucination scored as a valid faithful trial |
| minor | excluded trials were counted but never persisted | a 10-of-24 exclusion nobody could audit |

Fixes: hold when the declared magnitude appears anywhere in the claim; whitelist value
*shapes* rather than blacklisting refusal phrasings; compare units case-sensitively;
normalise hyphens and line breaks on both sides; tighten the numeric lookahead; and write
excluded trials to `results/domain_gate_bench.excluded.jsonl`. Measured numbers were
unchanged by all of it, which is the point: these were correctness defects that the
headline metric could not see.

## 7. A real business document

Every domain up to here was synthetic and authored for this project. `real_business_doc.json`
is built from a document that was not: a real product-strategy memo (kept private; not vendored),
with the 12 source quotes pulled programmatically from the file so they cannot drift.

| run | leak | over-block |
|---|---|---|
| first attempt | 0% | **100%** (11/11) |
| after currency support | 0% | 58% (7/12) |
| after declaring the document's own vocabulary | 0% | 33% (4/12) |
| after entity aliases + extraction fixes | **0%** (0/19) | **8%** (1/12) |

**The first run was a total collapse, and the cause was a limitation I had already
dismissed.** Every price in a business document is written `$249`, `$120M`. The parser
required the number first, so `parse_quantity("$249")` returned `None`, every comparison
came back INCOMPARABLE, and the gate held 11 of 11 correct values. I had found this two
sections earlier (`$25` rejected at load) and filed it as a minor usability note. On a real
financial document it was fatal. Currency now parses symbol-first, in word form, and with
magnitude suffixes, so `$120M` and `120 million dollars` compare equal.

**I twice reported a leak rate that was my own harness.** Currency magnitude expansion made
the corruption generator emit `3e+08usd`, which the model could not substitute, so it
rewrote a different number and left the value under test correct -- the gate verified it,
correctly, and 8 trials were scored as leaks (a false 28%). After fixing that, 29% again:
the harness's trial-validity check was a *copy* of `value_is_grounded` that had not
received the same surface-form fix, so five correct answers were misread as spontaneous
hallucinations. The copy is deleted. The lasting fix is general: a corrupted trial is valid
only if the wrong value is present **and** the declared value is gone.

**What the remaining gap actually was.** Three of four late over-blocks were entity mention
failures -- I had declared the entities as `mcp boilerplate` while the document says
`MCP Server Boilerplate`, with no aliases at all. Author error, not a library defect. The
last one is `$95-145`: a **range**, which the schema cannot express and the gate correctly
holds. Ranges are pervasive in real pricing documents (`$1,500-3,000`, `$300-500/mo`) and
are the clearest remaining scope limit.

Also fixed en route: slot answers now take the first line (models append commentary after
the value), grounding falls back to the leading quantity (`"$60(download)"` for a passage
reading `"$60 download"`), qualifier patterns may start with a non-word character (`/mo`),
and the transport retries transient failures a bounded number of times -- a single Ollama
timeout had killed an eight-minute run outright.

## 8. Ranges, and a document nobody chose

Two gaps closed together, because the second exposed the first.

**Ranges are now expressible.** Real documents state them constantly ("$1,500-3,000 setup",
"$300-500/mo", "5 to 10 mg/kg"), and a `kind: quantity` relation previously rejected them
at load, so such facts were simply dropped from every domain. The semantics are a
deliberate asymmetry:

| declared | claimed | verdict | why |
|---|---|---|---|
| range | point inside | VERIFIED | the document supports that figure |
| range | point outside | BLOCK | provably unsupported |
| range | the same range | VERIFIED | |
| range | overlapping range | HELD | neither provably same nor different |
| range | disjoint range | BLOCK | |
| **point** | **range containing it** | **HELD** | a range cannot confirm a specific value |

The last row is the safety-relevant one. If a protocol gives one dose and the model answers
with a range, confirming it would let a reader infer the whole span is protocol-supported.

**The document was selected mechanically.** Both earlier real-document tests used files
chosen after browsing the machine -- an operator selection, and exactly the kind that looks
harmless until the result depends on it. `scripts/select_eval_document.py` enumerates
candidates by numeric density and size, then picks by sorting on the SHA-256 of the file
path: deterministic, reproducible, and uncorrelated with content. The operator sees the
document only after it is chosen. It picked a Series-A pitch deck; 19 facts were declared
from it, 8 of them ranges that could not have been declared at all a day earlier.

```
BLIND, mechanically-selected document, qwen2.5:14b
LEAK RATE        0/24  =  0%   CI95 [0%, 14%]     fact-clustered 0/13 [0%, 23%]
OVER-BLOCK RATE  8/18  = 44%   CI95 [25%, 66%]
```

**44% is worse than the 8-33% band previously claimed, and that band was too narrow.** It
was derived from documents chosen by the operator. On a document nobody chose, blind
over-block is 44%. The honest range is **8-44%**, and the low end of it reflects
familiarity with the document, not the library.

### A third false leak, same family as the first two

The first run of this document reported a 3% leak. It was the harness again, in a new form:
corrupting `$4M-$8M` by scaling its first number produced `$8M-$8M`, whose leading value
is $8M -- still **inside** the declared range. The gate verified it correctly; the trial
was mislabeled.

The fix is now a general invariant rather than a third patch: `compare_values(declared,
corrupted)` must equal DIFFER before a trial runs at all. A "corruption" that is not
provably different from the declared value is not a corruption, and counting it as one is
precisely how every false leak in this project got reported.

## 9. Someone else's vocabulary, on a document the author never read

Every measurement to this point shared a flaw: the same person wrote the vocabulary, ran
the benchmark, and interpreted it. Declaring "blind" removes tuning-on-results, but not the
accumulated knowledge of this library's failure modes, which a first-time author does not
have.

That is separable. The document was selected mechanically (§8). The domain was then
declared by an agent given **only** `docs/AUTHORING.md` and the document, explicitly barred
from reading this file, the README, the tests, or any result. Its declaration was measured
unmodified.

```
BLIND vocabulary by a first-time author, mechanically-selected document, qwen2.5:14b
28 facts declared (19 of them ranges), 17 facts reached by the harness

LEAK RATE        0/34  =  0%   CI95 [0%, 10%]    fact-clustered 0/17 [0%, 18%]
OVER-BLOCK RATE 13/28  = 46%   CI95 [30%, 64%]
```

**The leak rate held at 0% on a vocabulary the maintainer did not write, for a document the
maintainer never read.** That is the strongest safety evidence in this project, and the
largest sample: 34 wrong-value trials.

Over-block was 46%, against 44% when the maintainer declared a mechanically-selected
document blind. The two are indistinguishable at this sample size, which is itself the
finding: **the maintainer's accumulated knowledge was worth nothing on an unfamiliar
document.** The 8% figures earlier in this file measure familiarity with the source, not
skill with the library, and should be read that way.

### What the first-time author reported, unprompted

The exercise was also a usability test, and it found real defects that months of
maintainer use had not:

- **The value grammar is enforced at load and was undocumented.** `~$2,000`, `~$100M+` and
  `$5k cloud credit` all raised `ValidationError`, costing the author two debugging cycles
  and forcing information-lossy edits.
- **A genuine asymmetry:** `12 weeks engineering` parsed while `$5k cloud credit` did not.
  Currency was the stricter case for no reason a reader could predict.
- **`$100M+` had no representation.** Parsing it as exactly `$100M` would verify that one
  figure and block every larger one, inverting the document's meaning.
- Two facts were dropped entirely because ratios (`2,000-4,000x cheaper`) had no documented
  shape.

All are fixed: currency accepts trailing words like any quantity, `~` is accepted and
ignored, `X+` parses as an open range, and `docs/AUTHORING.md` now documents the grammar as
a table. Every value the author was forced to mangle or discard now loads.

The lesson generalises past this library: the maintainer could not find these because the
maintainer already knew the answers.

### Attacking the 46%, without touching the vocabulary

The 13 held values split cleanly into library deficiencies and vocabulary gaps. Only the
first kind was fixed; **the author's domain file was not edited**, because tuning it would
have reintroduced exactly the bias the exercise removed.

| held value | cause | whose problem |
|---|---|---|
| 7 empty extractions | entity aliases copied from table cells never matched the model's prose | **library** |
| `12-16 weeks after v0.5` | a claimed RANGE with a trailing clause had no fallback, though points had one | **library** |
| `$100M+`, `$5k cloud credit` | load-time rejection forced lossy declarations (§9) | **library**, already fixed |
| `8% equity` | needs `equity` declared as a qualifier | **author** |
| `AX` vs `Accelerator X` | a missing alias no rule can infer | **author** |

The entity-matching rules added, each from a real declaration that failed:

- separator punctuation and spacing normalise on both sides, so `Agency A / Agency B Grant` matches
  `Agency A/Agency B Grant`
- a **trailing parenthetical** in an alias is a disambiguator the author added to a table
  row (`Pre-seed angels (early stage)`), not part of the name
- a name listing several parties (`Alpha Fund / Beta Fund / Gamma Partners`) matches when
  **all** parts are present, however joined. Requiring all of them is what stops a common
  word like "alpha" resolving on its own -- that case is tested explicitly.

```
same document, same vocabulary, library fixes only
OVER-BLOCK  13/28 = 46%   ->   5/28 = 18%   CI95 [8%, 36%]
LEAK RATE    0/34 =  0%   ->   0/34 =  0%   unchanged
```

Two thirds of what looked like an irreducible coverage cost was the library failing to
recognise names a person would read as identical. The residue is genuine vocabulary work,
and the author's own report said as much before any of this was measured.

**The same fixes moved the §8 pitch-deck domain not at all** -- still 44%, re-measured
rather than assumed. Its holds have a different cause, so the honest over-block range is
still **8-44%**. A fix that halves one document's coverage cost and does nothing for
another is a reminder that "over-block" is not one phenomenon with one remedy.

## 10. Closing the coverage gap: 46% -> 11%

Attacking the remaining holds separated into three kinds of work, and only the middle one
is the library's.

**Library gaps (fixed).** Two more forms the model produces and the parser did not accept:

- ranges written as prose. The document writes `$500K-$2M`; the model writes
  `between $500K and $2M`. Three of one domain's eight holds were this single form, and
  the slot filter discarded it before the gate saw it (five words, no leading digit).
- category errors. Asked "how much is the raise?" on a passage about runway, the extractor
  answered `18 months`. A currency value and a duration are not competing readings of one
  slot; comparing them is meaningless and DIFFER would wrongly imply a contradiction, so
  the pair is now INCOMPARABLE.

**Vocabulary (the author's, by design).** The residue is trailing text the domain has not
declared irrelevant: `$1.50 per customer query` against a declared `$1.50`. The library
must not strip that by inference -- declaring `per day` irrelevant on a per-dose value
would silently make a wrong value verify -- so the decision stays with the author.

**Tooling (new).** `factgate.domain.suggest.suggest_qualifiers` reports the exact trailing
text that caused each hold, as candidates to approve or reject. It only proposes a residue
whose removal actually rescues the claim (a wrong value is not a qualifier problem), and it
flags time or basis wording as risky. Nothing it returns changes a verdict.

Run against the independently-declared domain it proposed three items; two were genuine.

```
independently-declared domain, mechanically-chosen document, qwen2.5:14b

blind, as the author wrote it            46%  (13/28)     leak 0/34
+ library fixes, vocabulary untouched    18%  ( 5/28)     leak 0/34
+ author accepts 2 suggested qualifiers  11%  ( 3/28)     leak 0/34
```

25 of 28 correct values verify, every wrong value is caught, and the path from 46% to 11%
is a documented loop rather than a diagnostic exercise.

The other mechanically-chosen document moved 44% -> 28% on the library fixes alone; its
suggestion loop has not been run. **The honest span is 11-33%**, and the low end now
requires one review pass rather than familiarity with the document.

### Not every hold is an over-block

Pushing the second document's loop surfaced a measurement flaw worth more than the number
it moved. Two of its four remaining holds looked like this:

```
document   system unit cost: ~$1.50/day
model      "The daily unit cost is approximately $1.50 per customer query"
verdict    HELD
```

The magnitude is right and the **basis is not** -- per day against per query. The gate held
a figure the model had rebased. That is the gate being correct, and the over-block metric
counts it as a failure, because the metric assumes a FAITHFUL answer is semantically
identical to the declared fact. When the model rephrases with a different basis, it is not.

The benchmark now reports these separately rather than folding them into one number:

```
NOTE: 2 of 4 holds had the right magnitude with a DIFFERENT basis (the model rebased
the figure); holding those is correct, so the true over-block is lower than the headline.
```

This is also why the two suggestions the tool flagged were **rejected** on review:
declaring `per customer query` irrelevant would have made a per-query figure verify against
a per-day fact. The flag existed for exactly that case, and it earned its place.

### The loop, as an operator would run it

`render_suggestions` is now emitted by the benchmark itself, so the cycle is run, read,
declare, re-run -- no join script, no diagnostic session:

```
suggested value_qualifiers (review before declaring):
  paste into value_qualifiers:
    "to Series B metrics"
  REVIEW CAREFULLY -- these carry time or basis wording, and
  declaring one irrelevant can make a wrong value verify:
      'per customer query' x2  e.g. .../daily_unit_cost -> '$1.50 per customer query'
```

## 11. Property testing found a leak I had introduced

Every bug found up to here came from a hand-written case, which only covers what someone
thought of. `tests/test_invariants.py` generates inputs nobody chose and asserts the
properties that must hold for all of them:

| | invariant |
|---|---|
| I1 | VERIFIED requires equality re-derivable **without** the code under test |
| I2 | BLOCK requires provable difference |
| I3 | total: every input yields a verdict, no exception escapes |
| I4 | MATCH is symmetric except for the documented range/point asymmetry |
| I5 | reflexive |
| I6 | no absence marker ever verifies |

**It broke I1 within its first run**, on a leak I had introduced two sections earlier:

```
declared  US$5,547M+
claimed   US$5,547M+ per query
verdict   VERIFIED
```

The range paths were ignoring unexplained trailing text while the point path rejected it.
`"10 mg/kg PO"` against a declared `"10 mg/kg"` was INCOMPARABLE, but
`"12-16 weeks after v0.5"` against `"12-16 weeks"` was MATCH -- the same shape of input,
opposite verdicts, and the permissive one verified a claim the gate had not fully parsed.

Fixed by making the rule uniform: **unexplained trailing text may still support a DIFFER
(a provably other magnitude is other however it is dressed) but it can never support a
MATCH.** Recovering that coverage means declaring the qualifier, not ignoring the words.

**That cost coverage, and the cost is reported rather than absorbed:** the
independently-declared domain went 11% -> 18% over-block. I would rather pay seven points
than keep an inconsistency the fuzzer classified as a leak.

The fuzzer was also right where I was wrong. My first I4 asserted plain symmetry; it broke
on `"$5,173 million+"` vs `"$41.72B"`. That asymmetry is the design -- a point inside a
declared range is document-supported, a range never confirms a declared point -- so the
invariant was corrected, not the code. A property test that disagrees with the code is not
automatically right about which one to change.

### Campaign results after the fix

```
comparator   160,000 cases (40 seeds x 4,000)    0 violations across I1, I2, I3, I5, I6
gate + link   13,844 cases, hostile inputs        0 raises, 0 non-derivable VERIFIED,
                                                  0 out-of-domain not held
```

Absence of failures here is not proof of correctness. It is a search several orders of
magnitude wider than the example tests, and it found a real leak the example tests missed.

## 12. Can a stranger actually use it?

Every measurement to this point ran inside a venv that already had everything, from the
working tree, with the maintainer's environment. That proves nothing about someone who
clones the repo. `scripts/acceptance.py` builds a **fresh venv**, installs the package the
way the README says, and exercises the documented surface from a directory outside the repo.

It failed on the first run, and one failure was a real public-facing defect:

**Four test modules hard-required `rck`, which is not on PyPI.** A stranger running
`pytest tests/` got four collection ERRORS and zero tests executed -- a repo that looks
broken on arrival. Optional dependencies must skip, not explode. They now use
`pytest.importorskip`, so a fresh clone reports **654 passed, 4 skipped, 0 errors** (as
of v0.3.0; the figure was 292 when this section was first written and grows with the
suite).

(The other failure was in the acceptance script itself: it invoked pytest before installing
it. Worth recording, because a harness that reports a false failure trains you to ignore it.)

```
fresh venv builds                                PASS
pip install -e . succeeds                        PASS
documented API works from outside the repo       PASS
domain gate needs no knowledge-base engine       PASS
README quickstart test passes in the fresh env   PASS
every shipped demo domain loads clean            PASS
full test suite passes in the fresh env          PASS   654 passed, 4 skipped
```

## 13. Soaking the live pipeline

The benchmark measures rates. The property tests check invariants on generated strings.
Neither checks invariants on verdicts produced by a real model reading real prose, which is
the only configuration that will run in production. `scripts/soak.py` walks every domain,
drives live extraction, and asserts on **every live verdict**:

| | invariant |
|---|---|
| S1 | VERIFIED is re-derivable without the gate's own comparison |
| S2 | VERIFIED implies the value occurs in the passage it came from |
| S3 | BLOCK implies a provable difference |
| S4 | an out-of-domain entity is never anything but HELD |
| S5 | no verdict is absent and no call raises |
| S6 | every verdict carries the fact set fingerprint, for audit |

Rates are reported but are deliberately **not** the pass criterion. A slow gate is a tuning
problem; an unsound one is a defect.

First full campaign, all eight domains, production extractor:

```
SOAK  model=qwen2.5:14b  claims adjudicated=142
  verdicts: {VERIFIED: 105, BLOCK: 11, HELD: 26}
  INVARIANTS HELD on every live verdict
```

Re-run after the residue rule and the hardening in section 15, same 142 claims:

```
SOAK  model=qwen2.5:14b  claims adjudicated=142
  verdicts: {VERIFIED: 112, BLOCK: 11, HELD: 19}
  INVARIANTS HELD on every live verdict
```

Seven more correct answers confirmed, seven fewer held, BLOCK unchanged, and every
invariant still holding on every verdict a real model produced. The S1 check was widened
to admit the second route to VERIFIED -- a claim quoted from the fact's own source -- and
re-derives that independently of `residue.py` rather than trusting the rule to audit
itself.

142 verdicts from a real model reading real prose, every one of them re-derivable,
grounded, and fingerprinted. This is the first check in the project that exercises the
configuration production would actually run, rather than a rate on a fixed sample.

## 14. Production hardening

Five changes aimed at business use rather than at the benchmark. None moved the measured
numbers (0% leak, 8% over-block on clinical/qwen2.5:14b), which is the point: they close
failure modes the headline metric cannot see.

**Conditional facts.** Real protocols are conditional -- amoxicillin is 45 mg/kg standard
and 90 mg/kg for otitis media -- and the schema previously rejected that outright
(`conflict: declared as X and Y`), which excluded most real source documents. Facts now
carry `when: {"indication": "otitis media"}` against declared `conditions`.

The safety rule is what happens when the condition is *not* supplied:

| situation | verdict |
|---|---|
| condition supplied, value matches that variant | VERIFIED |
| condition supplied, value matches a different variant | BLOCK |
| **condition NOT supplied, value matches some variant** | **HELD** |
| value matches no variant, conditioned or not | BLOCK |

Confirming "90 mg/kg" with the indication unknown would confirm a 2x overdose in the
standard case, so an unconditioned query on a conditional slot can never verify.

**Adversarial extraction.** `value_is_grounded` closed the accidental case (§6). The
deliberate case is a decoy: plant the declared value somewhere harmless and state a
different operative one, so grounding and the gate are each individually satisfied.

```
"Give acetaminophen 30 mg/kg. (Reference standard: 15 mg/kg.)"
```

`ambiguous_candidates` holds when the passage carries more than one candidate value for
the slot. The first version was too blunt and cost 17 points of coverage by flagging
`"15 mg/kg ... with a maximum daily total of 75 mg/kg"` -- both numbers real and declared,
for different slots. It now excuses competing values that another declared fact for the
same entity accounts for, while still holding on variants of the *same* slot. Coverage
returned to 8% with the guard active.

**Qualifier lint.** `value_qualifiers` is the sharpest footgun in the schema: declaring
`"per day"` irrelevant would let a daily total verify against a per-dose fact. `fs.lint()`
reports as an **error** any qualifier that collapses two distinct declared values to the
same normalised form (provably unsafe), and as a **warning** any qualifier containing
time/rate wording. All shipped domains are error-clean.

**Audit fingerprint.** Every `Verdict` carries `factset_fingerprint`, a stable digest over
facts, conditions, qualifiers and unit aliases -- everything that can change a verdict.
Order-independent, so reordering facts is not a change. Without it you cannot prove after
the fact which fact set produced a given decision.

**Measured cost.** `gate_claim` is ~33 microseconds, about 30,000 verdicts/sec/core, with
no model and no network on the verdict path. Fact-set load is ~0.9 ms. The extraction step
is the expensive one and is a model call.

### Honest limits

- N is small (12 facts, 36 trials). The fact-clustered leak interval is [0%, 24%], so "0%"
  means "no leak observed at this sample size", not "leak-free". Cite the clustered
  interval, not the trial-level one.
- **8% over-block holds only for `qwen2.5:14b` on a domain whose vocabulary has been
  tuned.** `llama3.2:3b` gives 64% on the identical harness. Treat 8% as the best
  observed operating point, not as the library's behaviour.
- Two models, one synthetic domain, authored for this demo and explicitly **not** medical
  guidance. Nothing here has been tested on a real protocol.
- `value_qualifiers` and per-relation `question` phrasings were derived from this domain's
  own observed failures, so the over-block figure is optimistic in the textbook sense.
  It has never been measured on a domain whose vocabulary was declared blind.
- Most negatives are still instructed rather than spontaneous (1 of 25 arose unprompted).
  Instructed corruption is a weaker source of negatives than observing real hallucinations.
- The gate is only as good as the declared fact set. `validate_sources` checks every fact
  traces to a corpus quote, but nothing checks the corpus itself is correct, and
  `value_qualifiers` are trusted as declared -- declaring "per day" irrelevant would
  silently make a per-day dose verify against a per-dose fact.
- One extractor (llama3.2:3b) and one synthetic domain, authored for this demo and
  explicitly **not** medical guidance.
- 45% over-block is still high. It is an extraction problem with a known shape (entity
  mention detection and slot answering), not an architectural one.
- Most negatives are instructed rather than spontaneous. Only 1 of 15 arose unprompted,
  which is a weaker source of negatives than RAGTruth's real annotated spans.
- The gate is only as good as the declared fact set. `validate_sources` checks every fact
  traces to a corpus quote, but nothing checks the corpus itself is correct.

## 15. Cutting the over-block, and the review that rewrote the fix

Over-block sat at 18% (17 of 94 faithful trials). The largest single cause was a correct
number followed by a basis phrase the declaration omitted:

```
declared "35 dollars"   claimed "35 dollars per occurrence"
declared "2 percent"    claimed "2 percent of the amount advanced, deducted at closing"
```

Trailing text cannot simply be ignored -- that was a real leak once ("US$5,547M+" verified
against "US$5,547M+ per query"), and the fix that closed it cost seven points of coverage.
It cannot be accepted either: "$1.50 per day" and "$1.50 per query" are different prices.

### The rule that shipped, and the four that did not

The proposal admitted a claim two ways: contiguous substring of the source, OR every
residue word appearing somewhere in the source. It was sent to an adversarial review
BEFORE implementation. The review returned UNSOUND with five constructed leaks:

| attack | what it proved |
|---|---|
| `intended for adults at 5 mg; contraindicated for pregnant patients` -> `5 mg for pregnant patients` | token membership confirmed a dose for a population the source FORBIDS |
| `billing does not vary per customer or per query` -> `$550 per customer query` | residue assembled from the clause that negates it; token tests have no polarity |
| `The setup fee is 35 dollars. Per-occurrence surcharges may apply` -> `35 dollars per occurrence` | dropping the full stop made it contiguous ACROSS A SENTENCE BOUNDARY |
| `Start at 7 mg/kg and titrate to 14 mg/kg` -> whole phrase | residue containing a quantity is a second VALUE, not a basis |
| `Standard cardholders pay 25 dollars...; premium cardholders pay no late fee` | residue harvested from a different tier's clause |

The third is the one worth dwelling on. The token leg was the obviously risky half; the
contiguity leg was the one I considered self-evidently safe, and it was independently
unsound. A separate review pass then found a sixth, on real shipped data: declared `$100M`
against a source reading `GPT-4 cost ~$100M+.` -- the residue is a bare `+`, which carries
no digit but turns the claim into an open range, silently reversing the documented
guarantee that a range never confirms a point.

What shipped is one leg, not two: the claim must appear contiguously inside the clause of
the fact's own source sentence that states the declared value, with no negation in that
clause and no quantity in the residue. Every attack above is now a test in
`tests/test_residue.py`.

### Result

| domain | leak | over-block | was |
|---|---|---|---|
| consumer lending, hard | 0/24 | 0/12 | 33% |
| consumer lending, hard + tuned | 0/24 | 0/12 | 25% |
| consumer lending | 0/24 | 0/12 | 0% |
| clinical dosing | 0/24 | 1/12 | 8% |
| business document A | 0/24 | 4/18 | 22% |
| business document B | 0/34 | 5/28 | 18% |
| **total** | **0/154** | **10/94 = 11%** | 18% |

Four other defects surfaced on the way, each measured before it was fixed:

- **A fact set was a denial-of-service payload.** `value_qualifiers` went straight into
  `re.compile`, so `(a+)+b` took 4.84s on a 26-character value and doubled per character.
  Rejected by shape at load; real patterns like `every \d+ (?:to \d+ )?hours?` still work.
- **`entities={"drug": "alias"}` was accepted in silence.** Python iterates the string, so
  the drug gained the aliases `a,c,e,l,m,o,p,r,t` and `gate_claim(fs, "a", "dose",
  "15 mg/kg")` returned VERIFIED.
- **The unit was never grounded.** `value_is_grounded("500 zorkmids", "...500 dollars...")`
  was True because only the number was looked for.
- **Qualifier stripping left `"2 percent ,"`** -- a floating comma that stopped the value
  parsing, so an identical claim verified in one domain and was held in a sister domain
  that declared more qualifiers.

### What the remaining 10 actually are

Every run now classifies its own holds, blind to the verdict: 3 are the gate correctly
refusing an invented basis, 4 never reached the gate (missing entity aliases), and 3 are
genuine -- two of those being the local model emitting a Russian word mid-answer. The
headline rate is deliberately not adjusted by this breakdown; a metric that moved the
number it explains would be marking its own homework.

## 16. What five unseen genres found

Every number up to this point came from eight domains, three of them private. That is a
small sample to call a coverage cost from, and the categories it surfaced were the
categories those eight happened to contain. Five more were authored blind -- insurance,
freight, clinical lab, SaaS contract, construction -- by writers who were told not to read
the gate's source, and they ship with the repository so the rows are reproducible.

They found five defects in one afternoon that eight domains had never touched.

**A unit alias corrupted the value it was meant to normalise.** Aliases were applied by
blind substring replacement, so a rate sheet declaring `{"mi": "miles"}` turned the claim
`$2.85 per mile` into `$2.85 per miles le`. The claim was character-for-character identical
to the declared value and was HELD. The same sheet's `{"hr": "hour"}` would have made
`threshold` into `thoursesold`.

**Normalisation was applied to one side only.** The gate compared the RAW declared value
against the NORMALISED claim, so a lab sheet declaring `{"K/uL": "thousand per microliter"}`
compared `20 K/uL` against `20 thousand per microliter` and held an identical string. Three
of seven holds on that domain were this. The alias corruption above was the same asymmetry
wearing a different hat.

**Every conditional fact was invisible to the extractor.** `link_targeted` skipped any slot
where `lookup()` returned `None`, and `lookup()` returns `None` for a conditional slot
precisely because no context was supplied. A declared, documented, tested feature was never
exercised end to end in any domain.

**And behind that gap, a false BLOCK.** With conditional slots finally being extracted, a
hemoglobin range declared separately for males and females met the model's faithful answer
`13.5-17.5 g/dL and 12.0-15.5 g/dL` and returned:

```
BLOCK  matches none of the 2 declared values for 'hemoglobin'/'reference_range'
```

The conditional path blocked whenever no variant MATCHED, which folds INCOMPARABLE into
DIFFER -- the one collapse the three-valued design exists to prevent, and which the primary
path has always been careful about. Telling a clinician that a correct reference range
contradicts the protocol is worse than any number of holds. BLOCK now requires that every
variant provably differs.

The lesson is not that conditional extraction was risky. The coverage gap was HIDING the
safety bug, not preventing it: for as long as those claims were silently dropped, nothing
could observe that the verdict behind them was wrong.

**My own negation list cost coverage for no safety.** It listed contrast markers --
"other", "but", "however", "rather" -- alongside genuine negation, so a contract reading
"upon 60 days written notice to the other party" was held because of the word "other".
Clause scoping, not a word list, is what stops a residue being harvested from another
clause; all five constructed leak tests still pass with the shorter list.

### Result

| domain | leak | over-block |
|---|---|---|
| consumer lending, three variants | 0/72 | 0/36 |
| clinical dosing | 0/24 | 1/12 |
| construction bid schedule (blind) | 0/21 | 2/15 |
| freight rate sheet (blind) | 0/32 | 2/15 |
| commercial property policy (blind) | 0/18 | 2/14 |
| clinical lab reference ranges (blind) | 0/29 | 4/16 |
| SaaS master agreement (blind) | 0/30 | 5/15 |
| **total** | **0/226** | **16/123 = 13%** |

Of 123 faithful trials, ONE was the gate refusing a claim it should have confirmed. Fifteen
never reached the gate: mostly the model answering with several conditional values at once,
where refusing to pick is correct, and a few missing entity aliases the library now names.

The live soak over all twelve domains, including the three private ones:

```
SOAK  model=qwen2.5:14b  claims adjudicated=232
  verdicts: {VERIFIED: 177, BLOCK: 27, HELD: 28}
  INVARIANTS HELD on every live verdict
```

It did not pass first time. It reported three S1 violations on verdicts where declared and
claimed were BYTE-IDENTICAL, because S1 compared the raw declared value against the
normalised claim -- the same asymmetry that had just been fixed in the gate, in a harness
that had not been updated with it. Worth recording rather than quietly correcting: the
check failed loudly instead of agreeing, which is the only reason the alternative, that the
gate had begun verifying something it should not, could be ruled out by inspection.

## 17. Fifteen domains, and the first leak this project ever measured

Six more blind domains were authored in genres chosen for value shapes the earlier nine
never exercised -- compound units (cents/kWh, $/kW-month), plus-or-minus tolerances,
dual-unit intervals ("4,000 flight hours or 24 months, whichever comes first"), and rates
per unit per period. Fifteen public domains now ship, all reproducible from a clone.

They found nine defects, and one of them ended a streak.

### A leak, on the fourteenth domain

A payroll sheet declared a threshold TWICE: $200,000 unconditionally, and $250,000 when
filing status is married filing jointly. Queried with no filing status, the gate returned

```
VERIFIED  claimed '$200,000'   "linked claim matches the declared fact"
```

`all()` over an empty condition set is True, so the unconditional default matched every
query and `lookup()` returned it -- silently picking a variant, which its own docstring
says it must never do:

> Returning None for "ambiguous" is deliberate ... a conditional slot queried without its
> condition can never verify. Picking a variant would confirm an otitis-media dose in a
> standard-indication context.

The code had violated that contract for every slot mixing a default with conditional
variants. On a tax threshold the confirmed value is at least a real figure from the
document; in the dosing case the docstring warns about, it confirms the standard dose for
a patient who may qualify for the conditional one. Found by the benchmark's
spontaneous-hallucination arm, which measures what a model does unprompted rather than
what the harness told it to do.

### Two false BLOCKs in the comparator

```
declared "5 g"   claimed "5000 mg"   ->  DIFFER, and the gate BLOCKED
```

5000 mg is 5 g. The number was tested before the unit, so different units were called a
provable contradiction on the strength of their digits. `"5 g"` against `"5000 zz"` went
the same way, and nothing here knows what zz is. No conversion table was added: it would
need a tolerance to survive float arithmetic, and a tolerance is the tuned parameter this
verdict layer exists without.

Separately, one text value CONTAINING another was treated as a contradiction, so
`"450 inch-pounds"` was reported as contradicting a declared
`"450 inch-pounds (50.8 newton-metres)"`.

### A decimal point was a sentence boundary

Clause scoping split the source on a bare `.`, so

```
"...require the potency to fall within 95.0 to 105.0 percent of label claim..."
became   "...within 95"  |  "0 to 105"  |  "0 percent of label claim..."
```

No clause contained the declared value, so no value carrying a decimal could ever be
residue-matched -- most of pharmaceutical, clinical-chemistry, nutrition, utility-rate and
tax data. Nine domains never showed it because the genres that make it visible had not
been written yet.

### And three more

A **quadratic blowup reachable from a CLAIM** rather than from a fact set: a digit run cost
303 ms at 2,048 characters and one measurement reported 15.5 seconds for a single call.
Capped, now 0.77 ms at 200,000 characters.

A **unit alias could equate two different units**: an author writing `{"mcg": "mg"}` made
every microgram claim normalise onto a milligram fact, and the gate VERIFIED a 1000-fold
dosing error with lint clean.

The **value-shape filter rejected values longer than three words**, while an aviation
schedule declares a nine-word interval. Nine holds were this, and they were invisible -- no
claim reached the gate, so no suggestion could be made about them either.

### What it costs

Every value-shape and safety fix in this section came from genres that did not exist in the
corpus a day earlier. The rate of finding is not falling, and that is the honest headline:
the method works, and it is not finished.

## 18. What replaced the hunt

Five rounds of adversarial review found defects one at a time:

| round | leaks found | where |
|---|---|---|
| 1 | 0 | -- |
| 2 | 1 | conditional default; eleven domains never surfaced it |
| 3 | 1 | inside round 2's fix |
| 4 | 4 | three in code less than a day old |
| 5 | 1 | inside a fix from that morning |

The rate did not fall, and by round 4 most new leaks were inside the previous round's
fixes. Each notation added to reduce over-block created leak surface faster than review
closed it. That is not a process that terminates.

What replaced it is a set of proofs a machine re-runs in eighty seconds, over the four
places a verdict is actually decided:

  `compare_values`     an exact-rational oracle sharing no code with the implementation
  author configuration either lint refuses the fact set, or every VERIFIED is oracle-equal
  residue and lookup   construction-based cases, where the answer follows from the assembly
  extraction           the pipeline emits no claim its own guards reject, whatever the
                       model says -- driven by a hostile scripted model

And a mutation gate, which is the part that makes the rest mean anything. `harden.py`
breaks the code ten ways and requires each proof to notice. A surviving mutant fails the
build, because it marks a defence nothing tests.

That rule earned itself immediately. The config proof's FIRST mutation run caught **zero**,
which is how two defects in the oracle itself were found while it was passing green: it
answered DONT_KNOW for every differing unit, so it could never prove "15 mg" and "15 mcg"
unequal -- precisely the class it existed to cover -- and it read "5 mg" as magnitude "m"
plus unit "g". A proof that cannot fail manufactures confidence instead of providing it.

The proofs then found what five rounds of review had not: a false BLOCK on `"0.5"` against
`"1/2"`, on their first run; and a coverage hole where a word broken across a line as
`aceta-
minophen` matched nothing, making every fact about that drug silently
unextractable.

## 19. The blind round: what a fresh attacker found that the proofs did not

The proof suite raised a question it could not answer itself: has the method converged, or
has the hunt merely stopped? A proof catches regressions of KNOWN defects; it says nothing
about defects nobody has thought of. So the code was handed to reviewers who were given the
product claim and the source but DELIBERATELY NO HISTORY of what had already been fixed, and
asked to attack from first principles.

They found two, both reproduced independently, both the categories that matter:

  LEAK -- temperature was absent from the unit-dimension table. {"F": "C"} passed lint clean
  and "100 F" VERIFIED against a declared "100 C", values 62 degrees apart. The same class as
  the mcg/mg and fl-oz/oz leaks, in a dimension the table simply did not cover.

  FALSE BLOCK -- a declared "Board Certified" reported "board-certified" as a contradiction.
  The typography fold normalised dash glyphs but never treated a hyphen between letters as a
  word joiner.

Both are fixed, and both now live in the value-grammar oracle and the mutation set. But the
finding itself is the important result: five hardened rounds and a machine-checked proof
suite did NOT make the code leak-proof against a fresh perspective. The honest claim is
narrower than "provably correct" and stronger than "we stopped finding bugs": every defect,
once found, cannot silently return -- and a blind attacker is how you find the ones the
proofs do not yet cover. The general backstop added here (two short unit tokens with no
abbreviation relationship are two different units) is an attempt to catch the NEXT untabulated
dimension before a reviewer does, not instead of one.

## Reproduce

```bash
# the negative result (sections 3-4)
python scripts/fetch_ragtruth.py                              # 36MB, MIT
python scripts/run_hallugate_pilot.py --n 25 --task QA        # ~3 min, needs Ollama

# the architecture that works (section 6)
python scripts/run_domain_bench.py --model llama3.2:3b        # ~4 min, needs Ollama

python -m pytest tests/ -q                                    # 214 tests, no network
```

The harness had the same disease it was built to cure: version one installed from the
WORKING TREE, which still contains gitignored private domains that no downloader receives,
so a green run said nothing about whether the published tree was complete. It now exports
`git archive HEAD` -- byte for byte what a clone gets -- and asserts no private corpus
reached it. 9/9 checks pass against the published tree.

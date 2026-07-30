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
`pytest.importorskip`, so a fresh clone reports **292 passed, 4 skipped, 0 errors**.

(The other failure was in the acceptance script itself: it invoked pytest before installing
it. Worth recording, because a harness that reports a false failure trains you to ignore it.)

```
fresh venv builds                                PASS
pip install -e . succeeds                        PASS
documented API works from outside the repo       PASS
domain gate needs no knowledge-base engine       PASS
README quickstart test passes in the fresh env   PASS
every shipped demo domain loads clean            PASS
full test suite passes in the fresh env          PASS   292 passed, 4 skipped
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

## Reproduce

```bash
# the negative result (sections 3-4)
python scripts/fetch_ragtruth.py                              # 36MB, MIT
python scripts/run_hallugate_pilot.py --n 25 --task QA        # ~3 min, needs Ollama

# the architecture that works (section 6)
python scripts/run_domain_bench.py --model llama3.2:3b        # ~4 min, needs Ollama

python -m pytest tests/ -q                                    # 214 tests, no network
```

# Declaring a domain

A domain is a JSON file describing the facts you want a language model held to. The gate
adjudicates a model's claims against it with no learned parameters, so everything it knows
comes from this file.

Load it with `FactSet.from_json(path)`; check it with `fs.validate_sources(corpus)` and
`fs.lint()`.

## Shape

```json
{
  "domain": "dosing",
  "corpus": "the full source text, whitespace-normalised",
  "entities": { "acetaminophen": ["tylenol", "paracetamol"] },
  "relations": {
    "pediatric_dose": {
      "kind": "quantity",
      "description": "the amount to give per dose",
      "question": "what is the dose of {entity}?"
    }
  },
  "conditions": ["indication"],
  "value_qualifiers": ["PO", "IV", "every \\d+ hours?"],
  "unit_aliases": { "%": "percent" },
  "facts": [
    { "s": "acetaminophen", "r": "pediatric_dose", "o": "15 mg/kg",
      "source": "Give acetaminophen 15 mg/kg PO every 4 to 6 hours." }
  ]
}
```

## Fields

**A note first-time authors ask about:** a fact's `o` (the value) does **not** need to be a
literal substring of its `source` — declare it in the grammar below (`"15 mg/kg"`,
`"1-2 days"`) even if the sentence writes it another way. Only the `source` sentence itself
must appear verbatim in the corpus. And a relation's `kind` is fixed per relation name:
every fact using that relation shares it, so a slot cannot be quantity for one fact and
text for another.

**`corpus`** — the source text the facts came from. Every fact's `source` must appear in it
verbatim (whitespace-normalised), or `validate_sources` reports the fact as unquoted. This
is what stops a wrong fact entering the set unnoticed.

**`entities`** — a map from a canonical name to the other ways your document writes it.
Aliases matter more than they look: the gate only asks about an entity it can find named in
the model's answer, so if your document says "MCP Server Boilerplate" and you declare
`mcp boilerplate` with no aliases, nothing will ever link. An alias claimed by two entities
is rejected at load, because resolution would otherwise be arbitrary.

**`relations`** — the properties an entity can have.
- `kind`: `"quantity"` (a number, optionally with a unit or currency, or a range) or
  `"text"` (a short phrase like `oral`).
- `description`: free text; helps a reader, and is shown to the extraction model.
- `question`: how to ask for this slot, with `{entity}` substituted. Phrasing matters
  measurably — `"what is the dose of ibuprofen?"` gets answered where
  `"what is the pediatric_dose of ibuprofen?"` often returns nothing.

**`facts`** — `s` (entity), `r` (relation), `o` (the value), `source` (the quote), and
optionally `when`.

### The value grammar

**A `kind: quantity` value is validated when the file loads.** If it does not parse,
`FactSet.from_dict` raises `ValidationError` and the domain does not load at all — this is
not deferred to `validate_sources` or `lint`. The accepted forms:

| form | examples |
|---|---|
| plain quantity | `15 mg/kg`, `60 breaths per minute`, `92 percent`, `12 weeks engineering` |
| currency, symbol-first | `$199`, `$1,200`, `$150M`, `$5k cloud credit` |
| currency, in words | `150 million dollars`, `199 USD` |
| range | `$1,500-3,000`, `5 to 10 mg/kg`, `15-25%`, `2,000-4,000x cheaper` |
| open range | `$100M+`, `18 months+` — means "at least", and compares that way |
| approximate | `~$2,000`, `~15 mg/kg` — the `~` is accepted and **ignored** |

Trailing descriptive words are allowed on both plain and currency values, and become part
of the unit: `$5k cloud credit` and `$5k` are different values, exactly as `12 weeks
engineering` and `12 weeks` are.

Units are **case-sensitive** (`mg` and `Mg` are milligram and megagram). Non-ASCII numerals
do not parse. A reversed range (`10-5`) is rejected rather than swapped.

`~` is ignored rather than honoured because the gate has no numeric tolerance by design: a
declared `~$2,000` verifies `$2,000` and blocks `$2,100`. If you need slack, declare a
range instead.

**`conditions` + `when`** — for facts that depend on circumstance:

```json
"conditions": ["indication"],
"facts": [
  {"s": "amoxicillin", "r": "dose", "o": "45 mg/kg", "when": {"indication": "standard"},   "source": "..."},
  {"s": "amoxicillin", "r": "dose", "o": "90 mg/kg", "when": {"indication": "otitis media"}, "source": "..."}
]
```

A conditional slot **cannot be verified without its condition**: `gate_claim(..., context=
{"indication": "otitis media"})`. Without context the gate holds, because confirming one
variant blind would confirm the wrong one in the other case.

**`value_qualifiers`** — trailing text that does **not** change what a value means in your
domain. If your document writes `10 mg/kg PO every 6 hours` and the declared value is
`10 mg/kg`, declare `PO` and `every \d+ hours?` so the extra words are ignored.

Entries are regular expressions; anything that fails to compile is treated as a literal.
This is the sharpest tool in the file — declaring something irrelevant that *is* relevant
(`per day` on a per-dose value) silently makes a wrong value verify. `fs.lint()` reports
qualifiers that collapse two of your own declared values as an **error**, and flags
time/rate wording as a **warning**.

**`unit_aliases`** — equivalent unit spellings, e.g. `{"%": "percent"}`.

**`private`** — set `true` if the corpus is confidential. Benchmark artifacts then go to a
gitignored directory, since they quote the source.

## Verdicts

| verdict | meaning |
|---|---|
| `VERIFIED` | the claim is provably the declared value |
| `BLOCK` | the claim is provably a different value |
| `HELD` | cannot be decided |

The gate is fail-closed: anything not provably matching is `HELD`. Only a provable
difference earns a `BLOCK`, because telling a user their correct value contradicts the
source is worse than declining to confirm it.

## Practical advice

1. Start from the document, not from memory. Copy each `source` quote out of the file.
2. Declare aliases for every way the document names a thing.
3. Run `fs.validate_sources(corpus)` and fix anything unquoted before trusting a result.
4. Run `fs.lint()` and read the warnings.
5. Expect to add `value_qualifiers` after seeing which correct answers get held. An
   undeclared qualifier costs coverage, never safety.

"""Failures that only appear when someone other than the author operates the gate.

A fact set is DATA. In any deployment where the person who writes it is not the person who
runs it -- a customer uploading their own domain, a fact set pulled from a registry, a
config file edited by an analyst -- it is untrusted input, and every defect below was
reachable by writing an ordinary-looking JSON file.

Every case here was reproduced before it was fixed; the docstrings record the measurement.
"""
import time

import pytest

from factgate.domain.factset import FactSet, ValidationError
from factgate.domain.gate import VERIFIED, gate_claim
from factgate.domain.link import respell_from_passage, value_is_grounded

BASE = {"domain": "d", "entities": {"e": ["e"]},
        "relations": {"p": {"kind": "quantity"}},
        "facts": [{"s": "e", "r": "p", "o": "5 mg", "source": "It is 5 mg."}]}


# --------------------------------------------------------------- denial of service
@pytest.mark.parametrize("pattern", ["(a+)+b", "(a*)*b", r"(\d+)+", "([a-z]+)*x",
                                     r"(\w+\s?)+$"])
def test_exponential_qualifier_patterns_are_refused_at_load(pattern):
    """MEASURED before the fix: the qualifier "(a+)+b" took 4.84 SECONDS to normalise a
    26-character value, doubling per added character -- 40 characters is hours. Author
    strings went straight into re.compile, so a fact set was a denial-of-service payload
    delivered as data."""
    with pytest.raises(ValidationError, match="unbounded quantifier"):
        FactSet.from_dict({**BASE, "value_qualifiers": [pattern]})


@pytest.mark.parametrize("pattern", [r"every \d+ (?:to \d+ )?hours?", r"q\d+h", "PO",
                                     "/mo", "per year", "C++", "(ab|cd)+", "a{2,4}"])
def test_legitimate_qualifier_patterns_still_load(pattern):
    """The power is deliberate -- real clinical domains declare frequency patterns -- so
    the fix rejects the dangerous SHAPE, not regexes in general. "(?:to \\d+ )?" is safe
    because `?` matches at most once and cannot compound."""
    FactSet.from_dict({**BASE, "value_qualifiers": [pattern]})


def test_a_rejected_pattern_costs_nothing_to_reject():
    """The check is a linear scan that never executes the pattern."""
    start = time.perf_counter()
    for _ in range(200):
        with pytest.raises(ValidationError):
            FactSet.from_dict({**BASE, "value_qualifiers": ["(a+)+b"]})
    assert time.perf_counter() - start < 1.0


def test_an_absurdly_long_value_does_not_run_qualifier_matching():
    fs = FactSet.from_dict({**BASE, "value_qualifiers": ["PO", "per day", "monthly"]})
    huge = "5 mg " + "x" * 100_000
    start = time.perf_counter()
    assert fs.normalise_value(huge) == huge
    assert time.perf_counter() - start < 0.5


# ------------------------------------------------- structurally wrong fact sets
def test_a_bare_string_where_aliases_belong_is_refused():
    """SILENTLY ACCEPTED before the fix, and a correctness bug rather than an ergonomic
    one: Python iterates the string, so entities={"acetaminophen": "paracetamol"} gave the
    drug the aliases 'a','c','e','l','m','o','p','r','t'. Measured:
    gate_claim(fs, "a", "dose", "15 mg/kg") returned VERIFIED. One stray letter in a
    response resolved to a specific drug."""
    with pytest.raises(ValidationError, match="character by character"):
        FactSet.from_dict({**BASE, "entities": {"acetaminophen": "paracetamol"}})


def test_a_bare_string_where_qualifiers_belong_is_refused():
    """Also silently accepted, compiling the qualifiers \\bP\\b and \\bO\\b and stripping
    standalone letters out of every value in the domain."""
    with pytest.raises(ValidationError, match="character by character"):
        FactSet.from_dict({**BASE, "value_qualifiers": "PO"})


@pytest.mark.parametrize("spec,field", [
    ({"entities": ["e"]}, "entities"),
    ({"relations": {"p": "quantity"}}, "relations"),
    ({"facts": {"s": "e"}}, "facts"),
    ({"unit_aliases": ["x"]}, "unit_aliases"),
    ({"conditions": "tier"}, "conditions"),
    ({"domain": 7}, "domain"),
])
def test_malformed_structure_raises_validation_error_naming_the_field(spec, field):
    """Previously AttributeError/TypeError from inside the library ("'list' object has no
    attribute 'items'"), which a service wrapping this cannot catch by contract or explain
    to whoever sent the request."""
    with pytest.raises(ValidationError, match=field):
        FactSet.from_dict({**BASE, **spec})


def test_a_non_object_fact_set_is_refused():
    with pytest.raises(ValidationError):
        FactSet.from_dict("not a fact set")


def test_empty_spec_still_loads_as_an_empty_gate():
    """The shape check must not break the deliberate empty-gate default, which is safe:
    no facts means everything is HELD."""
    fs = FactSet.from_dict({})
    assert fs.facts == [] and fs.resolve_entity("anything") is None


# ---------------------------------------------------------- caller-facing types
@pytest.mark.parametrize("bad", [5, ["5 mg"], {"v": 1}, b"5 mg", 3.4])
def test_non_string_claim_raises_type_error_not_an_internal_crash(bad):
    """A service passes whatever its JSON contained. These previously surfaced as
    "object of type 'int' has no len()" from inside the quantity parser."""
    fs = FactSet.from_dict(BASE)
    with pytest.raises(TypeError, match="claimed_value must be a string"):
        gate_claim(fs, "e", "p", bad)


def test_non_dict_context_raises_type_error():
    fs = FactSet.from_dict(BASE)
    with pytest.raises(TypeError, match="context must be a dict"):
        gate_claim(fs, "e", "p", "5 mg", context=["tier", "standard"])


# ------------------------------------------------------- grounding the UNIT too
def test_a_fabricated_unit_is_not_grounded_by_a_real_number():
    """Found by probing the guard directly. It checked only that the NUMBER occurred in
    the passage, so any unit at all rode along. Harmless on its own -- a fabricated unit
    does not match the declared one -- but in a domain declaring unit_aliases, a passage
    reading "500 units of stock", an extractor answering "500 usd", and an alias
    usd -> dollars against a declared "500 dollars" VERIFIES a currency the document never
    mentions."""
    passage = "The account holds 500 units of stock."
    assert value_is_grounded("500 zorkmids", passage) is False
    assert value_is_grounded("500 dollars", passage) is False


@pytest.mark.parametrize("value,passage", [
    ("92 percent", "Escalate below 92% saturation."),
    ("150 million dollars", "Acme Corp ($150M ARR)"),
    ("$79", "the Skill Pack is $79 download."),
    ("7.5 mg/kg", "Give acetaminophen 7.5 mg/kg PO now."),
    ("38 degrees celsius", "Record 38 degrees celsius or above."),
    ("12 weeks", "It ships in 12 weeks."),
])
def test_units_spelled_differently_are_still_grounded(value, passage):
    """Requiring the unit is only safe because spelling genuinely varies and the check is
    alias-aware: "$" for dollars, "%" for percent, singular for plural."""
    assert value_is_grounded(value, passage) is True


# -------------------------------------------------- the document's own spelling
def test_a_glued_extraction_is_respelled_from_the_passage():
    """MEASURED, three runs out of three at temperature 0: asked for the minimum balance,
    qwen2.5:14b answers "500 dollarsmonthly" -- the right words with the space gone. The
    qualifier patterns are word-bounded, so nothing could strip "monthly" out of
    "dollarsmonthly", and a correct reading was held over one missing space."""
    passage = ("The savings account requires a minimum balance of 500 dollars monthly "
               "to avoid the service charge.")
    assert respell_from_passage("500 dollarsmonthly", passage) == "500 dollars monthly"
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"savings account": []},
        "relations": {"minimum_balance": {"kind": "quantity"}},
        "value_qualifiers": ["monthly"],
        "facts": [{"s": "savings account", "r": "minimum_balance", "o": "500 dollars",
                   "source": passage}]})
    assert gate_claim(fs, "savings account", "minimum_balance",
                      respell_from_passage("500 dollarsmonthly", passage)).status == VERIFIED


@pytest.mark.parametrize("value", ["500 euros", "900 dollarsmonthly", "50 dollars"])
def test_respelling_never_invents_a_value_the_passage_lacks(value):
    """Only whitespace may differ; every other character must match in order, so this can
    never turn one value into a different one."""
    passage = "The savings account requires a minimum balance of 500 dollars monthly."
    assert respell_from_passage(value, passage) == value


# ---------------------------------------------------------------- authoring aid
def test_lint_flags_a_fact_its_own_source_states_more_precisely():
    """REAL, from the shipped bench data: declared "$100M" against the source "GPT-4 cost
    ~$100M+.". The document says AT LEAST $100M; the declaration says exactly $100M. A
    model that reads the document correctly and answers "$100M+" is then held, because a
    range never confirms a point. The gate is right and the fact set is wrong -- the
    hardest kind of over-block for an author to diagnose, so the library says it."""
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"gpt-4": []},
        "relations": {"training_cost": {"kind": "quantity"}},
        "facts": [{"s": "gpt-4", "r": "training_cost", "o": "$100M",
                   "source": "GPT-4 cost ~$100M+."}]})
    warned = [p for p in fs.lint() if "open-ended range" in p["message"]]
    assert warned and "$100M+" in warned[0]["message"]


def test_lint_does_not_flag_a_declaration_that_already_matches_its_source():
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"gpt-4": []},
        "relations": {"training_cost": {"kind": "quantity"}},
        "facts": [{"s": "gpt-4", "r": "training_cost", "o": "$100M+",
                   "source": "GPT-4 cost ~$100M+."}]})
    assert [p for p in fs.lint() if "open-ended range" in p["message"]] == []


def test_a_currency_amount_with_a_worded_unit_is_still_grounded():
    """REGRESSION, caught by re-running the benchmark rather than by a test. Requiring the
    unit to appear in the passage broke currency values carrying worded units: parse_quantity
    renders "$30 per million tokens" as the single glued unit "usdpermilliontokens", where
    "usd" is a canonical code the document never spells. Two verdicts went from VERIFIED to
    no-claim-at-all. The currency and the wording are now checked separately."""
    assert value_is_grounded(
        "$30 per million tokens",
        "LLM inference costs approximately $30 per million tokens.") is True
    assert value_is_grounded(
        "$0.50 per million tokens",
        "Widget inference costs approximately $0.50 per million tokens.") is True
    # and the hole this whole check exists to close stays closed
    assert value_is_grounded(
        "$30 per billion tokens",
        "LLM inference costs approximately $30 per million tokens.") is False


# ------------------------------------------------------- entity-alias suggestions
def test_alias_suggested_when_the_value_was_stated_but_the_entity_never_matched():
    """MEASURED, the largest remaining category of hold once trailing text was handled:
    asked what equity an accelerator takes, the model answered "Y Combinator typically
    takes around 7% equity" against an entity declared only as "yc". The value was stated
    correctly and the gate never saw it. That reads as an over-block and is really a
    one-line gap in the fact set, so the library names the missing alias."""
    from factgate.domain.suggest import suggest_entity_aliases
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"yc": ["YC"]},
        "relations": {"equity": {"kind": "quantity"}},
        "facts": [{"s": "yc", "r": "equity", "o": "7%",
                   "source": "YC takes 7% equity."}]})
    out = suggest_entity_aliases(
        fs, [("yc", "equity", "Y Combinator typically takes around 7% equity.")])
    assert out and "Y Combinator" in out[0]["candidates"]


def test_alias_suggested_for_a_morphological_near_miss():
    from factgate.domain.suggest import suggest_entity_aliases
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"the raise": ["the round"]},
        "relations": {"amount": {"kind": "quantity"}},
        "facts": [{"s": "the raise", "r": "amount", "o": "$5M-$10M",
                   "source": "The raise is $5M-$10M."}]})
    out = suggest_entity_aliases(
        fs, [("the raise", "amount", "We are raising $5M-$10M to fund growth.")])
    assert out and "raising" in out[0]["candidates"]


def test_no_alias_suggested_when_the_answer_does_not_state_the_value():
    """The guard that keeps this from manufacturing links. If the model never stated the
    value, inventing an alias would only make a non-answer linkable."""
    from factgate.domain.suggest import suggest_entity_aliases
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"yc": ["YC"]},
        "relations": {"equity": {"kind": "quantity"}},
        "facts": [{"s": "yc", "r": "equity", "o": "7%",
                   "source": "YC takes 7% equity."}]})
    assert suggest_entity_aliases(
        fs, [("yc", "equity", "Accelerators vary in what they take.")]) == []


def test_no_alias_suggested_when_an_entity_did_match():
    """If something was matched, the miss is downstream and an alias would not fix it --
    the real case being a model that hedged rather than asserting."""
    from factgate.domain.suggest import suggest_entity_aliases
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"fever": ["febrile threshold"]},
        "relations": {"threshold": {"kind": "quantity"}},
        "facts": [{"s": "fever", "r": "threshold", "o": "38 degrees celsius",
                   "source": "Fever is 38 degrees celsius."}]})
    assert suggest_entity_aliases(
        fs, [("fever", "threshold",
              "A recording of 38 degrees celsius is considered a fever.")]) == []


def test_a_suggested_alias_would_not_collide_with_another_entity():
    """An alias attaches claims to an entity; one that already belongs to a DIFFERENT
    entity would attach them to the wrong one, and the loader refuses such a fact set
    outright. Never propose one."""
    from factgate.domain.suggest import suggest_entity_aliases
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"yc": ["YC"], "Y Combinator Continuity": []},
        "relations": {"equity": {"kind": "quantity"}},
        "facts": [{"s": "yc", "r": "equity", "o": "7%", "source": "YC takes 7% equity."}]})
    for item in suggest_entity_aliases(
            fs, [("yc", "equity", "Y Combinator Continuity takes around 7% equity.")]):
        assert "Y Combinator Continuity" not in item["candidates"]


# --------------------------------------------------- repairing a bad extraction
def test_foreign_words_finds_text_the_passage_never_used():
    """MEASURED, deterministic on three runs at temperature 0: asked for a wall time from
    an English passage reading "12-16 weeks after v0", qwen2.5:14b answers
    "12-16 weeks после v0" -- the Russian for "after". The magnitude is right and the
    wording is not the document's, so the claim was held over a translation artifact."""
    from factgate.domain.link import foreign_words
    passage = "| v0.5 (~1 month training) | ~$2,000 | 12-16 weeks after v0 | Yes |"
    assert foreign_words("12-16 weeks после v0", passage) == ["после"]
    assert foreign_words("12-16 weeks after v0", passage) == []
    assert foreign_words("12-16 weeks", passage) == []


def test_a_foreign_word_triggers_one_repair_attempt(monkeypatch):
    """The repair asks ONCE, showing the model its own answer. Repeating the original
    prompt cannot help: the transport runs at temperature 0, so it returns the same string
    verbatim -- which is why the bug was reproducible three times out of three."""
    import factgate.domain.link as link
    passage = "The wall time for v0.5 is 12-16 weeks after v0."
    calls = []

    def fake(model, prompt, timeout, **kw):
        calls.append(prompt)
        return "12-16 weeks после v0" if len(calls) == 1 else "12-16 weeks after v0"

    monkeypatch.setattr(link, "ollama", fake)
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"v0.5": []},
        "relations": {"wall_time": {"kind": "quantity"}},
        "facts": [{"s": "v0.5", "r": "wall_time", "o": "12-16 weeks",
                   "source": passage}]})
    assert link.link_targeted(passage, fs, "m") == [("v0.5", "wall_time",
                                                     "12-16 weeks after v0")]
    assert len(calls) == 2, "expected exactly one repair attempt"
    assert "Corrected value" in calls[1]


def test_a_repair_that_is_still_not_the_documents_wording_is_dropped(monkeypatch):
    """Fail-closed is preserved. This repairs EXTRACTION and never relaxes adjudication:
    if the second attempt is still not the document's own words, no claim is produced and
    the fact is HELD."""
    import factgate.domain.link as link
    passage = "The wall time for v0.5 is 12-16 weeks after v0."
    monkeypatch.setattr(link, "ollama",
                        lambda *a, **k: "12-16 weeks после v0")
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"v0.5": []},
        "relations": {"wall_time": {"kind": "quantity"}},
        "facts": [{"s": "v0.5", "r": "wall_time", "o": "12-16 weeks",
                   "source": passage}]})
    assert link.link_targeted(passage, fs, "m") == []


def test_the_repair_cannot_smuggle_in_an_ungrounded_value(monkeypatch):
    """A repaired answer is still subject to every guard. A second attempt that returns a
    plausible but absent value must not become a claim."""
    import factgate.domain.link as link
    passage = "The wall time for v0.5 is 12-16 weeks after v0."
    seq = iter(["12-16 weeks после v0", "99-99 weeks after v0"])
    monkeypatch.setattr(link, "ollama", lambda *a, **k: next(seq))
    fs = FactSet.from_dict({
        "domain": "d", "entities": {"v0.5": []},
        "relations": {"wall_time": {"kind": "quantity"}},
        "facts": [{"s": "v0.5", "r": "wall_time", "o": "12-16 weeks",
                   "source": passage}]})
    assert link.link_targeted(passage, fs, "m") == []

"""Tests for the HalluGate-Bench substrate layer (RAGTruth-grounded).

The two logic-bearing pieces are tested here before implementation:
  1. span -> sentence labelling (an off-by-one silently mislabels the gold class)
  2. lenient triple parsing (local models emit malformed JSON ~11-100% of the time,
     measured; a strict parser attributes model capability loss to the parser)
"""
import pytest

from factgate.hallugate.ragtruth import split_sentences, label_sentences
from factgate.hallugate.extract import parse_triples


# --------------------------------------------------------------- sentences
def test_split_sentences_preserves_offsets():
    text = "Alpha is one. Beta is two! Gamma is three?"
    spans = split_sentences(text)
    assert [text[s:e] for s, e in spans] == [
        "Alpha is one.", "Beta is two!", "Gamma is three?"]


def test_split_sentences_handles_no_terminal_punctuation():
    text = "A trailing clause with no period"
    assert [text[s:e] for s, e in split_sentences(text)] == [text]


def test_split_sentences_ignores_empty_text():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_does_not_break_on_decimals():
    """Found against real RAGTruth QA text, not synthesised: money amounts like
    "$23.70" were split after "$23.", fragmenting a gold hallucination span across
    two pseudo-sentences and inflating the sentence count."""
    text = "Pay is $23.70 per hour. That is high."
    assert [text[s:e] for s, e in split_sentences(text)] == [
        "Pay is $23.70 per hour.", "That is high."]


def test_split_sentences_does_not_break_mid_number_before_comma_group():
    text = "It rose to $38,900.50 last year. Then it fell."
    assert len(split_sentences(text)) == 2


def test_label_sentences_marks_only_overlapping_sentence():
    text = "Alpha is one. Beta is two. Gamma is three."
    # span covers "Beta is two" only
    labels = [{"start": text.index("Beta"), "end": text.index("Beta") + 11}]
    out = label_sentences(text, labels)
    assert [h for _, h in out] == [False, True, False]


def test_label_sentences_span_touching_boundary_does_not_bleed():
    """A span ending exactly where the next sentence starts must not mark it."""
    text = "Alpha is one. Beta is two."
    end = text.index(".") + 1          # end of sentence 1, exclusive
    out = label_sentences(text, [{"start": 0, "end": end}])
    assert [h for _, h in out] == [True, False]


def test_label_sentences_no_labels_means_all_faithful():
    text = "Alpha is one. Beta is two."
    assert [h for _, h in label_sentences(text, [])] == [False, False]


def test_label_sentences_multiple_spans():
    text = "Alpha is one. Beta is two. Gamma is three."
    labels = [{"start": 0, "end": 5},
              {"start": text.index("Gamma"), "end": text.index("Gamma") + 5}]
    assert [h for _, h in label_sentences(text, labels)] == [True, False, True]


# ----------------------------------------------------------------- parsing
def test_parse_triples_well_formed_json():
    raw = '[{"s": "robin", "r": "isa", "o": "bird"}]'
    trips, strict = parse_triples(raw)
    assert trips == [("robin", "isa", "bird")]
    assert strict is True


def test_parse_triples_malformed_colon_array_is_recovered():
    """Measured real output of llama3.2:3b: a JSON *array* using colons. Invalid
    JSON, but the s/r/o content is present and must not be discarded."""
    raw = '["s": "Montrachet", "r": "isa", "o": "burgundy"]'
    trips, strict = parse_triples(raw)
    assert trips == [("Montrachet", "isa", "burgundy")]
    assert strict is False


def test_parse_triples_markdown_fenced():
    raw = '```json\n[{"s": "a", "r": "isa", "o": "b"}]\n```'
    trips, _ = parse_triples(raw)
    assert trips == [("a", "isa", "b")]


def test_parse_triples_empty_array():
    assert parse_triples("[]") == ([], True)


def test_parse_triples_garbage_returns_empty_not_crash():
    trips, strict = parse_triples("I could not find any claims.")
    assert trips == []
    assert strict is False


def test_parse_triples_multiple_malformed():
    raw = ('["s": "x", "r": "isa", "o": "y"]\n'
           '["s": "p", "r": "can", "o": "q"]')
    trips, _ = parse_triples(raw)
    assert trips == [("x", "isa", "y"), ("p", "can", "q")]


def test_parse_triples_skips_objects_missing_keys():
    raw = '[{"s": "a", "r": "isa"}, {"s": "b", "r": "isa", "o": "c"}]'
    trips, _ = parse_triples(raw)
    assert trips == [("b", "isa", "c")]

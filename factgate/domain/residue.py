"""Admitting a claim that is the declared value plus text the document itself supplies.

THE PROBLEM. The largest single category of over-block is a correct number followed by a
basis or scope phrase the declaration omitted. Measured, all real:

    declared "35 dollars"  claimed "35 dollars per occurrence"
    declared "2 percent"   claimed "2 percent of the amount advanced, deducted at closing"

Trailing text cannot simply be ignored -- that was a real leak ("US$5,547M+" verified
against "US$5,547M+ per query"), and the fix that closed it cost seven points of coverage.
Trailing text can silently change what a number means: "$1.50 per day" and "$1.50 per
query" are different prices.

THE RULE. The residue is either the document's own words or the model's invention. Admit
MATCH only when the whole claim appears CONTIGUOUSLY inside the clause of the fact's own
source sentence that states the declared value.

WHAT AN ADVERSARIAL REVIEW CHANGED. A first version of this rule allowed two ways to pass:
contiguous substring OR "every residue token appears somewhere in the source". Five
distinct leaks were constructed against it, and they are the reason for every restriction
below:

  * "This drug is intended for adults at 5 mg; it is contraindicated for pregnant
    patients." -- claiming "5 mg for pregnant patients" satisfied the token test using
    words lifted from the clause that FORBIDS it. Token-set membership has no notion of
    which clause a word came from. The token leg is gone; only contiguity survives.

  * "Overage billing does not vary per customer or per query" -- the token test assembled
    "per customer query" out of the very words that negate it. Hence NEGATIONS.

  * "The setup fee is 35 dollars. Per-occurrence monitoring surcharges may apply..." --
    normalising punctuation away made "35 dollars per occurrence" contiguous ACROSS A
    SENTENCE BOUNDARY. Contiguity alone was not safe either. Hence CLAUSE SCOPING.

  * "Start at 7 mg/kg and titrate to 14 mg/kg as tolerated" -- claiming the whole phrase
    is contiguous by construction, but it asserts a second dose, not a basis. Hence the
    residue may not contain a quantity.

  * "Standard cardholders pay 25 dollars...; premium cardholders pay no late fee at all"
    -- residue harvested from a different population's clause. Clause scoping closes it.

Everything here is decidable text containment. No thresholds, no scoring, no model call:
the verdict layer stays parameter-free.
"""
from __future__ import annotations

import re

from factgate.domain.quantity import parse_range

# Sentence and clause terminators. Commas deliberately do NOT split: "35 dollars per
# occurrence, assessed at end of business day" is one clause about one fee, and splitting
# on commas would discard the very residue this exists to admit.
# A DECIMAL POINT is not a sentence boundary. Splitting on a bare "." shredded
# "...within 95.0 to 105.0 percent of label claim..." into "...within 95", "0 to 105" and
# "0 percent of label claim...", so no clause contained the declared value and NO value
# carrying a decimal point could ever be residue-matched. That is most of pharma, clinical
# chemistry, nutrition, utility rates and tax percentages.
#
# Split on sentence punctuation unless it sits between two digits.
_SEGMENT = re.compile(r"(?<!\d)[.;:!?]|(?<=\d)[.;:!?](?!\d)")

# If any of these appears in the clause, the clause is not a plain assertion of the value
# and no residue drawn from it can be trusted. Deliberately over-inclusive: the cost of a
# false hold is a review, the cost of a false verify is the product's whole premise.
# Only words that INVERT or EXCLUDE. The first version also listed contrast markers --
# "other", "but", "however", "rather", "instead", "whereas" -- which cost real coverage for
# no safety: a SaaS contract reading "upon 60 days written notice to the other party" was
# held because of the word "other". Contrast markers are ordinary prose; clause scoping,
# not a word list, is what keeps a residue from being harvested out of a different clause.
_NEGATION = frozenset({
    "not", "n't", "never", "no", "nor", "none", "except", "excepting", "unless",
    "excluding", "excluded", "exclude", "without", "waived", "waive", "exempt",
    "contraindicated", "contraindication",
    # Exemption and reduction words, which say the value does not apply to someone.
    # "The fee is 35 dollars, free for premium members" admitted its whole residue while
    # the identical sentence using "waived" was correctly held -- the list had one word
    # for the concept and not the others.
    "free", "complimentary", "reduced", "discounted", "halved", "prorated", "refunded",
    "forgiven", "abated", "unlimited", "gratis",
})

_WORD = re.compile(r"[a-z']+")
# Unicode-aware. The parser deliberately reads ASCII [0-9] only, but this check has the
# opposite job: it must SEE a second value in the residue, and an ASCII-only pattern is
# blind to one written in another script. "35 dollars, up to ٥٠ dollars for expedited
# processing" was admitted whole while the identical claim using "50" was held.
_DIGIT = re.compile(r"[0-9]|\d", re.UNICODE)


def _norm(s: str) -> str:
    """Casefold and collapse whitespace. Punctuation is deliberately PRESERVED: removing
    it is what let a match run across a sentence boundary in the review above."""
    return " ".join(str(s).casefold().split())


def source_grounded(declared: str, claimed: str, source: str,
                    raw_claimed: str | None = None) -> bool:
    """Does `source` itself state `claimed`, in the clause where it states `declared`?

    Returns False for anything it cannot prove, which the gate turns into HELD.
    """
    if not declared or not (claimed or raw_claimed) or not source:
        return False

    # The caller passes the QUALIFIER-NORMALISED claim, but the source is raw document
    # text that no normalisation has touched. MEASURED: a domain declaring "of the amount
    # advanced" and "deducted at closing" normalised the claim down to "2 percent", which
    # can no longer be found as a residue in its own source sentence -- the same claim
    # verified in a sister domain that happened to declare fewer qualifiers. Compare both
    # spellings against the source; either proving containment is enough.
    for candidate in (claimed, raw_claimed):
        if candidate and _grounded_one(declared, candidate, source, raw_claimed):
            return True
    return False


def _grounded_one(declared: str, claimed: str, source: str,
                  raw_claimed: str | None) -> bool:
    nd, nc = _norm(declared), _norm(claimed)
    if nd == nc or not nc.startswith(nd):
        # Not a residue case at all: either identical (handled by ordinary comparison) or
        # the claim does not begin with the declared value, so this rule has no opinion.
        return False

    residue = nc[len(nd):]
    if _DIGIT.search(residue) or any(ch.isnumeric() for ch in residue):
        # A second number in the residue is a second VALUE being asserted, not a basis for
        # this one. It must go through range/point comparison, which is deliberately
        # stricter, rather than being waved through as scope text.
        return False

    # A residue with no digit in it can still turn a point into a RANGE. Found by review
    # on real shipped data: declared "$100M", source "GPT-4 cost ~$100M+.", claimed
    # "$100M+" -- the residue is a single "+", which carries no digit but makes the claim
    # an open-ended range. Admitting it would silently reverse the documented asymmetry
    # that a range never confirms a declared point (test_i4b). Checked on the RAW strings
    # as well, because normalisation can strip the very symbol that matters.
    for candidate in (claimed, nc, raw_claimed or ""):
        if candidate and parse_range(candidate) is not None and parse_range(declared) is None:
            return False

    for segment in _SEGMENT.split(source):
        seg = _norm(segment)
        if nd not in seg or nc not in seg:
            continue
        if _NEGATION & set(_WORD.findall(seg)):
            continue
        return True
    return False

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


# Words that may BEGIN a fragment of an admissible residue. A residue is text that says
# under what scope or basis the declared value applies -- "per occurrence", "of the amount
# advanced", "monthly" -- and every real one measured across fifteen domains is a
# prepositional phrase, a frequency adverb, a participial phrase, or a route code.
_LEADERS = frozenset({
    "per", "of", "at", "in", "on", "for", "by", "to", "from", "with", "within",
    "before", "after", "during", "across", "under", "over", "up", "each", "every",
})
_ADVERBS = frozenset({
    "monthly", "daily", "weekly", "annually", "quarterly", "hourly", "nightly",
    "yearly", "once", "twice", "thrice", "flat", "total", "combined", "approximately",
    "about", "roughly", "nominal", "maximum", "minimum", "inclusive", "exclusive",
})
# A fragment introducing an ALTERNATIVE or a CONDITION is not a modifier of this value, it
# is a second value or a caveat. "or the accrued interest if greater" adds a competing
# figure; refusing it costs one real case across fifteen domains and is the right trade.
_STRUCTURAL = frozenset({
    "or", "unless", "if", "except", "when", "while", "but", "though", "although",
    "provided", "subject", "whereas", "otherwise", "instead", "rather", "versus",
})
_TOKEN = re.compile(r"[a-z0-9'/%°µ+-]+")
_TOKEN_RAW = re.compile(r"[A-Za-z]+")


def _is_modifier_phrase(residue: str, raw: str | None = None) -> bool:
    """Is this residue a scope or basis phrase, and nothing else?

    Fail-closed by construction: a fragment must be RECOGNISED to be admitted, so a
    construction nobody anticipated is refused instead of allowed. Fragments are split on
    commas because real residues chain them -- "of the amount advanced, deducted at
    closing" is two.
    """
    # ALL-CAPS codes are recognised from the RAW residue, where the capitals survive.
    codes = {w.lower() for w in _TOKEN_RAW.findall(raw or "")
             if w.isupper() and w.isalpha() and 1 < len(w) <= 4}
    text = residue.strip(" ,;:")
    if not text:
        return False
    for fragment in (f.strip() for f in text.split(",")):
        if not fragment:
            continue
        words = _TOKEN.findall(fragment.lower())
        if not words:
            return False
        if _STRUCTURAL & set(words):
            return False
        head = words[0]
        if head in _LEADERS or head in _ADVERBS:
            continue
        # NO PARTICIPLES. "deducted at closing" is a harmless continuation, but so is the
        # shape of "waived for premium members", "restricted to tier 2" and
        # "contraindicated for pregnant patients" -- all of which say the value does NOT
        # apply to someone. The negation list catches the first two of those, and would
        # miss the next verb nobody listed, which is precisely the fail-open behaviour this
        # rule exists to remove.
        #
        # The cost is one real case across fifteen domains: "2 percent of the amount
        # advanced, deducted at closing" is now HELD rather than verified, and
        # suggest_qualifiers proposes the wording so an author can declare it in one line.
        # Trading one verification for a rule that cannot be defeated by an unlisted verb
        # is the right side of that trade.
        # A short ALL-CAPS code is a route or method marker: PO, IV, IM, SC. The case is
        # the signal, so this must be given the residue with its capitals intact -- checking
        # a lowercased string here silently refused every route code, which the clinical
        # domain's own test caught.
        if head in codes:
            continue
        return False
    return True


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
    raw_residue = (raw_claimed or claimed)[-len(residue) - 4:] if residue else ""
    if not _is_modifier_phrase(residue, raw_residue):
        # FAIL-CLOSED. Admission previously required the ABSENCE of a blacklisted negation
        # word, so a word missing from the list admitted the clause -- the one fail-open
        # path in the system, and it leaked once ("free for premium members" passed while
        # the identical sentence using "waived" was held). A blacklist of natural-language
        # negation can never be shown complete.
        #
        # Admission now requires POSITIVE RECOGNITION: the residue must be a modifier
        # phrase, and an unrecognised construction is refused rather than waved through.
        # The negation list is kept below as defence in depth, not as the defence.
        return False
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

    segments = [_norm(x) for x in _SEGMENT.split(source)]
    # If the declared value appears in MORE THAN ONE clause of its own source, there is no
    # single clause that states it and the residue could be harvested from the wrong one:
    #
    #   "The base rate is 4.5 percent. Penalty rates of 4.5 percent per month apply..."
    #
    # would admit "4.5 percent per month" for the BASE rate. Caught by a test written to
    # prove the decimal-point fix had not weakened clause scoping -- it had.
    if sum(1 for seg in segments if nd in seg) != 1:
        return False
    for seg in segments:
        if nd not in seg or nc not in seg:
            continue
        if _NEGATION & set(_WORD.findall(seg)):
            continue
        return True
    return False

"""Doc: one text with lazily computed, cached views.

Rules read everything from a Doc and never re-tokenise. All offsets (sentence
spans, token starts, regex matches on `doc.text` / `doc.lower`) are offsets
into the NORMALISED text `doc.text`; `doc.to_orig(start, end)` maps them back
to the caller's raw input (used for highlighting spans).

Views (each computed once, on first access):
    raw            the input string, untouched
    text           normalised text (quotes/dashes/spaces unified, see text.py)
    lower          text.lower(), same length as text (regexes run on this)
    sent_spans     [(start, end)] of sentences in `text`
    sentences      [str] sentence strings
    n_sentences    int
    sent_tokens    [[str]] tokens per sentence, original case, punctuation included
    sent_tok_starts[[int]] start offset (in `text`) of each token
    sent_words     [[str]] lower-cased word tokens per sentence (no punctuation)
    words          [str]   all lower-cased word tokens
    n_words        int     number of word tokens (the denominator of every rate)
    counts         Counter of `words`
    first_words    [str]   lower-cased first word of each sentence ('' if none)
    punct          Counter of punctuation tokens
    is_address     [bool]  sentence is a vocative only ("Herr talman!")
"""
from collections import Counter

try:
    from functools import cached_property
except ImportError:  # pragma: no cover - Python < 3.8
    cached_property = property

from . import text as T


class Doc:
    def __init__(self, raw):
        self.raw = raw if isinstance(raw, str) else str(raw)

    # -- strings --------------------------------------------------------------
    @cached_property
    def text(self):
        return T.normalise(self.raw)

    @cached_property
    def lower(self):
        return T.lower_same_length(self.text)

    @cached_property
    def _offset_map(self):
        norm, starts, ends = T.normalise_with_map(self.raw)
        if norm != self.text:  # should never happen; tests guard this
            raise AssertionError("normalise() and normalise_with_map() disagree")
        return starts, ends

    def to_orig(self, start, end):
        """Map a [start, end) span of `text` to a span of `raw`."""
        starts, ends = self._offset_map
        if not starts:
            return 0, 0
        if end <= start:
            i = min(start, len(starts) - 1)
            return starts[i], starts[i]
        return starts[start], ends[min(end, len(ends)) - 1]

    # -- sentences ------------------------------------------------------------
    @cached_property
    def sent_spans(self):
        return T.split_sentences(self.text)

    @cached_property
    def sentences(self):
        t = self.text
        return [t[s:e] for s, e in self.sent_spans]

    @property
    def n_sentences(self):
        return len(self.sent_spans)

    # -- tokens ---------------------------------------------------------------
    @cached_property
    def _tok(self):
        t = self.text
        rx = T.TOKEN_RX
        all_toks, all_starts, all_words = [], [], []
        for s, e in self.sent_spans:
            toks, starts = [], []
            for m in rx.finditer(t, s, e):
                toks.append(m.group())
                starts.append(m.start())
            all_toks.append(toks)
            all_starts.append(starts)
            all_words.append([w.lower() for w in toks if w[0].isalnum()])
        return all_toks, all_starts, all_words

    @property
    def sent_tokens(self):
        return self._tok[0]

    @property
    def sent_tok_starts(self):
        return self._tok[1]

    @property
    def sent_words(self):
        return self._tok[2]

    @cached_property
    def words(self):
        return [w for ws in self._tok[2] for w in ws]

    @cached_property
    def n_words(self):
        return sum(len(ws) for ws in self._tok[2])

    @cached_property
    def counts(self):
        return Counter(self.words)

    @cached_property
    def first_words(self):
        return [ws[0] if ws else "" for ws in self._tok[2]]

    @cached_property
    def punct(self):
        return Counter(tok for toks in self._tok[0] for tok in toks if not tok[0].isalnum())

    @cached_property
    def is_address(self):
        low = self.lower
        return [T.ADDRESS_RX.match(low, s, e) is not None for s, e in self.sent_spans]

    # -- convenience ----------------------------------------------------------
    def sentence_index(self, pos):
        """Index of the sentence containing text offset `pos` (or -1)."""
        for i, (s, e) in enumerate(self.sent_spans):
            if s <= pos < e:
                return i
        return -1

    def __repr__(self):
        t = self.text
        return "Doc(%r%s)" % (t[:60], "..." if len(t) > 60 else "")

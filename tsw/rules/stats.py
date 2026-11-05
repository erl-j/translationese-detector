"""stats: simplification / normalisation statistics and Swedish readability.

Baker's "universals" operationalised as per-unit statistics (Laviosa 1998;
Volansky et al. 2015; literature section E and A6/C5/C6). Every statistic is
computed on the same word stream (`word_items`): word tokens containing a
letter, outside vocative-only sentences ("Herr talman!" is turn boilerplate),
ignoring the topic-mask placeholder, so masking does not shift the values.

Rules (all T-leaning on Europarl dev; see each rule's comment for the numbers)

  stats.long_words   share of words with > 6 letters (the LIX long-word term).
                     LIX's other term, words per sentence, is starter.sent_len
                     and is deliberately NOT repeated here (within-class r of the
                     long-word share with mean sentence length: .04 O / .11 T).
  stats.av_nominal   'av' after a definite nominalisation ('genomförandet av')
  stats.av_other     every other 'av' (disjoint: each 'av' counts once)

Length handling (LengthStat, below). A unit is a word (Bernoulli value). The
SD of a text's mean is modelled as sqrt(sd^2/n + tau^2): sd = pooled SD of a
single unit, tau = between-text SD that does not shrink with length (speaker,
topic, session). Short texts behave like independent words (for the av rules
this equals the Poisson RateRule: ~ln(rate_t/rate_o) per hit and the absence
term), long texts saturate at the real class separation instead of growing
with sqrt(n) as an independent-units model does, so no ad-hoc `weight` < 1 is
needed. tau was read off rates.py-style output (text-mean SD at 500 words minus
sd^2/n) and typed in rounded. A statistic that is undefined (no units) gives 0.

Minimum length (MIN_WORDS = 50). With the trn parameters, the class
separation of a text mean at n words is d'(n) = (mu_t - mu_o) / s(n):

    n words        20     50    100    500
    long_words   .094   .14   .18    .25
    av_nominal   .11    .18   .25    .49
    av_other     .15    .23   .31    .57

On a single sentence (~20 words) the long-word share has a spread of about
.12 around a class gap of .01, so a sentence's word lengths say next to
nothing about its class while their noise is shared with other length-related
rules. Below ~50 words all three rules are at d' <= .15 (AUC <= .54), so they
abstain there (contribution 0, "undefined" in describe). The threshold
follows from these trn noise figures, one value for the family; it was not
chosen by scoring candidate thresholds on any split.

Negative results on dev.500.bal (masked; O vs T text means, AUC of the raw
statistic, within-class r with mean sentence length) -- not turned into rules:
  - lexical variety: MATTR-50 .830 vs .828 (AUC .54 for O), MSTTR-50 same,
    OVIX 64.3 vs 63.5 (.53), hapax share in 50-word windows .706 vs .702 (.53):
    the class means are equal at 2 significant digits, so no rule can be
    written; EP translators repeat no more than the edited originals do.
    (mattr_units / ovix_units are kept for rates.py --stat.)
  - share of the 50 most frequent Swedish words: .410 vs .418 (AUC .58), but
    without 'av' and 'för' the shares are identical (.375 vs .375): the whole
    effect is the av rules' (and 'för'), not simplification.
  - mean word length 5.39 vs 5.46 letters (AUC .57) < long-word share (.60),
    and the same signal: long-word share chosen (LIX criterion).
  - very long words (>= 15 letters, compound proxy): .029 vs .027 (AUC .54 for
    O); s-genitive proxy (-ens/-ets/-arnas...) 8.9 vs 9.0 per 1000: no signal.
  - nominalisation suffixes (-ning/-het/-tion/-ande...) 41 vs 45 per 1000:
    weak and already inside long_words / av_nominal.
  - 'för' 18.6 vs 21.3 per 1000 (x1.15), but 'för att' rises as much as the
    rest and other prepositions go both ways (i, på, mot, utan lower in T;
    till, om, med, under higher): preposition-specific interference, left to
    a lexical family.
  - sentence-length dispersion (levelling out predicts T less variable): CV
    .510 vs .517, mean relative deviation .398 vs .406 -- T slightly MORE
    variable, AUC .53; share of sentences > 40 words (AUC .72) and < 8 words
    (.59 for O) correlate with mean sentence length (r .7 / -.5): double
    counting starter.sent_len.

Calibration: trn = the old dev period (sessions <= 2004-06-30), dev.500.bal
(dev.100.bal for tau), topic-masked, 2026-11-05; values typed by hand, <= 2
significant digits. Every value was also checked on the pre-2002-03 part of
trn alone (the comments quote both) and rounds the same, except av_nominal
mu_o (.00126 on all of trn -> .0013; .00124 on the early part -> .0012):
trn is used. val (2004-07..2006) and test are never inspected. The negative
results below are all-trn numbers; the early part of trn gives the same
picture (MATTR .828/.825, OVIX 63.7/63.1, top-48 .382/.381, sentence-length
CV .513/.524).
"""
import math

from .base import Hit, StatRule

FAMILY = "stats"
FAMILY_CAP = 2.0

WINDOW = 50          # MATTR window (words)
LONG = 6             # LIX: a long word has MORE than 6 letters
MIN_WORDS = 50       # every rule abstains on texts with fewer counted words (see "Minimum length")
PLACEHOLDER = "xtopic"  # evals/topicmask.py placeholder (lower-cased): not a word of the text

# Definite nominalisations ("genomförandet", "användningen", "tillämpningen", "definitionen",
# "öppenheten") as the head of an av-phrase: EN "the implementation of X".
_NOMINAL_SUFFIXES = ("ningen", "ningarna", "ningar", "andet", "endet", "heten", "tionen", "ionen")


# ---------------------------------------------------------------------------
# word stream shared by all statistics

def word_items(doc):
    """[(word, start, end, sent_idx)] for the words the statistics look at:
    lower-cased word tokens that contain a letter (numbers are skipped), not
    the topic-mask placeholder, outside vocative-only sentences ("Herr
    talman!" is turn boilerplate, see docs/ENGINE.md). Cached on the Doc."""
    cached = doc.__dict__.get("_stats_items")
    if cached is not None:
        return cached
    out = []
    addr = doc.is_address
    for si, (toks, starts) in enumerate(zip(doc.sent_tokens, doc.sent_tok_starts)):
        if addr[si]:
            continue
        for t, s in zip(toks, starts):
            if not t[0].isalnum():
                continue
            w = t.lower()
            if w == PLACEHOLDER or not any(c.isalpha() for c in w):
                continue
            out.append((w, s, s + len(t), si))
    doc.__dict__["_stats_items"] = out
    return out


def n_letters(w):
    return sum(1 for c in w if c.isalpha())


# ---------------------------------------------------------------------------
# unit functions: doc -> [(value, span)]  (usable with rates.py --stat tsw.rules.stats:<fn>)

def long_word_units(doc):
    """One unit per word: 1.0 if it has more than 6 letters (the LIX long-word
    criterion), else 0.0. Span = the word's sentence, so the rule reports
    per-sentence shares (StatRule groups units by span)."""
    spans = doc.sent_spans
    return [(1.0 if n_letters(w) > LONG else 0.0, spans[si]) for w, _, _, si in word_items(doc)]


def _is_nominal(w):
    return w.endswith(_NOMINAL_SUFFIXES) and len(w) > 7


def av_nominal_units(doc):
    """One unit per word: 1.0 if it is 'av' directly after a definite
    nominalisation ('genomförandet av', 'användningen av'), else 0.0.
    Span = 'nominalisation av' for hits, None (document level) otherwise."""
    items = word_items(doc)
    out = []
    prev = None
    for w, s, e, si in items:
        if w == "av" and prev is not None and prev[3] == si and _is_nominal(prev[0]):
            out.append((1.0, (prev[1], e)))
        else:
            out.append((0.0, None))
        prev = (w, s, e, si)
    return out


def av_other_units(doc):
    """One unit per word: 1.0 if it is 'av' NOT after a definite
    nominalisation (disjoint from av_nominal_units), else 0.0."""
    items = word_items(doc)
    out = []
    prev = None
    for w, s, e, si in items:
        if w == "av" and not (prev is not None and prev[3] == si and _is_nominal(prev[0])):
            out.append((1.0, (s, e)))
        else:
            out.append((0.0, None))
        prev = (w, s, e, si)
    return out


def mattr_units(doc, window=WINDOW):
    """Moving-average type-token ratio (Covington & McFall 2010), cut into
    independent-ish units. With N >= window words there are N - window + 1
    windows (starting at 0 .. N-window); unit j collects the windows that start
    in [j*window, (j+1)*window) and its value is their mean TTR. Only full
    blocks are units, so there are floor(N / window) units (0 when N < window:
    the statistic is undefined and the rule abstains); the windows of the last
    block may reach into the leftover words, so every word counts. The mean of
    the unit values is MATTR up to the weighting of the last block. Span = the
    block's own `window` words. Windows run across sentence boundaries."""
    items = word_items(doc)
    n = len(items)
    if n < window:
        return []
    ws = [w for w, _, _, _ in items]
    counts = {}
    for w in ws[:window]:
        counts[w] = counts.get(w, 0) + 1
    ttr = [len(counts)]
    for i in range(window, n):
        w = ws[i]
        counts[w] = counts.get(w, 0) + 1
        o = ws[i - window]
        c = counts[o] - 1
        if c:
            counts[o] = c
        else:
            del counts[o]
        ttr.append(len(counts))
    # ttr[i] = types in window starting at i, i = 0 .. n - window
    out = []
    for j in range(n // window):
        seg = ttr[j * window:(j + 1) * window]
        out.append((sum(seg) / (len(seg) * float(window)),
                    (items[j * window][1], items[(j + 1) * window - 1][2])))
    return out


def ovix_units(doc):
    """OVIX (Hultman & Westman 1977), Swedish lexical variation index, as one
    unit per text: log(N) / log(2 - log(V) / log(N)) with N words, V types.
    Undefined (no unit) when N < 2 or every word is distinct (V == N)."""
    ws = [w for w, _, _, _ in word_items(doc)]
    n, v = len(ws), len(set(ws))
    if n < 2 or v >= n:
        return []
    return [(math.log(n) / math.log(2.0 - math.log(v) / math.log(n)), None)]


# ---------------------------------------------------------------------------
# rule type: StatRule with a text-level variance floor

class LengthStat(StatRule):
    """StatRule whose class-conditional SD of the text mean is

        s(n) = sqrt(sd^2 / n + tau^2)

    instead of sd / sqrt(n). `sd` is the pooled SD of a single unit, `tau` the
    SD of the text-level mean that does NOT shrink with length (speaker, topic
    and session effects; hand-set from rates.py: tau^2 = observed text-mean
    variance at 500 words - sd^2/n). With pooled sd the LLR is linear:

        LLR = (mu_t - mu_o) / s(n)^2 * (x - (mu_o + mu_t) / 2)

    Short texts: ~ the independent-units (Poisson/binomial-like) LLR. Long
    texts: the LLR stays bounded at the separation the classes really have,
    so no `weight` fudge for dependence is needed. Per-unit shares
    (mu_t - mu_o) / (n s(n)^2) * (v - mid) sum exactly to the total; units with
    the same span are summed (words -> their sentence) before `span_min`.
    n < min_units or no units -> 0 (statistic undefined)."""

    def __init__(self, name, tau=0.0, scale=1.0, unit_label="", **kw):
        super().__init__(name, **kw)
        if self.sd_o != self.sd_t:
            raise ValueError("%s: LengthStat needs a pooled sd" % name)
        if tau < 0:
            raise ValueError("%s: tau must be >= 0" % name)
        self.tau = float(tau)
        self.scale = scale          # display only (e.g. 1000 -> per 1000 words)
        self.unit_label = unit_label

    def _var(self, n):
        return self.sd_o ** 2 / n + self.tau ** 2

    def _slope(self, n):
        return (self.mu_t - self.mu_o) / self._var(n)

    def _llr(self, x, n):
        return self._slope(n) * (x - 0.5 * (self.mu_o + self.mu_t))

    def evidence(self, doc, weight=None):
        vals = self._values(doc)
        n = len(vals)
        if n == 0 or n < self.min_units:
            return [], 0.0
        raw = self._llr(sum(v for v, _ in vals) / n, n)
        c, s = self._scale(raw, weight)
        if c == 0.0:
            return [], 0.0
        per = s * self._slope(n) / n
        mid = 0.5 * (self.mu_o + self.mu_t)
        groups, order = {}, []
        for v, sp in vals:
            if sp is None:
                continue
            sp = tuple(sp)
            if sp not in groups:
                groups[sp] = 0.0
                order.append(sp)
            groups[sp] += per * (v - mid)
        hits, used = [], 0.0
        for sp in order:
            share = groups[sp]
            if abs(share) >= self.span_min:
                hits.append(Hit(sp[0], sp[1], share, self.unit_label or self.description))
                used += share
        return hits, c - used

    def describe(self, doc, contribution=None):
        x, n = self.statistic(doc)
        c = self.contribution(doc) if contribution is None else contribution
        if n == 0 or n < self.min_units:
            return "%s: undefined (%d %ss) -> 0" % (self.description, n, self.unit_name)
        return ("%s = %s over %d %s%s (typical: %s original, %s translated) -> %+.2f toward %s"
                % (self.description, self.fmt % (x * self.scale), n, self.unit_name, "" if n == 1 else "s",
                   self.fmt % (self.mu_o * self.scale), self.fmt % (self.mu_t * self.scale), c,
                   "TRANSLATED" if c > 0 else "ORIGINAL"))

    def meta(self):
        d = super().meta()
        d.update(tau=self.tau)
        return d


RULES = [
    LengthStat(
        "stats.long_words",
        description="share of long words (> 6 letters, the LIX criterion)",
        units=long_word_units, unit_name="word", unit_label="long-word share of sentence",
        # trn before 2002-03-01 (dev.500.bal, masked), words with a letter outside
        # vocatives: O .2764 (sd of text means .038), T .2874 (.039); text-level AUC .57;
        # non-Commission speakers O .2756, T .2829 (Commission speech is wordier in both classes).
        # All of trn (1997-2004): .2776/.2925. Pooled unit sd .45 (Bernoulli). Text-mean sd at 500
        # words .038 vs .020 predicted -> tau .033 (at 100 words .040-.042): tau .035.
        # Within-class r with mean sentence length .01 O / .14 T.
        mu_o=0.28, mu_t=0.29, sd=0.45, tau=0.035, fmt="%.3f",
        direction="T",
        rationale="The long-word component of LIX, without its sentence-length term (starter.sent_len "
                  "already models that). Simplification predicts shorter words in T; on EP dev the "
                  "OPPOSITE holds: translations from English have more 7-13-letter words (nominal "
                  "style: derived abstract nouns and definite forms such as 'genomförandet', "
                  "'tillämpningen', EU terminology), while the very longest words (16+ letters, "
                  "closed compounds) are slightly more frequent in originals. T > O in every year "
                  "1997-2004, but the level rises for both classes from 2000 (drift risk on test).",
        source="Björnsson (1968) LIX; Volansky et al. (2015) mean word length 66% [direction from dev]; "
               "literature C6/E5",
        min_units=MIN_WORDS, cap=1.0),
    LengthStat(
        "stats.av_nominal",
        description="'av' after a definite nominalisation ('genomförandet av')",
        units=av_nominal_units, unit_name="word", unit_label="nominalisation + 'av' (EN 'the X-ing of')",
        # all of trn, masked, per word: O .00126, T .00243 (before 2002-03: .00124/.00238; text
        # AUC .62); non-Commission speakers .0012/.0022 (all trn: 42% vs 64% of texts;
        # T > O every year, O .7-1.5, T 1.7-2.6 per 1000; top-5 speakers 31% O / 24% T of hits).
        # Pooled unit sd .043; text-mean sd .0018/.0026 at 500 vs .0016/.0022 predicted ->
        # tau .0009-.0015: .0012. Within-class r with mean sentence length .10 / .07.
        mu_o=0.0013, mu_t=0.0024, sd=0.043, tau=0.0012, scale=1000, fmt="%.2f/1000w",
        direction="T",
        rationale="Transfer of English 'the implementation/use/introduction of X' as a definite "
                  "nominalisation + av-phrase; native Swedish prefers a verb clause ('att genomföra'), "
                  "a compound or an s-genitive. Disjoint from stats.av_other (every 'av' is counted "
                  "once).",
        source="Nilsson (2004); Danielsson (2003) (av among T keywords); literature A6/C5",
        min_units=MIN_WORDS, span_min=0.05, cap=1.0),
    LengthStat(
        "stats.av_other",
        description="'av' elsewhere (of-periphrasis, by-agent)",
        units=av_other_units, unit_name="word", unit_label="'av'",
        # trn before 2002-03, masked, per word: O .01272, T .01686 (all trn .01257/.01678) (12.7 vs 16.9 per 1000; text AUC .67);
        # non-Commission .0126/.0165. All 'av': 14.0 vs 19.3 per 1000 (ln ratio .32), T > O every
        # year 1997-2004 (full dev), top-5 speakers 28% O / 22% T of hits. Pooled unit sd .12;
        # text-mean sd .0062/.0073 at 500 vs .0050/.0058 predicted -> tau .0037-.0045 (100
        # words: .0047-.0057): tau .0045. Within-class r with mean sentence length .12 / .13.
        mu_o=0.013, mu_t=0.017, sd=0.12, tau=0.0045, scale=1000, fmt="%.1f/1000w",
        direction="T",
        rationale="English of-phrases ('a part of', 'the rights of', 'aspects of our culture') and "
                  "by-agents of be-passives are rendered with 'av' where Swedish prefers compounds, "
                  "s-genitives or s-passives without agent: the preposition is over-represented in "
                  "translations. A lexical rate, but kept here as the periphrasis/compounding "
                  "indicator of the simplification family.",
        source="Nilsson (2004); Danielsson (2003); literature A6",
        min_units=MIN_WORDS, span_min=0.05, cap=1.5),
]

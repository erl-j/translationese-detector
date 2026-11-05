"""Rule types. See docs/ENGINE.md for the maths and how to write a family.

Every rule maps a Doc to a signed contribution in log-odds units
(> 0 = evidence for "translated from English", < 0 = evidence for "original
Swedish"), multiplied by `weight` and clipped to +-`cap`. All numeric
parameters are typed in by hand; nothing here fits anything.

Common metadata (keyword arguments of every rule type)
    name         unique id, e.g. "starter.dem_choice" (unique across ALL families)
    description  one line, human readable ("denna/detta/dessa vs den/det/de här")
    direction    "T" (more in translations) or "O" (more in originals). For a
                 ChoiceRule it is the direction of variant A. Checked against
                 the parameters at construction time (catches sign typos).
    rationale    why the rule should work (mechanism), one or two sentences
    source       citation key from docs/literature.md, or "[exp]"
    cap          max |contribution| of this rule (after weight)
    weight       trust multiplier, default 1.0
    family       normally left None: filled in from the module (FAMILY or module name)
    enabled      False = off unless Engine(include_disabled=True)

Methods used by the engine
    contribution(doc) -> float                    fast path (scoring)
    evidence(doc) -> (hits, residual)             slow path (spans): hits is a
        list of Hit(start, end, share, label) in NORMALISED-text offsets whose
        shares plus `residual` (document-level part, e.g. the Poisson length
        term) add up to contribution(doc)
    describe(doc, contribution) -> str            one human-readable sentence
"""
import math
import re
from collections import Counter, namedtuple

Hit = namedtuple("Hit", "start end share label")

DIRECTIONS = ("T", "O")


def clip(x, cap):
    if x > cap:
        return cap
    if x < -cap:
        return -cap
    return x


def phrase_rx(phrase):
    """'den här' -> 'den\\s+här' (escaped, whitespace-tolerant)."""
    return r"\s+".join(re.escape(p) for p in phrase.split())


def words_rx(words):
    """Regex source matching any of `words` (single words or phrases) as whole
    words, i.e. not inside a longer word or hyphenated compound. Longest first."""
    alts = sorted({w.strip().lower() for w in words if w.strip()}, key=len, reverse=True)
    if not alts:
        raise ValueError("empty word list")
    return r"(?<![\w-])(?:%s)(?![\w-])" % "|".join(phrase_rx(w) for w in alts)


def _as_words(x):
    if isinstance(x, str):
        return x.split("|") if "|" in x else x.split(",") if "," in x else x.split()
    return list(x)


class Matcher:
    """A compiled pattern + which Doc string it runs on.

    pattern: regex source (str), or dict {label: regex} (each hit is labelled by
             the key of the alternative that matched; the key may contain {m}
             for the matched surface), or None when `words` is given.
    words:   list of words/phrases (or "a|b|c"), matched as whole words.
    case:    False (default) -> match on doc.lower (write lower-case regexes);
             True -> match on doc.text (case-sensitive rules).
    """

    def __init__(self, pattern=None, words=None, case=False, flags=0):
        self.labels = None
        if words is not None:
            if pattern is not None:
                raise ValueError("give pattern or words, not both")
            src = words_rx(_as_words(words))
        elif isinstance(pattern, dict):
            parts, self.labels = [], {}
            for i, (label, rx) in enumerate(pattern.items()):
                g = "_m%d" % i
                self.labels[g] = label
                parts.append("(?P<%s>%s)" % (g, rx))
            src = "|".join(parts)
        elif isinstance(pattern, (list, tuple, set, frozenset)):
            src = words_rx(pattern)
        elif isinstance(pattern, str):
            src = pattern
        else:
            raise ValueError("Matcher needs a pattern (str/dict) or words")
        self.source = src
        self.rx = re.compile(src, flags)
        self.case = case

    def target(self, doc):
        return doc.text if self.case else doc.lower

    def count(self, doc):
        n = 0
        for _ in self.rx.finditer(self.target(doc)):
            n += 1
        return n

    def finditer(self, doc):
        """Yield (start, end, label) in normalised-text offsets; label=None if unlabelled."""
        for m in self.rx.finditer(self.target(doc)):
            label = self.labels.get(m.lastgroup) if self.labels else None
            yield m.start(), m.end(), label


def _fmt_label(template, surface):
    if template is None:
        return "'%s'" % surface
    try:
        return template.format(m=surface)
    except (KeyError, IndexError, ValueError):
        return template


def _side(x):
    return "TRANSLATED" if x > 0 else "ORIGINAL"


class Rule:
    """Base class. Subclasses implement _raw(doc) and evidence(doc)."""

    kind = "rule"
    default_cap = 3.0

    def __init__(self, name, description="", direction=None, rationale="", source="[exp]",
                 cap=None, weight=1.0, family=None, enabled=True):
        if not name or not isinstance(name, str):
            raise ValueError("rule needs a name")
        if direction is not None and direction not in DIRECTIONS:
            raise ValueError("%s: direction must be 'T' or 'O', got %r" % (name, direction))
        self.name = name
        self.description = description or name
        self.direction = direction
        self.rationale = rationale
        self.source = source
        self.cap = float(self.default_cap if cap is None else cap)
        self.weight = float(weight)
        self.family = family
        self.enabled = enabled

    def _check_direction(self, implied):
        if self.direction is None:
            self.direction = implied
        elif implied is not None and self.direction != implied:
            raise ValueError("%s: direction=%r but its parameters point to %r"
                             % (self.name, self.direction, implied))

    # -- to implement ---------------------------------------------------------
    def _raw(self, doc):
        """Unweighted, uncapped log-likelihood ratio."""
        raise NotImplementedError

    def contribution(self, doc, weight=None):
        """clip(weight * raw LLR, cap). `weight` overrides self.weight (engine ablations)."""
        raw = self._raw(doc)
        if raw == 0.0:
            return 0.0
        return clip((self.weight if weight is None else weight) * raw, self.cap)

    def evidence(self, doc, weight=None):
        return [], self.contribution(doc, weight)

    def describe(self, doc, contribution=None):
        c = self.contribution(doc) if contribution is None else contribution
        return "%s: %+.2f toward %s" % (self.description, c, _side(c))

    def meta(self):
        return {"name": self.name, "kind": self.kind, "family": self.family,
                "description": self.description, "direction": self.direction,
                "rationale": self.rationale, "source": self.source, "cap": self.cap,
                "weight": self.weight, "enabled": self.enabled}

    def __repr__(self):
        return "<%s %s>" % (type(self).__name__, self.name)

    # helper for evidence(): scale raw per-hit terms the same way contribution()
    # scaled the raw total (weight, then cap).
    def _scale(self, raw, weight=None):
        if raw == 0.0:
            return 0.0, 0.0
        c = clip((self.weight if weight is None else weight) * raw, self.cap)
        return c, c / raw


# ---------------------------------------------------------------------------

class HitRule(Rule):
    """Fixed log-odds per match, capped. For rare, high-precision items (calques).

        contribution = clip(weight * sum of per-hit values, cap)

    Either one `per_hit` for every match of `pattern`/`words`, or a `lexicon`
    {label: (regex, per_hit)} with a value per entry (all of one sign). per_hit
    > 0 points to T, < 0 to O. `label` is a format string for span labels
    ({m} = matched text); with a dict pattern or lexicon the keys are labels.
    """

    kind = "hit"
    default_cap = 3.0

    def __init__(self, name, pattern=None, words=None, per_hit=None, lexicon=None, case=False,
                 label=None, **meta):
        super().__init__(name, **meta)
        self.label = label
        self.per_label = None
        if lexicon is not None:
            if pattern is not None or words is not None or per_hit is not None:
                raise ValueError("%s: give lexicon OR pattern/words + per_hit" % name)
            vals = [float(v) for _, v in lexicon.values()]
            if any(v == 0 for v in vals) or (min(vals) < 0 < max(vals)):
                raise ValueError("%s: lexicon per_hit values must be non-zero and of one sign" % name)
            self.matcher = Matcher({lab: rx for lab, (rx, _) in lexicon.items()}, case=case)
            self.per_label = {g: float(lexicon[lab][1]) for g, lab in self.matcher.labels.items()}
            self.per_hit = vals[0]
            sign = vals[0]
        else:
            if not per_hit:
                raise ValueError("%s: per_hit must be non-zero" % name)
            self.per_hit = float(per_hit)
            self.matcher = Matcher(pattern, words, case)
            sign = self.per_hit
        self._check_direction("T" if sign > 0 else "O")

    def _matches(self, doc):
        """[(start, end, label, per_hit)]"""
        out = []
        for m in self.matcher.rx.finditer(self.matcher.target(doc)):
            if self.per_label is not None:
                out.append((m.start(), m.end(), self.matcher.labels[m.lastgroup], self.per_label[m.lastgroup]))
            else:
                lab = self.matcher.labels.get(m.lastgroup) if self.matcher.labels else None
                out.append((m.start(), m.end(), lab, self.per_hit))
        return out

    def _raw(self, doc):
        if self.per_label is None:
            return self.matcher.count(doc) * self.per_hit
        return sum(v for _, _, _, v in self._matches(doc))

    def evidence(self, doc, weight=None):
        ms = self._matches(doc)
        if not ms:
            return [], 0.0
        c, s = self._scale(sum(v for _, _, _, v in ms), weight)
        hits = [Hit(a, b, s * v, _fmt_label(lab or self.label, doc.text[a:b])) for a, b, lab, v in ms]
        return hits, 0.0

    def describe(self, doc, contribution=None):
        ms = self._matches(doc)
        c = self.contribution(doc) if contribution is None else contribution
        what = Counter(_fmt_label(lab or self.label, doc.text[a:b]) for a, b, lab, _ in ms)
        return "%s: %s -> %+.2f toward %s" % (
            self.description, ", ".join("%s%s" % (k, " x%d" % n if n > 1 else "") for k, n in what.items())
            or "no hits", c, _side(c))

    def meta(self):
        d = super().meta()
        d.update(per_hit=self.per_hit, pattern=self.matcher.source)
        if self.per_label:
            d["lexicon"] = {self.matcher.labels[g]: v for g, v in self.per_label.items()}
        return d


class RateRule(Rule):
    """Poisson rate rule. rate_o / rate_t = expected matches per 1000 words in
    originals / translations (hand-set). With k matches in n words:

        LLR = k * ln(rate_t/rate_o) - n * (rate_t - rate_o) / 1000

    contribution = clip(weight * LLR, cap). k = 0 is evidence too (the second
    term), and it grows with text length. `min_words`: below this many words
    the rule abstains (0).
    """

    kind = "rate"
    default_cap = 3.0

    def __init__(self, name, pattern=None, words=None, rate_o=None, rate_t=None, case=False,
                 label=None, min_words=0, **meta):
        super().__init__(name, **meta)
        if not rate_o or not rate_t or rate_o <= 0 or rate_t <= 0:
            raise ValueError("%s: rate_o and rate_t must be > 0" % name)
        if rate_o == rate_t:
            raise ValueError("%s: rate_o == rate_t carries no evidence" % name)
        self.rate_o, self.rate_t = float(rate_o), float(rate_t)
        self.log_ratio = math.log(self.rate_t / self.rate_o)
        self.per_word = (self.rate_t - self.rate_o) / 1000.0
        self.matcher = Matcher(pattern, words, case)
        self.label = label
        self.min_words = min_words
        self._check_direction("T" if rate_t > rate_o else "O")

    def _raw(self, doc):
        n = doc.n_words
        if n == 0 or n < self.min_words:
            return 0.0
        return self.matcher.count(doc) * self.log_ratio - n * self.per_word

    def evidence(self, doc, weight=None):
        n = doc.n_words
        if n == 0 or n < self.min_words:
            return [], 0.0
        ms = list(self.matcher.finditer(doc))
        raw = len(ms) * self.log_ratio - n * self.per_word
        c, s = self._scale(raw, weight)
        hits = [Hit(a, b, s * self.log_ratio, _fmt_label(lab or self.label, doc.text[a:b]))
                for a, b, lab in ms]
        return hits, s * (-n * self.per_word)

    def describe(self, doc, contribution=None):
        n = doc.n_words
        k = self.matcher.count(doc)
        c = self.contribution(doc) if contribution is None else contribution
        return ("%s: %d hit%s in %d words (expected %.2g if original, %.2g if translated) -> %+.2f toward %s"
                % (self.description, k, "" if k == 1 else "s", n, n * self.rate_o / 1000.0,
                   n * self.rate_t / 1000.0, c, _side(c)))

    def meta(self):
        d = super().meta()
        d.update(rate_o=self.rate_o, rate_t=self.rate_t, pattern=self.matcher.source)
        return d


class ChoiceRule(Rule):
    """Binomial choice between two competing variants A and B.

    p_o = P(A | A or B chosen, original), p_t = same in translations.
    With kA, kB occurrences:

        LLR = kA * ln(p_t/p_o) + kB * ln((1-p_t)/(1-p_o))      (0 if kA+kB == 0)

    direction is the direction of variant A ("T" if p_t > p_o).
    Patterns/words for A and B must not overlap (one string, one variant).
    """

    kind = "choice"
    default_cap = 3.0

    def __init__(self, name, a=None, b=None, words_a=None, words_b=None, p_o=None, p_t=None,
                 label_a=None, label_b=None, case=False, **meta):
        super().__init__(name, **meta)
        for p in (p_o, p_t):
            if p is None or not 0.0 < p < 1.0:
                raise ValueError("%s: p_o and p_t must be in (0, 1)" % name)
        if p_o == p_t:
            raise ValueError("%s: p_o == p_t carries no evidence" % name)
        self.p_o, self.p_t = float(p_o), float(p_t)
        self.la = math.log(self.p_t / self.p_o)
        self.lb = math.log((1.0 - self.p_t) / (1.0 - self.p_o))
        self.matcher_a = Matcher(a, words_a, case)
        self.matcher_b = Matcher(b, words_b, case)
        self.label_a = label_a or "A"
        self.label_b = label_b or "B"
        self._check_direction("T" if p_t > p_o else "O")

    def counts(self, doc):
        return self.matcher_a.count(doc), self.matcher_b.count(doc)

    def _raw(self, doc):
        ka, kb = self.counts(doc)
        if ka + kb == 0:
            return 0.0
        return ka * self.la + kb * self.lb

    def evidence(self, doc, weight=None):
        ma = list(self.matcher_a.finditer(doc))
        mb = list(self.matcher_b.finditer(doc))
        if not ma and not mb:
            return [], 0.0
        c, s = self._scale(len(ma) * self.la + len(mb) * self.lb, weight)
        hits = [Hit(x, y, s * self.la, "%s: '%s'" % (lab or self.label_a, doc.text[x:y])) for x, y, lab in ma]
        hits += [Hit(x, y, s * self.lb, "%s: '%s'" % (lab or self.label_b, doc.text[x:y])) for x, y, lab in mb]
        hits.sort()
        return hits, 0.0

    def describe(self, doc, contribution=None):
        ka, kb = self.counts(doc)
        c = self.contribution(doc) if contribution is None else contribution
        return ("%s: %s x%d vs %s x%d (share of %s: %.0f%% in originals, %.0f%% in translations) "
                "-> %+.2f toward %s" % (self.description, self.label_a, ka, self.label_b, kb,
                                         self.label_a, 100 * self.p_o, 100 * self.p_t, c, _side(c)))

    def meta(self):
        d = super().meta()
        d.update(p_o=self.p_o, p_t=self.p_t, pattern_a=self.matcher_a.source,
                 pattern_b=self.matcher_b.source, label_a=self.label_a, label_b=self.label_b)
        return d


def _gauss_logpdf(x, mu, sd):
    z = (x - mu) / sd
    return -0.5 * z * z - math.log(sd)


class StatRule(Rule):
    """Gaussian rule on the mean of per-unit values (e.g. words per sentence).

    units(doc) -> list of (value, (start, end) or None), one per unit (usually
    sentence). Values are clipped to `unit_clip` (lo, hi) if given. With n units
    and mean x:

        LLR = log N(x; mu_t, sd_t/sqrt(n)) - log N(x; mu_o, sd_o/sqrt(n))

    sd_o/sd_t are the SDs of a SINGLE unit's value in each class (hand-set).
    Pass `sd` for a pooled SD: then the LLR is linear in x and equals the sum of
    the per-unit LLRs, so per-sentence span shares are exact (recommended:
    equal SDs avoid the quadratic blow-up of unequal-variance Gaussians on
    outliers). Units whose share is below `span_min` in magnitude are not
    reported as spans; their part goes to the document-level residual.
    """

    kind = "stat"
    default_cap = 2.0

    def __init__(self, name, units=None, mu_o=None, mu_t=None, sd=None, sd_o=None, sd_t=None,
                 unit_clip=None, min_units=1, unit_name="unit", fmt="%.1f", span_min=0.25, **meta):
        super().__init__(name, **meta)
        if units is None or not callable(units):
            raise ValueError("%s: units must be a function doc -> [(value, span)]" % name)
        sd_o = sd if sd_o is None else sd_o
        sd_t = sd if sd_t is None else sd_t
        if mu_o is None or mu_t is None or not sd_o or not sd_t or sd_o <= 0 or sd_t <= 0:
            raise ValueError("%s: need mu_o, mu_t and sd (or sd_o and sd_t) > 0" % name)
        if mu_o == mu_t and sd_o == sd_t:
            raise ValueError("%s: identical class distributions carry no evidence" % name)
        self.units = units
        self.mu_o, self.mu_t = float(mu_o), float(mu_t)
        self.sd_o, self.sd_t = float(sd_o), float(sd_t)
        self.unit_clip = unit_clip
        self.min_units = min_units
        self.unit_name = unit_name
        self.fmt = fmt
        self.span_min = span_min
        self._check_direction(None if mu_t == mu_o else ("T" if mu_t > mu_o else "O"))

    def _values(self, doc):
        vals = self.units(doc)
        if self.unit_clip:
            lo, hi = self.unit_clip
            vals = [(min(max(v, lo), hi), sp) for v, sp in vals]
        return vals

    def _llr(self, x, n):
        r = math.sqrt(n)
        return _gauss_logpdf(x, self.mu_t, self.sd_t / r) - _gauss_logpdf(x, self.mu_o, self.sd_o / r)

    def _raw(self, doc):
        vals = self._values(doc)
        n = len(vals)
        if n == 0 or n < self.min_units:
            return 0.0
        return self._llr(sum(v for v, _ in vals) / n, n)

    def evidence(self, doc, weight=None):
        vals = self._values(doc)
        n = len(vals)
        if n == 0 or n < self.min_units:
            return [], 0.0
        raw = self._llr(sum(v for v, _ in vals) / n, n)
        c, s = self._scale(raw, weight)
        hits, used = [], 0.0
        for v, sp in vals:
            share = s * self._llr(v, 1)
            if sp is not None and abs(share) >= self.span_min:
                hits.append(Hit(sp[0], sp[1], share,
                                "%s %s = %s" % (self.unit_name, self.description, self.fmt % v)))
                used += share
        return hits, c - used

    def statistic(self, doc):
        vals = self._values(doc)
        return (sum(v for v, _ in vals) / len(vals) if vals else float("nan")), len(vals)

    def describe(self, doc, contribution=None):
        x, n = self.statistic(doc)
        c = self.contribution(doc) if contribution is None else contribution
        return ("%s = %s over %d %s%s (typical: %s original, %s translated) -> %+.2f toward %s"
                % (self.description, self.fmt % x, n, self.unit_name, "" if n == 1 else "s",
                   self.fmt % self.mu_o, self.fmt % self.mu_t, c, _side(c)))

    def meta(self):
        d = super().meta()
        d.update(mu_o=self.mu_o, mu_t=self.mu_t, sd_o=self.sd_o, sd_t=self.sd_t, unit_clip=self.unit_clip)
        return d


class PredicateRule(Rule):
    """Escape hatch: arbitrary fn(doc) -> raw LLR (before weight and cap).

    spans_fn(doc) -> [(start, end, raw_share, label)] (optional, normalised-text
    offsets; raw_share in the same units as fn's result, ideally summing to it).
    The engine scales the shares exactly like the total (weight, cap); whatever
    the spans do not account for becomes document-level evidence.
    describe_fn(doc, contribution) -> str (optional).
    """

    kind = "predicate"
    default_cap = 3.0

    def __init__(self, name, fn=None, spans_fn=None, describe_fn=None, **meta):
        super().__init__(name, **meta)
        if fn is None or not callable(fn):
            raise ValueError("%s: fn must be callable" % name)
        if self.direction is None:
            raise ValueError("%s: PredicateRule needs an explicit direction" % name)
        self.fn = fn
        self.spans_fn = spans_fn
        self.describe_fn = describe_fn

    def _raw(self, doc):
        return float(self.fn(doc) or 0.0)

    def evidence(self, doc, weight=None):
        raw = self._raw(doc)
        c, s = self._scale(raw, weight)
        if self.spans_fn is None:
            return [], c
        hits, used = [], 0.0
        for a, b, share, label in self.spans_fn(doc) or ():
            hits.append(Hit(a, b, s * share, label))
            used += s * share
        return hits, c - used

    def describe(self, doc, contribution=None):
        c = self.contribution(doc) if contribution is None else contribution
        if self.describe_fn is not None:
            return self.describe_fn(doc, c)
        return super().describe(doc, c)


# ---------------------------------------------------------------------------
# reusable unit functions for StatRule

def sentence_lengths(doc, skip_address=True):
    """[(words in sentence, sentence span)] for each sentence; vocative-only
    sentences ("Herr talman!") are skipped (boilerplate, see docs/ENGINE.md)."""
    out = []
    addr = doc.is_address if skip_address else None
    for i, ws in enumerate(doc.sent_words):
        if addr is not None and addr[i]:
            continue
        if ws:
            out.append((len(ws), doc.sent_spans[i]))
    return out

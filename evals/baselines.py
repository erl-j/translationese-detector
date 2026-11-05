"""Trivial baselines that exercise the harness end-to-end.

Classifier contract (see evals/README in the top-level README):
    score(text) -> float          > 0 means "translated from English"
    explain(text) -> dict         optional, {rule_name: signed contribution}
    score_and_explain(text)       optional fast path returning (score, dict)

None of these is a real translationese rule; they are reference points.
"""
import hashlib
import re

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ\"“(])")


class Constant:
    """Always the same score. Default +1 = always "translated"."""

    def __init__(self, value=1.0):
        self.value = float(value)

    def score(self, text):
        return self.value


class Random:
    """Uniform random score in [-1, 1), deterministic per (seed, text)."""

    def __init__(self, seed=0):
        self.seed = str(seed)

    def score(self, text):
        h = hashlib.sha256((self.seed + "\x00" + text).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / 2.0 ** 63 - 1.0


def mean_sentence_length(text):
    sents = [s for s in _SENT.split(text) if s.strip()]
    if not sents:
        return 0.0
    return sum(len(s.split()) for s in sents) / len(sents)


class MeanSentLen:
    """Longer sentences => "translated".

    threshold = median mean-sentence-length of dev.100.bal (computed once by
    hand on dev and pasted here; it is a hand-set constant, not a fitted model).
    direction = +1 because on dev translated text has longer sentences on
    average (this is an observation on dev, not an explanation of it).
    """

    DEV_MEDIAN = 19.5  # dev.100.bal median; class means orig 18.8, trans 22.6

    def __init__(self, threshold=DEV_MEDIAN, direction=1.0):
        self.threshold = float(threshold)
        self.direction = float(direction)

    def score_and_explain(self, text):
        c = self.direction * (mean_sentence_length(text) - self.threshold) / 10.0
        return c, {"mean_sent_len": c}

    def score(self, text):
        return self.score_and_explain(text)[0]

    def explain(self, text):
        return self.score_and_explain(text)[1]


class TopicCheat:
    """DELIBERATELY CHEATING sanity check for --mask-topic.

    Votes "original" for Sweden/Swedish mentions and "translated" for
    UK/Ireland/English mentions. Masked metrics should collapse to chance;
    if they do not, the topic mask is leaking.
    """

    RULES = {
        "mentions_sweden": (re.compile(r"(?<![\w-])(sverige|svensk|stockholm|riksdag)", re.I), -1.0),
        "mentions_uk_ie": (re.compile(r"(?<![\w-])(storbritannien|brittisk|britt|england|engelsk|skottland|"
                                      r"wales|irland|irländ|nordirland|london|dublin|labour)", re.I), +1.0),
    }

    def score_and_explain(self, text):
        ex = {}
        for name, (rx, w) in self.RULES.items():
            n = len(rx.findall(text))
            if n:
                ex[name] = w * n
        return sum(ex.values()), ex

    def score(self, text):
        return self.score_and_explain(text)[0]

    def explain(self, text):
        return self.score_and_explain(text)[1]

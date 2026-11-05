"""The expert-system engine: sum of capped rule contributions, per-family caps.

    score = prior + sum over families f of clip(sum of f's rule contributions, +-cap_f)

> 0 means "translated from English". All rules are discovered automatically
from tsw/rules/*.py (see tsw/rules/__init__.py and docs/ENGINE.md).

Harness entry points (evals/run.py loads "module:Name(kw=literal, ...)"):
    tsw.engine:Default  (alias tsw.engine:Expert)        all enabled rules
    tsw.engine:Families(only="wordorder,calques")        only these families
    tsw.engine:Engine(exclude="starter.sent_len")        drop rules or families
    tsw.engine:Engine(family_caps={"stats": 1.0}, weights={"starter.skall_ska": 0.5})
"""
import json
import math

from .doc import Doc
from .rules import discover

DEFAULT_FAMILY_CAP = 4.0
CAP_PREFIX = "cap:"


def _names(x):
    if x is None:
        return None
    if isinstance(x, str):
        return {s.strip() for s in x.split(",") if s.strip()}
    return set(x)


def _clip(x, cap):
    return cap if x > cap else -cap if x < -cap else x


class Engine:
    """Rule engine.

    rules            explicit list of rule objects (default: auto-discovered)
    families         only these families ("a,b" or list)
    exclude          drop these rule names or family names ("a,b" or list)
    family_caps      {family: cap} overrides (else module FAMILY_CAP, else default)
    default_family_cap
    weights          {rule name: weight} overrides (ablation without editing files)
    prior            constant added to the score (log-odds)
    include_disabled also use rules defined with enabled=False
    """

    def __init__(self, rules=None, families=None, exclude=None, family_caps=None,
                 default_family_cap=DEFAULT_FAMILY_CAP, weights=None, prior=0.0,
                 include_disabled=False):
        if rules is None:
            rules, module_caps, _ = discover()
        else:
            module_caps = {}
            for r in rules:
                if r.family is None:
                    r.family = "custom"
        fams, excl = _names(families), _names(exclude) or set()
        sel = []
        for r in rules:
            if not r.enabled and not include_disabled:
                continue
            if fams is not None and r.family not in fams:
                continue
            if r.name in excl or r.family in excl:
                continue
            sel.append(r)
        if fams is not None:
            unknown = fams - {r.family for r in rules}
            if unknown:
                raise ValueError("unknown families: %s" % ", ".join(sorted(unknown)))
        unknown = excl - {r.name for r in rules} - {r.family for r in rules}
        if unknown:
            raise ValueError("exclude: unknown rules/families: %s" % ", ".join(sorted(unknown)))
        self.rules = sel
        self.weights = dict(weights or {})
        unknown = set(self.weights) - {r.name for r in rules}
        if unknown:
            raise ValueError("weights: unknown rules: %s" % ", ".join(sorted(unknown)))
        self.prior = float(prior)
        caps = dict(module_caps)
        caps.update(family_caps or {})
        self.default_family_cap = float(default_family_cap)
        self.family_caps = {r.family: float(caps.get(r.family, self.default_family_cap)) for r in sel}
        # group rules by family once (order preserved)
        self._by_family = {}
        for r in sel:
            self._by_family.setdefault(r.family, []).append(r)

    # -- core -----------------------------------------------------------------
    def _contrib(self, r, doc):
        return r.contribution(doc, self.weights.get(r.name))

    def _run(self, doc):
        """-> (score, {rule: contribution != 0}, {family: (raw, clipped)})"""
        total = self.prior
        contrib, fam = {}, {}
        for f, rules in self._by_family.items():
            s = 0.0
            for r in rules:
                c = self._contrib(r, doc)
                if c:
                    contrib[r.name] = c
                    s += c
            cs = _clip(s, self.family_caps[f])
            fam[f] = (s, cs)
            total += cs
        return total, contrib, fam

    def score(self, text):
        return self._run(Doc(text))[0]

    def score_and_explain(self, text):
        """(score, {rule: contribution}); when a family cap binds, an extra
        entry "cap:<family>" holds the (signed) amount removed, so the values
        always sum to score - prior."""
        total, contrib, fam = self._run(Doc(text))
        out = dict(contrib)
        for f, (s, cs) in fam.items():
            if cs != s:
                out[CAP_PREFIX + f] = cs - s
        return total, out

    def explain(self, text):
        return self.score_and_explain(text)[1]

    # -- explanation for humans -----------------------------------------------
    def _rule_by_name(self):
        return {r.name: r for r in self.rules}

    def verdict(self, text, top=5):
        doc = Doc(text)
        total, contrib, fam = self._run(doc)
        rules = self._rule_by_name()
        prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, total))))
        items = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
        ev_t = [rules[n].describe(doc, c) for n, c in items if c > 0][:top]
        ev_o = [rules[n].describe(doc, c) for n, c in items if c < 0][:top]
        capped = ["family '%s' capped: %+.2f -> %+.2f" % (f, s, cs) for f, (s, cs) in fam.items() if cs != s]
        return {"label": "translated" if total > 0 else "original",
                "score": total, "prob_translated": prob,
                "n_words": doc.n_words, "n_sentences": doc.n_sentences,
                "evidence_translated": ev_t, "evidence_original": ev_o,
                "family_caps_applied": capped}

    def explain_spans(self, text, _doc=None):
        """Span-level evidence. Offsets index the ORIGINAL `text`.

        [{start, end, text, rule, family, direction, contribution_share, label}]
        sorted by start. contribution_share is this span's signed share of its
        rule's contribution (before family caps).
        """
        doc = _doc or Doc(text)
        spans = []
        for r in self.rules:
            hits, _ = self._evidence(r, doc)
            for h in hits:
                if not h.share:
                    continue
                a, b = doc.to_orig(h.start, h.end)
                spans.append({"start": a, "end": b, "text": doc.raw[a:b], "rule": r.name,
                              "family": r.family, "direction": "T" if h.share > 0 else "O",
                              "contribution_share": h.share, "label": h.label})
        spans.sort(key=lambda d: (d["start"], d["end"], d["rule"]))
        return spans

    def _evidence(self, r, doc):
        return r.evidence(doc, self.weights.get(r.name))

    def annotate(self, text, top=5):
        """Everything at once, JSON-serialisable:
        {score, prob, label, rules, families, spans, doc_level, evidence_translated,
         evidence_original}"""
        doc = Doc(text)
        total, contrib, fam = self._run(doc)
        rules = self._rule_by_name()
        prob = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, total))))
        doc_level = []
        for r in self.rules:
            if r.name not in contrib:
                continue
            _, resid = self._evidence(r, doc)
            if abs(resid) > 1e-9:
                doc_level.append({"rule": r.name, "family": r.family,
                                  "direction": "T" if resid > 0 else "O",
                                  "contribution": resid, "label": r.describe(doc, contrib[r.name])})
        for f, (s, cs) in fam.items():
            if cs != s:
                doc_level.append({"rule": CAP_PREFIX + f, "family": f,
                                  "direction": "T" if cs - s > 0 else "O", "contribution": cs - s,
                                  "label": "family '%s' capped at %+.2f (raw sum %+.2f)" % (f, cs, s)})
        items = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
        out = {"score": total, "prob": prob, "label": "translated" if total > 0 else "original",
               "n_words": doc.n_words, "n_sentences": doc.n_sentences,
               "rules": contrib,
               "families": {f: {"raw": s, "clipped": cs} for f, (s, cs) in fam.items()},
               "spans": self.explain_spans(text, _doc=doc),
               "doc_level": doc_level,
               "evidence_translated": [rules[n].describe(doc, c) for n, c in items if c > 0][:top],
               "evidence_original": [rules[n].describe(doc, c) for n, c in items if c < 0][:top]}
        return out

    def describe_rules(self):
        return [dict(r.meta(), family_cap=self.family_caps[r.family]) for r in self.rules]

    def __repr__(self):
        return "<Engine %d rules, families=%s>" % (len(self.rules), sorted(self._by_family))


class Default(Engine):
    """All enabled rules from all discovered families."""


Expert = Default  # name used in README examples


class Families(Engine):
    """Only some families: Families(only="wordorder,calques", exclude=None)."""

    def __init__(self, only, exclude=None, **kw):
        super().__init__(families=only, exclude=exclude, **kw)


def to_json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1)

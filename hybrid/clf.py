"""Harness classifiers for the hybrid models (fitted on trn by hybrid/train.py and hybrid/stack.py).

    python3 evals/run.py --clf "hybrid.clf:Hybrid(model='lr_expert')" --split val --gran 500

One model is fitted per granularity. Pass gran='sent'|'100'|'500' to force
one; by default the model is chosen by whitespace-token length: < 90 tokens ->
sentence model, < 300 -> 100-token model, otherwise 500-token model. The cut
points sit away from 100/500 because topic masking merges multi-word names
(e.g. "Förenade kungariket" -> XTOPIC) and would otherwise push a 500-token
chunk into the 100-token model. 35 of 87,904 trn sentences are >= 90 tokens and
get routed to the 100-token model.

lr_expert / bow_lr / tree_d3 / tree_d4 load JSON from hybrid/models/ and need
only the standard library + numpy; gbt loads a pickle from
data/cache/hybrid/models/ and needs scikit-learn (use the venv python).
LR models explain a score as {feature: w_j * x_j} (0 when the feature is
absent); the intercept is not listed.
"""
import json
import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from hybrid import features as F  # noqa: E402

GRAN_BY_LEN = ((90, "sent"), (300, "100"), (None, "500"))
_MEMO = {}  # text -> computed inputs; shared by all models in one process (hybrid/eval_val.py)
_MEMO_MAX = 400000


def _memo(key, fn):
    v = _MEMO.get(key)
    if v is None:
        if len(_MEMO) > _MEMO_MAX:
            _MEMO.clear()
        v = _MEMO[key] = fn()
    return v


def expert_features(text):
    """Transformed literature-feature row for one text (memoised)."""
    return _memo(("feat", text), lambda: F.transform(np.asarray([F.raw_counts(text)], dtype=np.float64), np)[0])


def gran_for(text):
    n = len(text.split())
    for lim, g in GRAN_BY_LEN:
        if lim is None or n < lim:
            return g


class Hybrid:
    def __init__(self, model="lr_expert", gran=None, models_dir=None):
        self.name = model
        self.gran = {"sentence": "sent", "s": "sent"}.get(gran, gran)
        self.dir = models_dir or os.path.join(HERE, "models")
        self.models = {}
        self._primed = {}

    def prime(self, texts):
        """Batch-score GBT models (per-row sklearn calls cost ~1.5 ms); later score() calls on
        these exact texts return the stored value. No-op for the other models."""
        if self.name != "gbt":
            return
        by = {}
        for t in texts:
            by.setdefault(self.gran or gran_for(t), []).append(t)
        for g, ts in by.items():
            X = np.stack([expert_features(t) for t in ts])
            self._primed.update(zip(ts, map(float, self._model(g)["model"].decision_function(X))))

    def _model(self, g):
        if g not in self.models:
            if self.name == "gbt":
                with open(os.path.join(ROOT, "data", "cache", "hybrid", "models", "gbt.%s.pkl" % g), "rb") as f:
                    self.models[g] = pickle.load(f)
            else:
                with open(os.path.join(self.dir, "%s.%s.json" % (self.name, g))) as f:
                    m = json.load(f)
                if "w" in m:
                    m["w_arr"] = np.asarray(m["w"])
                self.models[g] = m
        return self.models[g]

    def _x(self, text, m):
        if m.get("kind") == "bow_linear":
            words = F.word_tokens(text)
            n = max(len(words), 1)
            idx = {w: j for j, w in enumerate(m["features"])}
            x = np.zeros(len(idx))
            for w in words:
                j = idx.get(w)
                if j is not None:
                    x[j] += 1000.0 / n
            return x
        return expert_features(text)

    def score_and_explain(self, text):
        if text in self._primed:
            return self._primed[text], {}
        g = self.gran or gran_for(text)
        m = self._model(g)
        if self.name == "gbt":
            x = self._x(text, {})
            return float(m["model"].decision_function(x[None, :])[0]), {}
        x = self._x(text, m)
        if m["kind"] in ("linear", "bow_linear"):
            c = x * m["w_arr"]
            ex = {m["features"][j]: float(c[j]) for j in np.flatnonzero(c)}
            return float(c.sum() + m["b"]), ex
        nodes, i, path = m["nodes"], 0, []
        while nodes[i]["left"] >= 0:
            nd = nodes[i]
            go_left = x[nd["f"]] <= nd["thr"]
            path.append(nd["feature"])
            i = nd["left"] if go_left else nd["right"]
        p = min(max(nodes[i]["p_trans"], 0.01), 0.99)
        s = math.log(p / (1 - p))
        return s, {"leaf%d" % i: s}

    def score(self, text):
        return self.score_and_explain(text)[0]


class Stacked:
    """Phase-B models (hybrid/stack.py): ML over the rule engine's explain() output.

        python3 evals/run.py --clf "hybrid.clf:Stacked(model='stack_rules')" --split val --gran 500

    Inputs named in the model JSON: "engine_score", "rule:<rule name>" (pre-cap
    contribution), "fam:<family>" (family subscore after its cap), "feat:<expert
    feature>" (hybrid/features.py). The engine is tsw.engine:Default as found in
    this checkout; the model JSON records the tsw hash it was fitted against.
    explain() returns the learned per-input contributions w_j * x_j.
    """

    def __init__(self, model="stack_rules", gran=None, models_dir=None):
        from tsw.engine import Default
        self.engine = Default()
        self.fam_of = {r.name: r.family for r in self.engine.rules}
        self.inner = Hybrid(model=model, gran=gran, models_dir=models_dir)
        self._primed = {}

    def _gbt(self, m):
        if "model" not in m:
            with open(os.path.join(ROOT, m["pickle"]), "rb") as f:
                m["model"] = pickle.load(f)
        return m["model"]

    def prime(self, texts):
        """Batch-score gbt_pickle models (see Hybrid.prime)."""
        by = {}
        for t in texts:
            by.setdefault(self.inner.gran or gran_for(t), []).append(t)
        for g, ts in by.items():
            m = self.inner._model(g)
            if m["kind"] != "gbt_pickle":
                return
            X = np.stack([self._inputs(t, m) for t in ts])
            self._primed.update(zip(ts, map(float, self._gbt(m).decision_function(X))))

    def _engine(self, text):
        from tsw.doc import Doc
        from hybrid.engine_cache import rule_stats, stat_names
        doc = Doc(text)
        s, contrib, famd = self.engine._run(doc)
        if math.isnan(s):
            s = 0.0
        ex = dict(contrib)
        for f_, (raw, cl) in famd.items():
            if cl != raw:
                ex["cap:" + f_] = cl - raw
        return s, ex, dict(zip(stat_names(self.engine), rule_stats(self.engine, doc)))

    def _inputs(self, text, m):
        s, ex, raw_stats = _memo(("engine", text), lambda: self._engine(text))
        smean = dict(zip(m.get("stat_cols", []), m.get("smean", [])))
        stats = {n_: (smean.get(n_, 0.0) if (v != v and m["kind"] != "gbt_pickle") else v)
                 for n_, v in raw_stats.items()}
        feats = {}
        if any(i.startswith("feat:") for i in m["inputs"]):
            feats = dict(zip(F.feature_names(), expert_features(text)))
        fam = {}
        for k, v in ex.items():
            f = k[4:] if k.startswith("cap:") else self.fam_of.get(k)
            fam[f] = fam.get(f, 0.0) + v
        x = []
        for i in m["inputs"]:
            if i == "engine_score":
                x.append(s)
            elif i.startswith("rule:"):
                x.append(ex.get(i[5:], 0.0))
            elif i.startswith("fam:"):
                x.append(fam.get(i[4:], 0.0))
            elif i.startswith("feat:"):
                x.append(feats[i[5:]])
            elif i.startswith("stat:"):
                x.append(stats[i[5:]])
        return np.asarray(x, dtype=np.float64)

    def score_and_explain(self, text):
        if text in self._primed:
            return self._primed[text], {}
        g = self.inner.gran or gran_for(text)
        m = self.inner._model(g)
        x = self._inputs(text, m)
        if m["kind"] == "gbt_pickle":
            return float(self._gbt(m).decision_function(x[None, :])[0]), {}
        if m["kind"] == "linear":
            c = x * m["w_arr"]
            return float(c.sum() + m["b"]), {m["inputs"][j]: float(c[j]) for j in np.flatnonzero(c)}
        nodes, i = m["nodes"], 0
        while nodes[i]["left"] >= 0:
            i = nodes[i]["left"] if x[nodes[i]["f"]] <= nodes[i]["thr"] else nodes[i]["right"]
        p = min(max(nodes[i]["p_trans"], 0.01), 0.99)
        s = math.log(p / (1 - p))
        return s, {"leaf%d" % i: s}

    def score(self, text):
        return self.score_and_explain(text)[0]

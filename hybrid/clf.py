"""Harness classifiers for the hybrid models (fitted by hybrid/train.py on dev-train).

    python3 evals/run.py --clf "hybrid.clf:Hybrid(model='lr_expert')" --split dev-val --gran 500

One model is fitted per granularity. Pass gran='sent'|'100'|'500' to force
one; by default the model is chosen by whitespace-token length: < 90 tokens ->
sentence model, < 300 -> 100-token model, otherwise 500-token model. The cut
points sit away from 100/500 because topic masking merges multi-word names
(e.g. "Förenade kungariket" -> XTOPIC) and would otherwise push a 500-token
chunk into the 100-token model. 35 of 87,904 dev sentences are >= 90 tokens and
get routed to the 100-token model (the cached evaluation in train.py uses
the sentence model for them).

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
        return F.transform(np.asarray([F.raw_counts(text)], dtype=np.float64), np)[0]

    def score_and_explain(self, text):
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

        python3 evals/run.py --clf "hybrid.clf:Stacked(model='stack_rules')" --split dev-val --gran 500

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

    def _inputs(self, text, m):
        s, ex = self.engine.score_and_explain(text)
        need_feat = any(i.startswith("feat:") for i in m["inputs"])
        feats = {}
        if need_feat:
            fx = F.transform(np.asarray([F.raw_counts(text)], dtype=np.float64), np)[0]
            feats = dict(zip(F.feature_names(), fx))
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
        return np.asarray(x, dtype=np.float64)

    def score_and_explain(self, text):
        g = self.inner.gran or gran_for(text)
        m = self.inner._model(g)
        x = self._inputs(text, m)
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

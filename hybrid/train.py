#!/usr/bin/env python3
"""Fit the Phase-A hybrid models on ALL of trn (cached features); no evaluation here.

    python3 -m hybrid.train                     # all grans, all models   (~1 CPU-minute)
    python3 -m hybrid.train --grans 500 --models lr_expert tree_d4

Models (one per granularity; `hybrid.clf:Hybrid` picks one by text length):
  lr_expert   L2 logistic regression on the expert features (hybrid/features.py),
              C chosen by 5-fold CV inside trn, folds grouped by session date
  tree_d3/4   decision tree (depth 3/4) on the same, un-scaled features, so its
              splits read as rules ("rate of ju per 1000 words <= 0.4")
  gbt         gradient-boosted trees (HistGradientBoosting; early stopping on an
              internal random 15% of trn)
  bow_lr      small reference only: L2 LR on relative frequencies of the N most
              frequent trn words (no expert input)

Models go to hybrid/models/ (JSON; GBT pickles in data/cache/hybrid/models/).
Evaluation on val runs live through the harness: hybrid/eval_val.py. Never
fit or tune on val.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals"))
sys.path.insert(0, ROOT)
import metrics  # noqa: E402
import run  # noqa: E402
from hybrid import cache  # noqa: E402
from hybrid import features as F  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "hybrid", "models")
REPORT_DIR = os.path.join(ROOT, "hybrid", "reports")
BIG_DIR = os.path.join(cache.CACHE_DIR, "models")  # gitignored (pickles)
ALL_MODELS = ["lr_expert", "tree_d3", "tree_d4", "gbt", "bow_lr"]
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
BOW_N = 300
SEED = 20261105
GBT_PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15, random_state=SEED)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def fit_lr(X, y, groups):
    """L2 LR on standardised X; C by grouped 5-fold CV (log-loss). Returns a dict model."""
    mu, sd = standardize_fit(X)
    Z = (X - mu) / sd
    best = None
    folds = list(GroupKFold(n_splits=5).split(Z, y, groups))
    for C in CS:
        ll = 0.0
        oof = np.zeros(len(y))
        for tr, te in folds:
            m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], y[tr])
            p = np.clip(m.predict_proba(Z[te])[:, 1], 1e-6, 1 - 1e-6)
            oof[te] = np.log(p / (1 - p))
            ll -= np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
        if best is None or ll < best[0]:
            best = (ll, C, oof)
    m = LogisticRegression(C=best[1], max_iter=2000).fit(Z, y)
    w = m.coef_[0] / sd  # weights on raw features
    b = float(m.intercept_[0] - np.sum(m.coef_[0] * mu / sd))
    return {"kind": "linear", "C": best[1], "cv_logloss": best[0] / len(y),
            "cv_auc": metrics.roc_auc(list(map(int, y)), list(best[2])),
            "mean": mu.tolist(), "scale": sd.tolist(),
            "coef_std": m.coef_[0].tolist(), "w": w.tolist(), "b": b}


def linear_score(model, X):
    return X @ np.asarray(model["w"]) + model["b"]


def linear_contrib(model, X):
    """Per-feature contributions w_j * x_j (0 when the feature's value is 0)."""
    return X * np.asarray(model["w"])


def fit_tree(X, y, depth):
    leaf = max(20, int(0.01 * len(y)))
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED).fit(X, y)
    return t


def tree_to_json(t, names):
    tr = t.tree_
    nodes = []
    for i in range(tr.node_count):
        v = tr.value[i][0]
        p = float(v[1] / v.sum())
        nodes.append({"feature": names[tr.feature[i]] if tr.children_left[i] >= 0 else None,
                      "f": int(tr.feature[i]), "thr": float(tr.threshold[i]),
                      "left": int(tr.children_left[i]), "right": int(tr.children_right[i]),
                      "p_trans": p, "n": int(tr.n_node_samples[i])})
    return {"kind": "tree", "depth": t.get_depth(), "nodes": nodes}


def logit(p, eps=0.01):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def tree_score_row(model, x):
    nodes, i = model["nodes"], 0
    while nodes[i]["left"] >= 0:
        i = nodes[i]["left"] if x[nodes[i]["f"]] <= nodes[i]["thr"] else nodes[i]["right"]
    return logit(nodes[i]["p_trans"]), i


def tree_rules(model, names, prior):
    """Root-to-leaf paths as readable rules, sorted by how far the leaf is from the prior."""
    nodes = model["nodes"]
    out = []

    def walk(i, conds):
        nd = nodes[i]
        if nd["left"] < 0:
            out.append((conds, nd["p_trans"], nd["n"]))
            return
        f, thr = nd["feature"], nd["thr"]
        walk(nd["left"], conds + [(f, "<=", thr)])
        walk(nd["right"], conds + [(f, ">", thr)])
    walk(0, [])
    out.sort(key=lambda r: -abs(r[1] - prior) * r[2])
    return out


def bow_matrix(B, cols):
    """Relative frequencies per 1000 words of the selected vocabulary columns."""
    tot = np.asarray(B.sum(1)).ravel()
    tot[tot == 0] = 1
    return np.asarray(B[:, cols].todense()) * (1000.0 / tot[:, None])


def fmt_rule(conds, _=None):
    return " AND ".join("%s %s %.3g" % c for c in conds)


def load_summary():
    p = os.path.join(REPORT_DIR, "training.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def save_summary(s):
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "training.json"), "w") as f:
        json.dump(s, f, indent=1, sort_keys=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grans", nargs="+", default=["sent", "100", "500"])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    args = ap.parse_args(argv)
    names = F.feature_names()
    for d_ in (MODEL_DIR, REPORT_DIR, BIG_DIR):
        os.makedirs(d_, exist_ok=True)
    summary = load_summary()
    for g in args.grans:
        g = run.GRANS[g]
        t0 = time.time()
        d = cache.load(g)
        X = F.transform(d["C"].astype(np.float64), np)
        y = d["labels"].astype(int)
        groups = d["dates"]
        print("== gran=%s  trn %d (orig %d / trans %d)  sessions %d  features %d"
              % (g, len(y), (y == 0).sum(), (y == 1).sum(), len(set(groups)), X.shape[1]))
        info = {"n": int(len(y)), "data_sha256": str(d["data_sha256"]), "features_hash": str(d["features_hash"])}
        for mname in args.models:
            t1 = time.time()
            if mname == "lr_expert":
                model = fit_lr(X, y, groups)
                model.update(features=names, gran=g, trained_on="trn")
                path = os.path.join(MODEL_DIR, "lr_expert.%s.json" % g)
                info[mname] = {"C": model["C"], "cv_auc_trn": model["cv_auc"]}
            elif mname.startswith("tree_d"):
                depth = int(mname[-1])
                model = tree_to_json(fit_tree(X, y, depth), names)
                model.update(features=names, gran=g, trained_on="trn")
                path = os.path.join(MODEL_DIR, "%s.%s.json" % (mname, g))
                info[mname] = {"max_depth": depth, "min_samples_leaf": max(20, int(0.01 * len(y)))}
            elif mname == "gbt":
                m = HistGradientBoostingClassifier(**GBT_PARAMS).fit(X, y)
                with open(os.path.join(BIG_DIR, "gbt.%s.pkl" % g), "wb") as f:
                    pickle.dump({"model": m, "features": names, "gran": g}, f)
                info[mname] = {"n_iter": int(m.n_iter_)}
                model = None
            elif mname == "bow_lr":
                freq = np.asarray(d["bow"].sum(0)).ravel()
                cols = np.argsort(-freq, kind="stable")[:BOW_N]
                model = fit_lr(bow_matrix(d["bow"], cols), y, groups)
                model.update(features=[d["vocab"][j] for j in cols], gran=g, kind="bow_linear", trained_on="trn")
                path = os.path.join(MODEL_DIR, "bow_lr.%s.json" % g)
                info[mname] = {"C": model["C"], "cv_auc_trn": model["cv_auc"], "top_n": BOW_N}
            else:
                sys.exit("unknown model %s" % mname)
            if model is not None:
                with open(path, "w") as f:
                    json.dump(model, f, indent=0)
            print("  %-10s %s  (%.1fs)" % (mname, info[mname], time.time() - t1))
        summary.setdefault(g, {}).update(info)
        print("  gran %s done in %.1fs" % (g, time.time() - t0))
    save_summary(summary)


if __name__ == "__main__":
    main()

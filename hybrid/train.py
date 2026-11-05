#!/usr/bin/env python3
"""Fit the hybrid / ML models on dev-train, evaluate on dev-val (cached features).

    python3 -m hybrid.train                     # all grans, all models, save results
    python3 -m hybrid.train --grans 500 --models lr_expert tree_d4 --no-save

Models (one per granularity; `hybrid.clf:Hybrid` picks one by text length):
  lr_expert   L2 logistic regression on the expert features (hybrid/features.py),
              C chosen by 5-fold CV on dev-train grouped by session date
  tree_d3/4   decision tree (depth 3/4) on the same, un-scaled features, so its
              splits read as rules ("rate of ju per 1000 words <= 0.4")
  gbt         gradient-boosted trees (reference ceiling for these features)
  bow_lr      pure-ML reference: L2 LR on relative frequencies of the N most
              frequent dev-train words (function-word style, no expert input)

Evaluation runs on the cached dev-val features, unmasked and topic-masked (the
masked text is exactly what `evals/run.py --mask-topic` would feed), and writes
harness-format results JSON (split="dev-val") via evals/run.py:save_result.
`hybrid.clf:Hybrid` re-computes the same scores from raw text through the
harness (see tests/README in hybrid/README.md for the consistency check).
"""
import argparse
import datetime
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
import devsplit  # noqa: E402
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
    for C in CS:
        ll = 0.0
        for tr, te in GroupKFold(n_splits=5).split(Z, y, groups):
            m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], y[tr])
            p = np.clip(m.predict_proba(Z[te])[:, 1], 1e-6, 1 - 1e-6)
            ll -= np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
        if best is None or ll < best[0]:
            best = (ll, C)
    m = LogisticRegression(C=best[1], max_iter=2000).fit(Z, y)
    w = m.coef_[0] / sd  # weights on raw features
    b = float(m.intercept_[0] - np.sum(m.coef_[0] * mu / sd))
    return {"kind": "linear", "C": best[1], "cv_logloss": best[0] / len(y), "mean": mu.tolist(), "scale": sd.tolist(),
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


# --------------------------------------------------------------------------
# evaluation / reporting
# --------------------------------------------------------------------------

def git_short():
    gi = run.git_info()
    return gi


def result_blob(clf_spec, gran, masked, labels, scores, data, extra, expl=None):
    gi = run.git_info()
    out = {"harness_version": run.HARNESS_VERSION,
           "timestamp": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
           "clf": clf_spec, "split": "dev-val", "gran": gran, "balanced": True, "mask_topic": bool(masked),
           "limit": None, "limit_seed": None, **gi,
           "data_file": "dev.%s.bal.jsonl" % gran, "data_sha256": str(data["data_sha256"]),
           "metrics": metrics.all_metrics(list(map(int, labels)), list(map(float, scores))),
           "rules": metrics.rule_diagnostics(list(map(int, labels)), expl) if expl is not None else None,
           "elapsed_sec": None, "argv": sys.argv, "subset_rule": devsplit.describe("dev-val"),
           "eval_mode": "cached features (hybrid/train.py); fitted on %s" % devsplit.describe("dev-train")}
    out.update(extra)
    return out


def expl_dicts(contrib, names, top=None):
    out = []
    for row in contrib:
        nz = np.flatnonzero(row)
        out.append({names[j]: float(row[j]) for j in nz})
    return out


def unseen_speaker_mask(gran, data):
    with open(os.path.join(cache.CACHE_DIR, "dev.%s.speakers.json" % gran), encoding="utf-8") as f:
        spk = json.load(f)
    tr = set(s for i in np.flatnonzero(~data["is_val"]) for s in spk[i] if s)
    vi = np.flatnonzero(data["is_val"])
    return np.array([not any(s in tr for s in spk[i] if s) for i in vi])


def fmt_rule(conds, feat_desc):
    parts = []
    for f, op, thr in conds:
        parts.append("%s %s %.3g" % (f, op, thr))
    return " AND ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grans", nargs="+", default=["sent", "100", "500"])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--no-save", action="store_true", help="do not write results JSON / leaderboard")
    args = ap.parse_args(argv)
    names = F.feature_names()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(BIG_DIR, exist_ok=True)
    summary_path = os.path.join(REPORT_DIR, "metrics.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    for g in args.grans:
        g = run.GRANS[g]
        t0 = time.time()
        d = cache.load(g)
        X = F.transform(d["C"].astype(np.float64), np)
        Xm_val = F.transform(d["C_masked"].astype(np.float64), np)
        tr, va = ~d["is_val"], d["is_val"]
        assert np.array_equal(np.flatnonzero(va), d["val_idx"])
        Xtr, ytr, Xva, yva = X[tr], d["labels"][tr].astype(int), X[va], d["labels"][va].astype(int)
        groups = d["dates"][tr]
        unseen = unseen_speaker_mask(g, d)
        prior = ytr.mean()
        print("== gran=%s  dev-train %d  dev-val %d (unseen-speaker units %d)  features %d"
              % (g, len(ytr), len(yva), unseen.sum(), X.shape[1]))
        # pure-ML BoW columns: top-N by dev-train token count
        bow_cols = None
        if "bow_lr" in args.models:
            freq = np.asarray(d["bow"][np.flatnonzero(tr)].sum(0)).ravel()
            bow_cols = np.argsort(-freq, kind="stable")[:BOW_N]
            bow_words = [d["vocab"][j] for j in bow_cols]

        for mname in args.models:
            t1 = time.time()
            expl_u = expl_m = None
            extra = {}
            if mname == "lr_expert":
                model = fit_lr(Xtr, ytr, groups)
                model.update(features=names, gran=g)
                s_u, s_m = linear_score(model, Xva), linear_score(model, Xm_val)
                expl_u = expl_dicts(linear_contrib(model, Xva), names)
                expl_m = expl_dicts(linear_contrib(model, Xm_val), names)
                with open(os.path.join(MODEL_DIR, "lr_expert.%s.json" % g), "w") as f:
                    json.dump(model, f, indent=0)
                extra["hyper"] = {"C": model["C"]}
            elif mname.startswith("tree_d"):
                depth = int(mname[-1])
                t = fit_tree(Xtr, ytr, depth)
                model = tree_to_json(t, names)
                model.update(features=names, gran=g)
                s_u = np.array([tree_score_row(model, x)[0] for x in Xva])
                s_m = np.array([tree_score_row(model, x)[0] for x in Xm_val])
                with open(os.path.join(MODEL_DIR, "%s.%s.json" % (mname, g)), "w") as f:
                    json.dump(model, f, indent=0)
                extra["hyper"] = {"max_depth": depth, "min_samples_leaf": max(20, int(0.01 * len(ytr)))}
            elif mname == "gbt":
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=15,
                                                   l2_regularization=1.0, early_stopping=True,
                                                   validation_fraction=0.15, random_state=SEED).fit(Xtr, ytr)
                s_u, s_m = m.decision_function(Xva), m.decision_function(Xm_val)
                with open(os.path.join(BIG_DIR, "gbt.%s.pkl" % g), "wb") as f:
                    pickle.dump({"model": m, "features": names, "gran": g}, f)
                extra["hyper"] = {"n_iter": int(m.n_iter_), "max_leaf_nodes": 15, "learning_rate": 0.08}
            elif mname == "bow_lr":
                Btr = bow_matrix(d["bow"][np.flatnonzero(tr)], bow_cols)
                Bva = bow_matrix(d["bow"][np.flatnonzero(va)], bow_cols)
                Bvm = bow_matrix(d["bow_masked"], bow_cols)
                model = fit_lr(Btr, ytr, groups)
                model.update(features=bow_words, gran=g, kind="bow_linear")
                s_u, s_m = linear_score(model, Bva), linear_score(model, Bvm)
                with open(os.path.join(MODEL_DIR, "bow_lr.%s.json" % g), "w") as f:
                    json.dump(model, f, indent=0)
                extra["hyper"] = {"C": model["C"], "top_n": BOW_N}
            else:
                sys.exit("unknown model %s" % mname)
            spec = "hybrid.clf:Hybrid(model='%s')" % mname
            mu = metrics.all_metrics(list(yva), list(s_u))
            mm = metrics.all_metrics(list(yva), list(s_m))
            mus = metrics.all_metrics(list(yva[unseen]), list(np.asarray(s_u)[unseen]))
            mus["note"] = "dev-val units none of whose speakers occur in dev-train (orig %d / trans %d)" % (
                int((yva[unseen] == 0).sum()), int((yva[unseen] == 1).sum()))
            extra["unseen_speaker_metrics"] = mus
            extra["n_unseen_speaker"] = int(unseen.sum())
            print("  %-10s unmasked acc=%.4f F1=%.4f AUC=%.4f | masked acc=%.4f F1=%.4f AUC=%.4f | unseen-spk AUC=%.4f"
                  " (%.1fs)" % (mname, mu["accuracy"], mu["macro_f1"], mu["roc_auc"], mm["accuracy"], mm["macro_f1"],
                                 mm["roc_auc"], mus["roc_auc"], time.time() - t1))
            for masked, s, ex in ((False, s_u, expl_u), (True, s_m, expl_m)):
                blob = result_blob(spec, g, masked, yva, s, d, dict(extra), ex)
                if masked:
                    blob["unmasked_metrics"] = mu
                    blob["topic_auc_drop"] = mu["roc_auc"] - mm["roc_auc"]
                if not args.no_save:
                    run.save_result(blob, quiet=True)
                summary.setdefault(g, {}).setdefault(mname, {})["masked" if masked else "unmasked"] = {
                    k: blob["metrics"][k] for k in ("n", "accuracy", "macro_f1", "roc_auc")}
            summary[g][mname]["unseen_speaker"] = {k: mus[k] for k in ("n", "accuracy", "macro_f1", "roc_auc")}
        print("  gran %s done in %.1fs" % (g, time.time() - t0))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()

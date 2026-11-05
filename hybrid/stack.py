#!/usr/bin/env python3
"""Phase B: ML on top of the rule engine (fit on dev-train, evaluate on dev-val).

    python3 -m hybrid.engine_cache --gran 500 100 sent   # once per engine version
    python3 -m hybrid.stack                              # seconds; re-run after every merge

All inputs are expert features: the engine's per-rule contributions (explain()),
its per-family subscores, the raw statistic each rule computes (hits per 1000
words, choice shares, unit means; hybrid/engine_cache.py), and the literature
word-list features of Phase A (hybrid/features.py).

  engine_calib    LR on the engine score alone: only re-sets threshold/scale, so
                  any gain over the raw engine is calibration, not re-weighting
  stack_rules     LR over the per-rule contributions (before family caps):
                  coefficient = learned multiplier on the hand-set contribution
                  (1.0 = hand-set weight is right, < 0 = the rule's sign flips)
  stack_families  LR over the per-family subscores (after caps)
  lr_engine       LR over rule contributions + rule statistics (engine-only inputs)
  gbt_engine      gradient-boosted trees over the same engine-only inputs
  tree_engine     depth-3 decision tree over rule statistics + engine score
                  (thresholds in natural units -> candidate rules)
  hybrid_plus     engine score + the 10 strongest Phase-A features of lr_expert
                  (picked from the dev-train fit)
  lr_all          LR over rule contributions + rule statistics + Phase-A features
  gbt_all         gradient-boosted trees over the same

Every model is written to hybrid/models/<name>.<gran>.json (applied from raw
text by hybrid.clf:Stacked) and evaluated like hybrid/train.py (harness JSON,
split="dev-val"). hybrid/reports/stacking_<gran>.md compares learned and
hand-set weights per rule.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals"))
sys.path.insert(0, ROOT)
import metrics  # noqa: E402
import run  # noqa: E402
from hybrid import cache, engine_cache  # noqa: E402
from hybrid import features as F  # noqa: E402
from hybrid import train as T  # noqa: E402

ALL = ["engine_calib", "stack_rules", "stack_families", "lr_engine", "gbt_engine", "tree_engine", "hybrid_plus",
       "lr_all", "gbt_all"]
N_PLUS = 10


def family_matrix(E, cols, fam_of):
    """Per-family subscores after caps: sum of the family's rule columns + its cap: column."""
    fams = []
    for c in cols:
        f = c[4:] if c.startswith(engine_cache_cap()) else fam_of[c]
        if f not in fams:
            fams.append(f)
    M = np.zeros((E.shape[0], len(fams)))
    for j, c in enumerate(cols):
        f = c[4:] if c.startswith(engine_cache_cap()) else fam_of[c]
        M[:, fams.index(f)] += E[:, j]
    return M, fams


def engine_cache_cap():
    return "cap:"


def fit_linear_raw(X, y, groups):
    """fit_lr but report weights on the raw input scale (multipliers)."""
    return T.fit_lr(X, y, groups)


def write_model(name, g, model):
    with open(os.path.join(T.MODEL_DIR, "%s.%s.json" % (name, g)), "w") as f:
        json.dump(model, f, indent=0)


def rule_report(g, ec, model, Etr, ytr, rule_cols):
    meta = {m["name"]: m for m in ec["rule_meta"]}
    lines = ["# Learned vs hand-set rule weights (gran=%s)\n" % g,
             "LR over the engine's per-rule contributions on dev-train (%d units; C=%g chosen by grouped CV). "
             "`multiplier` is the learned coefficient on the hand-set contribution: 1.0 means the hand-set "
             "weight is about right, >1 under-weighted, between 0 and 1 over-weighted, **< 0 means the sign "
             "flips** (the rule points the wrong way once the other rules are accounted for). `alone AUC` is the "
             "rule's contribution used alone as a score on dev-train; `fires` is the share of units with a "
             "non-zero contribution. Engine: `%s`, tsw hash `%s`, commit `%s`.\n"
             % (len(ytr), model["C"], ec["clf"], ec["tsw_hash"], str(ec["git_commit"])[:8]),
             "| rule | family | dir | hand weight | cap | fires | alone AUC | multiplier | std. coef | flag |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for j, c in enumerate(rule_cols):
        m = meta.get(c, {})
        w = model["w"][j]
        alone = metrics.roc_auc(list(ytr), list(Etr[:, j]))
        fires = float(np.mean(Etr[:, j] != 0))
        flag = "SIGN FLIP" if w < 0 else ("x%.1f" % w if (w > 2 or w < 0.5) else "")
        if alone < 0.5 and fires > 0:
            flag = (flag + "; alone AUC<0.5").lstrip("; ")
        lines.append("| `%s` | %s | %s | %.2f | %.2f | %.3f | %.3f | %+.3f | %+.3f | %s |"
                     % (c, m.get("family", ""), m.get("direction", ""), m.get("weight", 1.0), m.get("cap", 0),
                        fires, alone, w, model["coef_std"][j], flag))
    lines.append("\nLearned intercept (prior, log-odds): %+.3f (engine prior %+.3f).\n"
                 % (model["b"], float(ec["prior"])))
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grans", nargs="+", default=["sent", "100", "500"])
    ap.add_argument("--models", nargs="+", default=ALL)
    ap.add_argument("--tag", default="default", help="engine cache tag (hybrid/engine_cache.py --tag)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(T.REPORT_DIR, exist_ok=True)
    summary_path = os.path.join(T.REPORT_DIR, "metrics.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    names = F.feature_names()
    for g in args.grans:
        g = run.GRANS[g]
        t0 = time.time()
        ec = engine_cache.load(g, args.tag)
        fc = cache.load(g)
        pos = {i: k for k, i in enumerate(fc["ids"])}
        itr = np.array([pos[i] for i in ec["ids_train"]])
        iva = np.array([pos[i] for i in ec["ids_val"]])
        vpos = {i: k for k, i in enumerate(fc["val_idx"])}
        ivm = np.array([vpos[i] for i in iva])
        X = F.transform(fc["C"].astype(np.float64), np)
        Xtr, Xva = X[itr], X[iva]
        Xvm = F.transform(fc["C_masked"].astype(np.float64), np)[ivm]
        ytr, yva = ec["y_train"].astype(int), ec["y_val"].astype(int)
        assert np.array_equal(ytr, fc["labels"][itr]) and np.array_equal(yva, fc["labels"][iva])
        groups = ec["dates_train"]
        cols = ec["cols"]
        rule_cols = [c for c in cols if not c.startswith("cap:")]
        rj = [cols.index(c) for c in rule_cols]
        fam_of = {m["name"]: m["family"] for m in ec["rule_meta"]}
        Etr, Eva, Evm = ec["E_train"], ec["E_val"], ec["E_val_masked"]
        stat_cols = [str(c) for c in ec["stat_cols"]]
        Str, Sva, Svm = ec["S_train"], ec["S_val"], ec["S_val_masked"]
        smean = np.nanmean(Str, 0)  # NaN (StatRule with no units) -> dev-train mean, for LR/trees

        def imp(S):
            S = S.copy()
            r, c = np.nonzero(np.isnan(S))
            S[r, c] = smean[c]
            return S
        Sitr, Siva, Sivm = imp(Str), imp(Sva), imp(Svm)
        eng_in = ["rule:" + c for c in rule_cols] + ["stat:" + c for c in stat_cols]
        all_in = eng_in + ["feat:" + n for n in names]
        EStr, ESva, ESvm = (np.hstack([Etr[:, rj], Sitr]), np.hstack([Eva[:, rj], Siva]),
                            np.hstack([Evm[:, rj], Sivm]))
        str_, sva, svm = ec["s_train"], ec["s_val"], ec["s_val_masked"]
        # unseen-speaker mask in engine-cache val order
        unseen_fc = T.unseen_speaker_mask(g, fc)
        unseen = unseen_fc[ivm]
        print("== gran=%s  engine %s (%d rules, tsw %s)  train %d  val %d"
              % (g, ec["clf"], len(rule_cols), ec["tsw_hash"], len(ytr), len(yva)))
        base_meta = {"engine": str(ec["clf"]), "tsw_hash": str(ec["tsw_hash"]),
                     "engine_commit": str(ec["git_commit"]), "rule_cols": rule_cols}
        for mname in args.models:
            t1 = time.time()
            extra = {"engine": base_meta["engine"], "engine_tsw_hash": base_meta["tsw_hash"]}
            if mname == "engine_calib":
                model = T.fit_lr(str_[:, None], ytr, groups)
                model.update(inputs=["engine_score"])
                s_u, s_m = T.linear_score(model, sva[:, None]), T.linear_score(model, svm[:, None])
            elif mname == "stack_rules":
                model = T.fit_lr(Etr[:, rj], ytr, groups)
                model.update(inputs=["rule:" + c for c in rule_cols])
                s_u, s_m = T.linear_score(model, Eva[:, rj]), T.linear_score(model, Evm[:, rj])
                with open(os.path.join(T.REPORT_DIR, "stacking_%s.md" % g), "w") as f:
                    f.write(rule_report(g, ec, model, Etr[:, rj], ytr, rule_cols))
            elif mname == "stack_families":
                Ftr, fams = family_matrix(Etr, cols, fam_of)
                Fva, _ = family_matrix(Eva, cols, fam_of)
                Fvm, _ = family_matrix(Evm, cols, fam_of)
                model = T.fit_lr(Ftr, ytr, groups)
                model.update(inputs=["fam:" + f for f in fams])
                s_u, s_m = T.linear_score(model, Fva), T.linear_score(model, Fvm)
                extra["family_multipliers"] = dict(zip(fams, model["w"]))
            elif mname == "lr_engine":
                model = T.fit_lr(EStr, ytr, groups)
                model.update(inputs=eng_in)
                s_u, s_m = T.linear_score(model, ESva), T.linear_score(model, ESvm)
            elif mname in ("gbt_engine", "gbt_all"):
                Ztr, Zva, Zvm, inp = (EStr, ESva, ESvm, eng_in) if mname == "gbt_engine" else (
                    np.hstack([EStr, Xtr]), np.hstack([ESva, Xva]), np.hstack([ESvm, Xvm]), all_in)
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=15,
                                                   l2_regularization=1.0, early_stopping=True,
                                                   validation_fraction=0.15, random_state=T.SEED).fit(Ztr, ytr)
                s_u, s_m = m.decision_function(Zva), m.decision_function(Zvm)
                pk = os.path.join(T.BIG_DIR, "%s.%s.pkl" % (mname, g))
                with open(pk, "wb") as f:
                    pickle.dump(m, f)
                model = {"kind": "gbt_pickle", "pickle": os.path.relpath(pk, ROOT), "inputs": inp,
                         "n_iter": int(m.n_iter_)}
                extra["hyper"] = {"n_iter": int(m.n_iter_), "max_leaf_nodes": 15, "learning_rate": 0.08}
            elif mname == "tree_engine":
                inp = ["stat:" + c for c in stat_cols] + ["engine_score"]
                Ztr = np.hstack([Sitr, str_[:, None]])
                t = T.fit_tree(Ztr, ytr, 3)
                model = T.tree_to_json(t, inp)
                model["inputs"] = inp
                s_u = np.array([T.tree_score_row(model, x)[0] for x in np.hstack([Siva, sva[:, None]])])
                s_m = np.array([T.tree_score_row(model, x)[0] for x in np.hstack([Sivm, svm[:, None]])])
                with open(os.path.join(T.REPORT_DIR, "tree_engine_%s.md" % g), "w") as f:
                    f.write("# Depth-3 tree over the engine's rule statistics + engine score (gran=%s)\n\n"
                            "Fitted on dev-train (prior P(trans)=%.2f). `stat:<rule>:per1000` = hits per 1000 words, "
                            "`:share_a` = share of the rule's variant A among its choices, `:mean` = unit mean.\n\n"
                            % (g, ytr.mean()))
                    for conds, p, n in T.tree_rules(model, inp, ytr.mean()):
                        f.write("- IF %s THEN P(trans)=%.2f (n=%d)\n" % (T.fmt_rule(conds, None) or "(all)", p, n))
            elif mname == "hybrid_plus":
                lr = json.load(open(os.path.join(T.MODEL_DIR, "lr_expert.%s.json" % g)))
                top = list(np.argsort(-np.abs(lr["coef_std"]))[:N_PLUS])
                Ztr = np.hstack([str_[:, None], Xtr[:, top]])
                model = T.fit_lr(Ztr, ytr, groups)
                model.update(inputs=["engine_score"] + ["feat:" + names[j] for j in top])
                s_u = T.linear_score(model, np.hstack([sva[:, None], Xva[:, top]]))
                s_m = T.linear_score(model, np.hstack([svm[:, None], Xvm[:, top]]))
                extra["plus_features"] = [names[j] for j in top]
            elif mname == "lr_all":
                model = T.fit_lr(np.hstack([EStr, Xtr]), ytr, groups)
                model.update(inputs=all_in)
                s_u = T.linear_score(model, np.hstack([ESva, Xva]))
                s_m = T.linear_score(model, np.hstack([ESvm, Xvm]))
            else:
                sys.exit("unknown model %s" % mname)
            model.update(gran=g, smean=smean.tolist(), stat_cols=stat_cols, **base_meta)
            write_model(mname, g, model)
            if "C" in model:
                extra["hyper"] = {"C": model["C"]}
            T.emit(mname, "hybrid.clf:Stacked(model='%s')" % mname, g, yva, s_u, s_m, str(fc["data_sha256"]),
                   extra, unseen, summary, save=not args.no_save, t1=t1)
        # reference: the raw engine on the same records (already saved by engine_cache.py)
        me = metrics.all_metrics(list(yva), list(sva))
        mem = metrics.all_metrics(list(yva), list(svm))
        summary.setdefault(g, {})["engine_raw"] = {
            "unmasked": {k: me[k] for k in ("n", "accuracy", "macro_f1", "roc_auc")},
            "masked": {k: mem[k] for k in ("n", "accuracy", "macro_f1", "roc_auc")},
            "unseen_speaker": {k: v for k, v in metrics.all_metrics(list(yva[unseen]), list(sva[unseen])).items()
                               if k in ("n", "accuracy", "macro_f1", "roc_auc")},
            "engine": base_meta["engine"], "tsw_hash": base_meta["tsw_hash"]}
        print("  %-14s unmasked acc=%.4f F1=%.4f AUC=%.4f | masked acc=%.4f F1=%.4f AUC=%.4f   (reference)"
              % ("engine_raw", me["accuracy"], me["macro_f1"], me["roc_auc"], mem["accuracy"], mem["macro_f1"],
                 mem["roc_auc"]))
        print("  gran %s done in %.1fs" % (g, time.time() - t0))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()

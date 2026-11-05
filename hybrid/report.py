#!/usr/bin/env python3
"""Write the human-readable hybrid reports from the fitted models and caches (seconds).

    python3 -m hybrid.report

hybrid/reports/METRICS.md        dev-val table: every model x gran x {unmasked, masked},
                                 with the pure engine and MeanSentLen on the same records
hybrid/reports/coefficients.md   lr_expert coefficients in plain words vs the literature
hybrid/reports/literature_check.md  per-feature direction on dev-train vs the literature
hybrid/reports/tree_rules.md     decision-tree paths over the expert features (candidate rules)
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals"))
sys.path.insert(0, ROOT)
import metrics  # noqa: E402
from hybrid import cache  # noqa: E402
from hybrid import features as F  # noqa: E402
from hybrid import train as T  # noqa: E402

R = T.REPORT_DIR
GRANS = ["sent", "100", "500"]
ORDER = [("MeanSentLen", "baseline: mean sentence length (hand-set threshold)"),
         ("engine_raw", "pure rule engine `tsw.engine:Default`"),
         ("engine_calib", "engine score, LR-calibrated (threshold/scale only)"),
         ("stack_rules", "LR re-weighting of the engine's per-rule contributions"),
         ("stack_families", "LR over per-family subscores"),
         ("lr_engine", "LR over rule contributions + rule statistics"),
         ("gbt_engine", "GBT over rule contributions + rule statistics"),
         ("tree_engine", "depth-3 tree over rule statistics + engine score"),
         ("hybrid_plus", "engine score + 10 strongest literature features (LR)"),
         ("lr_expert", "LR over ~110 literature word-list features (Phase A)"),
         ("gbt", "GBT over the literature features (Phase A)"),
         ("tree_d3", "depth-3 tree over the literature features"),
         ("tree_d4", "depth-4 tree over the literature features"),
         ("lr_all", "LR over engine contributions + statistics + literature features"),
         ("gbt_all", "GBT over engine contributions + statistics + literature features"),
         ("bow_lr", "reference only: LR on the 300 most frequent words (no expert input)")]
DIRW = {1: "T↑", -1: "T↓", 0: "?"}


def latest(pattern):
    fs = sorted(glob.glob(os.path.join(ROOT, "results", pattern)))
    return json.load(open(fs[-1])) if fs else None


def metrics_md(summary):
    for g in GRANS:
        for masked in (False, True):
            o = latest("*_baselines_MeanSentLen_dev-val_%s%s.json" % (g, "_masked" if masked else ""))
            if o:
                summary.setdefault(g, {}).setdefault("MeanSentLen", {})["masked" if masked else "unmasked"] = {
                    k: o["metrics"][k] for k in ("n", "accuracy", "macro_f1", "roc_auc")}
    L = ["# Hybrid models on dev-val\n",
         "All models fitted on **dev-train** (dev sessions before 2002-03-01) and evaluated on **dev-val** "
         "(sessions 2002-03-01 .. 2004-05-05, balanced files; see hybrid/SPLIT.md). The pure engine and "
         "MeanSentLen are scored on exactly the same records. `masked` = topic tokens replaced by XTOPIC "
         "before scoring (evals/topicmask.py); models were always fitted on unmasked dev-train text. "
         "`unseen spk` = AUC on dev-val units none of whose speakers occur in dev-train "
         "(sent n=2139, 100 n=256; too few at 500). Numbers are acc / macro-F1 / AUC.\n"]
    eng = summary.get("500", {}).get("engine_raw", {})
    if eng:
        L.append("Engine: `%s`, tsw source hash `%s`.\n" % (eng.get("engine"), eng.get("tsw_hash")))
    for g in GRANS:
        n = summary.get(g, {}).get("engine_raw", {}).get("unmasked", {}).get("n")
        L += ["\n## gran = %s (n = %s)\n" % (g, n),
              "| model | what | unmasked acc / F1 / AUC | masked acc / F1 / AUC | unseen spk AUC |",
              "|---|---|---|---|---:|"]
        for k, desc in ORDER:
            r = summary.get(g, {}).get(k)
            if not r:
                continue
            u, m = r.get("unmasked", {}), r.get("masked", {})
            us = r.get("unseen_speaker", {}).get("roc_auc")
            f = lambda d: "%.3f / %.3f / **%.3f**" % (d["accuracy"], d["macro_f1"], d["roc_auc"]) if d else "-"
            L.append("| `%s` | %s | %s | %s | %s |" % (k, desc, f(u), f(m),
                                                      "-" if us is None or us != us else "%.3f" % us))
    open(os.path.join(R, "METRICS.md"), "w").write("\n".join(L) + "\n")


def univariate(g):
    d = cache.load(g)
    X = F.transform(d["C"].astype(np.float64), np)
    tr = ~d["is_val"]
    Xtr, y = X[tr], d["labels"][tr].astype(int)
    out = {}
    ci = {c: i for i, c in enumerate(F.COUNT_NAMES)}
    for j, n in enumerate(F.feature_names()):
        x = Xtr[:, j]
        base = n.split("_", 1)[1] if n.startswith(("rate_", "share_")) else None
        hits = float(d["C"][tr, ci[base]].sum()) if base in ci else float("inf")
        out[n] = {"mean_o": float(x[y == 0].mean()), "mean_t": float(x[y == 1].mean()),
                  "auc": metrics.roc_auc(list(y), list(x)), "hits": hits}
    return out


def coef_md(uni):
    L = ["# What the expert-feature LR learned (`lr_expert`, fitted on dev-train)\n",
         "Per granularity, the 25 features with the largest standardised coefficient (effect of +1 SD of "
         "the feature on the log-odds of *translated*). `lit` = predicted direction in docs/literature.md "
         "(T↑ more frequent in translations, T↓ less, ? = calibrate); `data` = direction of the class means "
         "on dev-train, with the feature's AUC alone (> .5 = higher in T); `LR` = sign of the coefficient "
         "given all other features. ✗ marks disagreement with the literature. LR signs of collinear "
         "features (e.g. a rate and its ratio) can differ from their univariate direction; trust `data` "
         "for the direction of a single cue, `LR` for its unique contribution.\n"]
    for g in GRANS:
        m = json.load(open(os.path.join(T.MODEL_DIR, "lr_expert.%s.json" % g)))
        cs = np.asarray(m["coef_std"])
        L += ["\n## gran = %s (C = %g)\n" % (g, m["C"]),
              "| # | feature | plain meaning | lit | data (AUC) | LR std. coef | |",
              "|---:|---|---|---|---|---:|---|"]
        for r, j in enumerate(np.argsort(-np.abs(cs))[:25]):
            n = m["features"][j]
            lit, d, desc = F.feature_meta(n)
            u = uni[g][n]
            dd = 1 if u["auc"] > 0.5 else -1
            bad = "✗ data" if (d and dd != d and abs(u["auc"] - 0.5) > 0.01) else ""
            if d and np.sign(cs[j]) != d:
                bad = (bad + " ✗ LR").strip()
            L.append("| %d | `%s` | %s (%s) | %s | %s (%.3f) | %+.3f | %s |"
                     % (r + 1, n, desc.replace("|", "\\|"), lit, DIRW[d], DIRW[dd], u["auc"], cs[j], bad))
    open(os.path.join(R, "coefficients.md"), "w").write("\n".join(L) + "\n")


def litcheck_md(uni):
    L = ["# Literature directions vs dev-train data\n",
         "For every expert feature with a predicted direction in docs/literature.md: the feature's AUC "
         "alone on dev-train at 100- and 500-token granularity (> .5 = higher in translations), the class "
         "means at 500 tokens and the T/O ratio of the class means (rare cues have AUC ≈ .5 because most "
         "units have zero hits, so the ratio matters for them). A granularity *agrees with T↑* if AUC ≥ .53 "
         "or T/O ≥ 1.25 (and conversely AUC ≤ .47 or T/O ≤ 0.8 for T↓); the ratio only counts for items with "
         "≥ 30 hits in dev-train at that granularity (`hits` column, 500). **Contradicted** = the data points "
         "the other way at both 100 and 500; **confirmed** = the predicted way at both; otherwise "
         "weak/mixed. Ratios (`r_*`) are the share of the T-leaning variant.\n",
         "| feature | meaning | lit id | lit | AUC 100 | AUC 500 | mean O (500) | mean T (500) | T/O (500) | hits | verdict |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    rows = []
    for n in F.feature_names():
        lit, d, desc = F.feature_meta(n)
        a1, a5 = uni["100"][n]["auc"], uni["500"][n]["auc"]

        def ratio(g):
            u = uni[g][n]
            return (u["mean_t"] + 1e-9) / (u["mean_o"] + 1e-9)

        def says(g):  # +1 data says T-up, -1 T-down, 0 unclear
            a, r = uni[g][n]["auc"], ratio(g)
            ok = uni[g][n]["hits"] >= 30  # ratio only for items with enough dev-train hits
            up, down = a >= .53 or (ok and r >= 1.25), a <= .47 or (ok and r <= 0.8)
            return 0 if up == down else (1 if up else -1)
        s1, s5 = says("100"), says("500")
        if d == 0:
            verdict = "no prediction; data: %s" % {1: "T↑", -1: "T↓", 0: "unclear"}[s5]
        elif s1 == s5 == -d:
            verdict = "**contradicted**"
        elif s1 == s5 == d:
            verdict = "confirmed"
        else:
            verdict = "weak/mixed"
        rows.append((verdict.startswith("**"), -abs(a5 - .5),
                     "| `%s` | %s | %s | %s | %.3f | %.3f | %.3g | %.3g | %.2f | %s | %s |"
                     % (n, desc.replace("|", "\\|"), lit, DIRW[d], a1, a5, uni["500"][n]["mean_o"],
                        uni["500"][n]["mean_t"], ratio("500"),
                        "%.0f" % uni["500"][n]["hits"] if uni["500"][n]["hits"] < 1e18 else "-", verdict)))
    rows.sort(key=lambda r: (not r[0], r[1]))
    L += [r[2] for r in rows]
    open(os.path.join(R, "literature_check.md"), "w").write("\n".join(L) + "\n")


def tree_md():
    L = ["# Decision-tree rules over the expert features (candidate rules for the expert system)\n",
         "Shallow trees (`tree_d3`, `tree_d4`) fitted on dev-train over the literature features "
         "(hybrid/features.py), un-scaled, so thresholds are in natural units: `rate_x` = hits per 1000 "
         "words, `share_x` = share of sentences, `r_x` = share of the T-leaning variant among the choices "
         "(smoothed: (k_T+.5)/(k_T+k_O+1)). Leaves are listed by how far they move the prior, weighted by "
         "size. P(trans) is the dev-train leaf purity (prior ≈ .47).\n"]
    for g in GRANS:
        for mn in ("tree_d4",):
            m = json.load(open(os.path.join(T.MODEL_DIR, "%s.%s.json" % (mn, g))))
            L.append("\n## %s, gran = %s\n" % (mn, g))
            for conds, p, n in T.tree_rules(m, m["features"], 0.47):
                txt = " AND ".join("%s %s %.3g" % c for c in conds)
                L.append("- IF %s THEN P(trans) = %.2f (n = %d)" % (txt, p, n))
    open(os.path.join(R, "tree_rules.md"), "w").write("\n".join(L) + "\n")


def main():
    summary = json.load(open(os.path.join(R, "metrics.json")))
    metrics_md(summary)
    uni = {g: univariate(g) for g in GRANS}
    coef_md(uni)
    litcheck_md(uni)
    tree_md()
    print("wrote", ", ".join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(R, "*.md")))))


if __name__ == "__main__":
    main()

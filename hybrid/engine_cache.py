#!/usr/bin/env python3
"""Cache the rule engine's per-rule contributions on dev-train / dev-val (Phase B input).

    python3 -m hybrid.engine_cache --gran 500 100 sent      # (+ --clf to pick an engine)

Runs `tsw.engine:Default` (or --clf; must be a tsw Engine) on dev-train
(unmasked) and dev-val (unmasked + topic-masked), with the same semantics as the
harness's evaluate(), and stores (a) the explain() dicts as a dense matrix (one
column per rule, plus the "cap:<family>" adjustments) and (b) the raw statistic
behind each rule (hits per 1000 words, choice shares, unit means), computed on
the same Doc. As a by-product it saves the official pure-rules dev-val results
(split="dev-val") exactly as `evals/run.py --split dev-val` would, so every
hybrid model is compared with the engine on the same records.

Cost: one engine pass over ~1.33x the dev tokens per granularity (dev-train +
2x dev-val). With the 5-rule starter engine (~4 us/token) all three grans take
~30 CPU-seconds; time it again when families are merged (it scales with the
number of rules) and use Slurm if it exceeds ~1 CPU-minute.

Output: data/cache/hybrid/engine.<gran>.npz (gitignored), tagged with the
engine's rule list, the tsw/ source hash and git commit.
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals"))
sys.path.insert(0, ROOT)
import devsplit  # noqa: E402
import metrics  # noqa: E402
import run  # noqa: E402
from hybrid import cache  # noqa: E402


def tsw_hash():
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(ROOT, "tsw", "**", "*.py"), recursive=True)):
        if "/tests/" in p:
            continue
        h.update(p[len(ROOT):].encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def path(gran, tag="default"):
    return os.path.join(cache.CACHE_DIR, "engine.%s.%s.npz" % (tag, gran))


def stat_names(clf):
    out = ["n_words"]
    for r in getattr(clf, "rules", []):
        k = type(r).__name__
        if k in ("HitRule", "RateRule"):
            out.append(r.name + ":per1000")
        elif k == "ChoiceRule":
            out += [r.name + ":share_a", r.name + ":choices_per1000"]
        elif k == "StatRule":
            out.append(r.name + ":mean")
    return out


def rule_stats(clf, doc):
    """The raw statistics each rule computes before turning them into an LLR:
    hits per 1000 words (Hit/Rate), share of variant A + choices per 1000 words
    (Choice), the unit mean (Stat; NaN if no units). Order = stat_names()."""
    n = doc.n_words
    out = [float(n)]
    for r in getattr(clf, "rules", []):
        k = type(r).__name__
        if k in ("HitRule", "RateRule"):
            out.append(1000.0 * r.matcher.count(doc) / max(n, 1))
        elif k == "ChoiceRule":
            ka, kb = r.counts(doc)
            out += [(ka + 0.5) / (ka + kb + 1.0), 1000.0 * (ka + kb) / max(n, 1)]
        elif k == "StatRule":
            out.append(float(r.statistic(doc)[0]))
    return out


def evaluate(clf, recs, mask_topic):
    """Same semantics as evals/run.py:evaluate (score_and_explain on the (masked)
    text, NaN -> 0) plus rule_stats on the same Doc."""
    from tsw.doc import Doc
    from tsw.engine import CAP_PREFIX
    import topicmask
    labels, scores, expl, stats = [], [], [], []
    for r in recs:
        text = topicmask.mask(r["text"]) if mask_topic else r["text"]
        doc = Doc(text)
        total, contrib, fam = clf._run(doc)
        e = dict(contrib)
        for f, (s, cs) in fam.items():
            if cs != s:
                e[CAP_PREFIX + f] = cs - s
        if math.isnan(total):
            total = 0.0
        labels.append(r["label"])
        scores.append(total)
        expl.append(e)
        stats.append(rule_stats(clf, doc))
    return {"metrics": metrics.all_metrics(labels, scores), "scores": scores, "labels": labels,
            "rules": metrics.rule_diagnostics(labels, expl), "explanations": expl,
            "stats": np.asarray(stats, np.float64)}


def build(gran, clf_spec, tag, save_results=True):
    t0 = time.time()
    clf = run.load_clf(clf_spec)
    dpath = run.data_path(run.data_dir(), "dev", gran, True)
    recs = run.load_records(dpath)
    tr = devsplit.select(recs, "dev-train")
    va = devsplit.select(recs, "dev-val")
    res = {"train": evaluate(clf, tr, False), "val": evaluate(clf, va, False),
           "val_masked": evaluate(clf, va, True)}
    rule_names = [r.name for r in getattr(clf, "rules", [])]
    cols = list(rule_names)
    for k in res.values():
        for e in k["explanations"]:
            for name in e:
                if name not in cols:
                    cols.append(name)
    ci = {c: j for j, c in enumerate(cols)}

    def dense(expl):
        M = np.zeros((len(expl), len(cols)), np.float64)
        for i, e in enumerate(expl):
            for k, v in e.items():
                M[i, ci[k]] = v
        return M

    fam = {r.name: r.family for r in getattr(clf, "rules", [])}
    meta = [r.meta() for r in getattr(clf, "rules", [])]
    gi = run.git_info()
    np.savez_compressed(
        path(gran, tag),
        E_train=dense(res["train"]["explanations"]), E_val=dense(res["val"]["explanations"]),
        E_val_masked=dense(res["val_masked"]["explanations"]),
        s_train=np.asarray(res["train"]["scores"]), s_val=np.asarray(res["val"]["scores"]),
        s_val_masked=np.asarray(res["val_masked"]["scores"]),
        y_train=np.asarray(res["train"]["labels"], np.int8), y_val=np.asarray(res["val"]["labels"], np.int8),
        ids_train=np.asarray([r["id"] for r in tr]), ids_val=np.asarray([r["id"] for r in va]),
        dates_train=np.asarray([r["date"] for r in tr]),
        S_train=res["train"]["stats"], S_val=res["val"]["stats"], S_val_masked=res["val_masked"]["stats"],
        stat_cols=np.asarray(stat_names(clf)),
        cols=np.asarray(cols), rule_family=np.asarray([fam.get(c, c.split(":", 1)[-1]) for c in cols]),
        rule_meta=json.dumps(meta), clf=clf_spec, tsw_hash=tsw_hash(), git_commit=gi["git_commit"] or "",
        prior=float(getattr(clf, "prior", 0.0)), data_sha256=run.sha256_file(dpath))
    if save_results:  # official pure-rules numbers on dev-val, same format as run_one
        for masked in (False, True):
            r = res["val_masked" if masked else "val"]
            out = {"harness_version": run.HARNESS_VERSION,
                   "timestamp": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"), "clf": clf_spec,
                   "split": "dev-val", "gran": gran, "balanced": True, "mask_topic": masked, "limit": None,
                   "limit_seed": None, **gi, "data_file": os.path.basename(dpath),
                   "data_sha256": run.sha256_file(dpath), "metrics": r["metrics"], "rules": r["rules"],
                   "elapsed_sec": None, "argv": sys.argv, "subset_rule": devsplit.describe("dev-val"),
                   "eval_mode": "harness evaluate() via hybrid/engine_cache.py"}
            if masked:
                out["unmasked_metrics"] = res["val"]["metrics"]
                out["topic_auc_drop"] = res["val"]["metrics"]["roc_auc"] - r["metrics"]["roc_auc"]
            run.save_result(out, quiet=True)
    m, mm = res["val"]["metrics"], res["val_masked"]["metrics"]
    print("gran=%s %s: %d rules, train %d val %d | dev-val AUC %.4f (masked %.4f)  %.1fs"
          % (gran, clf_spec, len(rule_names), len(tr), len(va), m["roc_auc"], mm["roc_auc"], time.time() - t0))


def load(gran, tag="default"):
    z = np.load(path(gran, tag), allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if str(d["tsw_hash"]) != tsw_hash():
        print("WARNING: engine cache %s was built from different tsw/ sources (rebuild with "
              "python3 -m hybrid.engine_cache --gran %s)" % (path(gran, tag), gran), file=sys.stderr)
    d["rule_meta"] = json.loads(str(d["rule_meta"]))
    d["cols"] = [str(c) for c in d["cols"]]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gran", nargs="+", default=["500", "100", "sent"])
    ap.add_argument("--clf", default="tsw.engine:Default")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--no-save", action="store_true", help="do not write the pure-rules results JSON")
    a = ap.parse_args()
    for g in a.gran:
        build(run.GRANS[g], a.clf, a.tag, save_results=not a.no_save)


if __name__ == "__main__":
    main()

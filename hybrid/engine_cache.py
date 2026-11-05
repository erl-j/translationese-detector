#!/usr/bin/env python3
"""Cache the rule engine's per-rule contributions and rule statistics on trn (Phase B input).

    python3 -m hybrid.engine_cache --gran 500 100 sent      # (+ --clf to pick an engine)

Runs `tsw.engine:Default` (or --clf; must be a tsw Engine) on every trn.<gran>.bal
record (unmasked), with the same semantics as the harness's evaluate(), and
stores (a) the explain() dicts as a dense matrix (one column per rule, plus the
"cap:<family>" adjustments), (b) the raw statistic behind each rule (hits per
1000 words, choice shares, unit means) computed on the same Doc, and (c) the
engine score. val/test are never touched here (they are scored live by
hybrid/eval_val.py).

Cost: one engine pass over trn (~4.8M tokens over the three grans). With the
5-rule starter engine (~6 us/token incl. statistics) about 30 CPU-seconds; it
scales with the number of rules, so after merging the six families run it via
Slurm (hybrid/slurm/retrain_val.sh).

Output: data/cache/hybrid/engine.<tag>.<gran>.npz (gitignored), tagged with the
engine's rule list, the tsw/ source hash and git commit.
"""
import argparse
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


def _kind(r):
    """Rule type by inheritance (subclasses such as stats.LengthStat count as their base)."""
    from tsw.rules.base import ChoiceRule, HitRule, RateRule, StatRule
    for k, cls in (("hitrate", HitRule), ("hitrate", RateRule), ("choice", ChoiceRule), ("stat", StatRule)):
        if isinstance(r, cls):
            return k
    return None


def stat_names(clf):
    out = ["n_words"]
    for r in getattr(clf, "rules", []):
        k = _kind(r)
        if k == "hitrate":
            out.append(r.name + ":per1000")
        elif k == "choice":
            out += [r.name + ":share_a", r.name + ":choices_per1000"]
        elif k == "stat":
            out.append(r.name + ":mean")
    return out


def rule_stats(clf, doc):
    """The raw statistics each rule computes before turning them into an LLR:
    hits per 1000 words (Hit/Rate), share of variant A + choices per 1000 words
    (Choice), the unit mean (Stat and subclasses; NaN if no units). Predicate
    rules have no statistic (their contribution is used). Order = stat_names()."""
    n = doc.n_words
    out = [float(n)]
    for r in getattr(clf, "rules", []):
        k = _kind(r)
        if k == "hitrate":
            out.append(1000.0 * r.matcher.count(doc) / max(n, 1))
        elif k == "choice":
            ka, kb = r.counts(doc)
            out += [(ka + 0.5) / (ka + kb + 1.0), 1000.0 * (ka + kb) / max(n, 1)]
        elif k == "stat":
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
            "explanations": expl,
            "stats": np.asarray(stats, np.float64)}


def build(gran, clf_spec, tag):
    t0 = time.time()
    clf = run.load_clf(clf_spec)
    dpath = run.data_path(run.data_dir(), "trn", gran, True)
    recs = run.load_records(dpath)
    res = evaluate(clf, recs, False)
    rule_names = [r.name for r in getattr(clf, "rules", [])]
    cols = list(rule_names)
    for e in res["explanations"]:
        for name in e:
            if name not in cols:
                cols.append(name)
    ci = {c: j for j, c in enumerate(cols)}
    E = np.zeros((len(recs), len(cols)), np.float64)
    for i, e in enumerate(res["explanations"]):
        for k, v in e.items():
            E[i, ci[k]] = v
    fam = {r.name: r.family for r in getattr(clf, "rules", [])}
    meta = [r.meta() for r in getattr(clf, "rules", [])]
    gi = run.git_info()
    os.makedirs(cache.CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        path(gran, tag), E=E, s=np.asarray(res["scores"]), y=np.asarray(res["labels"], np.int8),
        ids=np.asarray([r["id"] for r in recs]), dates=np.asarray([r["date"] for r in recs]),
        S=res["stats"], stat_cols=np.asarray(stat_names(clf)),
        cols=np.asarray(cols), rule_family=np.asarray([fam.get(c, c.split(":", 1)[-1]) for c in cols]),
        rule_meta=json.dumps(meta), clf=clf_spec, tsw_hash=tsw_hash(), git_commit=gi["git_commit"] or "",
        prior=float(getattr(clf, "prior", 0.0)), data_sha256=run.sha256_file(dpath))
    print("gran=%s %s: %d rules, trn %d | engine trn AUC %.4f  %.1fs"
          % (gran, clf_spec, len(rule_names), len(recs), res["metrics"]["roc_auc"], time.time() - t0))


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
    a = ap.parse_args()
    for g in a.gran:
        build(run.GRANS[g], a.clf, a.tag)


if __name__ == "__main__":
    main()

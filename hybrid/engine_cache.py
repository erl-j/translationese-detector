#!/usr/bin/env python3
"""Cache the rule engine's per-rule contributions on dev-train / dev-val (Phase B input).

    python3 -m hybrid.engine_cache --gran 500 100 sent      # (+ --clf to pick an engine)

Runs `tsw.engine:Default` (or --clf) through the harness's own evaluate()
on dev-train (unmasked) and dev-val (unmasked + topic-masked) and stores the
explain() dicts as a dense matrix (one column per rule, plus the "cap:<family>"
adjustments). As a by-product it saves the official pure-rules dev-val results
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


def build(gran, clf_spec, tag, save_results=True):
    t0 = time.time()
    clf = run.load_clf(clf_spec)
    dpath = run.data_path(run.data_dir(), "dev", gran, True)
    recs = run.load_records(dpath)
    tr = devsplit.select(recs, "dev-train")
    va = devsplit.select(recs, "dev-val")
    res = {"train": run.evaluate(clf, tr, False), "val": run.evaluate(clf, va, False),
           "val_masked": run.evaluate(clf, va, True)}
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

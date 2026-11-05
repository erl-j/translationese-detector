#!/usr/bin/env python3
"""Flatten every branch's eval results into the files the branch dashboard reads.

    python3 evals/export_dashboard.py [--out data/cache/dashboard]

Reads results/all/*.json (the shared copy every worktree writes to) and
data/manifest.json, and writes:
  runs.json    one row per eval run (branch, commit, clf, gran, mask, metrics)
  rules.json   per-rule diagnostics for the LATEST run of each
               (branch, clf, split, gran, mask, data) that has rules
  corpus.json  example and token counts per split, granularity and class
Stdlib only, a few seconds of CPU.
"""
import argparse
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = "./results/all"


def num(x, nd=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("TSW_SHARED_RESULTS", SHARED))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cache", "dashboard"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(ROOT, "data", "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    current = {k: v.get("sha256") for k, v in man.get("files", {}).items()}

    runs, latest, stale = [], {}, 0
    for p in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        if current.get(r.get("data_file")) not in (None, r.get("data_sha256")):
            stale += 1  # scored on an older build of the data
            continue
        m = r.get("metrics") or {}
        commit = (r.get("git_commit") or "")[:8]
        row = {
            "run_id": os.path.basename(p)[:-5],
            "time": r.get("timestamp"),
            "branch": r.get("git_branch") or "unknown",
            "commit": commit + ("+dirty" if r.get("git_dirty") else ""),
            "clf": r.get("clf"),
            "split": r.get("split"),
            "gran": str(r.get("gran")),
            "data": "bal" if r.get("balanced", True) else "all",
            "masked": bool(r.get("mask_topic")),
            "limited": bool(r.get("limit")),
            "n": m.get("n"),
            "accuracy": num(m.get("accuracy")),
            "macro_f1": num(m.get("macro_f1")),
            "auc": num(m.get("roc_auc")),
        }
        runs.append(row)
        if r.get("rules"):
            key = (row["branch"], row["clf"], row["split"], row["gran"], row["masked"], row["data"])
            if key not in latest or row["time"] > latest[key][0]["time"]:
                latest[key] = (row, r["rules"])

    rules = []
    for row, rr in latest.values():
        for name, s in rr.items():
            rules.append({
                "id": "|".join([row["branch"], row["clf"], row["split"], row["gran"],
                                "m" if row["masked"] else "u", row["data"], name]),
                "branch": row["branch"], "clf": row["clf"], "split": row["split"], "gran": row["gran"],
                "masked": row["masked"], "commit": row["commit"], "time": row["time"], "rule": name,
                "fire_orig": num(s.get("fire_orig")), "fire_trans": num(s.get("fire_trans")),
                "log_odds": num(s.get("log_odds"), 3), "auc": num(s.get("auc")),
                "sign_acc": num(s.get("sign_acc")), "n_fire": s.get("n_fire"),
            })

    corpus = []
    for fname, s in sorted(man.get("files", {}).items()):
        split, gran, data = fname.split(".")[:3]
        for cls, lab in (("orig", "original"), ("trans", "translated")):
            corpus.append({"split": split, "gran": gran, "data": data, "class": lab,
                           "examples": s.get("n_" + cls), "tokens": s.get("tokens_" + cls)})

    for name, obj in (("runs", runs), ("rules", rules), ("corpus", corpus)):
        with open(os.path.join(args.out, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print("skipped %d runs on stale data" % stale)
    print("wrote %d runs, %d rule rows, %d corpus rows to %s" % (len(runs), len(rules), len(corpus), args.out))


if __name__ == "__main__":
    main()

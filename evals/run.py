#!/usr/bin/env python3
"""Evaluate a rule-based classifier on the Europarl translationese splits.

Examples
    python3 evals/run.py --clf baselines:Constant --split dev --gran 100
    python3 evals/run.py --clf "baselines:Random(seed=3)" --split dev --gran sent --mask-topic
    python3 evals/run.py --clf baselines:MeanSentLen --gran 500 --show-errors 5
    python3 evals/run.py --clf tsw.engine:Expert --split test --gran 500 --i-mean-it

--clf is "module:Name" or "module:Name(kw=literal, ...)". Modules are looked up
in evals/ and in the repo root (so tsw.something works). If Name is a class it
is instantiated (with the given keyword arguments); otherwise it must already
be an object with score(text).

Classifier contract
    score(text) -> float               > 0 means "translated from English"
    explain(text) -> {rule: float}     optional, signed per-rule contributions
    score_and_explain(text)            optional fast path -> (score, explain-dict)

Output: a readable report on stdout, results/<ts>_<clf>_<split>_<gran>[_masked].json
in the repo root of the worktree you run from, a copy in the shared dir
($TSW_SHARED_RESULTS or ./results/all),
and one line appended to results/LEADERBOARD.md.
"""
import argparse
import ast
import datetime
import hashlib
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import devsplit  # noqa: E402
import metrics  # noqa: E402
import topicmask  # noqa: E402

SHARED_DEFAULT = "./results/all"
GRANS = {"sent": "sent", "sentence": "sent", "s": "sent", "100": "100", "500": "500"}
HARNESS_VERSION = 1


# --------------------------------------------------------------------------
# environment helpers
# --------------------------------------------------------------------------

def _git(*args, cwd=ROOT):
    try:
        return subprocess.run(["git"] + list(args), cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, universal_newlines=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def git_info():
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": _git("rev-parse", "HEAD") or None,
            "git_dirty": bool(status),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "worktree": _git("rev-parse", "--show-toplevel") or ROOT}


def data_dir(override=None):
    """data/processed of this worktree, else of the main checkout (worktrees
    share one built copy because data/processed is gitignored)."""
    cands = [override, os.environ.get("TSW_DATA"), os.path.join(ROOT, "data", "processed")]
    common = _git("rev-parse", "--git-common-dir")
    if common:
        common = os.path.normpath(os.path.join(ROOT, common))
        cands.append(os.path.join(os.path.dirname(common), "data", "processed"))
    for c in cands:
        if c and os.path.isdir(c):
            return c
    sys.exit("no data/processed found (tried %s); build it with: python3 -I data/build.py"
             % [c for c in cands if c])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# classifier loading
# --------------------------------------------------------------------------

def load_clf(spec):
    if ":" not in spec:
        sys.exit("--clf must look like module:Name or module:Name(kw=value)")
    mod_name, obj_spec = spec.split(":", 1)
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(\((.*)\))?\s*", obj_spec, re.S)
    if not m:
        sys.exit("bad classifier spec %r" % obj_spec)
    name, call, argstr = m.group(1), m.group(2), m.group(3)
    kwargs = {}
    if call and argstr.strip():
        node = ast.parse("f(%s)" % argstr, mode="eval").body
        if node.args:
            sys.exit("use keyword arguments only in --clf, e.g. Random(seed=3)")
        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, name)
    if isinstance(obj, type):
        obj = obj(**kwargs)
    elif kwargs:
        sys.exit("%s is not a class; cannot pass arguments" % name)
    if not hasattr(obj, "score") and not hasattr(obj, "score_and_explain"):
        sys.exit("%s has no score(text) method" % spec)
    return obj


def scorer(clf):
    """Return f(text) -> (score, explanation-dict or None)."""
    if hasattr(clf, "score_and_explain"):
        def f(t):
            s, e = clf.score_and_explain(t)
            return float(s), dict(e or {})
    elif hasattr(clf, "explain"):
        def f(t):
            return float(clf.score(t)), dict(clf.explain(t) or {})
    else:
        def f(t):
            return float(clf.score(t)), None
    return f


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def file_split(split):
    """Pseudo-splits dev-train / dev-val (evals/devsplit.py) live in the dev files."""
    return "dev" if split in devsplit.SUBSETS else split


def data_path(ddir, split, gran, balanced):
    return os.path.join(ddir, "%s.%s.%s.jsonl" % (file_split(split), gran, "bal" if balanced else "all"))


def load_records(path, limit=None, seed=0, subset=None):
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    if subset:
        recs = devsplit.select(recs, subset)
    if limit and limit < len(recs):
        # stratified random subset, deterministic, kept in corpus order
        rng = random.Random("limit|%d" % seed)
        idx = {0: [], 1: []}
        for i, r in enumerate(recs):
            idx[r["label"]].append(i)
        n1 = round(limit * len(idx[1]) / len(recs))
        keep = rng.sample(idx[0], min(len(idx[0]), limit - n1)) + rng.sample(idx[1], min(len(idx[1]), n1))
        recs = [recs[i] for i in sorted(keep)]
    return recs


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(clf, recs, mask_topic):
    f = scorer(clf)
    labels, scores, expl = [], [], []
    has_expl = True
    for r in recs:
        text = topicmask.mask(r["text"]) if mask_topic else r["text"]
        s, e = f(text)
        if math.isnan(s):
            s = 0.0
        labels.append(r["label"])
        scores.append(s)
        if e is None:
            has_expl = False
        expl.append(e)
    out = {"metrics": metrics.all_metrics(labels, scores), "scores": scores, "labels": labels}
    out["rules"] = metrics.rule_diagnostics(labels, expl) if has_expl else None
    out["explanations"] = expl if has_expl else None
    return out


def fmt_metrics(m):
    c = m["confusion"]
    return ("n=%d (orig %d / trans %d)  acc=%.4f  macroF1=%.4f  AUC=%.4f  pred_trans=%.3f\n"
            "confusion (rows=true, cols=pred):      pred_orig  pred_trans\n"
            "                          true_orig   %9d  %10d\n"
            "                          true_trans  %9d  %10d"
            % (m["n"], m["n_orig"], m["n_trans"], m["accuracy"], m["macro_f1"], m["roc_auc"],
               m["pred_trans_rate"], c["tn"], c["fp"], c["fn"], c["tp"]))


def show_errors(recs, res, n, mask_topic, width=600):
    errs = [i for i, (y, s) in enumerate(zip(res["labels"], res["scores"])) if (s > 0) != (y == 1)]
    # most confidently wrong first, alternating classes
    by = {0: sorted([i for i in errs if res["labels"][i] == 0], key=lambda i: -abs(res["scores"][i])),
          1: sorted([i for i in errs if res["labels"][i] == 1], key=lambda i: -abs(res["scores"][i]))}
    picked = []
    while len(picked) < n and (by[0] or by[1]):
        for lab in (0, 1):
            if by[lab] and len(picked) < n:
                picked.append(by[lab].pop(0))
    print("\n== %d misclassified examples (most confident first; %d errors total) ==" % (len(picked), len(errs)))
    for i in picked:
        r = recs[i]
        text = topicmask.mask(r["text"]) if mask_topic else r["text"]
        print("\n--- id=%s  true=%s  score=%+.3f  date=%s  speaker=%s"
              % (r["id"], "TRANS" if r["label"] else "ORIG", res["scores"][i], r["date"], r.get("speaker")))
        print(text[:width] + (" [...]" if len(text) > width else ""))
        if res["explanations"] is not None:
            e = res["explanations"][i]
            top = sorted(e.items(), key=lambda kv: -abs(kv[1]))[:12]
            print("    rules: " + (", ".join("%s=%+.3f" % kv for kv in top) if top else "(none fired)"))


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9.=+-]+", "_", s).strip("_")[:80]


def append_leaderboard(path, row):
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("# Leaderboard\n\nOne line per `evals/run.py` invocation (appended automatically). "
                    "AUC is ROC-AUC of the raw scores; `mask` = topic tokens masked.\n\n"
                    "| time (UTC) | branch | commit | clf | split | gran | data | mask | n | acc | macro-F1 | AUC |\n"
                    "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|\n")
        f.write(row + "\n")


def run_one(clf, clf_spec, split, gran, balanced=True, mask_topic=False, limit=None, seed=0,
            ddir=None, save=True, compare_unmasked=False, rules_top=40, errors=0, quiet=False,
            argv=None):
    """Evaluate one (split, gran, mask) condition; print and save. Returns result dict."""
    t0 = time.time()
    ddir = data_dir(ddir)
    path = data_path(ddir, split, gran, balanced)
    if not os.path.exists(path):
        sys.exit("missing %s; build it with: python3 -I data/build.py" % path)
    subset = split if split in devsplit.SUBSETS else None
    recs = load_records(path, limit, seed, subset=subset)
    res = evaluate(clf, recs, mask_topic)
    ref = evaluate(clf, recs, False) if (mask_topic and compare_unmasked) else None
    gi = git_info()
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = {"harness_version": HARNESS_VERSION, "timestamp": ts, "clf": clf_spec,
           "split": split, "gran": gran, "balanced": balanced, "mask_topic": bool(mask_topic),
           "limit": limit, "limit_seed": seed if limit else None, **gi,
           "data_file": os.path.basename(path), "data_sha256": sha256_file(path),
           "metrics": res["metrics"], "rules": res["rules"],
           "elapsed_sec": None, "argv": argv}
    if subset:
        out["subset_rule"] = devsplit.describe(subset)
    if ref is not None:
        out["unmasked_metrics"] = ref["metrics"]
        out["topic_auc_drop"] = ref["metrics"]["roc_auc"] - res["metrics"]["roc_auc"]
    out["elapsed_sec"] = round(time.time() - t0, 2)

    if not quiet:
        print("== %s | split=%s gran=%s data=%s mask_topic=%s | %s@%s%s"
              % (clf_spec, split, gran, "bal" if balanced else "all", mask_topic, gi["git_branch"],
                 (gi["git_commit"] or "?")[:8], "+dirty" if gi["git_dirty"] else ""))
        print(fmt_metrics(res["metrics"]))
        if ref is not None:
            print("unmasked: acc=%.4f macroF1=%.4f AUC=%.4f   (AUC drop from masking: %+.4f)"
                  % (ref["metrics"]["accuracy"], ref["metrics"]["macro_f1"], ref["metrics"]["roc_auc"],
                     out["topic_auc_drop"]))
        if res["rules"]:
            print("\n== per-rule diagnostics (O=original, T=translated; sorted by |log-odds| of firing) ==")
            print(metrics.format_rule_table(res["rules"], top=rules_top))
        if errors:
            show_errors(recs, res, errors, mask_topic)

    if save:
        save_result(out, quiet=quiet)
    return out


def save_result(out, quiet=False):
    """Write results/<name>.json, the shared copy and a LEADERBOARD line for a result dict
    built by run_one (or by an external evaluator that fills the same fields)."""
    ts, clf_spec, split, gran = out["timestamp"], out["clf"], out["split"], out["gran"]
    balanced, mask_topic, limit, gi = out["balanced"], out["mask_topic"], out["limit"], out
    name = "%s_%s_%s_%s%s%s.json" % (ts, safe_name(clf_spec), split, gran,
                                     "" if balanced else "_all", "_masked" if mask_topic else "")
    rdir = os.path.join(ROOT, "results")
    os.makedirs(rdir, exist_ok=True)
    blob = json.dumps(out, indent=1, sort_keys=True, allow_nan=True)
    with open(os.path.join(rdir, name), "w", encoding="utf-8") as f:
        f.write(blob + "\n")
    shared = os.environ.get("TSW_SHARED_RESULTS", SHARED_DEFAULT)
    try:
        os.makedirs(shared, exist_ok=True)
        sname = "%s_%s_%s" % (ts, safe_name(gi["git_branch"] or "nobranch"), name[len(ts) + 1:])
        with open(os.path.join(shared, sname), "w", encoding="utf-8") as f:
            f.write(blob + "\n")
    except OSError as e:
        print("WARNING: could not write shared copy to %s: %s" % (shared, e), file=sys.stderr)
    m = out["metrics"]
    append_leaderboard(os.path.join(rdir, "LEADERBOARD.md"),
                       "| %s | %s | %s%s | `%s` | %s | %s | %s%s | %s | %d | %.4f | %.4f | %.4f |"
                       % (ts, gi["git_branch"], (gi["git_commit"] or "?")[:8],
                          "+dirty" if gi["git_dirty"] else "", clf_spec, split, gran,
                          "bal" if balanced else "all", "" if not limit else " (limit %d)" % limit,
                          "masked" if mask_topic else "-", m["n"], m["accuracy"], m["macro_f1"],
                          m["roc_auc"]))
    if not quiet:
        print("\nwrote results/%s (+ shared copy, LEADERBOARD.md)" % name)


def check_test_guard(split, i_mean_it, show_errors_n=0):
    if split == "test" and not i_mean_it:
        sys.exit("REFUSING: the test split is locked. Develop on dev; evaluate on test only for a "
                 "final, pre-registered run, and then pass --i-mean-it.")
    if split == "test" and show_errors_n:
        sys.exit("REFUSING: --show-errors is not allowed on the test split (it would leak test examples).")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clf", required=True, help='e.g. baselines:Constant or "baselines:Random(seed=3)"')
    ap.add_argument("--split", default="dev", choices=["dev", "test"] + list(devsplit.SUBSETS),
                    help="dev | test | dev-train / dev-val (date halves of dev, see evals/devsplit.py)")
    ap.add_argument("--gran", default="100", help="sent | 100 | 500")
    ap.add_argument("--unbalanced", action="store_true", help="use *.all.jsonl instead of the balanced *.bal.jsonl")
    ap.add_argument("--mask-topic", action="store_true",
                    help="mask country/nationality/place/party tokens before scoring; also reports unmasked "
                         "metrics for comparison")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a stratified random subset of N examples")
    ap.add_argument("--seed", type=int, default=0, help="seed for --limit sampling")
    ap.add_argument("--show-errors", type=int, default=0, metavar="N", help="print N misclassified examples (dev only)")
    ap.add_argument("--rules-top", type=int, default=40, help="rows in the per-rule table (0 = all)")
    ap.add_argument("--no-save", action="store_true", help="do not write results/ or LEADERBOARD.md")
    ap.add_argument("--data-dir", default=None, help="override data/processed location")
    ap.add_argument("--i-mean-it", action="store_true", help="required to evaluate on the locked test split")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    check_test_guard(args.split, args.i_mean_it, args.show_errors)
    if args.gran not in GRANS:
        sys.exit("--gran must be one of sent, 100, 500")
    clf = load_clf(args.clf)
    run_one(clf, args.clf, args.split, GRANS[args.gran], balanced=not args.unbalanced,
            mask_topic=args.mask_topic, limit=args.limit, seed=args.seed, ddir=args.data_dir,
            save=not args.no_save, compare_unmasked=True, rules_top=args.rules_top or None,
            errors=args.show_errors, argv=sys.argv if argv is None else list(argv))


if __name__ == "__main__":
    main()

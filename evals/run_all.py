#!/usr/bin/env python3
"""Score one classifier on dev-train and dev-val at every granularity x {unmasked, masked}.

    python3 evals/run_all.py --clf baselines:MeanSentLen          # dev-train,dev-val (default)
    python3 evals/run_all.py --clf tsw.engine:Default --splits dev-val
    python3 evals/run_all.py --clf baselines:MeanSentLen --split dev   # old protocol: all of dev
    python3 evals/run_all.py --clf "baselines:Random(seed=1)" --grans 100,500 --limit 2000

dev-train / dev-val are the date halves of dev (evals/devsplit.py): calibrate on
dev-train, compare and select on dev-val; test stays locked (--i-mean-it).
Writes one results JSON per condition (12 by default) via evals/run.py, plus a
shared copy each, and one LEADERBOARD line each; prints a summary table.
Sentence-level dev has ~88k examples, so run this through Slurm when the
classifier is not trivially fast (see README).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run  # noqa: E402

SPLITS = ("dev-train", "dev-val", "dev", "test")
DEFAULT_SPLITS = "dev-train,dev-val"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clf", required=True)
    ap.add_argument("--grans", default="sent,100,500")
    ap.add_argument("--splits", default=None,
                    help="comma-separated splits from dev-train, dev-val, dev, test (default dev-train,dev-val)")
    ap.add_argument("--split", default=None, choices=SPLITS, help="a single split (same as --splits X)")
    ap.add_argument("--i-mean-it", action="store_true")
    ap.add_argument("--unbalanced", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules-top", type=int, default=25)
    ap.add_argument("--quiet", action="store_true", help="only print the summary table")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)
    if args.split and args.splits:
        ap.error("give --split or --splits, not both")
    splits = [s.strip() for s in (args.split or args.splits or DEFAULT_SPLITS).split(",") if s.strip()]
    bad = [s for s in splits if s not in SPLITS]
    if bad:
        ap.error("unknown split(s) %s; choose from %s" % (", ".join(bad), ", ".join(SPLITS)))
    for s in splits:
        run.check_test_guard(s, args.i_mean_it)
    clf = run.load_clf(args.clf)
    rows = []
    for split in splits:
        for g in args.grans.split(","):
            g = run.GRANS[g.strip()]
            for mask in (False, True):
                if not args.quiet:
                    print("\n" + "=" * 100)
                out = run.run_one(clf, args.clf, split, g, balanced=not args.unbalanced, mask_topic=mask,
                                  limit=args.limit, seed=args.seed, save=not args.no_save,
                                  compare_unmasked=False, rules_top=args.rules_top, quiet=args.quiet,
                                  argv=sys.argv if argv is None else list(argv))
                m = out["metrics"]
                rows.append((split, g, "masked" if mask else "-", m["n"], m["accuracy"], m["macro_f1"],
                             m["roc_auc"]))
    print("\n== summary: %s (%s) ==" % (args.clf, "all" if args.unbalanced else "balanced"))
    print("%-9s %-5s %-7s %8s %8s %9s %8s" % ("split", "gran", "mask", "n", "acc", "macroF1", "AUC"))
    for r in rows:
        print("%-9s %-5s %-7s %8d %8.4f %9.4f %8.4f" % r)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score the fitted hybrid models ONCE on val through the harness (metrics only).

    python3 -m hybrid.eval_val                 # Phase A + baselines (+ Phase B with --phase-b)

For every model x gran (sent, 100, 500) x {unmasked, masked} this calls
evals/run.py:run_one(split="val"), so results land in results/ (+ the shared
results/all copy, LEADERBOARD.md) with split="val" and this branch. Reference
classifiers baselines:MeanSentLen and tsw.engine:Default are scored on the same
records. Classifiers are wrapped score-only, so no per-feature/per-rule val
statistics are computed or stored: val is metrics-only. Feature and engine
computations are memoised per text across models (hybrid/clf.py).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evals"))
sys.path.insert(0, ROOT)
import run  # noqa: E402
import topicmask  # noqa: E402

PHASE_A = ["lr_expert", "gbt", "tree_d3", "tree_d4", "bow_lr"]
PHASE_B = ["engine_calib", "stack_rules", "stack_families", "lr_engine", "gbt_engine", "tree_engine",
           "hybrid_plus", "lr_all", "gbt_all"]
REFS = ["baselines:MeanSentLen", "tsw.engine:Default"]


class ScoreOnly:
    def __init__(self, clf):
        self.clf = clf

    def score(self, text):
        return self.clf.score(text)


def specs(phase_b, only=None):
    out = list(REFS) + ["hybrid.clf:Hybrid(model='%s')" % m for m in PHASE_A]
    if phase_b:
        out += ["hybrid.clf:Stacked(model='%s')" % m for m in PHASE_B]
    if only:
        out = [s for s in out if any(o in s for o in only)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase-b", action="store_true", help="also score the Phase-B (engine-stacking) models")
    ap.add_argument("--only", nargs="*", help="substring filter on classifier specs")
    ap.add_argument("--grans", default="sent,100,500")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on a subset (then nothing is saved)")
    a = ap.parse_args(argv)
    rows = []
    for spec in specs(a.phase_b, a.only):
        t0 = time.time()
        clf = ScoreOnly(run.load_clf(spec))
        for g in a.grans.split(","):
            g = run.GRANS[g.strip()]
            for mask in (False, True):
                if hasattr(clf.clf, "prime"):  # batch GBT scoring on exactly the texts run_one will score
                    recs = run.load_records(run.data_path(run.data_dir(), "val", g, True), a.limit, 0)
                    clf.clf.prime([topicmask.mask(r["text"]) if mask else r["text"] for r in recs])
                out = run.run_one(clf, spec, "val", g, balanced=True, mask_topic=mask, limit=a.limit,
                                  save=a.limit is None, compare_unmasked=False, quiet=True,
                                  argv=sys.argv if argv is None else list(argv))
                m = out["metrics"]
                rows.append((spec, g, "masked" if mask else "-", m["n"], m["accuracy"], m["macro_f1"], m["roc_auc"]))
                print("%-45s %-4s %-6s n=%6d acc=%.4f F1=%.4f AUC=%.4f" % rows[-1], flush=True)
        print("   (%s: %.1fs)" % (spec, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

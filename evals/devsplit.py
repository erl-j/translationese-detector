"""Deterministic date split of the dev split into dev-train / dev-val.

Protocol (README, docs/ENGINE.md): rule values are calibrated on dev-train
(tsw/tools/rates.py refuses anything else), rule sets are compared and selected
on dev-val, and the locked test split is scored once at the end. The hybrid
(ML) side track on branch `hybrid-ml` fits its models on dev-train with the
same cut, so results on dev-val are comparable across branches.

    dev-train = dev records with session date <  VAL_START  (1997-01-14 .. 2002-02-xx)
    dev-val   = dev records with session date >= VAL_START  (2002-03-xx .. 2004-05-05)

VAL_START is the first of the month after the 70% point of the dev time span
(1997-01-14 .. 2004-05-05 -> 2002-02-24), so dev-val is the last ~30% of the
span (~33% of the balanced records). Every record carries one session date, so
all units of a session fall on the same side. (hybrid-ml: hybrid/SPLIT.md.)
"""

VAL_START = "2002-03-01"
SUBSETS = ("dev-train", "dev-val")


def subset_of(rec):
    """'dev-train' or 'dev-val' for a dev record (by its session date)."""
    return "dev-val" if rec["date"] >= VAL_START else "dev-train"


def select(recs, subset):
    if subset not in SUBSETS:
        raise ValueError("unknown dev subset %r" % subset)
    return [r for r in recs if subset_of(r) == subset]


def describe(subset):
    return ("dev records with session date %s %s" % ("<" if subset == "dev-train" else ">=", VAL_START))

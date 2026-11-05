#!/usr/bin/env python3
"""Print the dev-train / dev-val split statistics (markdown) from the caches."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hybrid import cache  # noqa: E402


def main():
    print("| gran | subset | n | orig (0) | trans (1) | sessions | first date | last date |")
    print("|---|---|---:|---:|---:|---:|---|---|")
    overlap = {}
    for g in ("sent", "100", "500"):
        d = cache.load(g)
        with open(os.path.join(cache.CACHE_DIR, "dev.%s.speakers.json" % g), encoding="utf-8") as f:
            spk = json.load(f)
        for name, m in (("dev-train", ~d["is_val"]), ("dev-val", d["is_val"])):
            y, dates = d["labels"][m], d["dates"][m]
            print("| %s | %s | %d | %d | %d | %d | %s | %s |" % (g, name, m.sum(), (y == 0).sum(), (y == 1).sum(),
                                                               len(set(dates)), min(dates), max(dates)))
        trd = set(d["dates"][~d["is_val"]])
        assert not trd & set(d["dates"][d["is_val"]]), "a session date is on both sides"
        for lab in (0, 1):
            sp = {}
            for name, m in (("dev-train", ~d["is_val"]), ("dev-val", d["is_val"])):
                sp[name] = set(s for i in np.flatnonzero(m & (d["labels"] == lab)) for s in spk[i] if s)
            overlap[(g, lab)] = (len(sp["dev-train"]), len(sp["dev-val"]), len(sp["dev-train"] & sp["dev-val"]))
    print("\nSpeaker overlap (distinct speaker names in the `speakers` field):\n")
    print("| gran | class | dev-train speakers | dev-val speakers | in both | share of dev-val speakers also in dev-train |")
    print("|---|---|---:|---:|---:|---:|")
    for (g, lab), (a, b, c) in overlap.items():
        print("| %s | %s | %d | %d | %d | %.0f%% |" % (g, "orig" if lab == 0 else "trans", a, b, c, 100.0 * c / max(b, 1)))


if __name__ == "__main__":
    main()

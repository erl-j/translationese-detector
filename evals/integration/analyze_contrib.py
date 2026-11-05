"""Analyse a dump_contrib.py file (trn): within-class correlations of rule
contributions, family-cap binding, leave-one-family-out ablation.

    python3 evals/integration/analyze_contrib.py corr trn100.jsonl 0.3   # pairs with |r| > 0.3
    python3 evals/integration/analyze_contrib.py caps trn500.jsonl
    python3 evals/integration/analyze_contrib.py lofo trnsent.jsonl trn100.jsonl trn500.jsonl

LOFO from the dump is exact: score = sum of clipped family sums, so dropping a
family drops its term (same as Engine(exclude="family")).
"""
import json, math, os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import metrics as M

def load(p):
    return [json.loads(l) for l in open(p)]

def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0

def corr(recs, thr=0.3, min_fire=0.01):
    rules = sorted({k for r in recs for k in r["c"]})
    dirn = {}
    for k in rules:
        s = sum(r["c"].get(k, 0) for r in recs); dirn[k] = 1 if s > 0 else -1
    out = []
    for cls in (0, 1):
        sub = [r for r in recs if r["y"] == cls]
        cols = {k: [r["c"].get(k, 0.0) for r in sub] for k in rules}
        fire = {k: sum(1 for v in cols[k] if v) / len(sub) for k in rules}
        for i, a in enumerate(rules):
            for b in rules[i + 1:]:
                if fire[a] < min_fire or fire[b] < min_fire:
                    continue
                out.append((cls, a, b, pearson(cols[a], cols[b])))
    # combine: per pair r_O, r_T
    d = collections.defaultdict(dict)
    for cls, a, b, r in out:
        d[(a, b)][cls] = r
    return d, dirn

def cap_report(recs):
    fams = sorted({f for r in recs for f in r["f"]})
    y = [r["y"] for r in recs]
    rows = []
    for f in fams:
        clipped = [r["f"].get(f, [0, 0])[0] - r["f"].get(f, [0, 0])[1] for r in recs]  # amount removed (signed raw-clipped)
        bind = [1 if abs(c) > 1e-9 else 0 for c in clipped]
        nO = sum(1 for b, t in zip(bind, y) if b and t == 0); nT = sum(1 for b, t in zip(bind, y) if b and t == 1)
        NO = y.count(0); NT = y.count(1)
        # correctness of clipped evidence: among bound texts, does sign of removed raw amount point to true class?
        corr_ = sum(1 for c, t in zip(clipped, y) if abs(c) > 1e-9 and ((c > 0) == (t == 1)))
        nb = sum(bind)
        auc = M.roc_auc(y, clipped) if nb else float("nan")
        # AUC among bound texts only
        yb = [t for c, t in zip(clipped, y) if abs(c) > 1e-9]; cb = [c for c in clipped if abs(c) > 1e-9]
        aucb = M.roc_auc(yb, cb) if len(set(yb)) == 2 else float("nan")
        meanabs = sum(abs(c) for c in cb) / nb if nb else 0
        rows.append((f, nO / NO, nT / NT, corr_ / nb if nb else float("nan"), auc, aucb, meanabs))
    return rows

def scores(recs, drop=()):
    return [sum(v[1] for f, v in r["f"].items() if f not in drop) for r in recs]

def met(recs, drop=()):
    y = [r["y"] for r in recs]; s = scores(recs, drop)
    m = M.all_metrics(y, s)
    return m["accuracy"], m["roc_auc"]

def rule_auc(recs, names):
    y = [r["y"] for r in recs]
    for n in names:
        s = [r["c"].get(n, 0.0) for r in recs]
        fO = sum(1 for v, t in zip(s, y) if v and t == 0) / y.count(0)
        fT = sum(1 for v, t in zip(s, y) if v and t == 1) / y.count(1)
        print("%-40s AUC=%.3f fireO=%.3f fireT=%.3f" % (n, M.roc_auc(y, s), fO, fT))

def pair_auc(recs, a, b, wa=1.0, wb=1.0):
    y = [r["y"] for r in recs]
    s = [wa * r["c"].get(a, 0.0) + wb * r["c"].get(b, 0.0) for r in recs]
    return M.roc_auc(y, s)

def uncap_diag(recs):
    y = [r["y"] for r in recs]
    base = scores(recs)
    m0 = M.all_metrics(y, base)
    out = []
    for f in sorted({f for r in recs for f in r["f"]}):
        s = [b - r["f"].get(f, [0, 0])[1] + r["f"].get(f, [0, 0])[0] for b, r in zip(base, recs)]
        m = M.all_metrics(y, s)
        # bound texts misclassified by the full engine
        bound = [i for i, r in enumerate(recs) if abs(r["f"].get(f, [0, 0])[0] - r["f"].get(f, [0, 0])[1]) > 1e-9]
        wrong = sum(1 for i in bound if (base[i] > 0) != (y[i] == 1))
        out.append((f, len(bound), wrong, m["accuracy"] - m0["accuracy"], m["roc_auc"] - m0["roc_auc"]))
    return out

def matrix(recs, min_fire=0.01):
    """Full within-class correlation matrices {cls: {a: {b: r}}} of rule contributions."""
    rules = sorted({k for r in recs for k in r["c"]})
    out = {}
    for cls, nm in ((0, "O"), (1, "T")):
        sub = [r for r in recs if r["y"] == cls]
        cols = {k: [r["c"].get(k, 0.0) for r in sub] for k in rules}
        keep = [k for k in rules if sum(1 for v in cols[k] if v) / len(sub) >= min_fire]
        out[nm] = {a: {b: round(pearson(cols[a], cols[b]), 3) for b in keep} for a in keep}
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "corr":
        recs = load(sys.argv[2]); thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
        d, dirn = corr(recs)
        print("pairs with |r| > %.2f in either class (rO, rT, same-direction?)" % thr)
        for (a, b), v in sorted(d.items(), key=lambda kv: -max(abs(x) for x in kv[1].values())):
            if max(abs(x) for x in v.values()) > thr:
                print("%-40s %-40s rO=%+.2f rT=%+.2f %s" % (a, b, v.get(0, float('nan')), v.get(1, float('nan')),
                      "co-move(+)" if (v.get(0,0)+v.get(1,0))>0 else "cancel(-)"))
    elif mode == "caps":
        recs = load(sys.argv[2])
        print("%-10s %8s %8s %10s %8s %10s %8s" % ("family", "bind_O", "bind_T", "corr_dir", "AUC_all", "AUC_bound", "mean|cl|"))
        for row in cap_report(recs):
            print("%-10s %8.3f %8.3f %10.3f %8.3f %10.3f %8.2f" % row)
    elif mode == "uncap":       # diagnostic: the same engine without one family's cap
        for row in uncap_diag(load(sys.argv[2])):
            print("%-10s bound=%5d engine-wrong-among-bound=%4d  uncapped: dAcc=%+.4f dAUC=%+.4f" % row)
    elif mode == "lofo":
        for p in sys.argv[2:]:
            recs = load(p)
            fams = sorted({f for r in recs for f in r["f"]})
            acc, auc = met(recs)
            print(p, "full acc=%.4f auc=%.4f" % (acc, auc))
            for f in fams:
                a2, u2 = met(recs, (f,))
                print("  -%-10s acc=%.4f (%+.4f) auc=%.4f (%+.4f)" % (f, a2, a2 - acc, u2, u2 - auc))

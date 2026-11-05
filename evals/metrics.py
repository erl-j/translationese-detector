"""Stdlib-only metrics for binary classification (label 1 = translated).

All functions take plain lists. Predictions follow the classifier contract:
score > 0 means "translated" (label 1), anything else means "original" (0).
"""
import math


def predict(scores):
    return [1 if s > 0 else 0 for s in scores]


def confusion(labels, preds):
    """Return dict with tp, fp, tn, fn (positive class = 1 = translated)."""
    c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for y, p in zip(labels, preds):
        if p == 1:
            c["tp" if y == 1 else "fp"] += 1
        else:
            c["fn" if y == 1 else "tn"] += 1
    return c


def accuracy(labels, preds):
    if not labels:
        return float("nan")
    return sum(1 for y, p in zip(labels, preds) if y == p) / len(labels)


def f1_from_counts(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def macro_f1(labels, preds):
    c = confusion(labels, preds)
    f_pos = f1_from_counts(c["tp"], c["fp"], c["fn"])
    f_neg = f1_from_counts(c["tn"], c["fn"], c["fp"])  # class 0 as positive
    return (f_pos + f_neg) / 2


def roc_auc(labels, scores):
    """ROC-AUC via the Mann-Whitney U statistic with average ranks for ties.

    Equals P(score of a random positive > score of a random negative)
    + 0.5 * P(tie). Returns nan if one class is absent.
    """
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def all_metrics(labels, scores):
    preds = predict(scores)
    return {"n": len(labels),
            "n_orig": sum(1 for y in labels if y == 0),
            "n_trans": sum(1 for y in labels if y == 1),
            "accuracy": accuracy(labels, preds),
            "macro_f1": macro_f1(labels, preds),
            "roc_auc": roc_auc(labels, scores),
            "confusion": confusion(labels, preds),
            "pred_trans_rate": sum(preds) / len(preds) if preds else float("nan")}


def rule_diagnostics(labels, explanations, smoothing=0.5):
    """Per-rule statistics from explain() dicts.

    A rule "fires" on a text when its contribution is present and != 0.
    Returns {rule: stats} with
      fire_orig / fire_trans     firing rate in each class
      mean_orig / mean_trans     mean contribution per class (0 when not firing)
      log_odds                   log[(f_t+a)/(n_t-f_t+a)] - log[(f_o+a)/(n_o-f_o+a)],
                                 >0 = fires more on translated text
      auc                        ROC-AUC of the contribution alone as a score
      sign_acc                   among firings, share whose sign agrees with the
                                 label (+ on translated, - on original)
      n_fire                     total firings
    """
    n = {0: 0, 1: 0}
    for y in labels:
        n[y] += 1
    rules = set()
    for e in explanations:
        rules.update(e.keys())
    out = {}
    for r in sorted(rules):
        vals = [float(e.get(r, 0.0) or 0.0) for e in explanations]
        fire = {0: 0, 1: 0}
        tot = {0: 0.0, 1: 0.0}
        agree = 0
        for v, y in zip(vals, labels):
            tot[y] += v
            if v != 0:
                fire[y] += 1
                if (v > 0) == (y == 1):
                    agree += 1
        a = smoothing
        lo = (math.log((fire[1] + a) / (n[1] - fire[1] + a))
              - math.log((fire[0] + a) / (n[0] - fire[0] + a)))
        nf = fire[0] + fire[1]
        out[r] = {"fire_orig": fire[0] / n[0] if n[0] else float("nan"),
                  "fire_trans": fire[1] / n[1] if n[1] else float("nan"),
                  "mean_orig": tot[0] / n[0] if n[0] else float("nan"),
                  "mean_trans": tot[1] / n[1] if n[1] else float("nan"),
                  "log_odds": lo,
                  "auc": roc_auc(labels, vals),
                  "sign_acc": agree / nf if nf else float("nan"),
                  "n_fire": nf}
    return out


def format_rule_table(diag, top=None):
    rows = sorted(diag.items(), key=lambda kv: -abs(kv[1]["log_odds"]))
    if top:
        rows = rows[:top]
    w = max([len("rule")] + [len(r) for r, _ in rows])
    hdr = ("%-*s %8s %8s %9s %9s %8s %6s %7s %7s"
           % (w, "rule", "fire_O", "fire_T", "mean_O", "mean_T", "logodds", "auc", "signacc", "n_fire"))
    lines = [hdr, "-" * len(hdr)]
    for r, d in rows:
        lines.append("%-*s %8.3f %8.3f %9.4f %9.4f %+8.3f %6.3f %7.3f %7d"
                     % (w, r, d["fire_orig"], d["fire_trans"], d["mean_orig"], d["mean_trans"],
                        d["log_odds"], d["auc"], d["sign_acc"] if d["n_fire"] else float("nan"), d["n_fire"]))
    return "\n".join(lines)

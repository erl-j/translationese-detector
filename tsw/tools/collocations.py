#!/usr/bin/env python3
"""Discovery helper: n-grams (2-4 words) over- or under-represented in
translated vs original Swedish, on DEV ONLY.

It PROPOSES candidates; it never writes rules. A knowledge engineer reads the
list, decides which items are genuine English interference (and can name the
English source construction and a native alternative), and then checks each
accepted item with `tsw.tools.rates --rx ... --by-speaker --by-year --kwic`.

    python3 -m tsw.tools.collocations                       # top T-leaning 2-4-grams
    python3 -m tsw.tools.collocations --direction O --top 80
    python3 -m tsw.tools.collocations --n 3 --min-count 15 --min-speakers 10
    python3 -m tsw.tools.collocations --contains 'vilk' --direction both
    python3 -m tsw.tools.collocations --mask-topic --hide-topic

Split: only trn (= the old dev split, sessions <= 2004-06-30; 'dev' is an
alias) is read. 'val' (2004-07..2006) and 'test' (2007..2011) are refused:
never look at their n-grams.

Data: trn.sent.all by default (every sentence carries its own speaker, and the
unbalanced file has ~4.8x more translated text, which helps for rare items).
Counts are rate-normalised (per 1000 words in each class), so class imbalance
does not matter. N-grams never cross punctuation (they live inside one clause
chunk); with --bos a sentence-start marker `<s>` is prepended, so
sentence-initial phrases (`<s> först av allt`) can be found.

Columns
    cO, cT      raw counts per class
    rO, rT      per 1000 words
    lr          smoothed ln(rT/rO) with +ALPHA (default 0.5) added to both counts
    G2          log-likelihood ratio statistic (Dunning), for strength of evidence
    sO, sT      number of distinct speakers producing the n-gram in each class
    top%        share of the favoured class's hits from its single most frequent speaker
    uni         the strongest component-word unigram lr (same smoothing); if it is
                close to lr, the skew is inherited from one word (lexical/topic),
                not from the combination (phraseology)
    flags       P = contains a proper noun (capitalised mid-sentence in >= 50% of
                its occurrences), M = touched by evals/topicmask, D = digit,
                S = speaker quirk (favoured-class speakers < --min-speakers, or top% >= 30),
                I = inherited (|uni| >= 0.8 |lr|)

Cost: a full dev-train pass takes ~30 s and ~1.3 GB; run it through Slurm:
    timeout 900 srun -p berzelius-hopper-cpu --time=00:15:00 -c 1 --mem=8G \\
        python3 -m tsw.tools.collocations --top 150 > out.txt
"""
import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EVALS = os.path.join(ROOT, "evals")
for p in (ROOT, EVALS):
    if p not in sys.path:
        sys.path.insert(0, p)

from tsw.text import TOKEN_RX, is_word, normalise  # noqa: E402

CLASSES = ((0, "O"), (1, "T"))
BOS = "<s>"
SPLITS = {"trn": lambda date: date <= "2004-06-30", "dev": lambda date: date <= "2004-06-30"}


def data_path(gran, balanced, data_dir):
    import run as harness  # evals/run.py (resolves the shared data dir of worktrees)
    ddir = harness.data_dir(data_dir)
    name = "%s.%s.jsonl" % (gran, "bal" if balanced else "all")
    path = os.path.join(ddir, "trn." + name)
    if not os.path.exists(path):          # data built before the trn/val/test split
        path = os.path.join(ddir, "dev." + name)
    if "test" in os.path.basename(path):
        sys.exit("REFUSING: collocations.py works on dev only.")
    return path


def load(path, limit=None, seed=0, mask=False, split="trn"):
    keep = SPLITS[split]
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not keep(r.get("date") or ""):
                continue
            recs.append((r["label"], r.get("speaker") or "?", r["text"]))
    if limit and limit < len(recs):
        rng = random.Random("colloc|%d" % seed)
        recs = [recs[i] for i in sorted(rng.sample(range(len(recs)), limit))]
    if mask:
        import topicmask
        recs = [(y, s, topicmask.mask(t)) for y, s, t in recs]
    return recs


def chunks(text, bos=False):
    """Yield (lower-cased word runs between punctuation, case-kept run).
    The first run of each text gets the BOS marker when bos=True."""
    toks = [m.group() for m in TOKEN_RX.finditer(normalise(text))]
    run, run_c = ([BOS], [BOS]) if bos else ([], [])
    for t in toks:
        if is_word(t):
            run.append(t.lower())
            run_c.append(t)
        else:
            if run:
                yield run, run_c
            run, run_c = [], []
    if run:
        yield run, run_c


def ngrams(run, ns):
    L = len(run)
    for n in ns:
        for i in range(L - n + 1):
            yield " ".join(run[i:i + n])


def smoothed_lr(c_t, c_o, n_t, n_o, alpha):
    return math.log(((c_t + alpha) / n_t) / ((c_o + alpha) / n_o))


def g2(c_t, c_o, n_t, n_o):
    """Dunning log-likelihood (2x2: n-gram vs other words, T vs O)."""
    tot = c_t + c_o
    N = n_t + n_o
    out = 0.0
    for obs, n in ((c_t, n_t), (c_o, n_o)):
        exp = tot * n / N
        if obs > 0:
            out += obs * math.log(obs / exp)
    return 2.0 * out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gran", default="sent", choices=["sent", "100", "500"])
    ap.add_argument("--balanced", action="store_true", help="use trn.*.bal (default: trn.*.all)")
    ap.add_argument("--n", default="2,3,4", help="n-gram sizes, e.g. 2,3,4")
    ap.add_argument("--min-count", type=int, default=10, help="min hits in the favoured class")
    ap.add_argument("--min-speakers", type=int, default=5,
                    help="flag S when fewer distinct speakers produce it in the favoured class")
    ap.add_argument("--alpha", type=float, default=0.5, help="additive smoothing of counts")
    ap.add_argument("--direction", default="T", choices=["T", "O", "both"])
    ap.add_argument("--sort", default="lr", choices=["lr", "g2"])
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--contains", help="only n-grams matching this regex (on the space-joined n-gram)")
    ap.add_argument("--hide-topic", action="store_true", help="drop rows flagged P, M or D")
    ap.add_argument("--hide-quirks", action="store_true", help="drop rows flagged S")
    ap.add_argument("--bos", action="store_true", help="prepend a sentence-start marker <s>")
    ap.add_argument("--mask-topic", action="store_true", help="apply evals/topicmask before counting")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir")
    ap.add_argument("--split", default="trn",
                    help="only trn (old dev, sessions <= 2004-06-30; alias 'dev') is allowed; val and test are refused")
    args = ap.parse_args(argv)
    if args.split not in SPLITS:
        sys.exit("REFUSING: collocations.py works on trn only (got %r; val and test are held out)."
                 % args.split)
    ns = sorted({int(x) for x in args.n.split(",")})
    contains = re.compile(args.contains) if args.contains else None

    path = data_path(args.gran, args.balanced, args.data_dir)
    recs = load(path, args.limit, args.seed, args.mask_topic, args.split)

    # pass 1: n-gram and unigram counts per class; capitalisation of words
    cnt = (Counter(), Counter())
    uni = (Counter(), Counter())
    n_words = [0, 0]
    cap_mid = Counter()   # lower word -> capitalised occurrences not at sentence start
    all_mid = Counter()
    for y, _, text in recs:
        first = True
        for run, run_c in chunks(text, args.bos):
            words = [w for w in run if w != BOS]
            n_words[y] += len(words)
            uni[y].update(words)
            for j, (w, wc) in enumerate(zip(run, run_c)):
                if w == BOS or (first and j == (1 if args.bos else 0)):
                    continue
                all_mid[w] += 1
                if wc[:1].isupper():
                    cap_mid[w] += 1
            first = False
            c = cnt[y]
            for g in ngrams(run, ns):
                c[g] += 1
    n_o, n_t = n_words
    print("data: %s %s (%d texts%s); words O %d / T %d; n-grams %s; alpha %.2g"
          % (args.split, os.path.basename(path), len(recs), ", topic-masked" if args.mask_topic else "",
             n_o, n_t, ns, args.alpha))

    def uni_lr(w):
        return smoothed_lr(uni[1][w], uni[0][w], n_t, n_o, args.alpha)

    cands = []
    keys = set(cnt[0]) | set(cnt[1]) if args.direction != "T" else set(cnt[1])
    for g in keys:
        c_o, c_t = cnt[0][g], cnt[1][g]
        lr = smoothed_lr(c_t, c_o, n_t, n_o, args.alpha)
        fav = c_t if lr > 0 else c_o
        if fav < args.min_count:
            continue
        if args.direction == "T" and lr <= 0 or args.direction == "O" and lr >= 0:
            continue
        if contains and not contains.search(g):
            continue
        cands.append((g, c_o, c_t, lr))
    del keys
    key = (lambda r: -abs(r[3])) if args.sort == "lr" else (lambda r: -g2(r[2], r[1], n_t, n_o))
    cands.sort(key=key)
    # keep a margin: rows dropped by --hide-* are replaced from further down
    pool = cands[:max(args.top * 4, args.top + 200)]
    want = {g for g, _, _, _ in pool}
    del cnt

    # pass 2: speakers for the candidate pool only
    spk = defaultdict(lambda: (Counter(), Counter()))
    for y, who, text in recs:
        seen = set()
        for run, _ in chunks(text, args.bos):
            for g in ngrams(run, ns):
                if g in want:
                    spk[g][y][who] += 1
                    seen.add(g)

    try:
        import topicmask
        def touched(g):  # noqa: E306
            return topicmask.mask(g) != g
    except ImportError:  # pragma: no cover
        def touched(g):  # noqa: E306
            return False

    rows = []
    for g, c_o, c_t, lr in pool:
        ws = [w for w in g.split() if w != BOS]
        flags = ""
        if any(all_mid[w] >= 3 and cap_mid[w] >= 0.5 * all_mid[w] for w in ws):
            flags += "P"
        if touched(g):
            flags += "M"
        if any(ch.isdigit() for ch in g):
            flags += "D"
        s_o, s_t = spk[g]
        fav = s_t if lr > 0 else s_o
        n_fav = sum(fav.values())
        top_share = fav.most_common(1)[0][1] / n_fav if n_fav else 0.0
        if len(fav) < args.min_speakers or top_share >= 0.30:
            flags += "S"
        ulr = max((uni_lr(w) for w in ws), key=abs) if ws else 0.0
        if ulr * lr > 0 and abs(ulr) >= 0.8 * abs(lr):
            flags += "I"
        if args.hide_topic and any(f in flags for f in "PMD"):
            continue
        if args.hide_quirks and "S" in flags:
            continue
        rows.append((g, c_o, c_t, 1000.0 * c_o / n_o, 1000.0 * c_t / n_t, lr, g2(c_t, c_o, n_t, n_o),
                     len(s_o), len(s_t), top_share, ulr, flags))
        if len(rows) >= args.top:
            break

    print("%-38s %6s %6s %8s %8s %6s %8s %4s %4s %5s %6s %s"
          % ("n-gram", "cO", "cT", "rO", "rT", "lr", "G2", "sO", "sT", "top%", "uni", "flags"))
    for g, c_o, c_t, r_o, r_t, lr, gg, so, st, ts, ulr, flags in rows:
        print("%-38s %6d %6d %8.4f %8.4f %+6.2f %8.1f %4d %4d %4.0f%% %+6.2f %s"
              % (g[:38], c_o, c_t, r_o, r_t, lr, gg, so, st, 100 * ts, ulr, flags))


if __name__ == "__main__":
    main()

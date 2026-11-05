#!/usr/bin/env python3
"""Knowledge-engineering helper: per-class statistics of a pattern on DEV ONLY.

This prints descriptive statistics; a human reads them and types rounded
values (<= 2 significant digits) into a rule definition. Nothing here writes
or tunes rule parameters.

    python3 -m tsw.tools.rates --rx '\\b(emellertid|huruvida|således)\\b'
    python3 -m tsw.tools.rates --words "ju,nog,visst" --kwic 5 --mask-topic
    python3 -m tsw.tools.rates --choice '\\bskall\\b' '\\bska\\b' --by-year
    python3 -m tsw.tools.rates --rule starter.dem_choice --by-speaker
    python3 -m tsw.tools.rates --stat sentlen --gran sent
    python3 -m tsw.tools.rates --stat mymodule:units_fn          # fn(doc) -> [(value, span)]

Regexes run on the lower-cased normalised text (doc.lower) unless --case.
Rates are per 1000 word tokens as counted by tsw.doc.Doc (the same
denominator RateRule uses). Default data: dev.500.bal (all 1367+1367 chunks).
The test split is refused.
"""
import argparse
import importlib
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

from tsw.doc import Doc  # noqa: E402
from tsw.rules import base as B  # noqa: E402

CLASSES = ((0, "O"), (1, "T"))


def _data_file(gran, unbalanced, data_dir):
    import run as harness  # evals/run.py (also resolves worktree data dirs)
    ddir = harness.data_dir(data_dir)
    path = os.path.join(ddir, "dev.%s.%s.jsonl" % (gran, "all" if unbalanced else "bal"))
    if "test" in os.path.basename(path):
        sys.exit("REFUSING: rates.py works on dev only.")
    return path


def load(gran="500", unbalanced=False, limit=None, seed=0, mask=False, data_dir=None):
    path = _data_file(gran, unbalanced, data_dir)
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    if limit and limit < len(recs):
        rng = random.Random("rates|%d" % seed)
        recs = [recs[i] for i in sorted(rng.sample(range(len(recs)), limit))]
    if mask:
        import topicmask
        for r in recs:
            r["text"] = topicmask.mask(r["text"])
    return recs


def sig2(x):
    """Round to 2 significant digits (display only)."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, 1 - int(math.floor(math.log10(abs(x)))))


def kwic(text, a, b, width=45):
    left = text[max(0, a - width):a].rjust(width)
    right = text[b:b + width]
    return "%s[%s]%s" % (left, text[a:b], right)


def _resolve_rule(name):
    from tsw.rules import discover
    rules, _, _ = discover()
    for r in rules:
        if r.name == name:
            return r
    sys.exit("no rule named %r (known: %s)" % (name, ", ".join(r.name for r in rules)))


def _load_fn(spec):
    if spec == "sentlen":
        return B.sentence_lengths
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def analyse_matchers(recs, matchers, args):
    """matchers: list of (tag, Matcher). One tag = rate; two tags = choice (A, B)."""
    tot_words = Counter()
    n_texts = Counter()
    hits = defaultdict(Counter)        # tag -> class -> hits
    any_hit = defaultdict(Counter)     # tag -> class -> texts with >=1 hit
    any_choice = Counter()
    spk = defaultdict(Counter)         # class -> speaker -> hits (all tags)
    spk_words = defaultdict(Counter)
    year = defaultdict(lambda: defaultdict(Counter))  # class -> year -> tag/words
    surface = defaultdict(Counter)     # tag -> surface -> count
    examples = defaultdict(list)       # (tag, cls) -> [kwic]
    rng = random.Random(args.seed)
    seen = Counter()
    for r in recs:
        y = r["label"]
        d = Doc(r["text"])
        n = d.n_words
        tot_words[y] += n
        n_texts[y] += 1
        who = r.get("speaker") or "?"
        yr = (r.get("date") or "????")[:4]
        spk_words[y][who] += n
        year[y][yr]["_words"] += n
        chosen = False
        for tag, m in matchers:
            k = 0
            for a, b, _ in m.finditer(d):
                k += 1
                surface[tag][m.target(d)[a:b]] += 1
                key = (tag, y)
                seen[key] += 1
                # reservoir sample of KWIC lines
                if args.kwic:
                    line = kwic(d.text, a, b)
                    if len(examples[key]) < args.kwic:
                        examples[key].append(line)
                    else:
                        j = rng.randrange(seen[key])
                        if j < args.kwic:
                            examples[key][j] = line
            if k:
                hits[tag][y] += k
                any_hit[tag][y] += 1
                spk[y][who] += k
                year[y][yr][tag] += k
                chosen = True
        if chosen:
            any_choice[y] += 1

    print("data: %d texts (O %d / T %d), words O %d / T %d%s"
          % (len(recs), n_texts[0], n_texts[1], tot_words[0], tot_words[1],
             ", topic-masked" if args.mask_topic else ""))
    for tag, m in matchers:
        print("\n== %s: /%s/%s" % (tag, m.source if len(m.source) < 200 else m.source[:200] + "...",
                                    " (case-sensitive)" if m.case else ""))
        print("%-3s %8s %10s %12s %14s" % ("cls", "hits", "per1000w", "texts>=1hit", "obs (2 s.f.)"))
        rates = {}
        for y, c in CLASSES:
            rate = 1000.0 * hits[tag][y] / tot_words[y] if tot_words[y] else float("nan")
            rates[c] = rate
            print("%-3s %8d %10.4f %11.1f%% %14s" % (c, hits[tag][y], rate,
                                                     100.0 * any_hit[tag][y] / max(1, n_texts[y]), sig2(rate)))
        if rates["O"] > 0 and rates["T"] > 0:
            print("ratio T/O = %.2f   ln(T/O) = %+.2f per hit" % (rates["T"] / rates["O"],
                                                                 math.log(rates["T"] / rates["O"])))
        top = surface[tag].most_common(args.top_forms)
        if top:
            print("top forms: " + ", ".join("%s=%d" % kv for kv in top))
    if len(matchers) == 2:
        (ta, _), (tb, _) = matchers
        print("\n== choice A=%s vs B=%s" % (ta, tb))
        print("%-3s %7s %7s %9s %16s" % ("cls", "kA", "kB", "P(A)", "texts w/ choice"))
        for y, c in CLASSES:
            ka, kb = hits[ta][y], hits[tb][y]
            p = ka / (ka + kb) if ka + kb else float("nan")
            print("%-3s %7d %7d %9.3f %15.1f%%" % (c, ka, kb, p, 100.0 * any_choice[y] / max(1, n_texts[y])))
    if args.by_speaker:
        print("\n== speaker concentration (all tags pooled)")
        for y, c in CLASSES:
            tot = sum(spk[y].values())
            if not tot:
                print("%s: no hits" % c)
                continue
            top = spk[y].most_common(5)
            share = sum(v for _, v in top) / tot
            print("%s: %d hits from %d speakers; top-5 share %.0f%%: %s" % (
                c, tot, len(spk[y]), 100 * share,
                ", ".join("%s %d (%.2g/1000w)" % (s, v, 1000.0 * v / max(1, spk_words[y][s])) for s, v in top)))
    if args.by_year:
        print("\n== per year (per 1000 words; choice: P(A))")
        yrs = sorted(set(year[0]) | set(year[1]))
        for y, c in CLASSES:
            cells = []
            for yr in yrs:
                cell = year[y].get(yr)
                if not cell or not cell["_words"]:
                    continue
                if len(matchers) == 2:
                    ka, kb = cell[matchers[0][0]], cell[matchers[1][0]]
                    cells.append("%s:%s" % (yr, "%.2f" % (ka / (ka + kb)) if ka + kb else "-"))
                else:
                    k = sum(cell[t] for t, _ in matchers)
                    cells.append("%s:%.2f" % (yr, 1000.0 * k / cell["_words"]))
            print("%s  %s" % (c, "  ".join(cells)))
    if args.kwic:
        for tag, _ in matchers:
            for y, c in CLASSES:
                ex = examples.get((tag, y), [])
                if ex:
                    print("\n-- KWIC %s, class %s (%d random of %d)" % (tag, c, len(ex), seen[(tag, y)]))
                    for line in ex:
                        print("   " + line)


def analyse_stat(recs, fn, args):
    vals = defaultdict(list)
    per_text = defaultdict(list)
    for r in recs:
        d = Doc(r["text"])
        v = [x for x, _ in fn(d)]
        if args.clip:
            lo, hi = args.clip
            v = [min(max(x, lo), hi) for x in v]
        vals[r["label"]].extend(v)
        if v:
            per_text[r["label"]].append(sum(v) / len(v))
    print("unit statistic: %s%s" % (args.stat, " clipped to %s" % (args.clip,) if args.clip else ""))
    print("%-3s %8s %9s %9s %7s %7s %7s %7s %12s" % ("cls", "units", "mean", "sd", "p10", "p50", "p90",
                                                     "p99", "text-mean sd"))
    sds = {}
    for y, c in CLASSES:
        v = sorted(vals[y])
        if not v:
            continue
        n = len(v)
        mu = sum(v) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, n - 1))
        sds[c] = (n, sd)
        q = lambda p: v[min(n - 1, int(p * n))]  # noqa: E731
        tm = per_text[y]
        tmu = sum(tm) / len(tm)
        tsd = math.sqrt(sum((x - tmu) ** 2 for x in tm) / max(1, len(tm) - 1))
        print("%-3s %8d %9.3f %9.3f %7.2f %7.2f %7.2f %7.2f %12.3f" % (c, n, mu, sd, q(.1), q(.5), q(.9),
                                                                       q(.99), tsd))
    if len(sds) == 2:
        (n0, s0), (n1, s1) = sds["O"], sds["T"]
        pooled = math.sqrt(((n0 - 1) * s0 ** 2 + (n1 - 1) * s1 ** 2) / max(1, n0 + n1 - 2))
        print("pooled unit sd = %.3f (2 s.f. %s)" % (pooled, sig2(pooled)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rx", action="append", help="regex (repeatable: one block per regex)")
    g.add_argument("--words", help="comma- or |-separated words/phrases, matched as whole words")
    g.add_argument("--wordfile", help="file with one word/phrase per line (# comments)")
    g.add_argument("--choice", nargs=2, metavar=("A_RX", "B_RX"), help="two competing variants")
    g.add_argument("--rule", help="name of an existing rule: analyse its pattern(s)/units")
    g.add_argument("--stat", help="'sentlen' or module:function returning [(value, span)] per doc")
    ap.add_argument("--gran", default="500", choices=["sent", "100", "500"])
    ap.add_argument("--unbalanced", action="store_true", help="use dev.*.all instead of dev.*.bal")
    ap.add_argument("--mask-topic", action="store_true")
    ap.add_argument("--case", action="store_true", help="match on case-preserved text")
    ap.add_argument("--kwic", type=int, default=0, metavar="N", help="N random concordance lines per class")
    ap.add_argument("--by-speaker", action="store_true")
    ap.add_argument("--by-year", action="store_true")
    ap.add_argument("--top-forms", type=int, default=12)
    ap.add_argument("--clip", type=float, nargs=2, metavar=("LO", "HI"), help="clip unit values (--stat)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir")
    ap.add_argument("--split", default="dev", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.split != "dev":
        sys.exit("REFUSING: rates.py works on dev only (never look at test).")

    recs = load(args.gran, args.unbalanced, args.limit, args.seed, args.mask_topic, args.data_dir)
    if args.stat or (args.rule and isinstance(_resolve_rule(args.rule), B.StatRule)):
        if args.rule:
            r = _resolve_rule(args.rule)
            fn = r.units
            args.stat = args.rule
            if r.unit_clip and not args.clip:
                args.clip = r.unit_clip
            print("rule %s: mu_o=%s mu_t=%s sd_o=%s sd_t=%s" % (r.name, r.mu_o, r.mu_t, r.sd_o, r.sd_t))
        else:
            fn = _load_fn(args.stat)
        analyse_stat(recs, fn, args)
        return
    if args.rx:
        matchers = [("rx%d" % (i + 1), B.Matcher(rx, case=args.case)) for i, rx in enumerate(args.rx)]
    elif args.words:
        matchers = [("words", B.Matcher(words=args.words, case=args.case))]
    elif args.wordfile:
        with open(args.wordfile, encoding="utf-8") as f:
            ws = [l.split("#")[0].strip() for l in f]
        matchers = [("wordfile", B.Matcher(words=[w for w in ws if w], case=args.case))]
    elif args.choice:
        matchers = [("A", B.Matcher(args.choice[0], case=args.case)),
                    ("B", B.Matcher(args.choice[1], case=args.case))]
    else:
        r = _resolve_rule(args.rule)
        print("rule %s (%s): %s" % (r.name, r.kind, json.dumps({k: v for k, v in r.meta().items()
                                                                if not k.startswith("pattern")},
                                                               ensure_ascii=False, default=str)))
        if isinstance(r, B.ChoiceRule):
            matchers = [("A", r.matcher_a), ("B", r.matcher_b)]
        elif hasattr(r, "matcher"):
            matchers = [(r.name, r.matcher)]
        else:
            sys.exit("rule %s has no pattern to analyse" % r.name)
    analyse_matchers(recs, matchers, args)


if __name__ == "__main__":
    main()

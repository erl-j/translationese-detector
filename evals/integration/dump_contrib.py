"""Dump per-text rule contributions, family raw/clipped sums and span overlaps on TRN only.

    srun -p berzelius-hopper-cpu --time=00:15:00 -c 1 --mem=8G \
        python3 evals/integration/dump_contrib.py . 100 /abs/path/trn100.jsonl --spans

One JSON line per text: {id, y, n, c: {rule: contribution}, f: {family: [raw, clipped]}}.
With --spans, OUT.spans.json counts, for every pair of rules, the evidence spans
(Rule.evidence hits) that overlap in characters: [n, same sign, opposite sign,
sum of min |share|]. Topic-masked text unless --nomask. ~90 s per trn granularity
on one CPU. Used for docs/INTEGRATION.md; reads trn only (val/test never).
usage: dump_contrib.py WORKTREE GRAN OUT [--limit N] [--spans] [--engine-kw JSON]"""
import argparse, json, os, sys, time, collections
ap = argparse.ArgumentParser()
ap.add_argument("wt"); ap.add_argument("gran"); ap.add_argument("out")
ap.add_argument("--limit", type=int); ap.add_argument("--spans", action="store_true")
ap.add_argument("--engine-kw", default="{}"); ap.add_argument("--nomask", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.wt); sys.path.insert(0, os.path.join(a.wt, "evals"))
import run as R, topicmask
from tsw.engine import Engine
from tsw.doc import Doc
eng = Engine(**json.loads(a.engine_kw))
ddir = R.data_dir()
recs = R.load_records(R.data_path(ddir, "trn", a.gran, True), limit=a.limit)
t0 = time.time()
pairs = collections.defaultdict(lambda: [0, 0, 0, 0.0])  # n, same, opp, sum min|share|
with open(a.out, "w") as fo:
    for r in recs:
        text = r["text"] if a.nomask else topicmask.mask(r["text"])
        doc = Doc(text)
        contrib, fam = {}, {}
        hits = []
        for f, rules in eng._by_family.items():
            s = 0.0
            for rule in rules:
                if a.spans:
                    hs, _ = rule.evidence(doc)
                    c = rule.contribution(doc)
                    for h in hs:
                        if h.share:
                            hits.append((h.start, h.end, rule.name, h.share))
                else:
                    c = rule.contribution(doc)
                if c:
                    contrib[rule.name] = round(c, 5); s += c
            cap = eng.family_caps[f]
            fam[f] = [round(s, 5), round(max(-cap, min(cap, s)), 5)]
        if a.spans and hits:
            hits.sort()
            for i in range(len(hits)):
                a0, b0, n0, s0 = hits[i]
                for j in range(i + 1, len(hits)):
                    a1, b1, n1, s1 = hits[j]
                    if a1 >= b0:
                        break
                    if n1 == n0:
                        continue
                    k = tuple(sorted((n0, n1)))
                    p = pairs[k]; p[0] += 1
                    if (s0 > 0) == (s1 > 0): p[1] += 1
                    else: p[2] += 1
                    p[3] += min(abs(s0), abs(s1))
        fo.write(json.dumps({"id": r["id"], "y": r["label"], "n": r.get("n_tokens"), "c": contrib, "f": fam}) + "\n")
if a.spans:
    json.dump({"|".join(k): v for k, v in sorted(pairs.items(), key=lambda kv: -kv[1][0])},
              open(a.out + ".spans.json", "w"), indent=0)
print("done", len(recs), "texts in %.1fs" % (time.time() - t0), file=sys.stderr)

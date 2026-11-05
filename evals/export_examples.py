#!/usr/bin/env python3
"""Export annotated trn examples (TP/TN/FP/FN) for the dashboard.

    python3 evals/export_examples.py [--engine tsw.engine:Default] [--n 3000] [--seed 20261105]
                                     [--out data/cache/dashboard/examples.json]

For each granularity (single sentences, 100-token chunks) a random sample of
--n units of the balanced TRN file (trn = the former dev split, byte-identical;
never val or test: their texts must not be shown) is scored with the engine on
unmasked text, and each unit gets an outcome at threshold 0 (positive =
translated): TP, TN, FP, FN. Per granularity x outcome it picks 3 confident
examples (largest |score|) and 2 borderline ones (smallest |score|), preferring
units with at least one clue span and avoiding repeats of the same set of
firing rules, so both strong cases and near-misses are visible.

Each row: id, gran, outcome, truth, pred, score, prob, date, text, spans
(Engine.annotate spans, offsets into text), top (top rule contributions with
the rule's human-readable evidence sentence), doc_level, note (hand-written
analyst annotation from NOTES below, keyed by id; written after reading the
example). Rows whose id has no note get note "" and are listed on stderr.

Deterministic (fixed seed, no timestamps), stdlib only, a couple of CPU
seconds. Prints a short summary of the sampled errors.
"""
import argparse
import collections
import importlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tsw.doc import Doc  # noqa: E402

GRANS = [("sent", "sentence"), ("100", "100-token chunk")]
OUTCOMES = ["TP", "TN", "FP", "FN"]
N_CONFIDENT, N_BORDER = 3, 2
TOP_RULES = 5

# Analyst notes, keyed by example id (written after reading each example with
# the default seed and the INTEGRATED rule set of 2026-11-06, 52 -> 47 enabled
# rules in 7 families; the notes for the starter rule set are in git history,
# commit 627e006. Ids that drop out of the selection when rules change simply
# lose their note).
NOTES = {
    # --- sentences: TP
    "ep-02-01-15:24:1:s9": "Four 'kommer att' for English 'will' (the future-choice rule at its cap), 'när det rör' (EN 'as regards') and four address 'ni' make this long sentence clearly translated; the dashed aside and length add a little.",
    "ep-00-10-04:312:1:s34": "'Till sist, jag var' copies 'Finally, I was': the comma after the fronted adverbial, the missing V2 inversion and the calque connective 'till sist' all fire, each reading a different property (punctuation, word order, lexis) of one English construction.",
    "ep-01-01-17:15:1:s2": "'i synnerhet' (EN 'in particular'), two 'kommer att' and no comma before the clausal 'men vi' (English punctuation) mark it as translated.",
    "ep-97-04-07:37:1:s19": "Right by 0.002: the fronted clause with resumptive 'så' ('Om så är fallet så är jag', weakly T) is balanced by additive 'också' (O). No lexical calque fires.",
    "ep-02-04-10:252:1:s4": "'Om detta är sant, är det ...' mirrors 'If this is true, it is ...' (fronted clause, weakly T) against the inverted copular 'är det' (weakly O); right by 0.007.",
    # --- sentences: TN
    "ep-99-05-07:34:1:s13": "The particle 'nämligen', 'tycker' rather than 'anser', and the topicalised 'Personligen tycker jag' (V2) are three native clues without English triggers.",
    "ep-97-09-16:110:1:s13": "'ju', the fronted 'så' with V2 inversion ('så är det', now scored by the word-order rules) and a non-vocative exclamation are all native; sentence-initial 'Men' is the only T clue.",
    "ep-98-04-02:15:1:s2": "Modal particle 'ju' plus the stance adverb 'egentligen': neither has an English trigger, and together they decide the sentence.",
    "ep-00-02-15:315:1:s19": "No rule fires: 'i varje fall' is not in the stance list (which has 'i alla fall'), so the score is length and absence terms only (-0.004).",
    "ep-98-10-21:124:1:s2": "The fronted 'I dag genomfördes' is not classified (the verb is not in the closed list) and appositive ', nämligen' is excluded from the particle rule by design; right by 0.01 on absence terms.",
    # --- sentences: FP
    "ep-01-10-03:313:1:s14": "A Swedish speaker writes 'Till sist, för att ...': the comma rule (at its cap of 2) and the calque connective 'till sist' outweigh native 'skall'. Before a fronted purpose clause the comma is arguably normal Swedish.",
    "ep-99-10-06:212:1:s2": "'Det är skälet till att' (the 'this is the reason' connective), 'välkomnar' and 'säkerställa' in one long Council-style sentence. Swedish speakers use all three, but the per-hit values add up.",
    "ep-97-07-16:220:1:s3": "A polite 'Jag skulle vilja fråga' plus 'om vilka följder', an interrogative determiner that the pied-piped-relative rule misreads as a relative ('följder' is not in its noun list).",
    "ep-02-09-24:276:1:s27": "'Det är rimligt att ...' is neutral. With no lexical clue, the score comes only from the sentence opening and length terms (+0.001).",
    "ep-00-04-12:125:1:s18": "Native V2 with a fronted PP ('I Köln och i Helsingfors gjordes'), but fronted PPs carry almost no weight (-0.03); the absence terms tip it to T by 0.01.",
    # --- sentences: FN
    "ep-98-10-07:147:1:s3": "The translator chose spoken 'väldigt dåligt' and 'bra' and V2 'Då tänkte vi' (EN 'Then we thought'), three native-register clues inside a translation.",
    "ep-97-12-16:40:1:s2": "'tycker' and 'bra' in a translation (EN 'I think they have done well'); the register rules read both as native.",
    "ep-02-01-16:83:1:s9": "A translated list sentence with 'bland annat' and 'samt', two O-leaning unique items (T rates .02 and .25 per 1000 words); length cannot make up for them.",
    "ep-98-02-18:135:1:s1": "'Jag skulle vilja upprepa' ('I would like to repeat') points to T, but 'det är bra att' ('it is good to') cancels it; wrong by 0.003.",
    "ep-00-05-16:158:1:s5": "'Jag skulle vara glad att ge er' = 'I would be glad to give you' (native 'Jag ger er gärna') is a calque that the lexicon lacks; nothing fires.",
    # --- 100-token chunks: TP
    "c100:1998-03-10:1:3": "Saturated by several families: 'Slutligen,' with the English comma, 'i synnerhet', 'i allmänhet', 'Först och främst', five 'kommer att', 'skulle jag vilja' and 'hennes'. connpunct hits its cap.",
    "c100:2002-03-12:1:82": "'göra det möjligt för dem att' (EN 'make it possible for them to'), 'spenderar', 'Jag skulle vilja att', four 'kommer att' and 'mycket' + adjective; also 'om ni så vill' ('if you like') rendered with 'ni'.",
    "c100:2003-01-15:1:28": "'de senaste tio åren', two 'i synnerhet', 'Till sist,' with a comma, 'välkomnar' and nominal 'tillämpningen av' all point to T. v2_violation misses 'Till sist, Afghanistan är' because the subject is a proper noun.",
    "c100:2001-05-02:1:46": "Right by 0.002: four possessives ('våra revisorer', 'dess rapportering') and 'Detta är' point to T; two 'skall' and 'bra' point to O.",
    "c100:2003-02-10:1:41": "Report-style prose with long sentences (+0.67) and 'kommer att'. The native particle verb 'gick ut på' (-1.0) nearly flips it.",
    # --- 100-token chunks: TN
    "c100:1998-04-30:0:4": "Rhetorical native speech: 'ju', 'väl', 'egentligen', 'ändå', anaphoric 'då', 'bland annat', three 'skall' and two exclamations. The particles family hits its cap.",
    "c100:1997-10-21:0:22": "Two 'ju', two particle uses of 'nämligen' (now a rule), the fronted objects 'Det ser man' and 'Där ser man', and 'tycker ... bra': native from every angle.",
    "c100:2000-02-02:0:9": "'ju', 'väldigt', particle 'nämligen', 'självklart', 'tycker' and six generic 'man': a dense native register profile.",
    "c100:2000-07-05:0:35": "Formal native prose without particles: 'dock' and two 'också' (O) barely outweigh 'dess ändringsförslag' (T); right by 0.01.",
    "c100:2003-10-21:0:7": "Middle-field 'dock' and 'därför' (O) against 'säkerställa' and several 'av' (T); right by 0.01.",
    # --- 100-token chunks: FP
    "c100:2001-04-04:0:138": "A Council answer read by a Swedish speaker, in officialese: 'ta itu med', 'I synnerhet', 'kommer den att', 'anser'. No native clue fires.",
    "c100:2001-06-12:0:48": "Council-presidency prose by a Swedish speaker: 'i allmänhet ... i synnerhet', two 'kommer att', 'säkerställa', 'Låt mig få påpeka'. The abbreviation 's.k.' is the only native clue.",
    "c100:2002-04-09:0:27": "A Commission statement given in Swedish: four 'kommer att', 'säkerställa', nominal 'betänkandet av' and the zero relative 'det förslag vi kommer att lägga fram'. Its Swedish reads like translated Commission prose.",
    "c100:2002-09-25:0:16": "Plain written native prose with no lexical clue; 'Detta har' (T) tips it by 0.001. The V2 clause 'Det togs därigenom' is not classified.",
    "c100:1999-10-06:0:5": "Neutral native political prose: 'också' (O) is outweighed by length and absence terms by 0.004. The fronted PP 'I decennier har ...' carries almost no weight.",
    # --- 100-token chunks: FN
    "c100:2003-06-02:1:9": "A translation in native conventions: three 'skall', 'dvs.', 't.o.m.' and an exclamation; only 'För det andra kan jag' points to T.",
    "c100:2001-04-03:1:203": "A translated Commission reply with 'ju', 'tycker' and native V2 ('Därför måste jag', 'Förhoppningsvis kommer vi'); only 'i själva verket' and 'Detta är' point to T.",
    "c100:1998-07-01:1:44": "Generic 'you' rendered as 'ni' saturates ni_vs_man (+1.5), but the translator also used anaphoric 'då' twice, 'alltså' and V2 'Då har ni', which read as native; a TP under the starter rules, now a false negative.",
    "c100:2000-05-16:1:98": "Three 'förstås' (EN 'of course') count as the native choice against 'naturligtvis'. 'våra liv i deras händer' and 'det här' point to T; wrong by 0.001.",
    "c100:1999-03-09:1:36": "Possessives ('vår plikt', 'mina kolleger') point weakly to T and lose to absence terms. 'Inte nog med det' (EN 'Not only that') and 'på en mycket bredare skala' ('on a much broader scale') are calques that no rule covers.",
}


def load_engine(spec):
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)()


def outcome(truth, pred):
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(truth, pred)]


def pick(cands, n_conf, n_border):
    """cands: list of (abs_score, n_spans, ruleset, row). Confident first, then borderline."""
    chosen, seen_ids = [], set()

    def take(order, k):
        got, used = [], set()
        for pass_ in (0, 1, 2):
            for c in order:
                if len(got) >= k:
                    return got
                if c[3]["id"] in seen_ids:
                    continue
                if pass_ == 0 and (c[1] == 0 or c[2] in used):
                    continue          # first pass: has a clue span and a new rule set
                if pass_ == 1 and c[1] == 0:
                    continue          # second pass: has a clue span
                got.append(c)
                used.add(c[2])
                seen_ids.add(c[3]["id"])
        return got

    conf = take(sorted(cands, key=lambda c: (-c[0], c[3]["id"])), n_conf)
    chosen += [("confident", c) for c in conf]
    border = take(sorted(cands, key=lambda c: (c[0], c[3]["id"])), n_border)
    chosen += [("borderline", c) for c in border]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="tsw.engine:Default")
    ap.add_argument("--n", type=int, default=3000, help="units sampled per granularity")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--data", default=os.environ.get("TSW_DATA", os.path.join(ROOT, "data", "processed")))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cache", "dashboard", "examples.json"))
    args = ap.parse_args()

    eng = load_engine(args.engine)
    rules = {r.name: r for r in eng.rules}
    rows, summary = [], []
    for gran, gran_name in GRANS:
        path = os.path.join(args.data, "trn.%s.bal.jsonl" % gran)   # trn only, never val/test
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        rng = random.Random("%d:%s" % (args.seed, gran))
        sample = [json.loads(lines[i]) for i in sorted(rng.sample(range(len(lines)), min(args.n, len(lines))))]
        by_out = collections.defaultdict(list)
        fire = collections.defaultdict(collections.Counter)
        nospan = collections.Counter()
        for rec in sample:
            s, contrib = eng.score_and_explain(rec["text"])
            pred = 1 if s > 0 else 0
            out = outcome(int(rec["label"]), pred)
            ann = eng.annotate(rec["text"])
            ruleset = frozenset(sp["rule"] for sp in ann["spans"])
            by_out[out].append((abs(s), len(ann["spans"]), ruleset, rec, ann))
            for r in contrib:
                fire[out][r] += 1
            if not ann["spans"]:
                nospan[out] += 1
        summary.append("== %s (n=%d): %s" % (gran, len(sample),
                       "  ".join("%s %d" % (o, len(by_out[o])) for o in OUTCOMES)))
        for o in OUTCOMES:
            k = len(by_out[o]) or 1
            fr = ", ".join("%s %.0f%%" % (r.split(".")[-1], 100.0 * c / k) for r, c in fire[o].most_common(6))
            summary.append("   %s: no clue span %.0f%%; rules firing: %s" % (o, 100.0 * nospan[o] / k, fr))
        for o in OUTCOMES:
            cands = [(a, n, rs, rec) for a, n, rs, rec, _ in by_out[o]]
            anns = {rec["id"]: ann for _, _, _, rec, ann in by_out[o]}
            for kind, (_, _, _, rec) in pick(cands, N_CONFIDENT, N_BORDER):
                ann = anns[rec["id"]]
                doc = Doc(rec["text"])
                top = []
                for name, c in sorted(ann["rules"].items(), key=lambda kv: -abs(kv[1]))[:TOP_RULES]:
                    top.append({"rule": name, "contribution": round(c, 3),
                                "evidence": rules[name].describe(doc, c)})
                for d in ann["doc_level"]:
                    if d["rule"].startswith("cap:"):
                        top.append({"rule": d["rule"], "contribution": round(d["contribution"], 3),
                                    "evidence": d["label"]})
                rows.append({
                    "id": rec["id"], "split": "trn", "gran": gran, "gran_name": gran_name, "outcome": o, "kind": kind,
                    "truth": int(rec["label"]), "pred": 1 if ann["score"] > 0 else 0,
                    "score": round(ann["score"], 3), "prob": round(ann["prob"], 3),
                    "date": rec.get("date"), "n_words": ann["n_words"], "text": rec["text"],
                    "spans": [{"start": sp["start"], "end": sp["end"], "text": sp["text"],
                               "rule": sp["rule"], "family": sp["family"], "direction": sp["direction"],
                               "contribution_share": round(sp["contribution_share"], 3),
                               "label": sp["label"]} for sp in ann["spans"]],
                    "top": top,
                    "doc_level": [{"rule": d["rule"], "direction": d["direction"],
                                   "contribution": round(d["contribution"], 3), "label": d["label"]}
                                  for d in ann["doc_level"]],
                    "note": NOTES.get(rec["id"], ""),
                })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")
    missing = [r["id"] for r in rows if not r["note"]]
    print("\n".join(summary))
    print("wrote %d examples to %s" % (len(rows), os.path.relpath(args.out, ROOT)))
    if missing:
        print("no note for %d: %s" % (len(missing), " ".join(missing)), file=sys.stderr)


if __name__ == "__main__":
    main()

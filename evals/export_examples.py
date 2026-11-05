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
# the default seed and the starter rule set; ids that drop out of the selection
# when rules change simply lose their note).
NOTES = {
    # --- sentences: TP
    "ep-03-11-18:252:1:s26": "Two 'det här' for English 'this' (incl. 'med det här parlamentets stöd' = 'with this Parliament's support') saturate the demonstrative rule. Missed: 'redo för handling' is a calque of 'ready for action'.",
    "ep-02-12-18:101:1:s16": "Pseudo-cleft 'Vad vi behöver är action' (EN 'What we need is action', even keeping the English word), a semicolon-separated anaphoric 'för att' list and 'sist men inte minst' ('last but not least'). The engine only sees 'de här' and length; pseudo-cleft, semicolon and idiom rules would catch it.",
    "ep-01-02-12:94:1:s14": "'det faktum att' calque fires. 'denna kammare' (twice) renders the EP-English 'this House' yet counts as a weak original clue; a 'denna kammare / detta parlament' calque would flip it.",
    "ep-00-03-01:17:1:s25": "Coin flip: one 'denna' vs length. Missed: 'på grund av någon önskan hos Prodis kommission' mirrors 'because of any desire on the part of the Prodi Commission' (negative-polarity 'någon', 'hos' for 'on the part of').",
    "ep-04-01-14:25:1:s2": "Right by a hair: only 'dessa'/'skall' and length. The passive triplet 'luras, skrämmas eller pressas att lova' follows English 'fooled, bullied or pressured into promising'; 'pressas att' for 'pressured into' is a catchable calque.",
    # --- sentences: TN
    "ep-98-05-13:223:1:s4": "Modal particle 'ju' ('as we know') has no English trigger, so it is a clean original clue; short sentence adds a little.",
    "ep-97-09-16:110:1:s13": "Spoken native Swedish: 'ju', fronted 'så' with V2 inversion ('så är det') and an exclamation. Only 'ju' is used; the V2 'så är det' pattern is a second, uncaptured clue.",
    "ep-01-05-02:86:1:s13": "Left-dislocated att-clause picked up by 'det', a dashed aside '- det görs ju inte i dag -' and a tag 'är inte det en framgång?': native spoken syntax. The engine only sees 'ju'.",
    "ep-98-05-15:19:1:s6": "Native: negation raising 'anser vi inte att ... utan', late particle verb 'ta sig an' after a long parenthesis. No rule covers these; four denna/skall forms barely tip it to original.",
    "ep-02-09-24:149:1:s10": "Neutral administrative sentence ('Det återstår sex månader tills ...'); one 'skall' decides at -0.005. No real evidence either way.",
    # --- sentences: FP
    "ep-97-10-21:35:1:s1": "A Swedish speaker says 'först av allt': the calque was already established in EP Swedish, and its weight (2.0) lets one hit decide. Lower weight or require a second clue.",
    "ep-99-05-05:251:1:s8": "Native speaker using 'det faktum att'. Missed original clues: conditional 'vore det bra', 'i själva verket', hedging 'kanske'; the topic ('svenska företrädare', small languages) would be masked.",
    "ep-98-06-16:226:1:s4": "Formal written Swedish ('För det andra:', 'dvs.', V1 question with a long subject) where 'det faktum att' is native too (O rate 0.09/1000 words). One calque hit outweighs everything.",
    "ep-99-12-01:152:1:s5": "Colloquial native speech: particle verb 'få ihop', 'dvs.', 'några gånger'. Three denna-forms pull toward original, but sentence length tips it to translated by 0.003.",
    "ep-00-07-05:332:1:s34": "Two coordinated 'hur'-clauses as subject; no lexical clue at all, the score is neutral length only. Undecidable for the current rules.",
    # --- sentences: FN
    "ep-99-12-15:293:1:s2": "Regex bug: 'Paradoxalt nog' is a sentence adverb (EN 'paradoxically'), not the modal particle 'nog'. Exclude '<adjective>-t nog' (paradoxalt/märkligt/beklagligt nog).",
    "ep-02-05-14:54:1:s3": "Regex bug: 'väl förberedda' = 'well prepared', but the participle filter misses '-dda' forms. 'Vi vet alla att' ('We all know that') and 'någonsin' ('ever') are English-shaped.",
    "ep-00-05-16:281:1:s15": "'Detta är oacceptabelt.' = 'This is unacceptable.': three words, the only evidence is short length. Not decidable at sentence level.",
    "ep-00-03-15:133:1:s43": "Missed calque: 'Detta leder in mig på budgeten' = 'This leads me on to the budget' (native: 'Därmed kommer jag till budgeten'). Discourse-transition calques are a good rule target.",
    "ep-99-01-13:172:1:s5": "Missed calque: 'Det tycks mig som att' = 'It seems to me that' (native 'Det förefaller mig som om' / 'Jag tycker'). Two 'detta' push it to original.",
    # --- 100-token chunks: TP
    "c100:2001-02-01:1:20": "'det/den här' and 'Får jag först av allt tacka' ('May I first of all thank'). Missed: 'som ett preliminärt svar till det som sagts' uses 'svar till' for 'answer to' (native 'svar på').",
    "c100:2002-11-20:1:70": "Two 'det här' and 'det faktum att'. Also missed: 'Som svar till X' (native 'svar på'), passive 'bli övertygad av' ('be convinced by').",
    "c100:2001-02-01:1:41": "Four 'den/det här frågan' for English 'this issue' cap the rule. Also: 'ytterligare en hormon' (gender slip, 'ett hormon'), sentence-final 'till exempel?' copying English 'for example'.",
    "c100:1998-07-01:1:44": "Lively spoken source: generic 'you' rendered as 'ni' throughout ('Ni kan sedan gå ned ...'), loanword 'gate'. Engine sees nothing but length; generic 'ni' (native: 'man') would be a strong translation clue.",
    "c100:2003-12-18:1:5": "Missed calque 'Med detta sagt' = 'That said / With this said'; verbless 'För det första X och för det andra Y'. Three 'detta' almost flip it to original.",
    # --- 100-token chunks: TN
    "c100:2004-04-22:0:3": "Two concessive 'visst' plus 'tyvärr' and short sentences: typical native argumentation. Correct for the right reasons.",
    "c100:1997-10-21:0:22": "Two 'ju', two 'nämligen', fronting ('Det ser säkert vi', 'det kanske man inte ser'). 'nämligen' has no English trigger but is not yet a rule; a strong native clue.",
    "c100:1997-02-19:0:2": "Two 'ju', informal 'pratade', realia ('tiokronor och femkronor') and 'efter det att'. Correct; particles carry it.",
    "c100:2004-04-19:0:1": "Native: 'Först vill jag nämna' (not the calque 'först av allt'), 'ytterst' twice, 'beklämmande'. Only one 'detta' fires, so it is right by 0.0004; native intensifiers would help.",
    "c100:1998-05-15:0:3": "'Men även om ..., så borde' (resumptive 'så' after a fronted subordinate clause) is native Swedish with no English counterpart; also 'blotta misstanken', 'dvs.'. Engine only sees four denna-forms.",
    # --- 100-token chunks: FP
    "c100:2001-05-03:0:3": "A Swedish speaker using 'den här mobiliteten', 'det här området' and 'det faktum att': the dev-calibrated 'den här' direction fails for informal native speakers and saturates at 3.0.",
    "c100:2001-04-05:0:1": "Two 'det här' outweigh native signs: V2 'Därför behövs det', 'på sikt', 't.ex.', mid-position 'dock'. The English code-switch 'food safety-process' is a native-speaker anglicism, so a loanword rule would mislead here too.",
    "c100:2002-06-11:0:17": "Topicalised objects with a stranded preposition ('De mål vi sätter upp ... skall vi naturligtvis leva upp till', 'Allt detta bär vi med oss') are very native but undetected; 'det/den här' wins.",
    "c100:1997-04-09:0:14": "Native but no rule fires: 'inget som helst hot', 'Det framgick dock', 'enligt min mening'. Length alone tips it by 0.0003.",
    "c100:2002-06-12:0:19": "Formal native register ('i afton', 'Till att börja med'), but 'Tillåt mig att poängtera' looks like 'Allow me to stress'. Near zero; no decisive clue fires.",
    # --- 100-token chunks: FN
    "c100:2003-01-13:1:22": "Ironic speech: the translator rendered tag questions with 'väl' and 'ju', so particles can occur in translation. The '?Om' and '??' are garbled quote marks (a corpus artefact, not a clue).",
    "c100:2001-02-13:1:134": "Regex bug: 'långt nog' = 'far enough' (degree 'nog'). Missed translation clues: pseudo-cleft 'Vad vi försökte uppnå var', 'Inte bara ...; även ...' with a semicolon, 'Det finns många medlemsstater som anser'.",
    "c100:2003-11-18:1:142": "Regex bug: 'Beklagligt nog' ('regrettably') read as modal 'nog'. Missed: commissioner named in full with title, double negation 'inte skulle påstå att det inte är någon skillnad'.",
    "c100:2001-10-23:1:92": "Formal translations default to 'detta', so six denna-forms push it to original. Missed: 'Det är bara rätt och riktigt' ('only right and proper'), 'på allas våra vägnar' ('on behalf of all of us'), semicolon.",
    "c100:2002-09-25:1:8": "V2 violation: 'Med anknytning till detta, utvecklingen ... har varit positiv' (fronted adverbial, comma, subject before verb, as in English). A V2 rule would catch it; also 'pressas att', 'chef de file'.",
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

"""Write the Python reference results that web/parity_test.ts compares the JS port with.

    python3 -I web/export_parity.py [--n 2000] [--seed 20261106]

Texts (trn ONLY, never val or test):
  - n/2 sentences from data/processed/trn.sent.bal.jsonl and n/2 100-token
    chunks from trn.100.bal.jsonl, sampled with a fixed seed;
  - one very long text (40 trn 100-token chunks joined);
  - every string literal in tsw/tests/*.py (the unit-test inputs);
  - hand-written edge cases (curly quotes, dashes, NBSP, decomposed å, empty
    text, emoji, Greek final sigma, ...).

Output: web/parity/cases.json (gitignored: it contains Europarl text). For
each text, Engine.annotate() plus the normalised text, sentence spans and the
match spans of every rule regex (to test the regex translation in isolation).
Offsets are Python code-point offsets; the Deno test converts them.
"""
import argparse
import ast
import glob
import json
import os
import random
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tsw.doc import Doc  # noqa: E402
from tsw.engine import Default  # noqa: E402
from tsw.rules import base, wordorder, connpunct  # noqa: E402
from tsw import text as T  # noqa: E402

EDGE = [
    "",
    "   ",
    "\n\t  ",
    "Detta är ”viktigt” – och  det här är ‘bra’… Vi måste dessutom, i synnerhet nu, agera!",
    "Herr talman! Jag tycker att det faktum att vi är här är bra.",
    "Dét hår å mänga ö – décomposed.",   # decomposed å/ö/é
    "Vi har t.ex. fisket, bl.a. torsk, dvs. allt. Nu går vi vidare.",
    "Herr talman! Fru kommissionär, kära kolleger! Först av allt vill jag tacka. Dessutom, vi måste se till att detta "
    "genomförs på daglig basis. Det vet vi ju alla.",
    "soft­hyphen och zero​width och ⁠word joiner﻿",
    "em—dash, minus − 5, «guillemets» „låga” ‚enkla’ ″sekunder″",
    "Emoji 😀 i texten. Vi 👍🏽 tycker att det är bra! Och sedan 𝔘𝔫𝔦𝔠𝔬𝔡𝔢.",
    "ΑΣ ΟΔΟΣ. Σίσυφος. İstanbul ÅSA ÄR STORA BOKSTÄVER. DET FAKTUM ATT VI ÄR HÄR.",
    "Siffror ². Nästa mening ³ och ① cirkel. 1 000 000,5 kronor. 12:30. 2,3 procent. 1.5 gånger.",
    "alla små bokstäver utan versaler. det faktum att vi är här. först av allt.",
    "Vad vi behöver är tydlighet. Om vi inte agerar nu, kommer vi att förlora. Detta kan vi inte acceptera.",
    "Med andra ord, de är inte redo. Jag verkligen hoppas att vi kan inte vänta. Slutligen, kommissionen är ansvarig.",
    "Det är vi som bestämmer. Därför anser jag att det är bra. I dag har vi en debatt. Den frågan har vi diskuterat.",
    "Tabbar\tradbrytningar\r\n\r\nflera  mellanslag   och radavgränsare stycke.",
    "(Applåder) – Tack! ”Det är bra”, sa hon. ’Ja’ – nej... Vad?! Nåväl.",
    "a" * 3000,
    "Fru talman! " * 50,
    "Vi tycker också att man ska göra det. " * 200,
    "x" + "́" * 5 + " och é̂ samt ́ inledande kombinerande tecken.",
    "Kanske vi borde. Även vi måste. Bara du kan. Alltså, vi vet. Nu när vi vet det, kan vi gå vidare.",
    "Jag ser fram emot det. Vi välkomnar betänkandet. Vi måste säkerställa att EU:s regler följs i EU-länderna.",
    "Ärade ledamöter, jag har nämligen inte tid. Det är ju så att vi egentligen inte vet.",
]


def unit_test_strings():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "tsw", "tests", "test_*.py"))):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 2:
                if node.value not in out:
                    out.append(node.value)
    return out


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def regex_table(eng):
    """{name: (compiled regex, case)} for every rule matcher and family regex. Keyed by name,
    not source: words_rx() and TOKEN_RX order equal-length alternatives by set iteration,
    which depends on the per-process hash seed (same matches, different source text)."""
    tab = {}
    for r in eng.rules:
        for attr in ("matcher", "matcher_a", "matcher_b"):
            if hasattr(r, attr):
                m = getattr(r, attr)
                tab["%s:%s" % (r.name, attr)] = (m.rx, m.case)
    tab["wordorder:EMBEDDED_NEG_RX"] = (wordorder.EMBEDDED_NEG_RX, False)
    tab["text:token"] = (T.TOKEN_RX, True)
    tab["text:candidate"] = (T._CANDIDATE, True)
    tab["exclamation:end_rx"] = (connpunct._EXCL_END, True)
    return tab


def case(eng, rxs, cid, src, text):
    d = {"id": cid, "src": src, "text": text}
    try:
        a = eng.annotate(text)
    except Exception as e:  # recorded; the JS test expects an error too
        d["error"] = "%s: %s" % (type(e).__name__, e)
        return d
    doc = Doc(text)
    a["norm"] = doc.text
    a["lower"] = doc.lower
    a["sent_spans"] = doc.sent_spans
    a["n_tokens"] = sum(len(t) for t in doc.sent_tokens)
    a["is_address"] = doc.is_address
    a["openings"] = [[i, o.cat, o.a, o.b] for i, o in wordorder._openings_cached(doc)]
    m = {}
    for key, (rx, is_case) in rxs.items():
        spans = [[x.start(), x.end()] for x in rx.finditer(doc.text if is_case else doc.lower)]
        if spans:
            m[key] = spans
    a["regex"] = m
    d["py"] = a
    return d


def ranges(pred):
    out = []
    for c in range(sys.maxunicode + 1):
        if pred(chr(c)):
            if out and out[-1][1] == c - 1:
                out[-1][1] = c
            else:
                out.append([c, c])
    return out


def unicode_tables():
    """Python's view of every code point, for the JS code-point sweep."""
    import re
    word = re.compile(r"\w")
    space = re.compile(r"\s")
    return {
        "isalnum": ranges(str.isalnum), "isalpha": ranges(str.isalpha), "isupper": ranges(str.isupper),
        "isdigit": ranges(str.isdigit), "word": ranges(lambda ch: word.match(ch) is not None),
        "space": ranges(lambda ch: space.match(ch) is not None),
        "unassigned": ranges(lambda ch: unicodedata.category(ch) == "Cn"),
        "lower": {str(c): chr(c).lower() for c in range(sys.maxunicode + 1) if chr(c).lower() != chr(c)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--out", default=os.path.join(HERE, "parity", "cases.json"))
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(args.seed)
    proc = os.path.join(ROOT, "data", "processed")
    sent = load(os.path.join(proc, "trn.sent.bal.jsonl"))
    c100 = load(os.path.join(proc, "trn.100.bal.jsonl"))
    for recs in (sent, c100):
        assert all(r["id"] and r.get("date", "") <= "2004-06-30" for r in recs), "not trn"
    texts = []
    for name, recs, k in (("trn.sent", sent, args.n // 2), ("trn.100", c100, args.n - args.n // 2)):
        for r in rng.sample(recs, k):
            texts.append((r["id"], name, r["text"]))
    long_parts = rng.sample(c100, 40)
    texts.append(("long:40x100", "trn.100 x40", "\n\n".join(r["text"] for r in long_parts)))
    for i, s in enumerate(unit_test_strings()):
        texts.append(("unit:%d" % i, "unit-test string", s))
    for i, s in enumerate(EDGE):
        texts.append(("edge:%d" % i, "edge case", s))
    eng = Default()
    rxs = regex_table(eng)
    cases = [case(eng, rxs, cid, src, text) for cid, src, text in texts]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"n": len(cases), "seed": args.seed, "python": sys.version.split()[0],
                            "unicode": unicodedata.unidata_version,
                            "regex_keys": list(rxs), "unicode_tables": unicode_tables()},
                   "cases": cases}, f, ensure_ascii=False)
    print("wrote %s: %d cases (%d errors) in %.1fs"
          % (os.path.relpath(args.out, ROOT), len(cases), sum("error" in c for c in cases), time.time() - t0))


if __name__ == "__main__":
    main()

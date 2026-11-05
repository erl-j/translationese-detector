#!/usr/bin/env python3
"""Build labelled Swedish original-vs-translationese splits from Europarl v7.

One deterministic command (run it through Slurm, it takes a few CPU-minutes):

    python3 -I data/build.py            # download/extract if needed, then build
    python3 -I data/build.py --stats    # only print per-year counts, write nothing

Label 0 = original Swedish, label 1 = Swedish translated from English.

Labelling (see data/MANIFEST.md for the numbers):
  Europarl puts the LANGUAGE attribute on a speaker turn mostly in the
  *non-original* language versions; in the original-language file the tag is
  usually absent. So a Swedish original speech is untagged in txt/sv but tagged
  LANGUAGE="SV" in txt/en, and an English speech is tagged LANGUAGE="EN" in
  txt/sv but untagged in txt/en. We therefore collect, for every turn of
  txt/sv, the source-language evidence from four places:
      sv LANGUAGE attr, en LANGUAGE attr (same file, same SPEAKER ID),
      inline "(XX)" marker at the start of the sv text, and of the en text.
  A turn is kept iff all present evidence agrees on SV (label 0) or EN
  (label 1). Conflicting evidence -> dropped ("inconsistent"); no evidence ->
  dropped ("untagged"). Turns whose sv/en NAME attributes disagree are dropped
  (misalignment), as are turns whose sv text does not look Swedish.

Only the Python standard library is used. Downloaded files are treated as
untrusted data: nothing from them is executed.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "europarl")
TGZ = os.path.join(RAW, "europarl.tgz")
TXT = os.path.join(RAW, "txt")
OUT = os.path.join(ROOT, "data", "processed")
URL = "https://www.statmt.org/europarl/v7/europarl.tgz"
TGZ_SHA256 = "d109549ff35623ded3d375dc04040e3f0b399665680ed726281f877ff5995330"

SEED = 20261105
# Split by session date: dev = sessions up to and including DEV_LAST_DATE,
# test = sessions after it. The cut is the June 2004 European election (end of
# the 1999-2004 term, start of the post-enlargement parliament), which gives
# ~60% of the scarce original-Swedish tokens to dev and ~40% to test and keeps
# the two splits in different parliamentary terms (fewer shared speakers).
DEV_LAST_DATE = "2004-06-30"
GRANS = ("sent", "100", "500")
SENT_MIN_TOKENS = 3  # sentence granularity: drop fragments shorter than this

# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_raw():
    if not (os.path.isdir(os.path.join(TXT, "sv")) and os.path.isdir(os.path.join(TXT, "en"))):
        if not os.path.exists(TGZ):
            os.makedirs(RAW, exist_ok=True)
            print("downloading", URL, file=sys.stderr)
            urllib.request.urlretrieve(URL, TGZ)
        got = sha256(TGZ)
        if got != TGZ_SHA256:
            sys.exit("sha256 mismatch for %s: %s" % (TGZ, got))
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call([sys.executable, "-I", os.path.join(here, "extract_europarl.py"),
                               TGZ, TXT, "sv", "en"])


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

FNAME = re.compile(r"^ep-(\d\d)-(\d\d)-(\d\d)(?:-[0-9-]+)?\.txt$")
SPEAKER = re.compile(r"^<SPEAKER\b(.*)>\s*$")
ATTR = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+?))(?=\s|$)')
TAGLINE = re.compile(r"^<[A-Z/][^>]*>\s*$")
# "skriftlig. - (EN) ...", "för PPE-DE-gruppen. - (EN) ...", ". (EN) ...", "(EN) ..."
EU_LANGS = "BG CS DA DE EL EN ES ET FI FR GA HU IT LT LV MT NL PL PT RO SK SL SV"
INLINE_LANG = re.compile(r"^[^()]{0,250}?\((%s)\)\s*" % "|".join(EU_LANGS.split()))
ANY_LANG_MARK = re.compile(r"\s*\((?:%s)\)\s*" % "|".join(EU_LANGS.split()))
# "för PPE-DE-gruppen. - Herr talman! ..." (role prefix without language marker)
ROLE_PREFIX = re.compile(r"^(?:[a-zåäöé][^()!?]{0,250}?\.\s-\s)")
WHOLE_PAREN = re.compile(r"^\(.*\)\s*$")
PROC_PAREN = re.compile(
    r"\s*\((?:Applåd|Livliga|Skratt|Munterhet|Sorl|Protest|Blandade|Rop|Heckling|Talaren"
    r"|Parlamentet|Kammaren|Sammanträdet|Omröstning|Talmannen|Ordföranden|Protokollet"
    r"|Röstförklaring|Inlägget|Mikrofonen|Avbrott|Oro|Buanden|Visslingar|Ljudlig"
    r"|Ordningsfråga|Talmanskonferensen|Föredraganden|Debatten|Förslaget)[^()]*\)")


def parse_attrs(s):
    d = {}
    for k, v1, v2 in ATTR.findall(s):
        d[k.upper()] = v1 if v1 or not v2 else v2
    return d


def file_date(fname):
    m = FNAME.match(fname)
    if not m:
        return None
    yy, mm, dd = m.groups()
    y = int(yy)
    y = 1900 + y if y >= 90 else 2000 + y
    return "%04d-%s-%s" % (y, mm, dd)


def parse_file(path):
    """Return list of turns: dict(key, sid, name, lang, lines)."""
    turns = []
    seen = collections.Counter()
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if line.startswith("<"):
                m = SPEAKER.match(line)
                if m:
                    a = parse_attrs(m.group(1))
                    sid = a.get("ID", "")
                    seen[sid] += 1
                    cur = {"key": (sid, seen[sid]), "sid": sid, "name": a.get("NAME", ""),
                           "lang": a.get("LANGUAGE", "").upper() or None, "lines": []}
                    turns.append(cur)
                elif line.startswith("<CHAPTER"):
                    cur = None
                # <P> and other tags: paragraph break, ignore
                continue
            if cur is not None and line:
                cur["lines"].append(line)
    return turns


def norm_name(n):
    n = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in n if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z]+", " ", n).strip()


def names_agree(a, b):
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return True
    ta, tb = set(a.split()), set(b.split())
    return bool(ta & tb)


def clean_name(n):
    """'Eriksson (GUE/NGL ).' -> 'Eriksson'; 'Bolkestein,' -> 'Bolkestein'."""
    return re.sub(r"\s*\(.*$", "", n).strip(" ,.-")


def inline_lang(lines):
    if not lines:
        return None
    m = INLINE_LANG.match(lines[0])
    return m.group(1) if m else None


SV_STOP = set("och att det som en är på för med av inte till den har vi jag de om ett kan men så detta "
              "också mycket skall ska eller måste där när".split())
EN_STOP = set("the and of to is that in we this it for not are be have with on which as by "
              "should would there".split())


def looks_swedish(text):
    toks = re.findall(r"[a-zåäöé]+", text.lower())
    sv = sum(t in SV_STOP for t in toks)
    en = sum(t in EN_STOP for t in toks)
    return sv > en


def clean_lines(lines, stats):
    out = []
    for i, line in enumerate(lines):
        if WHOLE_PAREN.match(line):
            stats["drop_paren_line"] += 1
            continue
        m = INLINE_LANG.match(line)
        if m:
            line = line[m.end():]
            stats["strip_inline_lang_line%s" % ("0" if i == 0 else "N")] += 1
        elif i == 0:
            m = ROLE_PREFIX.match(line)
            if m:
                line = line[m.end():]
                stats["strip_role_prefix"] += 1
        # any remaining language marker anywhere (would leak the label)
        line, n = ANY_LANG_MARK.subn(" ", line)
        if n:
            stats["strip_inline_lang_midline"] += n
        new, n = PROC_PAREN.subn("", line)
        if n:
            stats["strip_proc_paren"] += n
        new = new.strip()
        if new:
            out.append(new)
    return out


# --------------------------------------------------------------------------
# sentence splitting (simple, rule-based, Swedish abbreviations)
# --------------------------------------------------------------------------

ABBREV = set("""t.ex bl.a m.m s.k dvs d.v.s osv o.s.v resp ca nr art kap jfr fr.o.m t.o.m o.d p.g.a
f.d m.fl m.a.o o.dyl e.d s ff st kl tel dr prof mr mrs hr fru ang betr enl inkl exkl vs ev
etc ex t.ex. bl jan feb mar apr jun jul aug sep sept okt nov dec""".split())
SENT_BOUND = re.compile(r"(?<=[.!?])[\"”»']?\s+(?=[\"“«(]?[A-ZÅÄÖÉ0-9])")


def split_sentences(par):
    out = []
    start = 0
    for m in SENT_BOUND.finditer(par):
        left = par[start:m.start()].rstrip("\"”»'")
        lastw = left.rsplit(None, 1)[-1] if left.split() else ""
        lw = lastw.lower().rstrip(".")
        if left.endswith(".") and (lw in ABBREV or re.fullmatch(r"[a-zåäö]", lw)
                                   or re.fullmatch(r"\d{1,2}", lw)):
            continue  # abbreviation, single letter, or ordinal like "1."
        out.append(par[start:m.start()].strip())
        start = m.end()
    tail = par[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def ntok(s):
    return len(s.split())


# --------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------


def label_turns(stats):
    svdir, endir = os.path.join(TXT, "sv"), os.path.join(TXT, "en")
    files = sorted(f for f in os.listdir(svdir) if FNAME.match(f))
    kept = []
    for fname in files:
        date = file_date(fname)
        sv_turns = parse_file(os.path.join(svdir, fname))
        enp = os.path.join(endir, fname)
        en_turns = parse_file(enp) if os.path.exists(enp) else None
        if en_turns is None:
            stats["files_without_en"] += 1
        en_by_key = {t["key"]: t for t in en_turns} if en_turns else {}
        for t in sv_turns:
            stats["turns_total"] += 1
            e = en_by_key.get(t["key"])
            ev = collections.OrderedDict()
            ev["sv_attr"] = t["lang"]
            ev["sv_inline"] = inline_lang(t["lines"])
            ev["en_attr"] = e["lang"] if e else None
            ev["en_inline"] = inline_lang(e["lines"]) if e else None
            langs = set(v for v in ev.values() if v)
            # what the brief's naive rule (sv attr only) would have said
            naive = {"SV": 0, "EN": 1}.get(t["lang"])
            if naive is not None:
                stats["naive_sv_attr_%s_total" % t["lang"]] += 1
            if not langs:
                stats["drop_untagged"] += 1
                continue
            if len(langs) > 1:
                stats["drop_inconsistent"] += 1
                if naive is not None:
                    stats["naive_sv_attr_%s_dropped_by_crosscheck" % t["lang"]] += 1
                if langs & {"SV", "EN"}:
                    stats["drop_inconsistent_sv_or_en"] += 1
                    stats["inconsistent_example:" + "/".join(sorted(langs))] += 1
                continue
            lang = langs.pop()
            if lang not in ("SV", "EN"):
                stats["drop_other_lang"] += 1
                continue
            label = 0 if lang == "SV" else 1
            cls = "orig" if label == 0 else "trans"
            if e is None:
                stats["%s_no_en_counterpart" % cls] += 1
            elif not names_agree(t["name"], e["name"]):
                stats["drop_name_mismatch_%s" % cls] += 1
                if naive is not None:
                    stats["naive_sv_attr_%s_dropped_by_crosscheck" % t["lang"]] += 1
                continue
            nsrc = sum(1 for v in ev.values() if v)
            stats["%s_evidence_sources_%d" % (cls, nsrc)] += 1
            if naive is not None and naive == label:
                stats["%s_naive_sv_attr_would_find" % cls] += 1
            lines = clean_lines(t["lines"], stats)
            text = " ".join(lines)
            if not text:
                stats["drop_empty_%s" % cls] += 1
                continue
            if not looks_swedish(text):
                stats["drop_not_swedish_%s" % cls] += 1
                if stats["drop_not_swedish_%s" % cls] <= 3:
                    stats["example_not_swedish_%s: %s" % (cls, text[:120])] += 1
                continue
            kept.append({"file": fname, "date": date, "sid": t["sid"], "occ": t["key"][1],
                         "speaker": clean_name(t["name"]), "label": label, "src_lang": lang,
                         "evidence": {k: v for k, v in ev.items() if v},
                         "paras": lines})
    # exact-duplicate turns (same text, same label) -> keep first
    seen = set()
    dedup = []
    for t in kept:
        h = hashlib.sha1(("\n".join(t["paras"])).encode("utf-8")).hexdigest()
        if h in seen:
            stats["drop_duplicate_turn_%s" % ("orig" if t["label"] == 0 else "trans")] += 1
            continue
        seen.add(h)
        dedup.append(t)
    return dedup


def split_of(date):
    return "dev" if date <= DEV_LAST_DATE else "test"


def make_units(turns):
    """Yield (split, gran, record) for all granularities."""
    units = {(s, g): [] for s in ("dev", "test") for g in GRANS}
    # sentences
    for t in turns:
        sp = split_of(t["date"])
        if sp is None:
            continue
        t["sents"] = [s for p in t["paras"] for s in split_sentences(p)]
        for i, s in enumerate(t["sents"]):
            if ntok(s) < SENT_MIN_TOKENS:
                continue
            units[(sp, "sent")].append(rec(t, "sent", s, "%s:%s:%d:s%d" % (t["file"][:-4], t["sid"], t["occ"], i),
                                           [t["speaker"]], 1))
    # chunks: consecutive sentences, same class, same session date, document order
    for g in ("100", "500"):
        target = int(g)
        bydate = collections.OrderedDict()
        for t in turns:
            bydate.setdefault((t["date"], t["label"]), []).append(t)
        for (date, label), ts in bydate.items():
            sp = split_of(date)
            if sp is None:
                continue
            buf, n, spk, nturns, last_turn, first_sid = [], 0, [], 0, None, None
            k = 0
            for t in ts:
                for s in t["sents"]:
                    if id(t) is not last_turn:
                        nturns += 1
                        last_turn = id(t)
                        if first_sid is None:
                            first_sid = t["sid"]
                        if t["speaker"] not in spk:
                            spk.append(t["speaker"])
                    buf.append(s)
                    n += ntok(s)
                    if n >= target:
                        units[(sp, g)].append(rec(ts[0], g, " ".join(buf),
                                                  "c%s:%s:%d:%d" % (g, date, label, k), spk, nturns,
                                                  date=date, label=label, sid=first_sid))
                        k += 1
                        buf, n, spk, nturns, last_turn, first_sid = [], 0, [], 0, None, None
            # leftover shorter than target is dropped
    return units


def rec(t, gran, text, uid, speakers, nturns, date=None, label=None, sid=None):
    label = t["label"] if label is None else label
    return {"id": uid, "text": text, "label": label,
            "src_lang": "SV" if label == 0 else "EN",
            "date": date or t["date"], "speaker": speakers[0] if speakers else t["speaker"],
            "speakers": speakers, "speaker_id": sid or t["sid"], "n_turns": nturns,
            "granularity": gran, "n_tokens": ntok(text)}


def balance(recs, split, gran):
    by = {0: [i for i, r in enumerate(recs) if r["label"] == 0],
          1: [i for i, r in enumerate(recs) if r["label"] == 1]}
    m = min(len(by[0]), len(by[1]))
    rng = random.Random("%s|%s|%d" % (split, gran, SEED))
    keep = set()
    for lab in (0, 1):
        keep.update(by[lab] if len(by[lab]) == m else rng.sample(by[lab], m))
    return [recs[i] for i in sorted(keep)]


def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256(path)


def per_year(turns):
    py = collections.defaultdict(lambda: [0, 0, 0, 0])  # turns0, tok0, turns1, tok1
    for t in turns:
        y = t["date"][:4]
        n = sum(ntok(p) for p in t["paras"])
        py[y][2 * t["label"]] += 1
        py[y][2 * t["label"] + 1] += n
    return py


BOILER = re.compile(r"^(Herr|Fru) (talman|ordförande|kommissionär|minister)", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true", help="only print label/year statistics")
    args = ap.parse_args()

    ensure_raw()
    stats = collections.Counter()
    turns = label_turns(stats)
    py = per_year(turns)
    print("\n== labelling stats ==")
    for k in sorted(stats):
        print("  %-60s %d" % (k, stats[k]))
    print("\n== per year: orig turns/tokens, trans turns/tokens ==")
    for y in sorted(py):
        print("  %s  orig %6d %9d   trans %6d %9d" % (y, *py[y]))
    if args.stats:
        return

    units = make_units(turns)
    os.makedirs(OUT, exist_ok=True)
    manifest = {"source": URL, "source_sha256": TGZ_SHA256, "seed": SEED,
                "dev_dates": "<= %s" % DEV_LAST_DATE, "test_dates": "> %s" % DEV_LAST_DATE,
                "sent_min_tokens": SENT_MIN_TOKENS, "labelling_stats": dict(stats),
                "per_year": {y: dict(zip(["orig_turns", "orig_tokens", "trans_turns", "trans_tokens"], v))
                             for y, v in sorted(py.items())},
                "files": {}}
    # boilerplate frequency per class (turn-initial address)
    bp = {0: [0, 0], 1: [0, 0]}
    for t in turns:
        bp[t["label"]][1] += 1
        if t["paras"] and BOILER.match(t["paras"][0]):
            bp[t["label"]][0] += 1
    manifest["boilerplate_turn_initial_address"] = {
        "orig": {"turns_starting_with_address": bp[0][0], "turns": bp[0][1]},
        "trans": {"turns_starting_with_address": bp[1][0], "turns": bp[1][1]}}
    # speaker overlap between splits
    spk = {"dev": set(), "test": set()}
    for t in turns:
        sp = split_of(t["date"])
        if sp:
            spk[sp].add((t["label"], norm_name(t["speaker"])))
    manifest["speaker_overlap"] = {
        lab: {"dev": len([s for s in spk["dev"] if s[0] == i]),
              "test": len([s for s in spk["test"] if s[0] == i]),
              "both": len([s for s in spk["dev"] & spk["test"] if s[0] == i])}
        for i, lab in ((0, "orig"), (1, "trans"))}

    for (sp, g), recs in sorted(units.items()):
        for variant, rr in (("all", recs), ("bal", balance(recs, sp, g))):
            name = "%s.%s.%s.jsonl" % (sp, g, variant)
            h = write_jsonl(os.path.join(OUT, name), rr)
            c = collections.Counter(r["label"] for r in rr)
            toks = collections.Counter()
            for r in rr:
                toks[r["label"]] += r["n_tokens"]
            manifest["files"][name] = {"sha256": h, "n": len(rr), "n_orig": c[0], "n_trans": c[1],
                                       "tokens_orig": toks[0], "tokens_trans": toks[1]}
            print("%-24s n=%7d  orig=%7d  trans=%7d  sha256=%s" % (name, len(rr), c[0], c[1], h[:12]))
    with open(os.path.join(ROOT, "data", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    write_manifest_md(manifest)


def write_manifest_md(m):
    L = ["# Data manifest", "",
         "Generated by `python3 -I data/build.py` (do not edit by hand; full numbers in "
         "`data/manifest.json`).", "",
         "- Source: %s" % m["source"], "- Source sha256: `%s`" % m["source_sha256"],
         "- Seed: %d; dev = session dates %s, test = session dates %s" % (m["seed"], m["dev_dates"], m["test_dates"]),
         "- Sentence granularity drops sentences with < %d whitespace tokens; chunk granularities "
         "(`100`, `500`) are consecutive sentences of one class within one session date, emitted as soon as "
         "they reach the target length (leftovers dropped)." % m["sent_min_tokens"], "",
         "## Files (data/processed/, gitignored)", "",
         "| file | n | orig (0) | trans (1) | tokens orig | tokens trans | sha256 |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for name, d in sorted(m["files"].items()):
        L.append("| %s | %d | %d | %d | %d | %d | `%s` |" % (name, d["n"], d["n_orig"], d["n_trans"],
                                                           d["tokens_orig"], d["tokens_trans"], d["sha256"]))
    L += ["", "## Labelling statistics (speaker turns in txt/sv)", "", "| counter | value |", "|---|---:|"]
    for k, v in sorted(m["labelling_stats"].items()):
        L.append("| %s | %d |" % (k.replace("|", "/"), v))
    L += ["", "## Per year (kept turns)", "", "| year | orig turns | orig tokens | trans turns | trans tokens |",
          "|---|---:|---:|---:|---:|"]
    for y, d in sorted(m["per_year"].items()):
        L.append("| %s | %d | %d | %d | %d |" % (y, d["orig_turns"], d["orig_tokens"], d["trans_turns"], d["trans_tokens"]))
    b = m["boilerplate_turn_initial_address"]
    L += ["", "## Boilerplate", "",
          "Turns starting with an address (`Herr/Fru talman/ordförande/kommissionär/minister`): "
          "orig %d/%d (%.1f%%), trans %d/%d (%.1f%%). Kept in the text." % (
              b["orig"]["turns_starting_with_address"], b["orig"]["turns"],
              100.0 * b["orig"]["turns_starting_with_address"] / max(1, b["orig"]["turns"]),
              b["trans"]["turns_starting_with_address"], b["trans"]["turns"],
              100.0 * b["trans"]["turns_starting_with_address"] / max(1, b["trans"]["turns"])),
          "", "## Speaker overlap between dev and test (distinct speaker names)", ""]
    for lab, d in m["speaker_overlap"].items():
        L.append("- %s: dev %d, test %d, in both %d" % (lab, d["dev"], d["test"], d["both"]))
    with open(os.path.join(ROOT, "data", "MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

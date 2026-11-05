#!/usr/bin/env python3
"""Export the feature catalogue of docs/literature.md (section 2) as a table.

    python3 evals/export_features.py [--out data/cache/dashboard/features.json]

One row per catalogue feature: id, name, family, direction, mechanism,
detection, example, source, priority, strength, status, ... Deterministic
(no timestamps; rows in document order), stdlib only, well under a second.

Parsing is best-effort over the doc's markdown (bold "**A1. Name**" entries in
sections A-D, pipe tables in E-G). Anything the parser gets wrong or cannot
find is fixed by hand in OVERRIDES below. Pure cross-reference entries
("A24 ... included in A3", "G1 = A13", ...) are folded into their target via
ALIASES, so they are not separate rows.

status = "implemented" if some rule file tsw/rules/*.py (base.py, __init__.py
and _*.py excluded) on ANY local branch, or in the main working tree
(including untracked files), mentions the feature id (or an alias id) as a
standalone token, or matches one of the KEYWORDS regexes of that feature
(case-insensitive). Otherwise "planned". Edit KEYWORDS to tune the mapping.
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "literature.md")

# --- hand-editable mappings -------------------------------------------------

# Dashboard family of each catalogue section; per-feature exceptions in FAMILY.
SECTION_FAMILY = {
    "A": "register/lexical",
    "B": "word order",
    "C": "calques/interference",
    "D": "connectives & explicitation",
    "E": "simplification stats",
    "F": "punctuation & orthography",
    "G": "Europarl-specific",
}
FAMILY = {
    # A splits three ways
    "A1": "particles & pronouns", "A5": "particles & pronouns", "A7": "particles & pronouns",
    "A9": "particles & pronouns", "A10": "particles & pronouns", "A11": "particles & pronouns",
    "A20": "particles & pronouns",
    "A6": "calques/interference", "A12": "calques/interference", "A13": "calques/interference",
    "A14": "calques/interference", "A16": "calques/interference", "A18": "calques/interference",
    "A19": "calques/interference",
    "A15": "Europarl-specific",
    # punctuation counts that the doc lists under E
    "E2": "punctuation & orthography", "E8": "punctuation & orthography",
}
FAMILY_ORDER = ["word order", "register/lexical", "particles & pronouns", "calques/interference",
                "connectives & explicitation", "punctuation & orthography",
                "simplification stats", "Europarl-specific"]

# Cross-reference entries folded into their target (alias -> target).
ALIASES = {"A24": "A3", "C7": "A17", "G1": "A13", "G2": "A14", "G9": "A15",
           "G10": "A7", "G11": "B8"}

# Top of the doc's priority list (section H); the first 16 are "high".
N_HIGH = 16

# Regexes (case-insensitive) that, found in a rule file, mark a feature as
# implemented, in addition to the feature id itself. Keep them specific: rule
# names ("starter.dem_choice"), descriptions or the distinctive pattern. Bare
# words also used in rationale prose or shared word lists (naturligtvis,
# huruvida, "i termer av" ...) give false positives; prefer pattern text.
KEYWORDS = {
    "A1": [r"dem_choice", r"demonstrative", r"denna\|detta\|dessa"],
    "A2": [r"skall_ska", r"ska_vs_skall", r"skall\s*(/|vs)\s*ska", r"\\bskall\\b"],
    "A3": [r"formal_words?", r"archaic", r"emellertid.{0,40}huruvida"],
    "A4": [r"colloquial", r"väldigt.{0,40}\blite\b"],
    "A5": [r"modal.?particles?"],
    "A6": [r"av_rate", r"\bav.?rate\b", r"of.?genitive"],
    "A7": [r"naturligtvis\s*(vs|/|_)"],
    "A8": [r"tycker_vs", r"tyck\w*\s*(vs|/)", r"tyck.{0,10}anse"],
    "A9": [r"generic.?man\b", r"man_rate"],
    "A10": [r"generic.?en\b", r"\\bens\\s"],
    "A11": [r"pronoun.?profile"],
    "A12": [r"faktum"],
    "A13": [r"skulle.?vilja"],
    "A14": [r"låt.?(mig|oss)", r"lat_mig"],
    "A15": [r"damer.?och.?herrar"],
    "A16": [r"calques?\b", r"calque_"],
    "A17": [r"kommer_att", r"att.?drop", r"kommer att\s*(vs|/)"],
    "A18": [r"pied.?pip", r"vilken\|vilket\|vilka"],
    "A19": [r"\\bvars\\b", r"vars_rate"],
    "A20": [r"reflexive.?poss", r"sin\|sitt\|sina"],
    "A21": [r"intensifier"],
    "A22": [r"hedges?\b", r"stance.?adv"],
    "A23": [r"short.?forms?", r"long.?forms?"],
    "A25": [r"formal.?register", r"samt\|såväl"],
    "B1": [r"subj(ect)?.?initial", r"subj_start", r"inver(sion|ted)", r"topicali[sz]"],
    "B2": [r"expletive"],
    "B3": [r"fronted.?obj", r"object.?det\b"],
    "B4": [r"resumptive.?så"],
    "B5": [r"comma.?after.?(adv|init)", r"init_adv.?comma", r"adverbial.?comma"],
    "B6": [r"\bv3\b", r"pre.?verbal"],
    "B7": [r"\bbiff\b", r"subordinate.?(clause)?.?order"],
    "B8": [r"\bclefts?\b"],
    "B9": [r"initial.?(men|och)\b", r"\bmen_och\b"],
    "B10": [r"positional", r"first.?token", r"last.?token"],
    "C1": [r"passive"],
    "C2": [r"participles?"],
    "C3": [r"indef(inite)?.?article", r"role.?nouns?"],
    "C4": [r"adj.?sak", r"sak_grej"],
    "C5": [r"s.?genitive"],
    "C6": [r"compound.?density", r"long.?words?"],
    "C8": [r"progressive"],
    "C9": [r"att.?retention", r"complementi[sz]er"],
    "C10": [r"prepositional.?calque", r"i\\s\+termer"],  # the regex itself, not prose
    "D1": [r"cohesive", r"t.?connectives?\b", r"connectives?_t\b"],
    "D2": [r"o.?connectives?\b", r"connectives?_o\b", r"discourse.?particles?"],
    "D3": [r"parenthes"],
    "D4": [r"explicit.?naming"],
    "E1": [r"sent(ence)?.?len", r"mean_sent"],
    "E2": [r"comma.?(density|rate|per)"],
    "E3": [r"\bmattr\b", r"\bttr\b"],
    "E4": [r"hapax"],
    "E5": [r"word.?len"],
    "E6": [r"word.?rank", r"top.?1000", r"freq(uency)?.?list"],
    "E7": [r"lexical.?density"],
    "E8": [r"semicolons?"],
    "E9": [r"exclamation", r"rhetorical"],
    "E10": [r"\bpmi\b", r"fixed.?expressions?"],
    "F1": [r"apostrophe"],
    "F2": [r"oxford", r"serial.?comma"],
    "F3": [r"comma.?after.?sub"],
    "F4": [r"capitali[sz]ed.?(nationalit|month|weekday)"],
    "F5": [r"decimal.?point"],
    "F6": [r"thousands"],
    "F7": [r"percent", r"\\d%"],
    "F8": [r"english.?quotes?", r"straight.?quotes?"],
    "F9": [r"särskriv", r"split.?compound"],
    "F10": [r"title.?case"],
    "G3": [r"vägnar"],
    "G4": [r"jag.?välkomnar", r"welcome_"],
    "G5": [r"vill tacka"],
    "G6": [r"uk.?topic", r"förenade kungariket"],
    "G7": [r"sv.?topic", r"swedish.?topic"],
    "G8": [r"procedural"],
}

# Hand fixes, applied last (field -> value). Keep short.
OVERRIDES = {
    "A1": {"example": "i denna fråga / i den här frågan"},
    "A2": {"example": "Vi skall / vi ska"},
    "A3": {"detection": r"\b(emellertid|huruvida|således)\b + extended formal list, per 1000",
           "example": "emellertid, huruvida, således"},
    "A4": {"detection": r"\b(väldigt|lite)\b, \bjätte\w+, ganska, nåt/nån, sånt …",
           "example": "väldigt viktigt, lite"},
    "A5": {"detection": r"\bju\b, \bnog\b, \bväl\b after finite verb, \bvisst\b, ändå, nämligen",
           "example": "det är ju så att"},
    "A6": {"detection": r"\bav\b per 1000 tokens", "example": "förslaget av kommissionen"},
    "A7": {"detection": r"\b(naturligtvis|givetvis)\b vs ju", "mechanism": "interference",
           "example": "naturligtvis också en gemensam grundval i denna vitbok"},
    "A8": {"detection": r"tyck- / (tyck- + anse- + tro-); \b(jag|vi)\s+(tror|anser)\s+att\b",
           "mechanism": "unique item", "example": "Jag tycker att det här …",
           "source": "[exp]; Danielsson (2003) O-cluster", "strength": "Medium-strong in speeches."},
    "A9": {"example": "Man måste komma ihåg att …", "source": "[exp]; analogy: Wintner (tutorial)"},
    "A10": {"example": "ens egen", "mechanism": "interference", "strength": "Rare."},
    "A11": {"direction": "?", "mechanism": "interference",
            "detection": "rates of jag / vi / du / ni (data-calibrated vector)",
            "source": "Volansky et al. (2015) (English analogue)",
            "strength": "Direction uncertain; calibrate on data."},
    "A12": {"example": "det faktum att"},
    "A13": {"example": "Jag skulle vilja tacka", "mechanism": "interference"},
    "A14": {"example": "Låt mig säga", "mechanism": "interference"},
    "A15": {"direction": "?", "mechanism": "interference", "detection": r"\bmina\s+damer\s+och\s+herrar\b",
            "example": "mina damer och herrar", "strength": "Unclear for EN→SV; check on data."},
    "A16": {"detection": "hand list of calques (§4.6), per-entry weights",
            "example": "ta plats, göra skillnad", "source": "[exp]"},
    "A17": {"example": "det kommer att bli / det kommer bli", "mechanism": "interference + normalization"},
    "A18": {"example": "i vilket", "mechanism": "interference"},
    "A19": {"example": "vars", "mechanism": "interference",
            "strength": "Medium-weak (native written, but EN whose/of which inflates it)."},
    "A20": {"example": "deras regeringar / sina regeringar", "mechanism": "interference + unique item"},
    "A21": {"example": "mycket viktigt / väldigt viktigt", "mechanism": "normalization + spoken register",
            "source": "Danielsson (2003) (väldigt)", "strength": "Calibrate."},
    "A22": {"direction": "?", "mechanism": "interference",
            "detection": "i själva verket, faktiskt, verkligen, helt enkelt, egentligen, inte minst",
            "example": "i själva verket", "source": "[exp]; Danielsson (2003) for inte minst",
            "strength": "Calibrate direction on data."},
    "A23": {"detection": "ratio short/long per pair: sa/sade, sen/sedan, nåt/något, dom/de …",
            "example": "sa / sade", "strength": "Rare in edited EP, strong in general text."},
    "A25": {"detection": r"\b(samt|såväl|respektive|i enlighet med|i syfte att)\b",
            "example": "i enlighet med", "strength": "Native bureaucratese too; calibrate."},
    "B1": {"detection": "INVERTED / (SUBJ_START + INVERTED) at sentence start",
           "mechanism": "interference", "example": "I dag har vi … / Vi har i dag …"},
    "B2": {"direction": "?", "mechanism": "unique item", "example": "Det finns",
           "strength": "Net direction uncertain."},
    "B3": {"example": "Det vet vi."},
    "B4": {"example": "Om vi inte agerar nu, så kommer …", "strength": "Rare in written T."},
    "B5": {"example": "Dessutom, vi måste …", "mechanism": "interference"},
    "B6": {"example": "Vi också vill", "mechanism": "interference", "source": "[exp]; Kunz et al. (2026)"},
    "B7": {"direction": "?", "mechanism": "interference",
           "detection": "SUBJUNCTION SUBJ SADV FIN (native) vs SUBJ FIN SADV",
           "example": "som kan inte", "strength": "Two-sided and rare; low weight."},
    "B8": {"example": "Det är därför …", "mechanism": "unique item",
           "strength": "Direction unverified; 'Det är därför' sub-pattern is T."},
    "B9": {"example": "Men …", "source": "Volansky et al. (2015) (English analogue)",
           "strength": "English analogue: O starts with But 2.25× more often."},
    "B10": {"detection": "hand lists of first-/last-token words with direction", "mechanism": "interference",
            "example": "sentence-final ju, alltså, också", "source": "Volansky et al. (2015)",
            "strength": "Best single interference family in Volansky et al. (97%)."},
    "C1": {"direction": "T", "detection": "periphrastic (bli/vara + participle) / (periphrastic + s-passive)",
           "example": "blev antagen / antogs", "source": "Fredriksson (2016); Volansky et al. (2015)"},
    "C2": {"example": "tittandes in i kameran", "mechanism": "interference", "strength": "Medium."},
    "C3": {"example": "Jag är en socialdemokrat", "mechanism": "interference"},
    "C4": {"example": "en fånig sak hände", "mechanism": "interference",
           "source": "Gellerstam (1986), via Kunz et al. (2026)", "strength": "Low frequency in EP."},
    "C5": {"example": "kommissionens förslag", "mechanism": "interference", "source": "Nilsson (2004)",
           "strength": "Weak proxy; combine with A6."},
    "C6": {"detection": "share of words ≥ 15 chars", "mechanism": "interference",
           "example": "gemenskapslagstiftningen", "source": "[exp]",
           "strength": "Use only after dev-set calibration (mean word length: 66%)."},
    "C8": {"detection": r"\bhåller\s+på\s+att\b", "mechanism": "interference",
           "example": "håller på att", "strength": "Weak."},
    "C9": {"detection": r"P(att | tror/tycker/vet/hoppas/menar/säger)", "example": "Jag tror det är viktigt",
           "mechanism": "universal (explicitation) + normalization", "strength": "Medium."},
    "C10": {"direction": "T", "mechanism": "interference", "detection": "fixed phrases (in the calque list)",
            "example": "i termer av, på basis av", "strength": "Keep to fixed phrases."},
    "D1": {"detection": "rate per 1000 of T-connectives list (§4.3a)", "example": "Dessutom …"},
    "D2": {"detection": "rate of O-connectives list (§4.3b); ratio vs D1", "example": "ju, ändå, nämligen",
           "mechanism": "unique item + spoken register",
           "strength": "Pair with D1 as a ratio for a length-robust signal."},
    "D3": {"strength": "English analogue (Volansky): parentheses typical of T.",
           "detection": r"( … ) after stripping EP stage directions"},
    "D4": {"direction": "?", "detection": "—", "source": "Volansky et al. (2015)",
           "strength": "Weak (explicit naming 58% in Volansky). Skip."},
    "E1": {"detection": "mean tokens per sentence", "mechanism": "interference"},
    "E2": {"detection": r"commas per token and per sentence (excl. \d,\d)", "mechanism": "interference"},
    "E3": {"detection": "MATTR, window 100"},
    "E4": {"detection": "hapax ratio in fixed windows"},
    "E5": {"detection": "mean chars per word"},
    "E6": {"detection": "share of tokens in a top-100/1000 Swedish word list"},
    "E7": {"detection": "content words / tokens"},
    "E8": {"detection": "; and : per sentence", "mechanism": "interference"},
    "E9": {"detection": "! and ? per sentence (strip Herr talman! first)", "mechanism": "spoken register"},
    "E10": {"detection": "hits from a hand list of native fixed expressions (§4.7)",
            "mechanism": "normalization"},
    "F1": {"example": "EU's"},
    "F2": {"example": "miljö, ekonomi, och sysselsättning"},
    "F4": {"detection": "capitalised Svensk…/Januari…/Måndag… after a lower-case word (case-sensitive)",
           "example": "i Januari"},
    "F5": {"example": "3.5 procent"},
    "F6": {"example": "1,000"},
    "F7": {"example": "5%"},
    "F8": {"detection": "\"…\" or “…” vs ”…”", "example": "“Bättre lagstiftning”"},
    "F9": {"detection": r"hand list of split EP compounds: \bmedlems\s+stat, \bmiljö\s+(fråg|politik) …",
           "example": "medlems stater"},
    "G3": {"example": "på vägnar av min grupp", "detection": r"\bpå\s+vägnar\s+av\b",
           "source": "[exp]"},
    "G4": {"example": "Jag välkomnar betänkandet", "detection": r"\b(jag|vi)\s+välkomnar\b"},
    "G5": {"example": "Jag vill tacka föredraganden", "detection": r"\b(jag|vi)\s+vill\s+tacka\b"},
    "G6": {"detection": "topic word list (UK/Ireland), off by default", "mechanism": "topic artefact",
           "example": "Förenade kungariket"},
    "G7": {"detection": "topic word list (Sweden, parties), off by default", "mechanism": "topic artefact",
           "example": "riksdagen, moderaterna"},
    "G8": {"detection": "procedural-sentence list; strip before scoring", "mechanism": "preprocessing",
           "example": "Debatten är avslutad."},
}

# ---------------------------------------------------------------------------

MECH = [("INT", "interference"), ("UNI", "universal"), ("UNQ", "unique item"),
        ("NORM", "normalization"), ("SPK", "spoken register")]
SECTION_MECH = {"D": "universal (explicitation)", "E": "universal (simplification)",
                "F": "interference", "G": "interference"}
# Author name found in an entry -> citation used when the entry has no "Source:" line.
AUTHORS = {"Gellerstam": "Gellerstam (1986)", "Danielsson": "Danielsson (2003)",
           "Aijmer": "Aijmer (1996)", "Nilsson": "Nilsson (2004)",
           "Volansky": "Volansky et al. (2015)", "Rabinovich": "Rabinovich & Wintner (2015)",
           "Kunz": "Kunz et al. (2026)", "Hasselgård": "Hasselgård (2004)",
           "Johansson": "Johansson (2004)", "Bohnacker": "Bohnacker & Rosén (2008)",
           "Fredriksson": "Fredriksson (2016)", "Blum-Kulka": "Blum-Kulka (1986)",
           "Kenny": "Kenny (2001)", "Wintner": "Wintner (tutorial)"}
EN_WORDS = set("the of to i would like you we one is that which in but be by this these people "
               "simply really indeed fact thus moreover besides however therefore of-phrases "
               "will think believe feel ladies gentlemen whose there it on behalf welcome "
               "a an let me us".split())
ENTRY_RX = re.compile(r"^\*\*([A-G]\d+)\.\s+(.*?)\*\*(?!\*)(.*)$")
ROW_RX = re.compile(r"^\|\s*([A-G]\d+)\s*\|")
CODE_RX = re.compile(r"`([^`]+)`")
ITAL_RX = re.compile(r"(?<![*\w])\*(?!\*)([^*\n]+?)\*(?![*\w])")


def plain(s):
    """Strip markdown emphasis/code markers and collapse whitespace."""
    s = re.sub(r"\*\*|\*|`", "", s)
    return re.sub(r"\s+", " ", s).strip()


def short(s, n=110):
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def direction_of(text):
    m = re.search(r"T↑|T↓|O↑", text)
    if not m:
        return "?"
    return "T" if m.group(0) == "T↑" else "O"


def mechanism_of(text, sec):
    found = [name for tag, name in MECH if re.search(r"\b%s\b" % tag, text)]
    return " + ".join(found) if found else SECTION_MECH.get(sec, "")


def example_of(texts):
    for t in texts:
        for m in ITAL_RX.finditer(t):
            span = m.group(1).strip().strip(".,;: ")
            words = re.findall(r"[\wåäöÅÄÖ'-]+", span.lower())
            if len(words) >= 2 and not (set(words) & EN_WORDS) and "→" not in span:
                return short(span, 60)
    return ""


def detection_of(text):
    for m in CODE_RX.finditer(text):
        c = m.group(1)
        if c in ("[lit]", "[exp]") or c in dict(MECH) or len(c) < 4:
            continue
        return short(c, 110)
    return ""


def bullets(text):
    """Markdown bullets with their continuation lines joined."""
    out = []
    for line in text.split("\n"):
        if line.startswith("- ") or not out:
            out.append(line[2:] if line.startswith("- ") else line)
        else:
            out[-1] += " " + line.strip()
    return out


def strength_of(text):
    for line in bullets(text):
        line = plain(line)
        for sent in re.split(r"(?<=\.)\s+", line):
            if re.match(r"(Strong|Medium|Weak|High precision|Potentially|Rare|Precise|Low|Very)\b", sent):
                return short(sent, 120)
    return ""


def source_of(text, tags):
    m = re.search(r"Sources?:\s*(.+?)(?:\n-|\n\n|$)", text, re.S)
    if m:
        return short(plain(m.group(1)).rstrip("."), 140)
    names = []
    for a, cite in AUTHORS.items():
        if a in text and cite not in names and not (a == "Wintner" and "Rabinovich" in text):
            names.append(cite)
    if names:
        return "; ".join(names)
    return "[exp]" if "[exp]" in tags or not tags else "[lit]"


def split_row(line):
    """Split a markdown table row on '|' that are not inside `code`."""
    cells, cur, code = [], "", False
    for ch in line.strip().strip("|"):
        if ch == "`":
            code = not code
        if ch == "|" and not code:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
    return cells + [cur]


def parse(doc_text):
    sec2 = doc_text.split("\n## 2.", 1)[1].split("\n## 3.", 1)[0]
    rows, order, section, sec_title, cur = {}, [], None, "", None
    prio_ids = []
    for line in sec2.split("\n"):
        h = re.match(r"^###\s+([A-H])\.\s+(.*)$", line)
        if h:
            section, sec_title, cur = h.group(1), h.group(2).strip(), None
            continue
        if section == "H":
            m = re.match(r"^\d+\.\s+\*\*([A-G]\d+)", line)
            if m:
                prio_ids.append(m.group(1))
            continue
        m = ENTRY_RX.match(line)
        if m:
            fid, name, tail = m.groups()
            cur = {"id": fid, "section": section, "section_title": sec_title,
                   "name": plain(name), "head": tail, "body": ""}
            rows[fid] = cur
            order.append(fid)
            continue
        m = ROW_RX.match(line)
        if m:
            cells = [c.strip() for c in split_row(line)]
            rows[m.group(1)] = {"id": m.group(1), "section": section, "section_title": sec_title,
                                "name": plain(cells[1]), "cells": cells}
            order.append(m.group(1))
            cur = None
            continue
        if cur is not None:
            cur["body"] += line + "\n"
    return rows, order, prio_ids


def doc_priority_of(ptxt, text):
    """P1/P2/P3 from the priority cell, else anywhere in the row; else artefact/skip/..."""
    m = re.search(r"P[123]", ptxt) or re.search(r"P[123]", text)
    if m:
        return m.group(0)
    m = re.search(r"artefact|preprocessing|Skip", text)
    return m.group(0).lower() if m else ""


def build(raw, prio_ids):
    sec = raw["section"]
    if "cells" in raw:  # table row (E, F, G)
        c = raw["cells"]
        text = " | ".join(c)
        if sec == "E":   # ID | Feature | Expected | Strength | Notes
            direction, strength, detection, ptxt = direction_of(c[2]), plain(c[3]), "", c[4]
        elif sec == "F":  # ID | Feature | Detection | Dir | Priority
            note = plain(re.sub(r"P[123]", "", c[4])).strip(" ()")
            strength = (note[:1].upper() + note[1:] + ". " if note else "") + \
                "Strong outside EP; EP editors normalise much of it."
            direction, detection, ptxt = direction_of(c[3]), c[2], c[4]
        else:            # G: ID | Feature | Dir | Notes | Priority
            direction, strength, detection, ptxt = direction_of(c[2]), plain(c[3]), "", c[4]
        dm = CODE_RX.search(detection)
        row = {
            "direction": direction,
            "mechanism": mechanism_of(text, sec),
            "detection": short(dm.group(1) if dm else plain(detection), 110),
            "example": example_of([c[1]]),
            "source": (source_of(text, []) if any(a in text for a in AUTHORS) else
                       AUTHORS["Volansky"] if sec == "E" and "%" in c[3] else "[exp]"),
            "evidence": "[lit]" if any(a in text for a in AUTHORS) or
                        (sec == "E" and "%" in c[3]) else "[exp]",
            "doc_priority": doc_priority_of(ptxt, text),
            "strength": short(strength, 120),
        }
    else:
        head, body = raw["head"], raw["body"]
        tags = re.findall(r"\[(?:lit|exp)\]", head)
        pm = re.search(r"P\d(?:/P\d)?", head)
        hm = re.search(r"`\[(?:lit|exp)\]`(?:\s*\([^)]*\))?\s*\(([^)]*)\)\s*$", head)
        hsrc = hm.group(1) if hm and any(a in hm.group(1) for a in AUTHORS) else ""
        src = source_of(body, tags)
        if src in ("[exp]", "[lit]") and hsrc:
            src = hsrc
        row = {
            "direction": direction_of(body if re.search(r"T↑|T↓|O↑", body) else head),
            "mechanism": mechanism_of(body, sec),
            "detection": detection_of(body),
            "example": example_of([head, body]),
            "source": src,
            "evidence": "+".join(sorted(set(tags))) or "",
            "doc_priority": pm.group(0) if pm else "",
            "strength": strength_of(body),
        }
    fid = raw["id"]
    row = dict({"id": fid, "name": raw["name"],
                "family": FAMILY.get(fid, SECTION_FAMILY[sec]),
                "section": "%s. %s" % (sec, raw["section_title"])}, **row)
    row["priority"] = "high" if fid in prio_ids[:N_HIGH] else "normal"
    row["aliases"] = sorted(a for a, t in ALIASES.items() if t == fid)
    row.update(OVERRIDES.get(fid, {}))
    return row


def git(*args):
    try:
        out = subprocess.run(["git", "-C", ROOT] + list(args), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.decode("utf-8", "replace")


def rule_files():
    """{label: text} of every rule family file on every local branch + the working tree."""
    files = {}

    def collect(out, label_fn):
        for line in out.splitlines():
            parts = line.split("\0")
            if len(parts) < 3:
                continue
            path, text = parts[-3], parts[-1]
            base = os.path.basename(path)
            if base in ("base.py", "__init__.py") or base.startswith("_"):
                continue
            key = label_fn(parts)
            files[key] = files.get(key, "") + text + "\n"

    pat = ["-I", "-z", "-n", "-e", "", "--", "tsw/rules/*.py"]
    for br in sorted(git("for-each-ref", "--format=%(refname:short)", "refs/heads/").split()):
        # with -z each line is "<branch>:<path>\0<lineno>\0<text>"
        out = git("grep", *pat[:-2], br, *pat[-2:])
        collect(out, lambda p: p[0])
    out = git("grep", "--untracked", *pat)
    collect(out, lambda p: "worktree:%s" % p[0])
    return files


def status_of(row, files):
    ids = [row["id"]] + row["aliases"]
    id_rx = re.compile(r"(?<![A-Za-z0-9])(%s)(?![0-9])" % "|".join(ids))
    kw = [re.compile(k, re.I) for k in KEYWORDS.get(row["id"], [])]
    hits = sorted(label for label, text in files.items()
                  if id_rx.search(text) or any(k.search(text) for k in kw))
    return ("implemented" if hits else "planned"), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=DOC)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cache", "dashboard", "features.json"))
    args = ap.parse_args()
    with open(args.doc, encoding="utf-8") as f:
        raws, order, prio_ids = parse(f.read())
    missing = [fid for fid in list(OVERRIDES) + list(KEYWORDS) + list(FAMILY)
               if fid not in raws or fid in ALIASES]
    if missing:
        sys.exit("unknown or aliased ids in mappings: %s" % ", ".join(sorted(set(missing))))
    files = rule_files()
    rows = []
    for fid in order:
        if fid in ALIASES:
            continue
        row = build(raws[fid], prio_ids)
        row["status"], row["implemented_by"] = status_of(row, files)
        rows.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
    by = {}
    for r in rows:
        by.setdefault(r["family"], [0, 0])[r["status"] == "implemented"] += 1
    print("%d features (%d catalogue ids, %d aliases folded), %d rule files scanned -> %s"
          % (len(rows), len(order), len(ALIASES), len(files), args.out))
    for fam in FAMILY_ORDER:
        if fam in by:
            print("  %-28s %2d planned, %2d implemented" % (fam, by[fam][0], by[fam][1]))


if __name__ == "__main__":
    main()

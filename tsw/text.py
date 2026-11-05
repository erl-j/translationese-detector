"""Text normalisation, Swedish sentence splitting and tokenisation.

Everything downstream (rules, spans) works on the *normalised* text. The
normalisation is not length-preserving (whitespace runs are collapsed, soft
hyphens dropped, "…" expanded to "..."), so `normalise_with_map` also returns
an offset map that sends every normalised character back to the original
input; `Doc.to_orig` uses it to turn rule spans into offsets into the text the
caller passed in.

Two code paths produce the normalised string:
    normalise(text)            fast (str.translate + one regex); used for scoring
    normalise_with_map(text)   char-by-char, also returns the offset map; only
                               used when spans are requested
They must agree exactly; tsw/tests/test_text.py checks that.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

_DQUOTES = "“”„‟«»″〝〞"     # “ ” „ ‟ « » ″
_SQUOTES = "‘’‚‛´`′"                  # ‘ ’ ‚ ‛ ´ ` ′
_DASHES = "—―‒⸺⸻"                               # — ― ‒ ⸺ ⸻ -> –
_HYPHENS = "‐‑−⁃"                                    # ‐ ‑ − ⁃     -> -
_SPACES = "\t\n\r\x0b\x0c         " \
          "        　\x85"
_DELETE = "­​‌‍⁠﻿"                        # soft hyphen, zero-width

_TABLE = {}
for _c in _DQUOTES:
    _TABLE[ord(_c)] = '"'
for _c in _SQUOTES:
    _TABLE[ord(_c)] = "'"
for _c in _DASHES:
    _TABLE[ord(_c)] = "–"
for _c in _HYPHENS:
    _TABLE[ord(_c)] = "-"
for _c in _SPACES:
    _TABLE[ord(_c)] = " "
for _c in _DELETE:
    _TABLE[ord(_c)] = None
_TABLE[ord("…")] = "..."   # …

_SPACE_RUN = re.compile(" {2,}")


def normalise(text):
    """NFC, unify quotes/dashes/spaces, drop soft hyphens, collapse whitespace."""
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
    text = text.translate(_TABLE)
    if "  " in text:
        text = _SPACE_RUN.sub(" ", text)
    return text.strip(" ")


def _clusters(text):
    """Yield (start, end) of base char + following combining marks."""
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        yield i, j
        i = j


def normalise_with_map(text):
    """Return (normalised, starts, ends): normalised[i] comes from text[starts[i]:ends[i]]."""
    if unicodedata.is_normalized("NFC", text):
        pieces = ((i, i + 1, c) for i, c in enumerate(text))
    else:
        pieces = ((s, e, unicodedata.normalize("NFC", text[s:e])) for s, e in _clusters(text))
    out, starts, ends = [], [], []
    for s, e, chunk in pieces:
        for c in chunk:
            r = _TABLE.get(ord(c), c)
            if r is None:
                continue
            for rc in r:
                if rc == " " and (not out or out[-1] == " "):
                    continue
                out.append(rc)
                starts.append(s)
                ends.append(e)
    while out and out[-1] == " ":
        out.pop()
        starts.pop()
        ends.pop()
    return "".join(out), starts, ends


def lower_same_length(text):
    """str.lower(), but never changes the length (so offsets stay valid)."""
    low = text.lower()
    if len(low) == len(text):
        return low
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


# ---------------------------------------------------------------------------
# sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations after which a period does NOT end a sentence (lower-cased,
# with their final period). Deliberately excludes real words that can end a
# sentence ("kom.", "ung.", "punkt.", stranded "i.").
NO_SPLIT_ABBR = frozenset("""
t.ex. bl.a. dvs. d.v.s. s.k. fr.o.m. t.o.m. resp. ca. nr. kap. art. prop. jfr. p.g.a. pga.
f.d. f.n. e.d. o.d. o.a. dr. mr. mrs. ms. st. sek. kl. tel. inkl. exkl. bl. sid. vol. bet.
dir. dnr. mom. stk. ff. ev. fig. tab. ex. mfl. jur. fil. prof. kungl. dåv. dvs t.ex bl.a s.k
""".split())

# Abbreviations that often END a sentence: split if a capital follows.
END_ABBR = frozenset("osv. o.s.v. etc. m.m. m.fl. o.dyl. m.a.o. e.t.c.".split())

_CANDIDATE = re.compile(r"(?:\.{3}|[.!?]+)['\")\]]*(?= )")
_OPENERS = " '\"([\u2013-"
_HAS_ALNUM = re.compile(r"[^\W_]")


def _is_abbrev_stop(text, pos):
    """pos = index of a '.' that might end a sentence. True if it is an abbreviation."""
    ws = text.rfind(" ", 0, pos)
    word = text[ws + 1:pos + 1].lstrip("(\"'[")
    low = word.lower()
    if low in NO_SPLIT_ABBR:
        return True
    if low in END_ABBR:
        return False
    core = word[:-1]
    # initials ("A. Smith") and dotted initialisms ("E.U.", "a.a.")
    if len(core) == 1 and core.isupper():
        return True
    if "." in core and all(0 < len(p) <= 2 for p in core.split(".")) and core.replace(".", "").isalpha():
        return True
    return False


def split_sentences(text):
    """Return a list of (start, end) spans of sentences in a normalised text.

    A sentence ends at . ! ? (or ...), optionally followed by closing quotes or
    brackets, when the next non-space, non-opening-punctuation character is an
    upper-case letter or a digit -- unless the '.' closes a known abbreviation
    (NO_SPLIT_ABBR) or an initial. "Herr talman! Jag ..." is two sentences.
    """
    n = len(text)
    cuts = []
    for m in _CANDIDATE.finditer(text):
        j = m.end()
        while j < n and text[j] in _OPENERS:
            j += 1
        if j >= n or not (text[j].isupper() or text[j].isdigit()):
            continue
        punct = m.group(0)
        if punct[0] == "." and punct[:2] != ".." and _is_abbrev_stop(text, m.start()):
            continue
        cuts.append(m.end())
    spans = []
    start = 0
    for c in cuts + [n]:
        s, e = start, c
        while s < e and text[s] == " ":
            s += 1
        while e > s and text[e - 1] == " ":
            e -= 1
        if e > s:
            if _HAS_ALNUM.search(text, s, e):
                spans.append((s, e))
            elif spans:
                # punctuation-only fragment: glue to the previous sentence
                spans[-1] = (spans[-1][0], e)
        start = c
    return spans


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------

_ABBR_TOKENS = sorted({a for a in NO_SPLIT_ABBR | END_ABBR if a.endswith(".") and len(a) > 2},
                      key=len, reverse=True)
TOKEN_RX = re.compile(
    r"(?<![^\W_])(?i:%s)(?![^\W_])|" % "|".join(re.escape(a) for a in _ABBR_TOKENS) +  # t.ex. dvs.
    r"(?<![^\W_])\d{1,3}(?: \d{3})+(?:,\d+)?(?![^\W_])"     # 1 000 000,5
    r"|\d+(?:[.,:]\d+)+(?![^\W_])"                           # 2,3  1.5  12:30
    r"|[^\W_]+(?:[-:'][^\W_]+)*(?:-(?= ))?"                  # words, EU-länder, EU:s, miljö-
    r"|\.{2,}|[^\w\s]"                                       # punctuation, one char ("..." kept)
)
WORD_CHAR = re.compile(r"[^\W_]")


def is_word(tok):
    """True for word/number/abbreviation tokens, False for punctuation."""
    return tok[0].isalnum()


def tokenize(text, start=0, end=None):
    """Return (tokens, starts) for text[start:end]; offsets are absolute."""
    if end is None:
        end = len(text)
    toks, starts = [], []
    for m in TOKEN_RX.finditer(text, start, end):
        toks.append(m.group())
        starts.append(m.start())
    return toks, starts


# vocative-only sentences ("Herr talman!", "Fru kommissionär, kära kolleger!")
ADDRESS_RX = re.compile(
    r"(?:herr|fru|ärade|bästa|kära)\s+(?:talman|ordförande|kommissionär|minister|rådsordförande|"
    r"kommissionsledamot|rådsminister|tjänstgörande|vice|premiärminister|kolleg|ledamöter)"
    r"[^.!?]{0,80}!$")

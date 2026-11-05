"""wordorder: V2 word order and information structure (docs/literature.md section B).

English is subject-first (SVO); Swedish is verb-second (V2) and freely fills the
prefield with adverbials, PPs, objects and fronted clauses (*I dag har vi ...*,
*Därför anser jag ...*, *Det vet vi*), with the subject inverted after the
finite verb. Translators tend to keep the English Theme (Hasselgård 2004), so a
translated text should open more of its main clauses with the subject and fewer
with a topicalised non-subject. The family also covers the few places where
English order leaks into Swedish clause structure (no inversion after a fronted
adverbial, pre-verbal sentence adverbs in main clauses, main-clause adverb order
in subordinate clauses, unshifted object pronouns) and fronted object *det*.

There is no tagger or parser: the clause-opening classifier below is a
precision-oriented heuristic over closed word lists (finite verbs, subject
pronouns, prepositions, subjunctions, fronted adverbs). Sentences it cannot
classify confidently are left out ("other"); every rule is a *choice* among the
classified openings or a rare-item count, never a rate of "other".

Calibration: see the comments next to each number (dev.500.bal unless noted,
`python3 -m tsw.tools.rates --stat tsw.rules.wordorder:<units_fn>` and KWIC
inspection), 2026-11-05.
"""
import math
import re

from .base import ChoiceRule, HitRule, PredicateRule  # noqa: F401  (rule types used below)

FAMILY = "wordorder"
FAMILY_CAP = 4.0

# ---------------------------------------------------------------------------
# word lists (lower case). Sources: docs/literature.md 4.1, 4.2, 4.5, extended
# with frequent Europarl verbs / adverbs (marked [exp]).

_w = lambda s: frozenset(s.split())  # noqa: E731

# Personal subject pronouns that unambiguously head a subject NP. 'dom' is
# left out (in EP it is mostly the noun 'dom' = verdict); 'det/den' are
# handled separately (expletive / demonstrative / determiner).
SUBJ_PERS = _w("jag vi ni du han hon hen de man")
# Subject-like tokens that may follow the finite verb in an inverted clause.
SUBJ_POST = SUBJ_PERS | _w("det den detta denna dessa denne")
COPULA = _w("är var blir blev")

FIN = _w("""
är var har hade kan kunde ska skall skulle vill ville måste bör borde får fick
blir blev kommer kom finns fanns gäller gällde anser ansåg tror trodde menar menade
vet visste tycker tyckte säger sa sade gör gjorde ser såg står stod går gick ger gav
tar tog handlar handlade innebär innebar behöver behövde verkar verkade föreslår föreslog
stöder stödjer stödde välkomnar välkomnade hoppas hoppades känner kände ligger låg
betyder betydde kräver krävde krävs krävdes behövs behövdes görs gjordes ges gavs
sägs anses ansågs brukar brukade tänker tänkte hör hörde talar talade
lever levde arbetar arbetade leder ledde fortsätter fortsatte börjar började
uppmanar uppmanade beklagar beklagade understryker underströk betonar betonade
lär tycks syns slutar slutade utgör utgjorde omfattar omfattade
riskerar riskerade saknar saknade skapar skapade ämnar torde må
förstår förstod delar delade accepterar accepterade godtar godtog instämmer instämde
konstaterar konstaterade noterar noterade välkomnar ber bad tackar tackade
röstar röstade måste vågar vågade önskar önskade väntar väntade återstår återstod
hade finner fann kallas kallades heter hette uppstår uppstod råder rådde
""")
# Modal / tense auxiliaries (subset of FIN)
AUX = _w("är var har hade kan kunde ska skall skulle vill ville måste bör borde får fick "
         "blir blev kommer lär torde må")

# Single-token fronted adverbials (literature 4.5 + [exp] additions). Words
# that are also subjunctions or conjunctions (då, sedan, nu, så, när) are
# handled separately.
ADV1 = _w("""
därför dessutom vidare slutligen emellertid dock således alltså följaktligen därmed
först här där idag igår imorgon tyvärr lyckligtvis naturligtvis givetvis självklart
faktiskt kanske verkligen egentligen samtidigt tidigare nyligen ofta sällan aldrig
också även sammanfattningsvis avslutningsvis därigenom därtill därutöver likaså ändå
visserligen troligen förmodligen säkert tydligen uppenbarligen återigen därefter
hittills fortfarande alltid redan snart äntligen likväl sålunda annars däremot
istället dessvärre härmed härigenom därav heller endast bara enbart inte
""")
# Fronted adverbs after which a following subject + verb is NOT a V2
# violation (focus particles on the subject, spoken conjunction-like uses):
# "Även vi måste ...", "Kanske vi borde ...", "Alltså, vi ...", "Bara du kan".
ADV_NO_V2_CHECK = _w("också även bara endast enbart inte kanske alltså heller här där säkert")

# Text-structuring conjuncts with a fixed one-to-one English equivalent that
# English puts first and sets off with a comma (Firstly, / Finally, / In
# conclusion, / However, / On the other hand, / Fortunately, / Once again,).
# Defined by that criterion, not by dev counts: (a) ordinal/enumerative
# markers, (b) formal concessive/consequential conjuncts. A translator who
# keeps the English Theme produces inversion here too, so this fronting does
# not reflect a Swedish topicalisation choice; it gets its own category.
CONJ_ADV = frozenset([
    "för det första", "för det andra", "för det tredje", "för det fjärde", "för det femte",
    "för det sista", "först och främst", "inledningsvis", "slutligen", "till sist",
    "sist men inte minst", "avslutningsvis", "sammanfattningsvis",
    "emellertid", "icke desto mindre", "inte desto mindre", "likväl", "således", "följaktligen",
    "å ena sidan", "å andra sidan", "lyckligtvis", "än en gång", "återigen",
])

# Multi-token fronted adverbials that do not start with a preposition.
ADV_MULTI = [tuple(s.split()) for s in (
    "icke desto mindre", "inte heller", "inte minst", "först och främst", "än en gång",
    "en gång till", "allra först", "allra sist", "sist men inte minst", "inte desto mindre",
    "inte bara", "trots allt", "över huvud taget", "för det första", "för det andra",
    "för det tredje", "för det fjärde", "för det femte", "för det sista", "för det mesta",
    "för min del", "för vår del", "för egen del", "framför allt", "å ena sidan", "å andra sidan",
    "till sist", "till slut", "till exempel", "därför inte",
)]
ADV_MULTI.sort(key=len, reverse=True)

PREP = _w("""
i på med för av till från under efter genom enligt vid mot inom utan trots över bland hos
kring före inför angående beträffande gentemot utöver bortsett jämfört förutom
via vad
""")
# 'om' is treated as a subjunction (the prepositional reading is rarely fronted);
# 'vad' only in 'vad gäller / vad beträffar / vad angår'.

SUBJN1 = _w("om när eftersom innan medan tills fast fastän ifall såvida emedan ehuru då sedan "
            "såsom liksom")
SUBJN_MULTI = [tuple(s.split()) for s in (
    "även om", "trots att", "i och med att", "så länge", "så snart", "för att", "genom att",
    "utan att", "när det gäller", "vad gäller", "vad beträffar", "vad angår", "om än",
    "nu när", "nu då", "i den mån", "på grund av att", "efter att", "därför att",
    "i stället för att", "istället för att",
)]
SUBJN_MULTI.sort(key=len, reverse=True)
# subjunctions whose clause verb is part of the subjunction itself
SUBJN_WITH_VERB = {("när", "det", "gäller"), ("vad", "gäller"), ("vad", "beträffar"), ("vad", "angår")}
# subjunctions that may introduce an infinitive (no subject, no finite verb)
SUBJN_INF = {("för", "att"), ("genom", "att"), ("utan", "att"), ("efter", "att"),
             ("i", "stället", "för", "att"), ("istället", "för", "att")}

# Main-clause pre-verbal sentence adverbs (V3, B6): "Vi också vill", "Jag
# verkligen hoppas". Not quantifiers (vi alla / vi båda / ni själva: floating
# quantifiers are grammatical in Swedish).
V3_ADV = _w("också även bara endast alltid aldrig inte verkligen faktiskt redan ofta "
            "fortfarande helt fullt starkt absolut definitivt därför dessutom uppriktigt "
            "varmt fullständigt helhjärtat helt enkelt")

# Words that open an embedded clause inside a would-be prefield: stop the scan.
_STOP = _w("som att vilket vilken vilka där när vad hur varför om eftersom då medan innan "
           "vars liksom")

_VOC_HEAD = _w("herr fru ärade kära bästa mina")
_INTERJ = _w("ja nej jo nåväl tack nå")
_COORD = _w("men och eller samt")

# ---------------------------------------------------------------------------
# clause-opening classifier


def _is_word(tok):
    return tok[:1].isalnum()


def _skip_prefix(toks, i):
    """Skip opening punctuation, a vocative ('Herr talman, ...'),
    an interjection ('Ja, ...') and a coordinating conjunction ('Men ...')."""
    n = len(toks)
    while i < n and not _is_word(toks[i]):
        i += 1
    # vocatives followed by a comma, possibly chained
    for _ in range(3):
        if i < n and toks[i] in _VOC_HEAD:
            j = i + 1
            while j < n and j <= i + 6 and toks[j] not in (",", "!", ".", ":"):
                j += 1
            if j < n and toks[j] == ",":
                i = j + 1
                continue
        break
    if i + 1 < n and toks[i] in _INTERJ and toks[i + 1] in (",", "!"):
        i += 2
    if i < n and toks[i] in _COORD:
        i += 1
        if i < n and toks[i] == ",":
            i += 1
    return i


def _match_seq(toks, i, seqs):
    """Length of the longest sequence in `seqs` that starts at toks[i], else 0."""
    for s in seqs:
        if tuple(toks[i:i + len(s)]) == s:
            return len(s)
    return 0


class Opening:
    """Result of classifying one sentence opening.

    cat: one of
      S_PRON    personal pronoun subject + finite verb     ("Vi har ...")
      S_DEM     demonstrative pronoun subject + verb        ("Detta är ...", "Det här är ...")
      S_DET     det + verb, no inversion                     ("Det är viktigt ...")
      CLEFT     det är/var X som                             ("Det är vi som ...", "Det är en fråga som ...")
      PSEUDO_CLEFT vad + subject ... är/var                  ("Vad vi behöver är ...")
      S_NP      lexical NP + verb, no pronoun after verb     ("Kommissionen har ...")
      INV_ADV   fronted adverb + finite verb                 ("Därför anser jag ...")
      INV_CONJ  fronted text-structuring conjunct + verb     ("Slutligen vill jag ...", CONJ_ADV)
      INV_PP    fronted PP + finite verb                     ("I dag har vi ...")
      INV_SUB   fronted subordinate clause + finite verb     ("Om vi ..., kommer vi ...")
      INV_OBJ   fronted object det/detta/så + verb + subject ("Det vet vi", "Detta kan vi inte ...")
      INV_NP    fronted lexical NP + verb + pronoun subject  ("Den frågan har vi ...")
      NOINV     fronted adverbial + subject + verb (V3)      ("Dessutom, vi måste ...")
      V3        subject + sentence adverb + verb (V3)        ("Vi också vill ...")
      SUB_SA    fronted clause + resumptive 'så' + verb      ("Om ..., så kommer vi ...")
      OTHER     anything else (questions, imperatives, unclassified)
    a, b: token index range of the evidence (prefield + finite verb [+ subject]).
    """
    __slots__ = ("cat", "a", "b", "fronted")

    def __init__(self, cat, a=0, b=0, fronted=""):
        self.cat, self.a, self.b, self.fronted = cat, a, b, fronted


_OTHER = Opening("OTHER")


# adverb-forming suffixes: inledningsvis, personligen, lyckligtvis, ...
_ADV_SUFFIX = re.compile(r"\w{3,}(?:vis|ligen|ligtvis|mässigt)$")
# finite-verb morphology (present -r, weak past -de/-te/-dde, s-forms); only
# used in positions where V2 grammar already predicts the finite verb.
_FIN_SUFFIX = re.compile(r"\w{2,}(?:ar|er|ir|yr|år|ör|dde|de|te|ades|des|tes)$")
# function words that can never be the finite verb in those positions
_NONVERB = _w("att som och eller men inte också även bara det den detta denna dessa de "
              "en ett i på av för till med om när där här nu då så under över efter")


# focus particles / exemplifiers that can attach to an NP ("Inte minst X har ..."):
# no verb guessing after them
_FOCUS_ADV = _w("exempelvis särskilt speciellt främst bara även också endast enbart huvudsakligen "
                "framförallt likaså")
_FOCUS_MULTI = {("inte", "minst"), ("inte", "bara"), ("framför", "allt"), ("till", "exempel"),
                ("inte", "heller")}
# determiners / pronouns that start a lexical subject NP after a comma
_NP_START = _w("den det de detta denna dessa en ett min vår din er hans hennes deras varje alla "
               "många flera vissa båda samtliga")
_DEF_SUFFIX = re.compile(r"\w{4,}(?:en|et|erna|arna|orna|ena)$")
_NOT_DEF = _w("mycket sedan även igen alltmer hittills tidigare egentligen tydligen självklart "
              "ordentligt gemensamt slutligen öppet direkt helt fortfarande samtidigt")


def _np_then_fin(toks, k, maxlen=4):
    """toks[k:] = lexical NP (determiner or definite noun first) + finite verb
    within `maxlen` tokens, no punctuation. Returns index of the verb or -1."""
    n = len(toks)
    if k >= n:
        return -1
    t = toks[k]
    if t in FIN or t in SUBJ_PERS or t in _STOP or t in _COORD or not _is_word(t):
        return -1
    if t not in _NP_START and (t in _NOT_DEF or not _DEF_SUFFIX.match(t)):
        return -1
    for j in range(k + 1, min(n, k + 1 + maxlen)):
        u = toks[j]
        if not _is_word(u) or u in _STOP or u in SUBJ_PERS or u in _COORD:
            return -1
        if u in FIN:
            return j
    return -1


def _finlike(t):
    return t in FIN or (t not in _NONVERB and _FIN_SUFFIX.match(t) is not None)


def _inv_end(toks, k):
    """Evidence end after a finite verb at k: include an inverted subject pronoun."""
    return k + 2 if (k + 1 < len(toks) and toks[k + 1] in SUBJ_POST) else k + 1


def _after_prefield(toks, p, j, kind, fronted, guess_verb=False):
    """Prefield toks[p:j] has ended (j = first token after it, maybe a comma).
    Decide V2 (inversion) vs V3 (no inversion) from what follows."""
    n = len(toks)
    k = j
    comma = k < n and toks[k] == ","
    if comma:
        k += 1
    if k >= n:
        return None
    t = toks[k]
    if t in FIN or (guess_verb and not comma and _finlike(t)):
        return Opening(kind, p, _inv_end(toks, k), fronted)
    if k + 1 < n and t in SUBJ_PERS and toks[k + 1] in FIN:
        return Opening("NOINV", p, k + 2, fronted)
    if kind == "INV_SUB" and comma and k + 1 < n and t == "så" and toks[k + 1] in FIN:
        return Opening("SUB_SA", p, k + 2, fronted)
    if comma and kind != "INV_SUB":
        # 'Med anknytning till detta, utvecklingen ... har varit' (V3, lexical subject)
        v = _np_then_fin(toks, k)
        if v > 0:
            return Opening("NOINV", p, v + 1, fronted)
    return None


def classify_opening(toks):
    """Classify the opening of one sentence. toks: lower-cased tokens,
    punctuation included (Doc.sent_tokens lower-cased). Returns an Opening."""
    n = len(toks)
    if n < 3 or "?" in toks[-3:]:
        return _OTHER
    p = _skip_prefix(toks, 0)
    if p >= n - 1:
        return _OTHER
    t0 = toks[p]
    t1 = toks[p + 1]
    t2 = toks[p + 2] if p + 2 < n else ""

    # --- pronoun subjects ---------------------------------------------------
    if t0 in SUBJ_PERS:
        if t1 in FIN:
            return Opening("S_PRON", p, p + 2)
        if t1 in V3_ADV and t2 in FIN:
            return Opening("V3", p, p + 3)
        if t0 != "de":           # 'de' may be a determiner: 'De mål vi sätter upp ...'
            return _OTHER

    # demonstrative 'det här / den här / de här' as a pronoun
    if t0 in ("det", "den", "de") and t1 in ("här", "där") and t2 in FIN:
        t3 = toks[p + 3] if p + 3 < n else ""
        if t3 in SUBJ_PERS and t2 not in COPULA:
            return Opening("INV_OBJ", p, p + 4, t0 + " " + t1)
        return Opening("S_DEM", p, p + 3)

    if t0 == "det":
        if t1 in ("är", "var"):
            # cleft / presentational 'Det är X som ...' (B8)
            for j in range(p + 3, min(n, p + 12)):
                u = toks[j]
                if u == "som":
                    return Opening("CLEFT", p, j + 1, "det " + t1)
                if not _is_word(u) or u in ("att", "när", "om", "eftersom", "då", "för"):
                    break
        if t1 in FIN:
            if t2 in SUBJ_PERS and t1 not in COPULA and t1 not in ("finns", "fanns", "gäller", "handlar"):
                return Opening("INV_OBJ", p, p + 3, "det")
            return Opening("S_DET", p, p + 2)
        return _OTHER

    if t0 in ("detta", "denna", "dessa", "den", "denne") and t1 in FIN:
        if t2 in SUBJ_PERS and t1 not in COPULA:
            return Opening("INV_OBJ", p, p + 3, t0)
        return Opening("S_DEM", p, p + 2)

    if t0 in ("så", "sådant") and t1 in FIN and t2 in SUBJ_POST:
        return Opening("INV_OBJ", p, p + 3, t0)

    # --- fronted subordinate clause -----------------------------------------
    m = _match_seq(toks, p, SUBJN_MULTI)
    is_subjn = m > 0 or t0 in SUBJN1
    if m == 0 and t0 in ("då", "sedan", "nu"):
        # adverb if a finite verb follows, subjunction if a subject follows
        if t1 in FIN:
            return Opening("INV_ADV", p, _inv_end(toks, p + 1), t0)
        is_subjn = t0 != "nu" and t1 in SUBJ_POST
    if is_subjn:
        return _classify_subclause(toks, p, p + (m or 1))

    # --- fronted adverbs ----------------------------------------------------
    m = _match_seq(toks, p, ADV_MULTI)
    if m or t0 in ADV1 or (t0 not in FIN and _ADV_SUFFIX.match(t0)):
        j = p + (m or 1)
        guess = tuple(toks[p:j]) not in _FOCUS_MULTI and t0 not in _FOCUS_ADV
        r = _after_prefield(toks, p, j, "INV_ADV", " ".join(toks[p:j]), guess_verb=guess)
        if r is None or (r.cat == "NOINV" and not m and t0 in ADV_NO_V2_CHECK):
            return _OTHER
        if r.cat == "INV_ADV" and r.fronted in CONJ_ADV:
            r.cat = "INV_CONJ"
        return r

    # --- pseudo-cleft: 'Vad vi behöver är ...' (EN 'What we need is ...') ---
    if t0 == "vad" and t1 in SUBJ_POST:
        for j in range(p + 2, min(n, p + 12)):
            if toks[j] in (".", ";", ":", "!", "?"):
                break
            if toks[j] in COPULA and j > p + 2 and toks[j - 1] not in SUBJ_POST:
                return Opening("PSEUDO_CLEFT", p, j + 1, "vad")
        return _OTHER

    # --- fronted PP ---------------------------------------------------------
    if t0 in PREP:
        if t0 == "vad" and t1 not in ("gäller", "beträffar", "angår"):
            return _OTHER
        if t0 == "för" and t1 in SUBJ_POST:      # 'för' = 'because'
            return _OTHER
        j = p + 1
        while j < n and j <= p + 10:
            t = toks[j]
            if t in _STOP or t in (".", ";", ":", "!", "(", "–", '"'):
                return _OTHER
            if j > p + 1:
                if t in FIN:
                    return Opening("INV_PP", p, _inv_end(toks, j), " ".join(toks[p:j]))
                if t in SUBJ_PERS:
                    # 'I dag, vi har ...' (V3 after a comma-marked PP); without
                    # a comma it is usually a contact relative clause
                    # ('Med den tillväxt vi har ...'): give up.
                    if toks[j - 1] == "," and j + 1 < n and toks[j + 1] in FIN:
                        return Opening("NOINV", p, j + 2, " ".join(toks[p:j - 1]))
                    # a short bare PP ('I dag vi har', 'På sikt vi måste') cannot
                    # host a contact relative (that needs a determiner + noun)
                    if j <= p + 3 and j + 1 < n and toks[j + 1] in FIN and \
                            toks[j - 1] not in ADV_NO_V2_CHECK and \
                            not any(u in _NP_START or not _is_word(u) for u in toks[p + 1:j]):
                        return Opening("NOINV", p, j + 2, " ".join(toks[p:j]))
                    return _OTHER
                if t == "," and j + 1 < n and toks[j + 1] in FIN:
                    return Opening("INV_PP", p, _inv_end(toks, j + 1), " ".join(toks[p:j]))
                if t == "," and j <= p + 8:
                    v = _np_then_fin(toks, j + 1)
                    if v > 0:
                        return Opening("NOINV", p, v + 1, " ".join(toks[p:j]))
            j += 1
        return _OTHER

    # --- lexical NP first (subject, or fronted object/predicative) ----------
    if t0 in FIN or t0 in _STOP or t0 in _COORD or not _is_word(t0):
        return _OTHER
    j = p + 1
    while j < n and j <= p + 5:
        t = toks[j]
        if t == "," and j > p + 1:
            # left dislocation: 'Den här frågan, den är ...' / 'X, det är ...'
            # (left dislocation 'X, det är ...' was tried: 3 O / 3 T sentences
            # on dev.500.bal, mostly comma splices; not kept)
            return _OTHER
        if t in SUBJ_PERS and j + 1 < n and _finlike(toks[j + 1]):
            # NP + contact relative clause ('De mål vi sätter upp ...'): the
            # NP is a fronted object if 'FIN + pronoun' follows the relative
            return _contact_relative(toks, p, j)
        if t in _STOP or not _is_word(t) or t in SUBJ_PERS:
            return _OTHER
        if t in FIN:
            if j + 1 < n and toks[j + 1] in SUBJ_PERS and toks[j + 1] != "de":
                return Opening("INV_NP", p, j + 2, " ".join(toks[p:j]))
            return Opening("S_NP", p, j + 1)
        j += 1
    return _OTHER


def _contact_relative(toks, p, j):
    """NP toks[p:j] + relative clause without 'som' starting at j (pronoun subject)."""
    n = len(toks)
    for k in range(j + 2, min(n - 1, j + 14)):
        t = toks[k]
        if t in (".", ";", ":", "!", "?", ",", "–") or t in _STOP:
            return _OTHER
        if t in FIN and toks[k + 1] in SUBJ_PERS and toks[k + 1] != "de" and toks[k - 1] not in SUBJ_POST:
            return Opening("INV_NP", p, k + 2, " ".join(toks[p:j]))
    return _OTHER


def _classify_subclause(toks, p, q):
    """toks[p:q] is a subjunction: find the end of the fronted clause and
    check the main clause that follows."""
    n = len(toks)
    sub = tuple(toks[p:q])
    fronted = " ".join(sub)
    infinitival = False
    if sub in SUBJN_WITH_VERB:
        own_fin = q - 1
    elif sub in SUBJN_INF and q < n and toks[q] not in SUBJ_POST:
        # infinitive clause ('För att uppnå detta måste vi ...'): no own finite verb
        own_fin = q
        infinitival = True
    elif q + 1 < n and toks[q] in SUBJ_POST:
        # pronoun subject: the clause verb follows (maybe after one adverb)
        own_fin = q + 1 if _finlike(toks[q + 1]) or toks[q + 1] not in V3_ADV else q + 2
    else:
        own_fin = -1
        for j in range(q, min(n, q + 7)):
            if toks[j] in FIN:
                own_fin = j
                break
            if toks[j] in (".", ";", ":", "!"):
                return _OTHER
        if own_fin < 0:
            return _OTHER
    j = own_fin + 1
    while j < n and j <= p + 40:
        t = toks[j]
        if t in (".", ";", ":", "!", "?"):
            return _OTHER
        if t == ",":
            r = _after_prefield(toks, p, j, "INV_SUB", fronted)
            if r is not None:
                return r
        elif t in FIN and j > own_fin + 1 and j + 1 < n and toks[j + 1] in SUBJ_PERS \
                and (infinitival or toks[j - 1] not in SUBJ_POST) and toks[j - 1] not in _STOP:
            return Opening("INV_SUB", p, j + 2, fronted)
        j += 1
    return _OTHER


def openings(doc):
    """[(sentence index, Opening)] for every non-vocative sentence."""
    out = []
    addr = doc.is_address
    for i, toks in enumerate(doc.sent_tokens):
        if addr[i]:
            continue
        out.append((i, classify_opening([t.lower() for t in toks])))
    return out


# ---------------------------------------------------------------------------
# rule 1: which opening was chosen (multinomial "choice" over opening types)
#
# P(category | sentence has a classifiable opening), per class. dev-train
# (dev.500.bal, sessions < 2002-03-01; 18171 O / 13191 T classified
# sentences; topic-masked shares identical to 3 decimals; top-5 speakers
# carry <= 34% of every category in either class). SUB_SA (resumptive 'så')
# is merged into INV_SUB: B4 predicts O, dev-train shows no O lean
# (0.12 vs 0.14 per 1000 words).
OPENING_P = {
    #  category        p_O      p_T     dev-train shares O / T      ln(T/O)
    "S_PRON":       (0.30,   0.37),   # .2972 / .3736   +0.23  subject pronoun first (B1)
    "S_NP":         (0.26,   0.23),   # .2566 / .2274   -0.12
    "S_DET":        (0.16,   0.14),   # .1581 / .1434   -0.10  'Det är/finns ...' (B2)
    "S_DEM":        (0.033,  0.054),  # .0333 / .0541   +0.49  'Detta är ...' (EN 'This is')
    "CLEFT":        (0.020,  0.015),  # .0200 / .0149   -0.30  'Det är X som' (B8)
    "PSEUDO_CLEFT": (0.0012, 0.0022), # .0012 / .0022   +0.64  'Vad vi behöver är' (EN 'What we need is')
    "INV_ADV":      (0.091,  0.035),  # .0912 / .0349   -0.96  'Därför anser jag' (B1)
    "INV_CONJ":     (0.018,  0.028),  # .0175 / .0279   +0.47  'Slutligen vill jag' (EN 'Finally, I')
    "INV_PP":       (0.073,  0.067),  # .0735 / .0666   -0.10  'I dag har vi'
    "INV_SUB":      (0.031,  0.046),  # .0311 / .0462   +0.40  'Om vi ..., kommer vi' (EN 'If we ..., we')
    "INV_NP":       (0.010,  0.0067), # .0100 / .0067   -0.40  'Den frågan har vi ...'
    "INV_OBJ":      (0.010,  0.0021), # .0103 / .0021   -1.6   'Det vet vi' (B3)
}
_OPEN_ALIAS = {"SUB_SA": "INV_SUB"}
OPENING_LLR = {c: math.log(pt / po) for c, (po, pt) in OPENING_P.items()}

_OPEN_LABEL = {
    "S_PRON": "subject pronoun first",
    "S_NP": "subject NP first",
    "S_DET": "'det' subject first",
    "S_DEM": "demonstrative subject first ('Detta är')",
    "CLEFT": "cleft 'det är X som'",
    "PSEUDO_CLEFT": "pseudo-cleft 'vad ... är' (EN 'what ... is')",
    "INV_ADV": "topicalised adverb + inversion",
    "INV_CONJ": "fronted structuring conjunct (EN 'Finally,') + inversion",
    "INV_PP": "fronted PP + inversion",
    "INV_SUB": "fronted subordinate clause (EN order)",
    "INV_NP": "fronted object/adverbial NP + inversion",
    "INV_OBJ": "fronted object 'det/detta' + inversion",
}


def _span(doc, i, a, b):
    """Text span of tokens a..b-1 of sentence i."""
    toks, starts = doc.sent_tokens[i], doc.sent_tok_starts[i]
    b = min(b, len(toks))
    if b <= a:
        b = a + 1
    return starts[a], starts[b - 1] + len(toks[b - 1])


def _openings_cached(doc):
    o = doc.__dict__.get("_wordorder_openings")
    if o is None:
        o = openings(doc)
        doc.__dict__["_wordorder_openings"] = o
    return o


def _opening_spans(doc):
    out = []
    for i, o in _openings_cached(doc):
        c = _OPEN_ALIAS.get(o.cat, o.cat)
        llr = OPENING_LLR.get(c)
        if llr is None:
            continue
        a, b = _span(doc, i, o.a, o.b)
        out.append((a, b, llr, _OPEN_LABEL[c]))
    return out


def _opening_llr(doc):
    return sum(s for _, _, s, _ in _opening_spans(doc))


def _opening_describe(doc, c):
    cnt = {}
    for i, o in _openings_cached(doc):
        k = _OPEN_ALIAS.get(o.cat, o.cat)
        if k in OPENING_LLR:
            cnt[k] = cnt.get(k, 0) + 1
    top = ", ".join("%s %d" % (_OPEN_LABEL[k], v) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:4])
    return "sentence openings (%s): %+.2f toward %s" % (top, c, "TRANSLATED" if c > 0 else "ORIGINAL")


# ---------------------------------------------------------------------------
# rule 2: English order leaking into Swedish clause structure (rare errors)

_SUBJN_RX = r"(?:att|om|eftersom|när|innan|medan|ifall|fastän|även\s+om|trots\s+att|så\s+att|därför\s+att)"
_SUBJ_RX = r"(?:jag|vi|ni|du|han|hon|de|man|det|den|detta)"
_FIN_RX = "(?:%s)" % "|".join(sorted(FIN, key=len, reverse=True))
# 'att vi kan inte ...' (main-clause negation order in a subordinate clause);
# not constituent negation ('inte bara/minst/heller ...').
EMBEDDED_NEG_RX = re.compile(
    r"(?<![\w-])" + _SUBJN_RX + r"\s+" + _SUBJ_RX + r"\s+" + _FIN_RX + r"\s+(?:inte|aldrig)(?![\w-])"
    r"(?!\s+(?:bara|minst|heller|enbart|endast|längre|så|alls|ens)(?![\w-]))")

# per-hit LLRs (judgement values, see rule rationale)
V2_VIOLATION_HIT = {"NOINV": 1.5, "V3": 1.5, "EMB": 1.5}
_V2_LABEL = {
    "NOINV": "no inversion after fronted adverbial (EN order, 'Dessutom, vi måste')",
    "V3": "adverb between subject and verb ('Vi också vill')",
    "EMB": "main-clause negation order in subordinate clause ('att vi kan inte')",
}


def _v2_spans(doc):
    out = []
    for i, o in _openings_cached(doc):
        if o.cat in ("NOINV", "V3"):
            a, b = _span(doc, i, o.a, o.b)
            out.append((a, b, V2_VIOLATION_HIT[o.cat], _V2_LABEL[o.cat]))
    for m in EMBEDDED_NEG_RX.finditer(doc.lower):
        out.append((m.start(), m.end(), V2_VIOLATION_HIT["EMB"], _V2_LABEL["EMB"]))
    return out


def _v2_llr(doc):
    return sum(s for _, _, s, _ in _v2_spans(doc))


RULES = [
    PredicateRule(
        "wordorder.opening_type",
        description="sentence opening: topicalised + inverted vs subject-first (V2 prefield choice)",
        fn=_opening_llr, spans_fn=_opening_spans, describe_fn=_opening_describe,
        direction="O",
        rationale="Hasselgård (2004): V2 languages put non-subjects in the prefield far more often "
                  "than English, and translators keep the English Theme. Each non-vocative, non-question "
                  "sentence whose opening the closed-list heuristic can classify contributes "
                  "ln(p_T/p_O) of its opening type (a multi-way 'which variant was chosen' rule; "
                  "unclassifiable openings contribute nothing, so it is robust to length). O side: "
                  "topicalised deictic/causal adverbs (därför, nu, då, samtidigt, här, däremot: "
                  "-0.96), fronted object det/detta (-1.6, B3), fronted NPs, clefts (B8). T side: "
                  "subject pronoun first (B1), 'Detta är' (EN 'This is'), pseudo-clefts (EN 'What we "
                  "need is'), and inversion after structuring conjuncts that mirror a sentence-initial "
                  "English adverbial ('Firstly,' 'Finally,' 'However,'), and fronted adverbial "
                  "clauses ('If ..., we ...'), both forced/allowed by English order. Two "
                  "literature directions are refined: fronting per se is not O (fronted clauses "
                  "and conjuncts are T), and resumptive 'så' (B4) is not O on dev-train.",
        source="literature B1, B2, B3, B4, B8; Hasselgård (2004); Bohnacker & Rosén (2008); "
               "[exp] for the INV_CONJ/INV_SUB/PSEUDO_CLEFT split",
        # dev-train dispersion: text-level sd of this LLR is 1.12x (500 words) and
        # 1.05x (100 words) the independent-sentence prediction -> weight 0.8
        weight=0.8, cap=3.0),
    PredicateRule(
        "wordorder.v2_violation",
        description="English word order leaking into Swedish (V3 / no inversion / embedded negation)",
        fn=_v2_llr, spans_fn=_v2_spans,
        direction="T",
        rationale="Interference with English SVO: a fronted adverbial followed by subject + verb "
                  "('Med andra ord, de är ...', 'Slutligen, kommissionen är ...'), a sentence "
                  "adverb between subject and finite verb ('Jag verkligen hoppas'), and main-clause "
                  "negation order after a subjunction ('att jag har inte', native 'att jag inte "
                  "har'). Rare in professional translation but precise. dev-train dev.500.bal: "
                  "no-inversion openings O 3 / T 20 (0.006 / 0.046 per 1000 words); embedded "
                  "negation order dev.500.all O 0 / T 39 (0 / 0.011); V3 O 0 / T 2. Per hit 1.5 "
                  "(judgement; observed ln ratio ~2.4, O counts too small to trust).",
        source="literature B5, B6, B7; pitfalls 5.5.9",
        cap=3.0),
]

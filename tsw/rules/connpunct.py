"""connpunct: connectives / explicitation and punctuation / orthography.

What the family covers
- Connectives. The explicitation hypothesis (Blum-Kulka 1986; Volansky et al.
  2015: *moreover*, *thus* over-used in translated English) predicts MORE
  connectives in T. On Europarl dev the opposite holds for the native Swedish
  connective adverbs (dessutom, därför, dock, därmed, således, framför allt,
  inte minst ...: 6.5 vs 2.5 per 1000 words): the Swedish speeches are
  argumentative and place connectives in the middle field (V2: "Vi måste därför
  ..."), and translators add one only where the English has one (unique-items
  effect, Tirkkonen-Condit 2004). What T over-uses are the fixed Swedish
  equivalents of English discourse markers (i synnerhet = in particular, i
  själva verket = in fact, icke desto mindre = nevertheless ...): interference
  rather than explicitation.
- Clause linking: att-drop after verbs of saying/thinking (O), zero object
  relative without 'som' (T: English contact clauses, the reverse of the
  'som-retention' explicitation claim), sentence-initial Och/Men (T: English
  And/But openings).
- Punctuation: comma after a sentence-initial adverbial (T, English habit),
  missing comma before clausal men/eftersom/medan/vilket (T, English
  punctuation of because/while/which kept), parenthetical dashes and
  explanatory parentheses (T), non-vocative exclamation marks (O, spoken
  rhetoric + transcription convention), abbreviations t.ex./dvs./bl.a. vs
  written-out forms (O; house style).

Normalised away by EP house style on dev (measured, not used): apostrophe
genitive (EU's: 5 vs 18 hits, all English names like Lloyd's), capitalised
months/weekdays/nationalities (0 real hits), decimal point and thousands comma
(near zero; '11.1' are article numbers), '%' without space, quote style,
overall comma density per word (32.8 vs 32.9), colon and question-mark rates,
Oxford comma (regex mostly hits vocatives/enumerations; 0.13 vs 0.21, too
noisy), semicolons (0.40 vs 0.48 and drifting in O: 0.9 -> 0.2 per 1000 words
between 1997 and 2001), 'inte bara ... men/även' (1 vs 6 hits).

Calibration: python3 -m tsw.tools.rates ... on dev.500.bal (O 697k / T 699k
words), 2026-11-05, for pattern design, KWIC and speaker/year checks; the
numbers in the rules were then re-measured on dev-train only (sessions before
2002-03-01; O 497k / T 434k words) and typed by hand, <= 2 significant digits.
Overlaps:
the wordorder family owns the V2 violation after a fronted adverbial
("Dessutom, vi måste"); connpunct.initial_adverbial_comma scores only the comma.
The particles family owns ändå/nämligen/alltså (also connective-like, also O).
"""
import math
import re

from .base import ChoiceRule, PredicateRule, RateRule

FAMILY = "connpunct"
FAMILY_CAP = 4.0

# --- shared word lists -------------------------------------------------------

_SUBJ = r"(?:jag|vi|ni|de|man|du|han|hon)"
# finite verbs that typically follow the subject of a complement/relative clause
_FIN = (r"(?:är|var|har|hade|kan|kunde|ska|skall|skulle|blir|blev|finns|fanns|måste|bör|borde|"
        r"vill|ville|gör|gjorde|går|kommer|får|fick|behöver|står|ligger|tar|innebär|handlar|"
        r"gäller|saknas|behövs)")
_SENT_START = r"(?:(?<=[.!?] )|(?<=[.!?]\" )|^)"


def _rx(words):
    """Alternation of lower-case words/phrases (spaces -> \\s+), longest first."""
    return "(?:%s)" % "|".join(w.replace(" ", r"\s+") for w in sorted(words, key=len, reverse=True))


# --- connectives -------------------------------------------------------------

_NATIVE_CONNECTIVES = (
    r"(?<![\w-])(?:dessutom|därför(?!\s+att\b)|dock|därmed|således|framför\s+allt|inte\s+minst|"
    r"samtidigt(?!\s+som\b)|i\s+stället|för\s+övrigt|i\s+alla\s+fall)(?![\w-])")

_CALQUE_CONNECTIVES = {
    "'{m}' (EN in particular)": r"(?<![\w-])i\s+synnerhet(?![\w-])",
    "'{m}' (EN in fact)": r"(?<![\w-])i\s+själva\s+verket(?![\w-])",
    "'{m}' (EN nevertheless)": r"(?<![\w-])icke\s+desto\s+mindre(?![\w-])",
    "'{m}' (EN in summary)": r"(?<![\w-])sammanfattningsvis(?![\w-])",
    "'{m}' (EN thus/consequently)": r"(?<![\w-])(?:sålunda|följaktligen)(?![\w-])",
    "'{m}' (EN first of all)": r"(?<![\w-])först\s+och\s+främst(?![\w-])",
    "'{m}' (EN finally/at last)": r"(?<![\w-])till\s+sist(?![\w-])",
    "'{m}' (EN anyway/in any case)": r"(?<![\w-])(?:hur\s+som\s+helst|i\s+vilket\s+fall(?:\s+som\s+helst)?)(?![\w-])",
    "'{m}' (EN for this reason/this is why)":
        r"(?<![\w-])(?:av\s+detta\s+skäl|av\s+den\s+anledningen|(?:detta|det)\s+är\s+skälet)(?![\w-])",
    "'{m}' (EN in general)": r"(?<![\w-])i\s+allmänhet(?![\w-])",
    "'{m}' (EN in other words)": r"(?<![\w-])med\s+andra\s+ord(?![\w-])",
    "'{m}' (EN against this background)": r"(?<![\w-])mot\s+bakgrund\s+av\s+detta(?![\w-])",
}

# --- clause linking ----------------------------------------------------------

_SAY_THINK = r"(?:tror|tycker|anser|menar|hoppas|säger|vet|tänker|förmodar|antar|befarar|fruktar)"
_ATT_SUBJ = (r"(?:jag|vi|ni|de|man|du|han|hon|det|den|detta|dessa|denna|alla|många|"
             r"kommissionen|rådet|parlamentet)")
_ATT_DROP = r"(?<![\w-])%s(?:\s+%s)?\s+%s\s+(?:inte\s+)?%s(?![\w-])" % (_SAY_THINK, _SUBJ, _ATT_SUBJ, _FIN)

_DET = r"(?:den|det|de|denna|detta|dessa|samma|alla|varje|många|några|ett|en)"
# a noun-like word: not a function word, verb of saying/thinking or subjunction
_NOUNISH = (r"(?!(?:som|att|och|eller|när|om|förrän|innan|eftersom|då|där|tror|tycker|anser|menar|"
            r"vet|hoppas|gäller|säger|sade|sa|är|var|har|hade|kan|vill|ska|skall|måste|bör|inte|"
            r"också|nu|här|redan|bara|ju|så|tänker|kommer|blir|finns)(?![\w-]))[a-zåäöé]+")
_REL_FIN = (r"(?:är|var|har|hade|kan|kunde|ska|skall|skulle|blir|blev|måste|bör|borde|vill|ville|"
            r"gör|gjorde|kommer|får|fick|behöver|tar|talar|diskuterar|ser|vet|säger|sade|sa|hört|"
            r"hörde|står|lägger|lade|röstar|röstade|antar|antog|debatterar|behandlar)")
_SOM_DROP = r"(?<![\w-])%s\s+(?:%s\s+)?%s\s+%s\s+%s(?![\w-])" % (_DET, _NOUNISH, _NOUNISH, _SUBJ, _REL_FIN)
_SOM_KEPT = r"(?<![\w-])%s\s+(?:%s\s+)?%s\s+som\s+%s\s+%s(?![\w-])" % (_DET, _NOUNISH, _NOUNISH, _SUBJ, _REL_FIN)

# --- punctuation -------------------------------------------------------------

_INIT_ADV = _rx([
    "dessutom", "därför", "vidare", "slutligen", "emellertid", "dock", "således", "alltså",
    "följaktligen", "därmed", "först", "sedan", "nu", "då", "här", "idag", "i dag", "tyvärr",
    "lyckligtvis", "naturligtvis", "givetvis", "självklart", "självfallet", "faktiskt", "kanske",
    "verkligen", "egentligen", "samtidigt", "tidigare", "nyligen", "också", "även",
    "sammanfattningsvis", "avslutningsvis", "för det första", "för det andra", "för det tredje",
    "för det fjärde", "med andra ord", "i själva verket", "å ena sidan", "å andra sidan",
    "trots detta", "trots det", "trots allt", "icke desto mindre", "i synnerhet", "framför allt",
    "i detta sammanhang", "enligt min mening", "enligt min uppfattning", "först och främst",
    "för övrigt", "till sist", "till slut", "i stället", "därefter", "ändå", "men", "så",
    "i så fall", "i praktiken", "för närvarande", "hur som helst", "i vilket fall",
    "i allmänhet", "kort sagt", "personligen", "uppenbarligen", "dessvärre", "på samma sätt",
])
_ADV_COMMA = _SENT_START + _INIT_ADV + r"\s*,"
_ADV_NOCOMMA = _SENT_START + _INIT_ADV + r"(?=\s+[^\s,])"

_CL_SUBJ = r"(?:det|jag|vi|ni|de|man|den|detta|han|hon|dessa|denna)"
_CL_CONJ = (r"(?:eftersom|medan|men\s+%s|vilket\s+(?:innebär|betyder|gör|är|var|har|skulle|kan|"
            r"kommer|leder|visar|också|inte|enligt|jag|vi))(?![\w-])" % _CL_SUBJ)
# word before the conjunction must be a clause-final word, not a coordinator/focus
# adverb/preposition ("och eftersom", "just eftersom", "i vilket")
_CL_STOP = (r"(?:och|men|eller|särskilt|just|bara|inte|helt|främst|delvis|även|också|framför|dels|"
            r"allt|enbart|endast|mest|minst|både|i|på|av|om|för|med|till|från|genom|enligt|under|"
            r"över|vid|mot|oavsett|utan|efter|inom|hur|vad|som|att|nämligen|alltså|då|så)")
_CL_NOCOMMA = r"(?<![\w-])(?!%s(?![\w-]))[a-zåäö]+\s+%s" % (_CL_STOP, _CL_CONJ)
_CL_COMMA = r",\s+%s" % _CL_CONJ

_ABBR = (r"(?<![\w.])(?:bl\.a\.?|t\.ex\.?|dvs\.?|d\.v\.s\.|m\.m\.|osv\.|s\.k\.|m\.fl\.|fr\.o\.m\.|"
         r"t\.o\.m\.)")
_FULL = (r"(?<![\w-])(?:bland\s+annat|till\s+exempel|det\s+vill\s+säga|med\s+mera|och\s+så\s+vidare|"
         r"så\s+kallad[e]?|med\s+flera|från\s+och\s+med|till\s+och\s+med)(?![\w-])")

# dash between two words (not inside document references "C5-0104/2001 - 2001/2053")
_DASH = r"(?<=[a-zåäö,]) [–-] (?=[a-zåäö])"

# parenthesis without digits or '/', not a document code, political group or
# Council configuration ("rådet (allmänna frågor)"), not a committee name
_PAREN = (r"(?<!rådet )(?<!rådets )\((?![^()]*[\d/])(?!utskottet)"
          r"(?!(?:eg|eeg|euratom|eksg|cns|cod|avc|syn|ini|cos|ack|bud|delvis|ppe-de|ppe|pse|eldr|elr|"
          r"gue/ngl|uen|edd|tdi|ni|verts/ale)\))[^()]{2,120}\)")


# --- exclamation outside vocatives (PredicateRule: needs doc.is_address) -------

_EXCL_RATE_O, _EXCL_RATE_T = 0.68, 0.21      # per 1000 words (dev-train), see rule comment
_EXCL_LOG = math.log(_EXCL_RATE_T / _EXCL_RATE_O)
_EXCL_PER_WORD = (_EXCL_RATE_T - _EXCL_RATE_O) / 1000.0
_EXCL_END = re.compile(r"!+[\"')\]]*\s*$")


def _exclamations(doc):
    """[(start, end)] of the '!' ending each non-vocative sentence."""
    out = []
    for i, (a, b) in enumerate(doc.sent_spans):
        if doc.is_address[i]:
            continue
        m = _EXCL_END.search(doc.text, a, b)
        if m:
            out.append((m.start(), m.start() + 1))
    return out


def _excl_llr(doc):
    n = doc.n_words
    if n == 0:
        return 0.0
    return len(_exclamations(doc)) * _EXCL_LOG - n * _EXCL_PER_WORD


def _excl_spans(doc):
    return [(a, b, _EXCL_LOG, "exclamation mark ending a non-vocative sentence")
            for a, b in _exclamations(doc)]


RULES = [
    # ---------------------------------------------------------------- connectives
    RateRule(
        "connpunct.native_connectives",
        description="native connective adverbs (dessutom, därför, dock, därmed, inte minst ...)",
        pattern=_NATIVE_CONNECTIVES,
        # dev.500.bal (masked, same unmasked) per 1000 words: O 6.5, T 2.5 (ln -0.97/hit).
        # därför (not 'därför att') 2.0/1.0, dock 1.3/.41, dessutom .61/.32, i stället .48/.19,
        # samtidigt (not 'samtidigt som') .48/.19, inte minst .55/.11, framför allt .39/.15,
        # därmed .36/.056, således .13/.05. 99 O speakers, top-5 share 27%; per year O 10-13,
        # T 5.3-6.6 (stable). Bursty (argumentative speakers) -> weight 0.5.
        # dev-train (date < 2002-03-01) re-check: O 6.4, T 2.2 -> values below.
        rate_o=6.4, rate_t=2.2,
        direction="O",
        rationale="Explicitation predicts more connectives in T, but on EP dev originals use ~2.6x "
                  "more native connective adverbs: Swedish argumentative speech places them in the "
                  "middle field (V2), while translators only render connectives present in the "
                  "English source (unique-items effect). Reversed w.r.t. Volansky's English result.",
        source="literature D1/D2 [direction reversed on EP dev]; Tirkkonen-Condit (2004)",
        weight=0.5, cap=2.0),
    RateRule(
        "connpunct.additive_ocksa",
        description="additive 'också'",
        words=["också"],
        # dev.500.bal per 1000 words: O 5.3, T 2.2 (texts with >=1: 90% / 60%); top-5 speakers
        # 29% (O) / 21% (T). 'även' is only 1.9/1.6, so translators do not simply swap the word.
        # dev-train: O 5.5, T 2.2.
        rate_o=5.5, rate_t=2.2,
        direction="O",
        rationale="Additive cohesion ('vi måste också ...') is a native Swedish argumentative habit "
                  "in the middle field; English 'also/too/as well' is less frequent and often left "
                  "unrendered. Same mechanism as native_connectives, kept separate because of its "
                  "frequency (bursty) -> low weight.",
        source="[exp] (cohesion: literature D2)",
        weight=0.4, cap=1.5),
    RateRule(
        "connpunct.calque_connectives",
        description="Swedish equivalents of English discourse markers (i synnerhet, i själva verket ...)",
        pattern=_CALQUE_CONNECTIVES,
        # dev.500.bal (masked) per 1000 words: O .38, T 1.4. i synnerhet .065/.54, i själva verket
        # .036/.17, till sist, först och främst, följaktligen .046/.084, i allmänhet,
        # sammanfattningsvis .019/.067, icke desto mindre .0086/.039, hur som helst, av detta skäl.
        # 51 O speakers (top-5 34%), 159 T speakers (21%). dev-train: O .34, T 1.5.
        rate_o=0.34, rate_t=1.5,
        direction="T",
        rationale="Interference: fixed English discourse markers (in particular, in fact, "
                  "nevertheless, first of all, in summary, for this reason) are rendered by their "
                  "dictionary equivalents, which native speakers use far less (they use framför "
                  "allt, faktiskt, ändå, först, kort sagt instead).",
        source="literature D1 (Volansky et al. 2015 cohesive markers), 4.3a",
        weight=0.8, cap=2.0),

    # ---------------------------------------------------------------- clause linking
    RateRule(
        "connpunct.att_drop",
        description="complementizer 'att' dropped after tror/tycker/anser/hoppas/säger ...",
        pattern=_ATT_DROP, label="att dropped: '{m}'",
        # dev.500.bal: P(drop | drop or kept) O 119/2318 = 0.051, T 29/1591 = 0.018. Most O drops
        # are 'tycker jag är' / 'tror jag är' / 'tycker det är'. First modelled as a ChoiceRule,
        # but 'att kept' is itself MORE frequent in O (3.2 vs 2.2 per 1000), so the B term pushed
        # O texts to T (rule AUC .46 on --limit 500): now a rate on the drop alone.
        # dev-train per 1000 words: O 71/497k = .14, T 26/434k = .060.
        rate_o=0.14, rate_t=0.060,
        direction="O",
        rationale="Explicitation: Swedish allows dropping 'att' after verbs of saying and thinking "
                  "('Jag tror det är viktigt'); translators retain it (English 'that' is also often "
                  "absent, so this is normalisation to the written norm). Also spoken register.",
        source="literature C9 / A17 (att-retention) [exp]",
        cap=1.5),
    ChoiceRule(
        "connpunct.som_drop",
        description="object relative without 'som' ('det arbete ni har gjort')",
        a=_SOM_DROP, b=_SOM_KEPT, label_a="zero relative", label_b="'som' relative",
        # dev.500.bal: P(no som) O 186/630 = 0.30, T 299/644 = 0.46. Top forms: 'det arbete han
        # har', 'de problem vi står', 'det sista jag vill' vs 'det arbete som han har'.
        # dev-train: O 111/445 = 0.25, T 190/415 = 0.46. O drifts upward over the years
        # (.09 in 1997-99, ~.4 from 2001), so weight 0.7.
        p_o=0.25, p_t=0.46,
        direction="T",
        rationale="The literature expects 'som'-retention (explicitation) in T; EP dev shows the "
                  "reverse: English contact clauses ('the work you have done') are rendered with a "
                  "zero relative ('det arbete ni har gjort'), i.e. interference beats explicitation.",
        source="literature D1/3 ('som' retained) [direction reversed on EP dev; exp]",
        weight=0.7, cap=1.5),
    RateRule(
        "connpunct.initial_och",
        description="sentence-initial 'Och'",
        pattern=_SENT_START + r"Och(?![\w-])", case=True,
        # dev.500.bal: O 8 hits (.011 per 1000), T 100 (.14). Spread over 58 T speakers
        # (top-5 23%). dev-train: O .016, T .17.
        rate_o=0.016, rate_t=0.17,
        direction="T",
        rationale="Interference: English speakers open sentences with 'And'; Swedish written and "
                  "edited norm avoids sentence-initial 'Och', so the Swedish transcripts almost "
                  "never have it while translators keep the source's sentence division.",
        source="literature B/positional tokens (Volansky et al. 2015) [exp]",
        cap=2.0),
    RateRule(
        "connpunct.initial_men",
        description="sentence-initial 'Men'",
        pattern=_SENT_START + r"Men(?![\w-])", case=True,
        # dev.500.bal: O .37, T .75 per 1000 words (texts 16% / 30%). Both classes decline over
        # the years (O .77 -> ~.2, T 1.05 -> ~.6). dev-train: O .45, T .85.
        rate_o=0.45, rate_t=0.85,
        direction="T",
        rationale="Interference: English 'But' at sentence start is kept as 'Men'; Swedish "
                  "speakers more often coordinate within the sentence (', men') or use 'dock' in "
                  "the middle field. (Volansky found the opposite for English O vs T: the "
                  "direction is source-language specific.)",
        source="Volansky et al. (2015) positional tokens [direction differs]; [exp]",
        weight=0.7, cap=1.5),

    # ---------------------------------------------------------------- punctuation
    ChoiceRule(
        "connpunct.initial_adverbial_comma",
        description="comma after a sentence-initial adverbial ('Dessutom, ...')",
        a=_ADV_COMMA, b=_ADV_NOCOMMA, label_a="comma after initial adverbial",
        label_b="no comma after initial adverbial",
        # dev.500.bal: P(comma) O 47/3730 = 0.013, T 259/2908 = 0.089 (rule pattern;
        # base list without the multiword additions gave .012/.085). Top T: 'För det första,',
        # 'Slutligen,', 'Till sist,'. Year-stable in O (0.00-0.02).
        # dev-train: O 27/2731 = 0.010, T 164/1812 = 0.091.
        p_o=0.010, p_t=0.091,
        direction="T",
        rationale="English sets off fronted adverbials with a comma; Swedish V2 does not "
                  "('Därför anser jag'). Only the punctuation is scored here: the V2 violation "
                  "that often follows ('Dessutom, vi måste') belongs to the wordorder family.",
        source="literature B5",
        cap=2.0),
    ChoiceRule(
        "connpunct.clause_comma_omission",
        description="no comma before clausal men/eftersom/medan/vilket",
        a=_CL_NOCOMMA, b=_CL_COMMA, label_a="no comma before conjunction",
        label_b="comma before conjunction",
        # dev.500.bal (masked): P(no comma) O 102/1960 = 0.052, T 303/1777 = 0.17. eftersom alone
        # .095/.31, men+subject .033/.078. Diachronic: O rises 0.00 (1997-99) -> 0.11 (2001-04).
        # dev-train: O 61/1405 = 0.043, T 185/1122 = 0.165.
        p_o=0.043, p_t=0.17,
        direction="T",
        rationale="English does not put a comma before restrictive 'because', 'while', 'which' "
                  "and often not before 'but'; translators follow the source punctuation, while the "
                  "Swedish norm of the period puts a comma before these clause linkers. The O rate "
                  "rises over time (looser Swedish comma norms), hence weight 0.7.",
        source="literature E2 (comma habits) [exp]",
        weight=0.7, cap=1.5),
    ChoiceRule(
        "connpunct.written_out_abbrev",
        description="'till exempel/bland annat/det vill säga' written out vs 't.ex./bl.a./dvs.'",
        a=_FULL, b=_ABBR, label_a="written out", label_b="abbreviation",
        # dev.500.bal: P(written out) O 538/1919 = 0.28, T 550/821 = 0.67. O 81 speakers,
        # top-5 29%. Per year P(abbr) O .54-.81, T .06-.55.
        # dev-train: O 415/1413 = 0.29, T 301/490 = 0.61.
        p_o=0.29, p_t=0.61,
        direction="T",
        rationale="NORM / house style: translators follow the writing guidelines (Myndigheternas "
                  "skrivregler) and write abbreviations out in running text; the Swedish "
                  "transcripts (and MEPs' written explanations of vote) keep t.ex., dvs., bl.a. "
                  "This is an orthographic convention of the two EP services rather than of "
                  "translation as such: low weight, will not generalise beyond EP.",
        source="Språkrådet/Myndigheternas skrivregler (literature 1.3); [exp, EP convention]",
        weight=0.5, cap=1.5),
    RateRule(
        "connpunct.parenthetical_dash",
        description="dash between words (parenthetical insertion)",
        pattern=_DASH,
        # dev.500.bal: word-dash-word O 1.4, T 3.1 per 1000 words (any spaced dash 1.6/3.4); per year
        # O 0.9-2.1, T 2.6-4.0 (stable). En dash vs hyphen is a 2003-04 house-style switch: not used.
        # dev-train: O 1.6, T 3.1.
        rate_o=1.6, rate_t=3.1,
        direction="T",
        rationale="Source shining through: English speeches (as transcribed) make heavy use of "
                  "dash-delimited asides; translators keep them, while Swedish transcribers break "
                  "sentences or use commas. Partly a transcription convention of the English "
                  "verbatim report, so low weight.",
        source="Volansky et al. (2015) punctuation [exp for the dash]",
        weight=0.5, cap=1.5),
    RateRule(
        "connpunct.explanatory_parens",
        description="explanatory parentheses (no numbers, not document codes)",
        pattern=_PAREN,
        # dev.500.bal: O .11, T .27 per 1000 words (texts 4.5% / 11%); acronym expansions
        # (SADC), glosses (färskt kött, ...), asides (och jag förväntar mig inte ...).
        # dev-train: O .078, T .26.
        rate_o=0.078, rate_t=0.26,
        direction="T",
        rationale="Explicitation: parenthetical glosses and acronym expansions are typical of "
                  "translations (Volansky: parentheses mark T).",
        source="literature D3; Volansky et al. (2015)",
        cap=1.0),
    PredicateRule(
        "connpunct.exclamation",
        description="exclamation marks outside vocatives",
        # dev.500.bal: sentence-final '!' in non-address sentences, O 443 (.64 per 1000 words),
        # T 129 (.18). dev-train: O .68, T .21 (values in _EXCL_RATE_O/_T above).
        # Poisson LLR computed in _excl_llr with these rates.
        fn=_excl_llr, spans_fn=_excl_spans,
        direction="O",
        rationale="Spoken rhetoric ('Vi lynchar inte Pinochet - vi dömer honom!', 'Tack för "
                  "svaret!'): Swedish transcribers keep the exclamation; translations of English "
                  "verbatim reports rarely have one outside the vocative. Partly a transcription "
                  "convention (SPK), so low weight and small cap.",
        source="literature E9 [SPK; transcription convention]",
        weight=0.5, cap=1.5),
]

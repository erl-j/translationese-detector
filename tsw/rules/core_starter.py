"""Starter family: six rules that prove the pipeline end to end.

Deliberately small. The full families (wordorder, register,
particles_pronouns, calques, connectives_punct, stats) are written on their
own branches as separate modules; when they cover these phenomena, this module
can be excluded (Engine(exclude="starter")) or deleted.

Calibration (2026-11-05, `python3 -m tsw.tools.rates ...` on dev.500.bal
unless noted; values typed in by hand, rounded to <= 2 significant digits).
Two literature expectations do NOT hold on Europarl dev; see the rationale
fields and docs/ENGINE.md ("Calibration findings").
"""
from .base import ChoiceRule, HitRule, RateRule, StatRule, sentence_lengths

FAMILY = "starter"
FAMILY_CAP = 6.0

_DEM_HAR = r"(?<![\w-])(?:den|det|de|dom)\s+här(?![\w-])"
_DEM_DENNA = r"(?<![\w-])(?:denna|detta|dessa|denne)(?![\w-])"

# Modal particles, with the disambiguating contexts of docs/literature.md A5.
# Revised 2026-11-05 (particles branch, after the coordinator's error analysis):
# 'nog' after a -t adverb/adjective ("tråkigt nog", "långt nog") or as degree
# "enough" is not the modal particle; 'väl' + -da/-de/-ta participles
# ("väl förberedda", "väl värda") is the adverb "well"; "ju mer/fler ... desto".
_FIN = r"(?:är|var|har|hade|kan|kunde|ska|skall|blir|blev|finns|fanns|måste|bör|vill|gör)"
_SUBJ = r"(?:det|den|vi|ni|de|han|hon|man|jag|du|detta)"
_MODAL_PARTICLES = {
    # not comparative "ju större/mer/fler ... desto"
    "modal particle 'ju'": r"(?<![\w-])ju(?![\w-])"
                           r"(?!\s+(?:\w+(?:re|are)|mer|mera|fler|flera|förr|mest)\b)",
    # not "nog med/för/att/av" (= enough), not "<adverb in -t> nog" ("tråkigt nog",
    # "långt nog", but "det nog" is kept), "glädjande/bra/nära/mer än/inte nog",
    # and not clause-final "... nog." (= enough)
    "modal particle 'nog'": r"(?<![\w-])(?:(?<!t )|(?<=det ))(?<!nde )(?<!bra )(?<!nära )(?<!än )"
                            r"(?<!inte )(?<!aldrig är )nog(?![\w-])"
                            r"(?!\s+(?:med|för|att|av)\b)(?!\s*[.,;:!?])",
    # only after a finite verb (+ subject), not "väl fungerande/medveten/förberedda/värda/kända",
    # not "väl så <adj>" (= quite as); 'inte/heller/knappast/...' after väl are particle contexts
    "modal particle 'väl'": r"(?<![\w-])" + _FIN + r"\s+(?:" + _SUBJ + r"\s+)?väl(?![\w-])"
                            r"(?!\s+(?!(?:inte|heller|knappast|just|egentligen)\b)"
                            r"\w+(?:ad|at|ade|en|et|na|d|t|da|de|ta|te)\b)"
                            r"(?!\s+så\s+(?!att\b)\w)|(?<![\w-])väl\s*\?",
    # not the determiner "ett visst (antal)", "i visst (mån)", "med visst (beklagande)"
    "modal particle 'visst'": r"(?<!ett )(?<!\bi )(?<!med )(?<![\w-])visst(?![\w-])",
}

RULES = [
    ChoiceRule(
        "starter.dem_har_vs_denna",
        description="demonstrative 'den/det/de här' vs 'denna/detta/dessa'",
        a=_DEM_HAR, b=_DEM_DENNA, label_a="den/det/de här", label_b="denna/detta/dessa",
        # dev.500.bal: share of 'den här'-type: O 117/8837 = 0.013, T 631/10440 = 0.060
        p_o=0.013, p_t=0.060,
        direction="T",
        rationale="Literature (A1) predicts the OPPOSITE (denna = T, den här = O). On Europarl dev the "
                  "Swedish-original transcripts are edited to written norm (denna 98.7%), while "
                  "translators of spoken English ('this') use 'den/det här' more often (6%). Kept with "
                  "the dev direction; expect it to be weaker outside EP. 'den/det där' excluded "
                  "(ambiguous with locative 'där').",
        source="Danielsson (2003) [direction reversed on EP dev]",
        # disabled: replaced by register.dem_har_rate (choice assumption violated, AUC .505)
        enabled=False, cap=3.0),
    ChoiceRule(
        "starter.ska_vs_skall",
        description="'ska' vs 'skall'",
        words_a=["ska"], words_b=["skall"], label_a="ska", label_b="skall",
        # dev.500.bal: P(ska) O 10/4483 = 0.0022, T 27/1963 = 0.014 (T 1997-98 only)
        p_o=0.0022, p_t=0.014,
        direction="T",
        rationale="Literature (A2) predicts skall = T. In EP 1997-2004 house style both classes use "
                  "'skall' (>98%); the few 'ska' are in translations. Strongly diachronic (Språkrådet "
                  "recommends 'ska'; EP switched later), so low weight and a small cap.",
        source="Danielsson (2003) [direction reversed on EP dev; diachronic]",
        # disabled: replaced by register.kommer_att_vs_ska (spelling-robust future choice)
        enabled=False, weight=0.5, cap=1.0),
    RateRule(
        "starter.modal_particles",
        description="modal particles ju/nog/väl/visst",
        pattern=_MODAL_PARTICLES,
        # dev.500.bal per 1000 words (revised regexes, 2026-11-05): ju .94/.041, nog .12/.019,
        # väl .080/.011, visst .063/.014; total O 842 hits 1.2, T 60 hits .086.
        # (before the fix: ju .96/.054, nog .18/.12, väl .10/.020, visst .066/.017; total 1.3/.21)
        # Particles inside questions (EN tag questions): ju O 10 / T 0, väl O 13 / T 5 hits on
        # dev, so no separate question discount.
        # dev-train (< 2002-03-01): O 725 hits 1.5 (1997-99 2.0-2.7, 2000-02 .88-.95), T 40 .092.
        # O typed from 2000-02 (strong drift toward the test period).
        rate_o=0.92, rate_t=0.092,
        direction="O",
        rationale="Unique items: Swedish modal particles have no English trigger, so translators "
                  "rarely create them (of course -> naturligtvis, not ju).",
        source="Aijmer (1996, 2015); Tirkkonen-Condit (2004); literature A5",
        cap=3.0),
    RateRule(
        "starter.formal_connectives",
        description="formal connectives emellertid/huruvida",
        words=["emellertid", "huruvida"],
        # dev.500.bal per 1000 words: O 0.64, T 0.72 (emellertid .55/.59, huruvida .089/.13).
        # 'således' dropped: reversed on dev (O .13, T .050).
        rate_o=0.64, rate_t=0.72,
        direction="T",
        rationale="Normalisation/conservatism: translators choose formal connectives. On EP dev the "
                  "effect is ~nil (edited originals are equally formal), so DISABLED by default; kept "
                  "as a documented negative result and a template.",
        source="Danielsson (2003); Kenny (2001); literature A3",
        enabled=False, cap=1.0),
    StatRule(
        "starter.sent_len",
        description="words per sentence",
        units=sentence_lengths, unit_name="sentence",
        # dev.500.bal, words per non-vocative sentence clipped to [1, 60]:
        # O mean 18.3 sd 9.7, T mean 21.5 sd 11.6, pooled sd 10.6
        mu_o=18, mu_t=21, sd=11, unit_clip=(1, 60),
        direction="T",
        rationale="Translations from English keep long source sentences and add explicitation; the "
                  "originals are edited speech. Weak (65% alone in the literature) and the sentence "
                  "means vary by speaker more than independence assumes (text-mean sd at 500 words "
                  "~3 vs ~2 predicted), hence weight 0.5.",
        source="Volansky et al. (2015) [weak]; literature E1",
        weight=0.5, cap=2.0),
    HitRule(
        "starter.calques",
        description="English calques",
        lexicon={
            # dev.500.all hits O/T (per 1000 words O/T): 65/1356 (.093/.37)
            "calque 'det faktum att' (EN 'the fact that')": (r"\b(?:det\s+faktum|faktumet)\s+att\b", 1.3),
            # 2/143 (.0029/.039)
            "calque 'först av allt' (EN 'first of all')": (r"\bförst\s+av\s+allt\b", 2.0),
            # 7/116 (.010/.031)
            "anglicism 'spendera' (EN 'spend')": (r"\bspender(?:a|ar|ade|at|as|ades)\b", 1.1),
            # 0/20; literal readings exist, so conservative
            "calque 'på marken' (EN 'on the ground')": (r"\bpå\s+marken\b(?!\s+(?:i|vid|nära)\b)", 1.0),
        },
        direction="T",
        rationale="Interference: word-for-word renderings of English multi-word expressions where "
                  "Swedish has a native alternative (att + clause, först och främst, lägga pengar, "
                  "på plats). 'i termer av', 'på basis av', 'i ljuset av', 'implementera' were checked "
                  "and show no T lean on EP dev, so they are not included.",
        source="literature A12, A16 / §4.6 (c19, c20, c09, c25)",
        # superseded by the calques family (tsw/rules/calques.py), which absorbs these entries
        enabled=False, cap=3.0),
]

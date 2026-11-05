"""register: lexical choice and register.

Which of two native competitors a writer picked (ChoiceRule wherever a real
alternative exists), and the frequency of register-marked items that have no
frequent English trigger (RateRule). Two mechanisms:

- NORM / conservatism: translators choose the formal or dictionary variant
  (även for 'also', anser for 'think', mycket for 'very', naturligtvis for
  'of course', av for 'of') and the literal politeness formula
  ('I would like to' -> jag skulle vilja).
- Unique items / spoken register: Swedish-specific evaluative and hedging
  words that English does not trigger (tycker, väldigt, självklart, bra,
  bland annat, samt) are under-represented in translations.

Several literature directions are REVERSED on Europarl dev, because the Swedish
originals are edited transcripts written to formal norm: den/det här is a T
marker, samt and naturligtvis/självklart (as a group) are O. Those rules say
so in their rationale. Owned replacements for the starter rules
starter.dem_har_vs_denna (-> register.dem_har_rate) and starter.ska_vs_skall
(-> register.kommer_att_vs_ska), which are disabled in core_starter.py.

Calibration: python3 -m tsw.tools.rates ... --mask-topic on dev.500.bal
(O 697k / T 699k words), 2026-11-05. Values typed by hand, <= 2 s.f.
Speaker checks (--by-speaker): no rule below has a top-5 speaker share above
~35% on the O side; Wallström speaks in both classes and shows the same
direction for ska/skall (O 7.0 vs T 3.3 /1000w), även/också (also 7.6 vs 3.8
for också) and av (16 vs 20).
"""
from .base import ChoiceRule, RateRule, words_rx

FAMILY = "register"
FAMILY_CAP = 4.0

_W = r"(?<![\w-])"   # left word boundary that also refuses hyphen compounds
_E = r"(?![\w-])"    # right word boundary

# future 'kommer att' (with at most one adverb/pronoun in between) vs 'ska/skall'
_KOMMER_ATT = (_W + r"kommer\s+(?:(?:inte|också|även|nog|säkert|sannolikt|förmodligen|troligen|"
               r"troligtvis|alltid|aldrig|därför|då|nu|snart|därmed|dessutom|jag|vi|ni|de|det|den|"
               r"detta|man|han|hon)\s+)?att" + _E)
_SKA = _W + r"(?:ska|skall)" + _E

_SKULLE_VILJA = (_W + r"(?:jag|vi)\s+skulle\s+(?:vilja|önska)" + _E + "|"
                 + _W + r"skulle\s+(?:jag|vi)\s+(?:vilja|önska)" + _E)
_VILL = _W + r"(?:jag|vi)\s+vill" + _E + "|" + _W + r"vill\s+(?:jag|vi)" + _E

# opinion verbs: 'tycka' (not 'tycks' = seems) vs 'anse' (not passive 'anses' = is considered)
_TYCKA = _W + r"tyck(?:er|te|a|t)" + _E
_ANSE = _W + r"(?:anse|anser|ansåg|ansett)" + _E

# degree adverb before an adjective/adverb: 'mycket' + adjective-like word
_MYCKET_ADJ = (_W + r"mycket\s+(?:\w+(?:ig|igt|iga|lig|ligt|liga|sam|samt|samma|isk|iskt|iska|bar|bart|"
               r"bara|ad|at|ade|ande)|bra|stor\w*|viktig\w*|god\w*|svår\w*|allvarlig\w*|positiv\w*|"
               r"tydlig\w*|starkt?|hög\w*|lång\w*|liten|litet|små)" + _E)

_DEM_HAR = _W + r"(?:den|det|de|dom)\s+här" + _E

RULES = [
    ChoiceRule(
        "register.kommer_att_vs_ska",
        description="future/intention auxiliary 'kommer att' vs 'ska/skall'",
        a=_KOMMER_ATT, b=_SKA, label_a="kommer att", label_b="ska/skall",
        # dev.500.bal masked: kommer att O 1340 (1.9/1000w), T 3000 (4.3); ska+skall O 4483 (6.4),
        # T 1963 (2.8). P(kommer att) O 0.230, T 0.604; a choice in 97% / 93% of texts.
        # Per year flat (ska+skall O 6.1-6.9, T 2.4-3.7). Same speaker across classes:
        # Wallström ska/skall 7.0 (O) vs 3.3 (T) per 1000 words.
        p_o=0.23, p_t=0.60,
        direction="T",
        rationale="Swedish marks intended/planned future and obligation with 'ska' and keeps 'kommer "
                  "att' for prediction; English 'will' covers both, and translators render it with the "
                  "dictionary equivalent 'kommer att' (NORM + INT). Both spellings of ska are counted, "
                  "so the rule is robust to the house-style switch skall -> ska (which made the old "
                  "starter.ska_vs_skall a spelling detector). Risk: att-drop ('kommer bli') grows in "
                  "later originals; it is not counted as 'kommer att', which only weakens the rule.",
        source="Danielsson (2003) skall/ska keyword [re-framed]; literature A2, A17, C7; [exp]",
        cap=2.5),
    ChoiceRule(
        "register.skulle_vilja_vs_vill",
        description="'jag/vi skulle vilja' vs 'jag/vi vill'",
        a=_SKULLE_VILJA, b=_VILL, label_a="skulle vilja", label_b="vill",
        # dev.500.bal masked: skulle vilja O 196 (0.28/1000w), T 656 (0.94); vill O 1991 (2.9),
        # T 1390 (2.0). P(skulle vilja) O 0.090, T 0.32. T hits from 195 speakers, top-5 share 16%
        # (not just Commission openings); per year O 0.06-0.12, T 0.23-0.41.
        p_o=0.09, p_t=0.32,
        direction="T",
        rationale="EN 'I would like to (thank/say/stress)' is rendered literally as 'jag skulle vilja'; "
                  "the native performative is 'jag vill tacka'. Polite O speakers also use it, so the "
                  "evidence is a shift in share, not a calque detector. Overlaps calque c42: keep it "
                  "in one family only.",
        source="literature A13 / G1 / c42",
        cap=2.0),
    ChoiceRule(
        "register.tycker_vs_anser",
        description="opinion verb 'tycka' vs 'anse'",
        a=_TYCKA, b=_ANSE, label_a="tycka", label_b="anse",
        # dev.500.bal masked: tycker/tyckte/tycka/tyckt O 928 (1.3/1000w), T 156 (0.22);
        # anse/anser/ansåg/ansett O 779 (1.1), T 753 (1.1). P(tycka) O 0.54, T 0.17.
        # O hits from 68 speakers (top-5 34%); per year O 0.9-1.9, T 0.14-0.33 per 1000 words.
        # 'tycks' (= seems) excluded (O 0.08, T 0.11); 'jag tror' is also O-leaning on dev
        # (0.76 vs 0.35), against the literature, so 'tro' is not put on the T side.
        p_o=0.54, p_t=0.17,
        direction="O",
        rationale="'tycka' (personal evaluative opinion) is a Swedish-specific distinction with no "
                  "English counterpart; EN 'think/believe/consider/feel' is rendered with the formal "
                  "'anse'. Danielsson's O cluster 'tycker att det här'.",
        source="literature A8; Danielsson (2003) O cluster",
        cap=2.0),
    ChoiceRule(
        "register.valdigt_vs_mycket",
        description="degree adverb 'väldigt' vs 'mycket' + adjective",
        a=_W + r"väldigt" + _E, b=_MYCKET_ADJ, label_a="väldigt", label_b="mycket + adj",
        # dev.500.bal masked: väldigt O 536 (0.77/1000w), T 110 (0.16); mycket + adjective O 874
        # (1.3), T 1045 (1.5). P(väldigt) O 0.38, T 0.095. O hits from 52 speakers, top-5 35%.
        # väldigt declines in O over 1997-2004 (1.4 -> 0.3 /1000w): small cap.
        p_o=0.38, p_t=0.095,
        direction="O",
        rationale="EN 'very' -> the formal 'mycket'; the spoken-register 'väldigt' is a native choice "
                  "translators avoid (NORM + SPK). Diachronic: väldigt is falling in edited EP "
                  "originals, so the cap is small.",
        source="Danielsson (2003) O keyword väldigt; literature A4, A21",
        cap=1.5),
    ChoiceRule(
        "register.sjalvklart_vs_naturligtvis",
        description="'självklart' vs 'naturligtvis/givetvis/självfallet'",
        a=_W + r"självklart" + _E, b=_W + r"(?:naturligtvis|givetvis|självfallet)" + _E,
        label_a="självklart", label_b="naturligtvis/givetvis/självfallet",
        # dev.500.bal masked: självklart O 208 (0.30/1000w), T 30 (0.043); naturligtvis/givetvis/
        # självfallet O 693 (0.99), T 349 (0.50). P(självklart) O 0.23, T 0.079. O hits from 41
        # speakers (top-5 28%); T from 26 speakers.
        p_o=0.23, p_t=0.079,
        direction="O",
        rationale="EN 'of course / naturally' maps onto the dictionary equivalents 'naturligtvis/"
                  "givetvis'; 'självklart' (adjective-derived, more colloquial) is the native choice. "
                  "The literature expects naturligtvis = T; on EP the whole group is more frequent in "
                  "originals, so only the choice within the group is used.",
        source="literature A7 / G10 [direction refined on EP dev]",
        cap=1.5),
    ChoiceRule(
        "register.aven_vs_ocksa",
        description="additive 'även' vs 'också'",
        words_a=["även"], words_b=["också"], label_a="även", label_b="också",
        # dev.500.bal masked: även O 1323 (1.9/1000w), T 1099 (1.6); också O 3722 (5.3), T 1547 (2.2).
        # P(även) O 0.26, T 0.42 (per year O 0.16-0.33 rising, T 0.30-0.52). Wallström (both
        # classes): också 7.6 (O) vs 3.8 (T). The rate of 'också' carries more than the share
        # (see register.ocksa_rate, disabled) but has no clean linguistic story.
        p_o=0.26, p_t=0.42,
        direction="T",
        rationale="Two native renderings of EN 'also/too'; translators prefer the formal written "
                  "'även' (NORM, like skall/denna/emellertid in Danielsson), originals the neutral "
                  "'också'. P(även) rises in originals over time, so weight 0.7.",
        source="Danielsson (2003) NORM mechanism; literature §3; [exp]",
        weight=0.7, cap=1.5),
    RateRule(
        "register.dem_har_rate",
        description="demonstrative 'den/det/de här'",
        pattern=_DEM_HAR, label="'{m}'",
        # dev.500.bal masked: O 117 (0.17/1000w, 6.7% of texts), T 631 (0.90, 22.6%).
        # denna/detta/dessa: O 12.5, T 14.0 (also higher in T, so a choice against them fails:
        # starter.dem_har_vs_denna had AUC .505). T top-5 speakers 23% but bursty (one speaker
        # 3.3/1000w), hence weight 0.5.
        rate_o=0.17, rate_t=0.90,
        direction="T",
        rationale="Literature (A1) predicts den här = O. EP originals are edited to written norm "
                  "(denna/detta), while translators of oral English 'this' often keep the spoken "
                  "'den/det här'. Direction reversed on EP; expect it to be weaker or flip outside EP.",
        source="Danielsson (2003) [direction reversed on EP dev]; docs/ENGINE.md findings",
        # cap 1.0: starter error analysis showed O speakers who say 'den/det här' saturating the
        # old ChoiceRule (+3.0). Only the 'här' variant is counted, so formal 'detta' and fixed
        # formulas ('denna kammare' = EP 'this House') no longer push toward O.
        weight=0.5, cap=1.0),
    RateRule(
        "register.bland_annat",
        description="'bland annat / bl.a.'",
        pattern=_W + r"bland\s+annat" + _E + "|" + _W + r"bl\.\s?a\.",
        label="'{m}'",
        # dev.500.bal masked: O 398 (0.57/1000w, 24% of texts), T 24 (0.034, 1.7%).
        # O from 65 speakers, top-5 31%; per year O 0.29-0.90, T 0.00-0.08.
        rate_o=0.57, rate_t=0.034,
        direction="O",
        rationale="Unique item: 'bland annat' is Swedish's routine marker of a non-exhaustive list; "
                  "English marks it differently ('including', 'such as', rare 'inter alia'), which "
                  "translators render 'inklusive/däribland/t.ex.' (inklusive/däribland O 0.11 vs T "
                  "0.30 /1000w).",
        source="Tirkkonen-Condit (2004) unique items; [exp]",
        cap=2.0),
    RateRule(
        "register.samt",
        description="coordinator 'samt'",
        words=["samt"],
        # dev.500.bal masked: O 542 (0.78/1000w, 29% of texts), T 186 (0.27, 12%). O from 63
        # speakers, top-5 29%; per year O 0.44-1.1, T 0.16-0.51.
        rate_o=0.78, rate_t=0.27,
        direction="O",
        rationale="Unique item: formal Swedish 'samt' ('and also', list-final) has no distinct English "
                  "trigger; EN 'and' -> 'och'. Literature A25 guessed T (bureaucratese); on EP dev it is "
                  "an O marker.",
        source="Tirkkonen-Condit (2004) unique items; literature A25 [direction reversed]",
        cap=1.5),
    RateRule(
        "register.bra",
        description="evaluative 'bra'",
        words=["bra"],
        # dev.500.bal masked: O 866 (1.2/1000w, 41% of texts), T 264 (0.38, 16%). O from 69
        # speakers, top-5 27%; per year O 1.0-1.6, T 0.26-0.44. (god/gott/goda O 0.65, T 0.41.)
        rate_o=1.2, rate_t=0.38,
        direction="O",
        rationale="Spoken/neutral evaluative 'bra' ('det är bra att', 'ett bra betänkande') is the "
                  "native default; translators render EN 'good/welcome/positive' with more formal or "
                  "more specific adjectives.",
        source="literature A4 (bra as evaluative) [exp]",
        cap=1.5),
    RateRule(
        "register.av_rate",
        description="preposition 'av'",
        words=["av"],
        # dev.500.bal masked: O 9468 (14/1000w), T 13176 (19). Present in ~100% of texts.
        # Wallström (both classes): 16 (O) vs 20 (T). Overdispersed (speaker/topic nominal
        # style), hence weight 0.5.
        rate_o=14, rate_t=19,
        direction="T",
        rationale="EN 'X of Y' and agentive 'by' are rendered with 'av' where Swedish prefers a "
                  "compound or s-genitive (INT).",
        source="Danielsson (2003) T keyword av; Nilsson (2004); literature A6",
        weight=0.5, cap=1.5),

    # ---- disabled alternatives (documented; compare with Engine(include_disabled=True)) ----
    RateRule(
        "register.skall_rate",
        description="'ska/skall' rate",
        pattern=_SKA,
        # dev.500.bal masked: O 6.4, T 2.8 per 1000 words. Subsumed by register.kommer_att_vs_ska.
        rate_o=6.4, rate_t=2.8,
        direction="O",
        rationale="Same mechanism as register.kommer_att_vs_ska, as a rate (alternative formulation).",
        source="Danielsson (2003) [re-framed]",
        weight=0.5, cap=1.5, enabled=False),
    RateRule(
        "register.ocksa_rate",
        description="'också' rate",
        words=["också"],
        # dev.500.bal masked: O 5.3, T 2.2 per 1000 words (90% / 60% of texts).
        rate_o=5.3, rate_t=2.2,
        direction="O",
        rationale="Alternative to register.aven_vs_ocksa. Disabled: the 2.4x rate difference is "
                  "larger than any 'also' shift EN would explain and lacks a clear mechanism.",
        source="[exp]",
        weight=0.5, cap=1.5, enabled=False),
    RateRule(
        "register.hedges",
        description="hedging/degree 'ganska', 'faktiskt'",
        words=["ganska", "faktiskt"],
        # dev.500.bal masked: ganska O 0.43, T 0.22; faktiskt O 0.62, T 0.38; together 1.05 / 0.61.
        rate_o=1.0, rate_t=0.61,
        direction="O",
        rationale="Spoken-register hedges (A4 ganska, A22 faktiskt). Weak (ln ratio -0.5); disabled "
                  "unless it adds on 500-word texts.",
        source="literature A4, A22 [exp]",
        cap=1.0, enabled=False),
]

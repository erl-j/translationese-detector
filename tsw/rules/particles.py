"""particles: modal and discourse particles, stance adverbs, Swedish-specific
particle verbs, and the pronoun profile (generic *man*, copular *det*,
possessive determiners, address *ni*).

Two mechanisms meet here.
- Unique items (Tirkkonen-Condit 2004): Swedish resources with no
  straightforward English trigger (modal particles, *nämligen*, *ändå*,
  anaphoric *då*, generic *man*, idiomatic particle verbs such as *slå fast*)
  are under-produced in translation -> presence = O evidence, absence (over
  long texts) = weak T evidence.
- Interference in the pronoun system: English possessive determiners
  (*my/our/his/its/their*) and address *you* are obligatory where Swedish uses
  a definite noun or a title, so translations carry more *vår, hans, dess,
  deras, ni/er*; English *of course* is rendered by the dictionary default
  *naturligtvis* rather than *självklart/givetvis/förstås*.

starter.modal_particles already covers *ju/nog/väl/visst*; nothing here
matches those words. Spoken-vs-written confound (literature §5.8): most O
rules here are also spoken-register markers, so they would over-fire on
informal written Swedish; the family cap keeps the group bounded.

Calibration: python3 -m tsw.tools.rates ... on dev.500.bal (O 697k / T 699k
words), 2026-11-05, then RE-CHECKED on dev-train only (sessions < 2002-03-01,
1823 chunks) after the trn/val protocol change; the typed values follow
dev-train ('dev-train:' comments). For drifting O items the 2000-02 dev-train
rate is typed (closer to the test period). All values typed by hand, <= 2 significant digits.
Checked and NOT used (see report): *liksom* (O .24 / T .24, no particle use in
EP), *det finns* (1.3/1.3), *sin/sitt/sina* vs *deras* after a 3rd-person
subject pronoun in the same clause (P(deras) .12/.13: translators do not get
the reflexive wrong), *ens* (.087/.096), *när/eftersom/sedan* (no difference),
*jag* (12/13), *du* (rare), generic *en* (0 hits).
"""
from .base import ChoiceRule, RateRule

FAMILY = "particles"
FAMILY_CAP = 4.0

_B = r"(?<![\w-])"   # left word boundary that also excludes hyphen compounds
_E = r"(?![\w-])"    # right word boundary

_FIN = (r"(?:är|var|har|hade|kan|kunde|ska|skall|blir|blev|finns|fanns|måste|bör|borde|vill|ville|"
        r"gör|gjorde|går|skulle|kommer|får|fick|menar|anser|tycker|tror|talar|handlar|gäller|vet|"
        r"ser|behöver|står|ligger)")
_SUBJ = r"(?:det|den|vi|ni|de|han|hon|man|jag|du|detta|denna|dessa)"

RULES = [
    # ------------------------------------------------------------------ particles
    RateRule(
        "particles.namligen",
        description="clause-internal particle 'nämligen' (explanatory 'you see')",
        # only the particle use (det är nämligen så att ...), not the appositive
        # ', nämligen X' (= EN 'namely'), which has an English trigger.
        pattern=_B + r"(?<![,:;–(\-]\s)(?<![,:;–(\-])nämligen" + _E,
        # dev.500.bal: O 284 hits .41/1000, T 15 hits .021/1000 (ln -2.9/hit); 60 O speakers,
        # top-5 share 32% (pooled with stance group). Appositive ', nämligen': .50 / .12 (not used).
        # dev-train: O 210 .42, T 4 hits .0092 (too few: judgement .02, conservative)
        rate_o=0.42, rate_t=0.02,
        direction="O",
        rationale="UNQ: the explanatory modal particle 'nämligen' (Jag anser nämligen att ...) has no "
                  "English counterpart (EN 'because/for' or nothing), so translators almost never "
                  "create it; the appositive 'namely' use is excluded.",
        source="literature A5, D2; Tirkkonen-Condit (2004); Aijmer (1996)",
        cap=2.5),
    RateRule(
        "particles.anda",
        description="concessive particle 'ändå'",
        words=["ändå"],
        # dev.500.bal: O 396 .57/1000, T 129 .18/1000 (ln -1.1); top-5 speaker share O 30%, T 15%.
        # by year O 1997-99 .65-.82, 2000-04 .45-.52; T flat .12-.23.
        # dev-train: O .58 (1997-99 .65-.82, 2000-02 .27-.52 ~.45), T .18; O typed from 2000-02
        rate_o=0.45, rate_t=0.18,
        direction="O",
        rationale="UNQ + SPK: 'ändå' is both a concessive adverb and a modal particle of shared "
                  "expectation (Det är ändå viktigt); EN 'still/yet/nevertheless' are often rendered "
                  "by 'dock/emellertid/trots detta', and the particle use has no trigger. O rate "
                  "typed from 2000-04 (drift toward the test period).",
        source="literature A5, D2; Aijmer (1996)",
        cap=2.0),
    RateRule(
        "particles.alltsa",
        description="inferential particle 'alltså'",
        words=["alltså"],
        # dev.500.bal: O 322 .46/1000, T 78 .11/1000 (ln -1.4); top-5 share O 29% (Wibe 2.7/1000).
        # strong drift in O: 1997-98 .66/1.06, 2000-04 .21-.43 (~.33); T .05-.18.
        # dev-train: O .53 (2000-02 .33-.47 ~.40), T .13; O typed from 2000-02
        rate_o=0.40, rate_t=0.13,
        direction="O",
        rationale="UNQ + SPK: 'alltså' as a resumptive/inferential particle (Problemet är alltså ...) "
                  "is a spoken-Swedish discourse marker; translators use it only for EN 'thus/so' and "
                  "prefer 'således/därför'. O rate typed from 2000-04 because of drift; low cap.",
        source="literature A5 (lower weight advised), D2",
        cap=1.5),
    RateRule(
        "particles.stance",
        description="stance adverbs 'egentligen/visserligen/i och för sig/i alla fall'",
        pattern={"stance adverb '{m}'":
                 _B + r"(?:egentligen|visserligen|i\s+och\s+för\s+sig|i\s+alla\s+fall)" + _E},
        # dev.500.bal: O 370 .53/1000, T 78 .11/1000 (ln -1.6); egentligen .34/.072,
        # visserligen .080/.023, i alla fall .075/.017, i och för sig .034/0.
        # dev-train: O 256 .51, T 51 .12
        rate_o=0.51, rate_t=0.12,
        direction="O",
        rationale="UNQ: Swedish-specific stance/concession markers ('egentligen' = 'actually, "
                  "when it comes down to it'; 'visserligen ... men'; 'i och för sig') with no stable "
                  "English trigger; EN 'actually/admittedly/in any case' map to other items.",
        source="literature A22 (egentligen O), D2 / 4.3b; Tirkkonen-Condit (2004)",
        cap=2.0),
    RateRule(
        "particles.da_anaphoric",
        description="anaphoric/inferential 'då' ('then, in that case')",
        pattern={
            # (sentence-initial 'Då' + finite verb is NOT counted here since the integration:
            # wordorder.fronted_theme scores it as a topicalised adverb, INV_ADV)
            # inverted: 'finns det då', 'skall vi då'
            "verb + subject + 'då'": _B + _FIN + r"\s+" + _SUBJ + r"\s+då" + _E,
            # straight: 'det är då', 'vi får då' (not 'då' + subject = subjunction)
            "subject + verb + 'då'": _B + _SUBJ + r"\s+" + _FIN + r"\s+då" + _E
                                     + r"(?!\s+" + _SUBJ + r"\b)",
        },
        # dev.500.bal: O 259 .37/1000, T 52 .074/1000 (ln -1.6). Initial 'Då'+FIN .19/.016,
        # FIN SUBJ då .080/.037, SUBJ FIN då .10/.020. Flat over years (O .80-.94 incl. all då-forms
        # per 1000? see report). All 'då' .136/.074; initial 'Då' + subject = subjunction (T-ish).
        # dev-train: O 193 .39 (2000-02 .20-.35 ~.28), T 29 .067; O typed near 2000-02
        # Integration (2026-11-06), without sentence-initial 'Då' + FIN, trn.500.bal masked:
        # O 129 .19 (2000-04 .12-.18 ~.15), T 41 .059; O typed from 2000-04 as before.
        rate_o=0.15, rate_t=0.059,
        direction="O",
        rationale="UNQ + SPK: 'då' as an anaphoric connective/particle linking to the previous "
                  "statement (Då måste vi ...; Vad finns det då för ...) is typical of Swedish "
                  "argumentation; English mostly has no word or 'so/then'. The temporal/causal "
                  "subjunction ('Då vi ...', ', då de ...') is excluded by requiring a finite verb "
                  "right after sentence-initial 'Då' and no subject after post-verbal 'då'.",
        source="literature A5 (då), 4.3b; [exp] for the contexts",
        cap=2.0),
    ChoiceRule(
        "particles.of_course",
        description="'of course': 'naturligtvis' vs 'givetvis/förstås'",
        a=_B + r"naturligtvis" + _E,
        # Integration: 'självklart' removed from B; register.sjalvklart counts it (rate, O)
        b=_B + r"(?:givetvis|förstås(?!\s+som\b))" + _E,
        label_a="naturligtvis", label_b="givetvis/förstås",
        # dev.500.bal: P(naturligtvis) O 506/923 = .55, T 306/393 = .78. Rates: naturligtvis
        # .73/.44, others .60/.12. O speakers 66 (top-5 27%). P(A) by year O .42-.69, T .59-.91.
        # dev-train: P(naturligtvis) O 365/668 = .55, T 231/299 = .77
        # Integration (2026-11-06), B without 'självklart', trn.500.bal masked: P(naturligtvis)
        # O 506/713 = .71, T 306/362 = .85 (per year O .55-.95, T .66-.97).
        p_o=0.71, p_t=0.85,
        direction="T",
        rationale="Literature A7 predicts naturligtvis = T (EN 'of course' -> dictionary default "
                  "'naturligtvis'). On EP dev 'naturligtvis' is NOT more frequent in T, but the "
                  "choice among the 'of course' adverbs is: native speakers spread over "
                  "självklart/givetvis/förstås. Gives T presence evidence.",
        source="literature A7; Aijmer (2013)",
        cap=1.5),
    RateRule(
        "particles.dvs",
        description="reformulation marker 'dvs.' / 'det vill säga'",
        pattern=_B + r"(?:dvs\.?|d\.v\.s\.?|det\s+vill\s+säga)" + r"(?![\w-])",
        # dev.500.bal: O 418 .60/1000, T 100 .14/1000 (ln -1.4); dvs.=437, det vill säga=70.
        # by year O 1997-98 ~1.0, 2000-04 .32-.67 (~.48); T .06-.24.
        # dev-train: O 332 .67 (2000-02 .45-.67 ~.55), T 64 .15; O typed from 2000-02
        rate_o=0.55, rate_t=0.15,
        direction="O",
        rationale="Swedish speakers reformulate and specify on the fly with 'dvs.' (Swedish speech "
                  "and edited speech); the English source rarely has 'that is/i.e.' at those points, "
                  "so translators seldom add it. Pointed out by the error analysis of the starter "
                  "engine. Possible overlap with a connectives family: small cap. O rate from 2000-04.",
        source="[exp] (coordinator error analysis); literature D2 (O-leaning discourse markers)",
        cap=1.5),
    # ------------------------------------------------------------------ particle verbs
    RateRule(
        "particles.particle_verbs",
        description="Swedish-specific particle verbs (slå fast, ställa sig bakom, lyfta fram, ...)",
        pattern={"particle verb '{m}'": _B + (
            r"(?:(?:slå|slår|slog|slagit|slås|slogs|slagits)\s+(?:\w+\s+)?(?:fast|vakt\s+om)"
            r"|(?:ställa|ställer|ställde|ställt|stå|står|stod|stått)\s+(?:\w+\s+)?(?:sig\s+)?bakom"
            r"|(?:lyfta|lyfter|lyfte|lyft|lyfts)\s+(?:\w+\s+)?fram"
            r"|(?:gå|går|gick|gått)\s+ut\s+på"
            r"|(?:ta|tar|tog|tagit)\s+(?:\w+\s+)?tag\s+i"
            r"|(?:sätta|sätter|satte|satt|dra|drar|drog|dragit)\s+(?:\w+\s+)?igång"
            r"|(?:komma|kommer|kom|kommit)\s+(?:\w+\s+)?fram\s+till)") + _E},
        # list fixed a priori (literature 4.7, items without an English phrasal-verb image).
        # dev.500.bal: O 370 .53/1000, T 140 .20/1000 (ln -0.97); O 60 speakers top-5 27%;
        # by year O .40-.64, T .11-.28. (The 4.7 items WITH an English image go the other way:
        # ta itu med .052/.37, lägga fram .43/.78, ta upp .59/.79.)
        # dev-train: O 253 .51, T 76 .18
        rate_o=0.51, rate_t=0.18,
        direction="O",
        rationale="UNQ: idiomatic particle verbs whose image has no English counterpart (slå fast "
                  "'establish', ställa sig bakom 'endorse', lyfta fram 'highlight', gå ut på 'aim "
                  "at') are produced by native speakers but not triggered by the English Latinate "
                  "verb, which translators render with a single verb (fastställa, stödja, betona).",
        source="literature 4.7, A16 note; Tirkkonen-Condit (2004) [exp for the item list]",
        cap=1.5),
    RateRule(
        "particles.ta_itu_med",
        description="'ta itu med' (EN tackle/address/deal with)",
        pattern=_B + r"(?:ta|tar|tog|tagit|tas|togs|tagits)\s+(?:\w+\s+)?itu\s+med" + _E,
        # dev.500.bal: O 36 .052/1000, T 262 .37/1000 (ln +2.0); T 113 speakers, top-5 24%
        # (Patten 31); by year T flat.
        # dev-train: O 26 .052, T 158 .36
        rate_o=0.052, rate_t=0.36,
        direction="T",
        rationale="Translation-equivalent over-use: 'ta itu med' is the standard rendering of EN "
                  "'tackle/address/deal with', which are very frequent in EP English; native "
                  "speakers spread over hantera, komma till rätta med, ta tag i, lösa.",
        source="[exp]; literature 4.7 (particle verbs; direction reversed for this item)",
        cap=2.0),
    # ------------------------------------------------------------------ pronouns
    RateRule(
        "particles.generic_man",
        description="generic pronoun 'man'",
        # not the noun 'en/den man', 'man och kvinna'
        pattern=_B + r"(?<!en )(?<!den )man" + _E + r"(?!\s+och\s+kvinn)",
        # dev.500.bal (masked): O 3062 4.4/1000, T 1687 2.4/1000 (ln -0.60); O 92 speakers top-5
        # 32%, T 172 speakers top-5 19%. Drift in O: 1997-99 ~6, 2000-04 3.2-4.1 (~3.7); T flat 2.4.
        # Overdispersed (texts with >= 1 hit: O 76% vs 89% Poisson) -> weight 0.5.
        # dev-train: O 4.7 (1997-99 ~6, 2000-02 3.0-4.0 ~3.5), T 2.4; O typed from 2000-02
        rate_o=3.5, rate_t=2.4,
        direction="O",
        # Integration: weight 0.5 -> 0.25. particles.ni_vs_man counts every 'man' as its B
        # variant, so each 'man' was O evidence twice (within-class r of the two contributions
        # .68 O / .44 T on trn.100); halving the rate rule leaves the choice rule in charge.
        rationale="UNQ: 'man' is the Swedish generic pronoun; English uses you/we/one/people/passive, "
                  "which translators render as vi, ni, en, människor or a passive. O rate typed from "
                  "2000-04 because of drift; bursty per speaker, hence weight 0.5.",
        source="literature A9; Tirkkonen-Condit (2004)",
        weight=0.25, cap=1.5),
    RateRule(
        "particles.copular_det",
        description="copular 'det är/var' and 'är/var det' (expletive, cleft, anaphoric det)",
        # Integration: sentence-initial 'Det är/var' (also after Men/Och) is not counted:
        # wordorder.subject_theme scores that opening (S_DET, CLEFT). Inverted 'är det' stays.
        pattern=_B + r"(?:(?<!^)(?<![.!?] )(?<![.!?]\" )(?<!^men )(?<!^och )(?<![.!?] men )"
                     r"(?<![.!?] och )det\s+(?:är|var)|(?:är|var)\s+det)" + _E,
        # dev.500.bal (masked): O 5089 7.3/1000, T 3349 4.8/1000 (ln -0.42); O 95 speakers top-5
        # 27%; by year O 8.6 (1997-99) -> 6.3-7.4 (2000-04), T 5.5 -> 3.9-4.7 (both drift down).
        # 'det är ADJ att' 1.2/.62, cleft 'det är ... som' .81/.48; 'detta/det här är' 1.0/1.3 (T).
        # dev-train: O 7.5 (2000-02 ~7.0), T 5.0 (2000-02 ~4.7)
        # Integration (2026-11-06), without sentence-initial 'Det är/var', trn.500.bal masked:
        # O 2661 3.8 (2000-02 3.3-3.9 ~3.5), T 1682 2.4 (2000-02 ~2.3); typed from 2000-02 as before.
        rate_o=3.5, rate_t=2.3,
        direction="O",
        rationale="Swedish fills the prefield and the subject slot with low-information 'det' "
                  "(extraposition 'Det är viktigt att', clefts 'Det är vi som', anaphoric 'Det är "
                  "bra'); English 'this is / we must / it is' renderings give 'detta är' or a "
                  "personal subject in translation. High-frequency item: weight 0.5.",
        source="literature B2 (det-initial, direction [exp]), B3; [exp]",
        weight=0.5, cap=1.5),
    RateRule(
        "particles.poss_1st",
        description="1st-person possessive determiners 'min/mitt/mina/vår/vårt/våra'",
        pattern=_B + r"(?:min|mitt|mina|vår|vårt|våra)" + _E,
        # dev.500.bal (masked): O 2652 3.8/1000, T 3906 5.6/1000 (ln +0.38); top-5 share O 27%,
        # T 18% (Patten 7/1000); by year flat (O 3.5-4.3, T 5.1-6.2). texts >= 1: O 78% (Poisson 85%).
        # dev-train: O 3.7, T 5.7
        rate_o=3.7, rate_t=5.7,
        direction="T",
        rationale="INT: English possessive determiners are near-obligatory (our citizens, my "
                  "country, our industry) where Swedish uses the definite form or a compound "
                  "(medborgarna, hemlandet, läkemedelsindustrin); translators keep 'vår/min'.",
        source="literature A11 (pronoun profile, [exp] direction); Volansky et al. (2015) pronouns",
        weight=0.6, cap=1.5),
    RateRule(
        "particles.poss_3rd",
        description="3rd-person non-reflexive possessives 'hans/hennes/deras/dess'",
        # not 'innan/sedan/till dess' (temporal 'until/since then')
        pattern=_B + r"(?:hans|hennes|deras|(?<!innan )(?<!sedan )(?<!till )dess)" + _E,
        # dev.500.bal (masked): O 755 1.1/1000, T 1946 2.8/1000 (ln +0.94); O 69 speakers top-5 29%,
        # T 192 speakers top-5 16%; by year O .86-1.4, T 2.4-3.0. Reflexive sin/sitt/sina 2.3/2.7.
        # dev-train: O 486 .98, T 1203 2.8
        rate_o=0.98, rate_t=2.8,
        direction="T",
        rationale="INT: EN his/her/its/their are rendered as hans/hennes/dess/deras where Swedish "
                  "would use the definite form, a compound, or reflexive 'sin'. A cheaper 'sin vs "
                  "deras' choice (same clause as a 3rd-person subject) showed no difference "
                  "(professional translators get the reflexive right), so a rate is used.",
        source="literature A20 (as a rate), A11",
        weight=0.8, cap=2.0),
    RateRule(
        "particles.han_hon",
        description="3rd-person personal pronouns 'han/hon'",
        words=["han", "hon"],
        # trn (dev <= 2004-06, masked): O 552 .79/1000, T 1260 1.8/1000 (ln +0.82); dev-train
        # subset .77/2.0; T 181 speakers, top-5 share 15%. Bursty (texts >= 1 hit: O 24% vs 32%
        # Poisson) -> weight 0.5. Also flagged by the hybrid-ML coefficients (dev-train fit).
        rate_o=0.79, rate_t=1.8,
        direction="T",
        rationale="INT/explicitation: English debate speakers refer back to the rapporteur, the "
                  "Commissioner or a colleague with he/she, kept as han/hon in translation; Swedish "
                  "speakers more often repeat the title (föredraganden, kommissionären) or use the "
                  "name. Turn-structure dependent (replies to named members): low weight.",
        source="literature A11 (pronoun profile, direction [exp]); hybrid-ML coefficients",
        weight=0.5, cap=1.0),
    ChoiceRule(
        "particles.ni_vs_man",
        description="'ni' vs generic 'man'",
        a=_B + r"ni" + _E,
        b=_B + r"(?<!en )(?<!den )man" + _E + r"(?!\s+och\s+kvinn)",
        label_a="ni", label_b="man",
        # dev.500.bal (masked): P(ni) O 540/3559 = .15, T 897/2545 = .35; by year O .10-.21
        # (rising: 2001-04 ~.19), T .23-.50. p_o typed .17 for drift.
        # dev-train: P(ni) O 371/2691 = .14 (2000-02 .13-.20 ~.17), T .38 (2000-02 ~.33)
        p_o=0.17, p_t=0.35,
        direction="T",
        rationale="INT: EN generic 'you' (You can then go down to ...) is rendered 'ni' where a "
                  "native speaker uses generic 'man'; EN address 'you' adds more 'ni'. Conditioning "
                  "on the ni/man choice is less sensitive to how much a text addresses someone than "
                  "a raw 'ni' rate, though EP address still inflates 'ni' in both classes.",
        source="literature A9, A11; [exp] (coordinator error analysis)",
        cap=1.5),
    RateRule(
        "particles.address_ni",
        description="2nd-person plural/polite address 'ni/er/ert/era'",
        pattern=_B + r"(?:ni|er|ert|era)" + _E,
        # dev.500.bal (masked): O 840 1.2/1000, T 1591 2.3/1000 (ln +0.64); O 52 speakers top-5 39%,
        # T 156 speakers top-5 20% (Cox, the EP President, 6.7/1000). Very bursty: texts >= 1 hit
        # O 26% (Poisson 45%), T 39% (Poisson 68%) -> weight 0.3, cap 1.
        rate_o=1.2, rate_t=2.3,
        direction="T",
        rationale="INT: EN 'you' (to the Commissioner, the President, colleagues) is rendered 'ni/er'; "
                  "Swedish speakers tend to avoid the polite 'ni' and address by title "
                  "(kommissionären, rådsordföranden). Strongly speaker/turn dependent: low weight.",
        source="literature A11; [exp]",
        # dev.100 masked (2000 texts, Default): AUC .7926 with, .7930 without; superseded by the
        # ni/man choice (particles.ni_vs_man), kept as a documented alternative.
        enabled=False, weight=0.3, cap=1.0),
]

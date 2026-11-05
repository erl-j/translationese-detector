"""wordorder family: detector minimal pairs, metamorphic tests, spans.

The minimal pairs are hand-written in the spirit of Swe-BLiMP (V2, V3,
embedded-clause adverb placement); the Swe-BLiMP sample in BirgerMoell/complin
has no licence, so no items are copied from it.
"""
import unittest

from tsw.doc import Doc
from tsw.engine import Families
from tsw.rules import wordorder as W


def cat(sentence):
    return W.classify_opening([t.lower() for t in Doc(sentence).sent_tokens[0]]).cat


def fam_score(text):
    return Families(only="wordorder").score(text)


# (native V2 sentence, its category) / (non-native or English-order twin, category)
MINIMAL_PAIRS = [
    ("I dag har vi ett stort problem.", "INV_PP", "I dag vi har ett stort problem.", "NOINV"),
    ("I dag har vi ett stort problem.", "INV_PP", "I dag, vi har ett stort problem.", "NOINV"),
    ("Därför anser jag att förslaget är dåligt.", "INV_ADV", "Därför jag anser att förslaget är dåligt.", "NOINV"),
    ("Dessutom måste vi tänka på miljön.", "INV_ADV", "Dessutom, vi måste tänka på miljön.", "NOINV"),
    ("Slutligen vill jag tacka föredraganden.", "INV_CONJ", "Slutligen, jag vill tacka föredraganden.", "NOINV"),
    ("För det första måste vi agera.", "INV_CONJ", "För det första, vi måste agera.", "NOINV"),
    ("Med andra ord är de tvungna att ansluta sig.", "INV_PP", "Med andra ord, de är tvungna att ansluta sig.", "NOINV"),
    ("Med anknytning till detta har utvecklingen varit positiv.", "INV_PP",
     "Med anknytning till detta, utvecklingen har varit positiv.", "NOINV"),
    ("Om vi inte agerar nu, kommer vi att förlora.", "INV_SUB", "Om vi inte agerar nu, vi kommer att förlora.", "NOINV"),
    ("Vi vill också tacka kommissionen.", "S_PRON", "Vi också vill tacka kommissionen.", "V3"),
    ("Jag hoppas verkligen att det går bra.", "S_PRON", "Jag verkligen hoppas att det går bra.", "V3"),
]


class TestMinimalPairs(unittest.TestCase):
    def test_pairs(self):
        for good, cg, bad, cb in MINIMAL_PAIRS:
            with self.subTest(good=good):
                self.assertEqual(cat(good), cg)
            with self.subTest(bad=bad):
                self.assertEqual(cat(bad), cb)

    def test_violation_rule_fires_only_on_bad(self):
        r = next(x for x in W.RULES if x.name == "wordorder.v2_violation")
        for good, _, bad, _ in MINIMAL_PAIRS:
            with self.subTest(good=good):
                self.assertEqual(r.contribution(Doc(good)), 0.0)
                self.assertGreater(r.contribution(Doc(bad)), 0.0)

    def test_embedded_negation(self):
        r = next(x for x in W.RULES if x.name == "wordorder.v2_violation")
        native = ["Jag beklagar att vi inte kan träffa honom.", "Det beror på att det inte finns pengar.",
                  "Om vi inte gör något händer ingenting.",
                  "Vi måste se till att det är inte bara staterna som ansvarar."]  # constituent negation
        nonnative = ["Jag beklagar att vi kan inte träffa honom.", "Det beror på att det finns inte pengar.",
                     "Jag måste säga att jag har inte mottagit något.", "Han sa att han kunde aldrig komma."]
        for s in native:
            with self.subTest(s=s):
                self.assertEqual(r.contribution(Doc(s)), 0.0)
        for s in nonnative:
            with self.subTest(s=s):
                self.assertAlmostEqual(r.contribution(Doc(s)), W.V2_VIOLATION_HIT["EMB"])


class TestOpenings(unittest.TestCase):
    CASES = [
        ("Det vet vi.", "INV_OBJ"),
        ("Det tycker jag är bra.", "INV_OBJ"),
        ("Detta kan vi inte acceptera.", "INV_OBJ"),
        ("Det här kan vi inte godta.", "INV_OBJ"),
        ("Så är det.", "INV_OBJ"),
        ("Vi vet det.", "S_PRON"),
        ("Det är viktigt att vi agerar.", "S_DET"),
        ("Det finns många problem.", "S_DET"),
        ("Det är vi som måste betala.", "CLEFT"),
        ("Detta är ett viktigt betänkande.", "S_DEM"),
        ("Det här är ett viktigt betänkande.", "S_DEM"),
        ("Kommissionen har lagt fram ett förslag.", "S_NP"),
        ("Den frågan har vi diskuterat länge.", "INV_NP"),
        ("De mål vi sätter upp skall vi leva upp till.", "INV_NP"),
        ("Vad vi behöver är en ny strategi.", "PSEUDO_CLEFT"),
        ("Nu har vi chansen.", "INV_ADV"),
        ("Då måste parlamentet veta skälen.", "INV_ADV"),
        ("Tyvärr misslyckades de.", "INV_ADV"),               # verb guessed from morphology
        ("Personligen anser jag att det är fel.", "INV_ADV"),  # -ligen adverb
        ("När det gäller jordbruket vill jag säga något.", "INV_SUB"),
        ("Eftersom vi inte har pengar kan vi inte göra något.", "INV_SUB"),
        ("För att uppnå detta måste vi samarbeta.", "INV_SUB"),
        ("Om vi inte gör något, så kommer det att gå illa.", "SUB_SA"),
        ("Enligt min mening är detta fel.", "INV_PP"),
        ("Herr talman, därför anser jag att vi måste agera.", "INV_ADV"),  # vocative skipped
        ("Men nu har vi en chans.", "INV_ADV"),                            # conjunction skipped
        # precision guards: unclassified rather than wrong
        ("Varför gör vi inte detta?", "OTHER"),                            # question
        ("Med den tillväxt vi har i dag kommer vi inte långt.", "OTHER"),  # contact relative in PP
        ("Även vi måste ta ansvar.", "OTHER"),                             # focus particle + subject
        ("Kanske vi borde vänta.", "OTHER"),                               # 'kanske' licenses V3
        ("I framtiden kanske vi kan gå vidare.", "OTHER"),
        ("För jag tror att det är fel.", "OTHER"),                         # 'för' = because
        ("Inte minst kvinnorna har drabbats.", "OTHER"),                   # focus particle + NP subject
        ("Låt mig säga något.", "OTHER"),                                  # imperative
    ]

    def test_cases(self):
        for s, c in self.CASES:
            with self.subTest(s=s):
                self.assertEqual(cat(s), c)

    def test_vocative_only_sentences_skipped(self):
        d = Doc("Herr talman! Därför anser jag att vi måste agera.")
        ops = W.openings(d)
        self.assertEqual([i for i, _ in ops], [1])
        self.assertEqual(ops[0][1].cat, "INV_ADV")


class TestMetamorphic(unittest.TestCase):
    """Rewriting a sentence from topicalised-inverted to subject-first moves the
    score toward TRANSLATED; inserting an English-order violation moves it
    further."""
    PAIRS = [
        ("Därför anser jag att förslaget måste ändras.", "Jag anser därför att förslaget måste ändras."),
        ("Nu har vi en historisk möjlighet.", "Vi har nu en historisk möjlighet."),
        ("Det vet vi alla.", "Vi vet det alla."),
        ("Detta kan vi inte acceptera.", "Vi kan inte acceptera detta."),
        ("Samtidigt måste vi skydda miljön.", "Vi måste samtidigt skydda miljön."),
        ("Den frågan har vi diskuterat länge.", "Vi har diskuterat den frågan länge."),
    ]

    def test_topicalised_to_subject_first(self):
        for topic, subj in self.PAIRS:
            with self.subTest(topic=topic):
                self.assertLess(fam_score(topic), 0.0)
                self.assertGreater(fam_score(subj), fam_score(topic))
                self.assertGreater(fam_score(subj), 0.0)

    def test_violation_moves_further(self):
        native = "Dessutom måste vi tänka på miljön. Vi måste agera nu."
        engl = "Dessutom, vi måste tänka på miljön. Vi måste agera nu."
        self.assertGreater(fam_score(engl), fam_score(native) + 1.0)

    def test_longer_text_accumulates(self):
        o = " ".join(t for t, _ in self.PAIRS)
        t = " ".join(s for _, s in self.PAIRS)
        self.assertLess(fam_score(o), -2.0)
        self.assertGreater(fam_score(t), 0.2)   # subject-first is only weakly T (lumped SUBJ_FIRST)


class TestSpansAndParams(unittest.TestCase):
    def test_spans_sum_and_offsets(self):
        text = "Herr  talman! Därför anser jag att vi måste agera. Vi vet det. Slutligen, vi måste säga att vi kan inte vänta."
        doc = Doc(text)
        for r in W.RULES:
            hits, resid = r.evidence(doc)
            self.assertTrue(hits, r.name)
            self.assertAlmostEqual(sum(h.share for h in hits) + resid, r.contribution(doc))
        spans = Families(only="wordorder").explain_spans(text)
        got = {s["text"] for s in spans}
        self.assertIn("Därför anser jag", got)
        self.assertIn("Vi vet", got)
        self.assertIn("Slutligen, vi måste", got)
        self.assertIn("att vi kan inte", got)
        for s in spans:
            self.assertEqual(text[s["start"]:s["end"]], s["text"])

    def test_opening_probabilities(self):
        for k in ("O", "T"):
            tot = sum(p[0 if k == "O" else 1] for p in W.OPENING_P.values())
            self.assertAlmostEqual(tot, 1.0, delta=0.01)
        for c in W.OPENING_P:
            self.assertIn(c, W._OPEN_LABEL)

    def test_no_topic_words_in_lists(self):
        banned = ("sverige", "svensk", "europa", "kommission", "parlament", "rådet", "eu")
        lists = [W.FIN, W.ADV1, W.PREP, W.SUBJN1, W.SUBJ_PERS, W.V3_ADV, W.CONJ_ADV]
        for lst in lists:
            for w in lst:
                self.assertFalse(any(w.startswith(b) for b in banned), w)


if __name__ == "__main__":
    unittest.main()

"""connpunct family: pattern precision, maths, and metamorphic tests."""
import math
import unittest

from tsw.doc import Doc
from tsw.engine import Families
from tsw.rules import connpunct as C
from tsw.rules import discover

R = {r.name: r for r in C.RULES}


def contrib(name, text):
    return R[name].contribution(Doc(text))


def labels(name, text):
    hits, _ = R[name].evidence(Doc(text))
    return [h.label for h in hits]


BASE = ("Herr talman! Vi har diskuterat betänkandet i utskottet. Förslaget gäller fisket i "
        "Östersjön. Jag har läst texten noga. Kommissionen har lagt fram en plan.")


class TestDiscovery(unittest.TestCase):
    def test_family_registered(self):
        rules, caps, _ = discover()
        names = [r.name for r in rules if r.family == "connpunct"]
        self.assertEqual(len(names), len(C.RULES))
        self.assertTrue(all(n.startswith("connpunct.") for n in names))
        self.assertEqual(caps["connpunct"], C.FAMILY_CAP)

    def test_metadata(self):
        for r in C.RULES:
            self.assertTrue(r.rationale and r.source, r.name)
            self.assertIn(r.direction, ("T", "O"))


class TestMaths(unittest.TestCase):
    def test_choice_llr(self):
        r = R["connpunct.initial_adverbial_comma"]
        d = Doc("Dessutom, vi måste agera. Därför anser jag att det är bra. Slutligen vill jag tacka.")
        ka, kb = r.counts(d)
        self.assertEqual((ka, kb), (1, 2))
        raw = math.log(r.p_t / r.p_o) + 2 * math.log((1 - r.p_t) / (1 - r.p_o))
        self.assertAlmostEqual(r.contribution(d), min(r.weight * raw, r.cap))

    def test_rate_llr(self):
        r = R["connpunct.native_connectives"]
        d = Doc("Vi måste därför agera, dock inte nu. " + " ".join(["ord"] * 92) + ".")
        n = d.n_words
        raw = 2 * math.log(r.rate_t / r.rate_o) - n * (r.rate_t - r.rate_o) / 1000
        self.assertAlmostEqual(r.contribution(d), max(-r.cap, r.weight * raw))

    def test_exclamation_poisson_and_spans(self):
        r = R["connpunct.exclamation"]
        d = Doc("Herr talman! Vi dömer honom! Det är fel.")
        n = d.n_words
        raw = C._EXCL_LOG - n * C._EXCL_PER_WORD
        self.assertAlmostEqual(r.contribution(d), r.weight * raw)
        hits, resid = r.evidence(d)
        self.assertEqual(len(hits), 1)                     # vocative '!' not counted
        self.assertEqual(d.text[hits[0].start:hits[0].end], "!")
        self.assertAlmostEqual(sum(h.share for h in hits) + resid, r.contribution(d))

    def test_shares_add_up(self):
        text = ("Dessutom, vi har i synnerhet t.ex. fisket - och jordbruket - att tänka på. "
                "Och vi röstar nej eftersom det är fel! Jag tror det är så. Det arbete ni gjort.")
        d = Doc(text)
        for r in C.RULES:
            hits, resid = r.evidence(d)
            self.assertAlmostEqual(sum(h.share for h in hits) + resid, r.contribution(d), msg=r.name)


class TestPatterns(unittest.TestCase):
    def test_native_connectives(self):
        self.assertEqual(len(labels("connpunct.native_connectives", "Vi måste därför agera, dock inte nu.")), 2)
        # causal 'därför att' and 'samtidigt som' are subjunctions, not connective adverbs
        self.assertEqual(labels("connpunct.native_connectives", "Det är därför att vi vill, samtidigt som de vill."), [])

    def test_calque_connectives(self):
        self.assertEqual(len(labels("connpunct.calque_connectives", "I själva verket vill jag i synnerhet tacka.")), 2)

    def test_att_drop(self):
        self.assertEqual(len(labels("connpunct.att_drop", "Jag tror det är viktigt.")), 1)
        self.assertEqual(labels("connpunct.att_drop", "Jag tror att det är viktigt."), [])

    def test_som_drop(self):
        self.assertEqual(labels("connpunct.som_drop", "Tack för det arbete ni har gjort.")[0][:13], "zero relative")
        self.assertEqual(labels("connpunct.som_drop", "Tack för det arbete som ni har gjort.")[0][:5], "'som'")

    def test_initial_och_case_and_position(self):
        self.assertEqual(len(labels("connpunct.initial_och", "Vi kom. Och sedan gick vi.")), 1)
        self.assertEqual(labels("connpunct.initial_och", "Vi kom och gick."), [])

    def test_clause_comma_omission(self):
        self.assertTrue(contrib("connpunct.clause_comma_omission", "Vi röstar nej eftersom det är fel.") > 0)
        self.assertTrue(contrib("connpunct.clause_comma_omission", "Vi röstar nej, eftersom det är fel.") < 0)
        # 'och eftersom', pied-piped 'i vilket' are not comma slots
        self.assertEqual(contrib("connpunct.clause_comma_omission", "Och eftersom vi vet det, i vilket land."), 0.0)

    def test_parens_skip_document_codes(self):
        self.assertEqual(labels("connpunct.explanatory_parens", "Betänkande (A5-0127/2001) av Katiforis."), [])
        self.assertEqual(labels("connpunct.explanatory_parens", "Rådet (allmänna frågor) möts."), [])
        self.assertEqual(len(labels("connpunct.explanatory_parens", "Programmet (som gäller små företag) är bra.")), 1)

    def test_dash_skips_document_codes(self):
        self.assertEqual(labels("connpunct.parenthetical_dash", "KOM(2000) 607 - C5-0104/2001 - 2001/2053"), [])
        self.assertEqual(len(labels("connpunct.parenthetical_dash", "Detta är - och det vet vi - viktigt.")), 2)


class TestMetamorphic(unittest.TestCase):
    """Edits that should move the family score in a known direction."""

    def setUp(self):
        self.eng = Families(only="connpunct")

    def s(self, text):
        return self.eng.score(text)

    def test_comma_after_initial_adverbial_raises(self):
        a = BASE + " Dessutom måste vi agera."
        b = BASE + " Dessutom, vi måste agera."
        self.assertGreater(self.s(b), self.s(a))

    def test_native_connective_lowers(self):
        self.assertLess(self.s(BASE.replace("Jag har läst", "Jag har därför läst")), self.s(BASE))

    def test_calque_connective_raises(self):
        self.assertGreater(self.s(BASE.replace("Jag har läst", "Jag har i synnerhet läst")), self.s(BASE))

    def test_abbreviation_lowers(self):
        a = BASE + " Det gäller till exempel torsken."
        b = BASE + " Det gäller t.ex. torsken."
        self.assertLess(self.s(b), self.s(a))

    def test_att_drop_lowers(self):
        a = BASE + " Jag tror att det är fel."
        b = BASE + " Jag tror det är fel."
        self.assertLess(self.s(b), self.s(a))

    def test_sentence_initial_och_raises(self):
        a = BASE + " Vi stöder den och vi röstar ja."
        b = BASE + " Vi stöder den. Och vi röstar ja."
        self.assertGreater(self.s(b), self.s(a))

    def test_removing_comma_before_eftersom_raises(self):
        a = BASE + " Vi röstar nej, eftersom det är fel."
        b = BASE + " Vi röstar nej eftersom det är fel."
        self.assertGreater(self.s(b), self.s(a))

    def test_exclamation_lowers_but_vocative_does_not(self):
        self.assertLess(self.s(BASE + " Det är fel!"), self.s(BASE + " Det är fel."))
        self.assertAlmostEqual(self.s("Fru talman! " + BASE), self.s("Fru talman. " + BASE), places=1)

    def test_dash_aside_raises(self):
        a = BASE + " Detta är, som vi vet, viktigt."
        b = BASE + " Detta är - som vi vet - viktigt."
        self.assertGreater(self.s(b), self.s(a))


if __name__ == "__main__":
    unittest.main()


class TestIntegrationExclusions(unittest.TestCase):
    """Overlaps removed when the families were integrated (docs/INTEGRATION.md)."""

    def test_native_connectives_not_sentence_initial(self):
        r = R["connpunct.native_connectives"]
        n = lambda t: len(list(r.matcher.finditer(Doc(t))))  # noqa: E731
        self.assertEqual(n("Därför anser jag detta. Men dessutom vill vi mer. Samtidigt måste vi agera."), 0)
        self.assertEqual(n("Vi måste därför agera, och vi kan dessutom vänta. Det är dock svårt."), 3)
        self.assertEqual(n("Vi gör det i alla fall."), 0)        # particles.stance owns 'i alla fall'

    def test_written_out_abbrev_without_bla_dvs(self):
        r = R["connpunct.written_out_abbrev"]
        self.assertEqual(r.counts(Doc("Det gäller bl.a. fisket, dvs. kvoterna, bland annat i norr.")), (0, 0))
        self.assertEqual(r.counts(Doc("Det gäller t.ex. fisket och till exempel kvoterna.")), (1, 1))

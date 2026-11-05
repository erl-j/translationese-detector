"""particles family: maths, disambiguation, metamorphic tests (and the revised
starter.modal_particles regexes)."""
import math
import unittest

from tsw.doc import Doc
from tsw.engine import Default, Families
from tsw.rules import core_starter, particles

R = {r.name: r for r in particles.RULES}
MP = {r.name: r for r in core_starter.RULES}["starter.modal_particles"]


def n_hits(rule, text):
    return rule.matcher.count(Doc(text))


def choice_counts(rule, text):
    return rule.counts(Doc(text))


class TestModule(unittest.TestCase):
    def test_names_and_metadata(self):
        self.assertEqual(particles.FAMILY, "particles")
        for r in particles.RULES:
            self.assertTrue(r.name.startswith("particles."), r.name)
            self.assertTrue(r.rationale and r.source, r.name)
            self.assertIn(r.direction, ("O", "T"))

    def test_family_discovered(self):
        eng = Families(only="particles")
        self.assertTrue(eng.rules)
        self.assertTrue(all(r.family == "particles" for r in eng.rules))

    def test_no_overlap_with_starter_particles(self):
        # ju/nog/väl/visst belong to starter.modal_particles only
        text = "Det är ju så. Det är nog bra. Det är väl så. Visst är det så."
        for r in particles.RULES:
            if hasattr(r, "matcher"):
                self.assertEqual(n_hits(r, text) if r.name != "particles.copular_det" else 0, 0, r.name)


class TestMaths(unittest.TestCase):
    def test_rate_namligen(self):
        r = R["particles.namligen"]
        d = Doc("Jag anser nämligen att " + " ".join(["ord"] * 96) + ".")
        self.assertEqual(d.n_words, 100)
        ro, rt = r.rate_o, r.rate_t
        raw = math.log(rt / ro) - 100 * (rt - ro) / 1000
        self.assertAlmostEqual(r.contribution(d), max(-r.cap, raw))

    def test_rate_weight_man(self):
        r = R["particles.generic_man"]
        d = Doc("Man kan säga att " + " ".join(["ord"] * 96) + ".")
        ro, rt = r.rate_o, r.rate_t
        raw = math.log(rt / ro) - 100 * (rt - ro) / 1000
        self.assertAlmostEqual(r.contribution(d), 0.25 * raw)

    def test_choice_of_course(self):
        r = R["particles.of_course"]
        # 'Självklart' is register.sjalvklart's since the integration
        d = Doc("Naturligtvis är det så. Självklart måste vi, förstås, och givetvis kan vi.")
        self.assertEqual(r.counts(d), (1, 2))
        po, pt = r.p_o, r.p_t
        exp = math.log(pt / po) + 2 * math.log((1 - pt) / (1 - po))
        self.assertAlmostEqual(r.contribution(d), max(-r.cap, exp))

    def test_spans_sum_to_contribution(self):
        eng = Default()
        text = ("Herr talman! Det är nämligen så att man inte kan göra det. Då måste vi ändå "
                "slå fast att våra medborgare och deras barn, dvs. alla, egentligen har rätt.")
        ann = eng.annotate(text)
        for name in R:
            if name not in ann["rules"]:
                continue
            spans = sum(s["contribution_share"] for s in ann["spans"] if s["rule"] == name)
            doc_level = sum(v for k, v in ann["doc_level"].items() if k == name) \
                if isinstance(ann["doc_level"], dict) else \
                sum(e.get("contribution", 0) for e in ann["doc_level"] if e.get("rule") == name)
            self.assertAlmostEqual(spans + doc_level, ann["rules"][name], places=6, msg=name)


class TestDisambiguation(unittest.TestCase):
    def test_namligen_particle_not_appositive(self):
        r = R["particles.namligen"]
        self.assertEqual(n_hits(r, "Det är nämligen så."), 1)
        self.assertEqual(n_hits(r, "Jag anser nämligen att det är fel."), 1)
        self.assertEqual(n_hits(r, "En viktig fråga, nämligen miljön."), 0)
        self.assertEqual(n_hits(r, "Ett mål - nämligen att bevara bestånden."), 0)
        self.assertEqual(n_hits(r, "Ett mål: nämligen tillväxt."), 0)

    def test_da_anaphoric_not_subjunction(self):
        r = R["particles.da_anaphoric"]
        # sentence-initial 'Då' + verb is wordorder.fronted_theme's since the integration
        self.assertEqual(n_hits(r, "Det är dyrt. Då måste vi agera."), 0)
        self.assertEqual(n_hits(r, "Det är dyrt. Måste vi då agera?"), 1)
        self.assertEqual(n_hits(r, "Vad finns det då för samband?"), 1)
        self.assertEqual(n_hits(r, "Det är då viktigt att minnas detta."), 1)
        # subjunction 'då' = as/since/when
        self.assertEqual(n_hits(r, "Det är dyrt. Då vi begrundar reformerna ser vi detta."), 0)
        self.assertEqual(n_hits(r, "Vi röstade nej, då det finns en klausul."), 0)
        self.assertEqual(n_hits(r, "Det var då vi kom."), 0)

    def test_man_not_noun(self):
        r = R["particles.generic_man"]
        self.assertEqual(n_hits(r, "Man måste agera."), 1)
        self.assertEqual(n_hits(r, "Det var en man från byn."), 0)
        self.assertEqual(n_hits(r, "Den man som kom."), 0)
        self.assertEqual(n_hits(r, "Både man och kvinna."), 0)
        self.assertEqual(n_hits(r, "Mannen och manliga."), 0)

    def test_poss_3rd_not_temporal_dess(self):
        r = R["particles.poss_3rd"]
        self.assertEqual(n_hits(r, "Unionen och dess medborgare, deras barn och hans fru."), 3)
        self.assertEqual(n_hits(r, "Sedan dess har inget hänt, och innan dess, till dess att."), 0)
        self.assertEqual(n_hits(r, "Dessutom."), 0)

    def test_copular_det(self):
        r = R["particles.copular_det"]
        # sentence-initial 'Det är/var' is wordorder.subject_theme's since the integration
        self.assertEqual(n_hits(r, "Det är viktigt. Är det så? Det var bra."), 1)
        self.assertEqual(n_hits(r, "Jag tror att det är viktigt. Därför var det bra."), 2)
        self.assertEqual(n_hits(r, "Detta är viktigt. Det här är bra."), 0)

    def test_particle_verbs_and_ta_itu_med(self):
        pv, tim = R["particles.particle_verbs"], R["particles.ta_itu_med"]
        self.assertEqual(n_hits(pv, "Vi slår fast att vi kan ställa oss bakom och lyfta fram detta."), 3)
        self.assertEqual(n_hits(pv, "Vi har kommit fram till ett beslut."), 1)
        self.assertEqual(n_hits(pv, "Vi slår inte till."), 0)
        self.assertEqual(n_hits(tim, "Vi måste ta itu med detta, och vi tar nu itu med det."), 2)

    def test_stance_and_dvs(self):
        self.assertEqual(n_hits(R["particles.stance"],
                                "Vad är egentligen målet? Det är i och för sig bra, i alla fall."), 3)
        self.assertEqual(n_hits(R["particles.dvs"], "Hamnarna, dvs. lotsningen, det vill säga allt."), 2)

    def test_ni_vs_man(self):
        r = R["particles.ni_vs_man"]
        self.assertEqual(choice_counts(r, "Ni kan sedan gå ned. Man kan också åka."), (1, 1))
        self.assertEqual(choice_counts(r, "Nittio procent, nio av tio."), (0, 0))


class TestStarterModalParticlesFix(unittest.TestCase):
    """Revised regexes of starter.modal_particles (allowed fix on this branch)."""

    def test_nog(self):
        self.assertEqual(n_hits(MP, "Det är nog så att vi får vänta."), 1)
        self.assertEqual(n_hits(MP, "Man kan nog säga det."), 1)
        self.assertEqual(n_hits(MP, "Jag tror att det nog inte är jordbruket."), 1)
        for t in ("Paradoxalt nog fungerar det.", "Beklagligt nog kom han inte.",
                  "Märkligt nog har inget hänt.", "Det går inte långt nog.",
                  "Det är tillräckligt nog.", "Det fungerar inte bra nog.",
                  "Glädjande nog var hon där.", "Det är nog med det.", "Det är inte nog.",
                  "Det kan inte nog understrykas.", "Mer än nog uppfyllt.", "Nära nog fullständig."):
            self.assertEqual(n_hits(MP, t), 0, t)

    def test_val(self):
        self.assertEqual(n_hits(MP, "Det är väl så att vi vet."), 1)
        self.assertEqual(n_hits(MP, "Vi vill väl inte upprepa misstaget?"), 1)
        self.assertEqual(n_hits(MP, "Det kan man väl säga."), 1)
        for t in ("De är väl förberedda.", "Förslagen är väl värda att stödja.",
                  "Vi är väl bekanta med detta.", "Problemen är väl kartlagda.",
                  "Åtgärder är väl så effektiva.", "Om allt går väl i morgon.",
                  "Resultaten är väl avvägda."):
            self.assertEqual(n_hits(MP, t), 0, t)

    def test_ju_desto(self):
        self.assertEqual(n_hits(MP, "Det är ju bra."), 1)
        for t in ("Ju större EU blir, desto svårare.", "Ju mer pengar, desto bättre.",
                  "Ju fler desto bättre.", "Ju förr desto bättre."):
            self.assertEqual(n_hits(MP, t), 0, t)

    def test_visst_determiner(self):
        self.assertEqual(n_hits(MP, "Visst kan man göra det."), 1)
        for t in ("Ett visst antal.", "I visst mån.", "Det är med visst beklagande."):
            self.assertEqual(n_hits(MP, t), 0, t)


class TestMetamorphic(unittest.TestCase):
    """Editing one item must move the Default score in the predicted direction."""

    BASE = ("Herr talman! Vi har diskuterat frågan länge i utskottet och i kammaren. "
            "{s1} Kommissionen har lagt fram ett förslag om fiske och miljö. "
            "{s2} Medborgarna i unionen väntar på svar från rådet och parlamentet.")

    def setUp(self):
        self.eng = Default()

    def score(self, s1, s2="Vi stöder förslaget."):
        return self.eng.score(self.BASE.format(s1=s1, s2=s2))

    def assertMovesToT(self, original, edited, **kw):
        a, b = self.score(original, **kw), self.score(edited, **kw)
        self.assertGreater(b, a, "%r -> %r: %.3f -> %.3f" % (original, edited, a, b))

    def test_delete_namligen(self):
        self.assertMovesToT("Det är nämligen så att förslaget är bra.",
                            "Det är så att förslaget är bra.")

    def test_delete_ju(self):
        self.assertMovesToT("Det är ju så att förslaget är bra.",
                            "Det är så att förslaget är bra.")

    def test_delete_anda_and_alltsa(self):
        self.assertMovesToT("Förslaget är ändå bra.", "Förslaget är bra.")
        self.assertMovesToT("Förslaget är alltså bra.", "Förslaget är bra.")

    def test_man_to_ni(self):
        self.assertMovesToT("Man kan sedan gå ned till hamnen.", "Ni kan sedan gå ned till hamnen.")

    def test_sjalvklart_to_naturligtvis(self):
        self.assertMovesToT("Självklart stöder vi förslaget.", "Naturligtvis stöder vi förslaget.")

    def test_definite_to_possessive(self):
        s1 = "Vi stöder förslaget."
        a = self.eng.score(self.BASE.format(s1=s1, s2="Medborgarna och regeringarna väntar."))
        b = self.eng.score(self.BASE.format(s1=s1, s2="Våra medborgare och deras regeringar väntar."))
        self.assertGreater(b, a)

    def test_slå_fast_to_fastställa(self):
        self.assertMovesToT("Vi slår fast att förslaget är bra.", "Vi fastställer att förslaget är bra.")

    def test_hantera_to_ta_itu_med(self):
        self.assertMovesToT("Vi måste hantera problemet.", "Vi måste ta itu med problemet.")

    def test_da_to_i_det_fallet(self):
        self.assertMovesToT("Då måste vi agera.", "I det fallet måste vi agera.")

    def test_det_ar_to_detta_ar(self):
        self.assertMovesToT("Det är ett bra förslag.", "Detta är ett bra förslag.")


if __name__ == "__main__":
    unittest.main()

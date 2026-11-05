"""register family: maths on hand-checked examples + metamorphic tests."""
import math
import unittest

from tsw.doc import Doc
from tsw.engine import Default, Engine, Families
from tsw.rules import register as R

RULES = {r.name: r for r in R.RULES}


def rule(name):
    return RULES[name]


def c(name, text):
    return rule(name).contribution(Doc(text))


FILLER = " ".join(["Vi diskuterar frågan om fisket i dag."] * 12)  # 84 neutral words


class TestFamily(unittest.TestCase):
    def test_metadata(self):
        self.assertEqual(R.FAMILY, "register")
        for r in R.RULES:
            self.assertTrue(r.name.startswith("register."), r.name)
            self.assertTrue(r.rationale and r.source, r.name)
        fam = Families(only="register")
        self.assertTrue(all(r.family == "register" for r in fam.rules))

    def test_replaced_starter_rules_disabled(self):
        names = {r.name for r in Default().rules}
        self.assertNotIn("starter.dem_har_vs_denna", names)
        self.assertNotIn("starter.ska_vs_skall", names)
        self.assertIn("register.dem_har_rate", names)
        self.assertIn("register.kommer_att_vs_ska", names)
        # documented alternatives stay off by default
        for n in ("register.skall_rate", "register.ocksa_rate", "register.hedges"):
            self.assertNotIn(n, names)
            self.assertIn(n, {r.name for r in Engine(include_disabled=True).rules})


class TestMaths(unittest.TestCase):
    def test_future_choice(self):
        r = rule("register.kommer_att_vs_ska")
        la, lb = math.log(0.60 / 0.24), math.log(0.40 / 0.76)
        self.assertAlmostEqual(c(r.name, "Det kommer att bli bra."), la)
        self.assertAlmostEqual(c(r.name, "Vi skall göra det och ni ska hjälpa till."), 2 * lb)
        self.assertAlmostEqual(c(r.name, "Det kommer inte att bli bra, men vi skall försöka."), la + lb)
        # 'kommer överens' / 'kommer fram till att' are not future 'kommer att'
        self.assertEqual(c(r.name, "Vi kommer överens. Vi kommer fram till att det är bra."), 0.0)
        # cap
        self.assertAlmostEqual(c(r.name, " ".join(["Vi skall."] * 10)), -2.5)

    def test_opinion_choice(self):
        r = "register.tycker_vs_anser"
        self.assertAlmostEqual(c(r, "Jag tycker att det är fel."), math.log(0.20 / 0.58))
        self.assertAlmostEqual(c(r, "Jag anser att det är fel."), math.log(0.80 / 0.42))
        # 'tycks' (seems) and passive 'anses' are not opinion verbs here
        self.assertEqual(c(r, "Det tycks vara fel, och det anses vara fel."), 0.0)

    def test_dem_har_rate(self):
        r = rule("register.dem_har_rate")
        d = Doc("Det här betänkandet är bra. " + FILLER)
        n = d.n_words
        raw = math.log(0.81 / 0.17) - n * (0.81 - 0.17) / 1000
        self.assertAlmostEqual(r.contribution(d), 0.5 * raw)
        # formal 'detta/denna' (incl. 'denna kammare') is NOT counted any more
        d2 = Doc("Detta är viktigt för denna kammare. " + FILLER)
        self.assertAlmostEqual(r.contribution(d2), -0.5 * d2.n_words * (0.81 - 0.17) / 1000)
        # cap 1.0 (speakers who say 'det här' a lot cannot saturate far)
        self.assertAlmostEqual(r.contribution(Doc("Det här och det här och den här och de här.")), 1.0)

    def test_bland_annat_forms(self):
        r = rule("register.bland_annat")
        self.assertEqual(r.matcher.count(Doc("Vi har bl.a. fisket och bland annat jordbruket.")), 2)

    def test_mycket_adj(self):
        r = rule("register.valdigt_vs_mycket")
        self.assertEqual(r.counts(Doc("Det är mycket viktigt och väldigt svårt, men mycket pengar.")), (1, 1))


def uncapped():
    """The register family alone, without the family cap (so single swaps are visible)."""
    return Families(only="register", family_caps={"register": 100.0})


class TestMetamorphic(unittest.TestCase):
    """Swapping one native variant for its translationese competitor moves the score toward T."""

    base = ("Herr talman! Jag {vill} tacka föredraganden. {Jag_tycker} att betänkandet är {intens} viktigt. "
            "Vi {fut} {inf} detta {aven} i utskottet. Det är {sjalv} en fråga om pengar. " + FILLER)
    native = dict(vill="vill", Jag_tycker="Jag tycker", intens="väldigt", fut="skall", inf="stödja",
                  aven="också", sjalv="självklart")
    swaps = {
        "vill": "skulle vilja",
        "Jag_tycker": "Jag anser",
        "intens": "mycket",
        "fut": "kommer att",
        "aven": "även",
        "sjalv": "naturligtvis",
    }

    def test_each_swap_moves_toward_T(self):
        eng = uncapped()
        s0 = eng.score(self.base.format(**self.native))
        for k, v in self.swaps.items():
            alt = dict(self.native, **{k: v})
            s1 = eng.score(self.base.format(**alt))
            self.assertGreater(s1, s0, "swap %s -> %s" % (k, v))

    def test_all_swaps_flip_family_sign(self):
        eng = Families(only="register")
        self.assertLess(eng.score(self.base.format(**self.native)), 0)
        self.assertGreater(eng.score(self.base.format(**dict(self.native, **self.swaps))), 0)

    def test_removing_native_items_moves_toward_T(self):
        eng = Families(only="register")
        t0 = "Detta är bra, samt viktigt, och bland annat fisket berörs. " + FILLER
        t1 = "Detta är positivt, och viktigt, och fisket berörs bland flera. " + FILLER
        self.assertGreater(eng.score(t1), eng.score(t0))

    def test_den_har_moves_toward_T(self):
        eng = Families(only="register")
        a = "Vi måste stödja detta förslag. " + FILLER
        b = "Vi måste stödja det här förslaget. " + FILLER
        self.assertGreater(eng.score(b), eng.score(a))

    def test_av_periphrasis_moves_toward_T(self):
        eng = Families(only="register")
        a = "Kommissionens förslag om miljöskydd. " + FILLER
        b = "Förslaget av kommissionen om skyddet av miljön. " + FILLER
        self.assertGreater(eng.score(b), eng.score(a))


if __name__ == "__main__":
    unittest.main()

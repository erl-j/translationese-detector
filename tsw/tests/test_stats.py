"""stats family: statistics on hand-checked examples, LengthStat maths, metamorphic tests."""
import math
import unittest

from tsw.doc import Doc
from tsw.engine import Families
from tsw.rules import base as B
from tsw.rules import discover
from tsw.rules import stats as S


def rule(name):
    return next(r for r in S.RULES if r.name == name)


class TestWordStream(unittest.TestCase):
    def test_skips_vocatives_numbers_placeholder(self):
        d = Doc("Herr talman! Vi har 15 XTOPIC förslag om EU:s budget 2003.")
        self.assertEqual([w for w, _, _, _ in S.word_items(d)], ["vi", "har", "förslag", "om", "eu:s", "budget"])

    def test_offsets_point_at_words(self):
        d = Doc("Detta är  viktigt.")
        for w, a, b, _ in S.word_items(d):
            self.assertEqual(d.lower[a:b], w)

    def test_n_letters(self):
        self.assertEqual(S.n_letters("eu:s"), 3)
        self.assertEqual(S.n_letters("eu-länderna"), 10)
        self.assertEqual(S.n_letters("genomförandet"), 13)

    def test_empty(self):
        for t in ("", "Herr talman!", "2003."):
            d = Doc(t)
            self.assertEqual(S.word_items(d), [])
            for r in S.RULES:
                self.assertEqual(r.contribution(d), 0.0, (t, r.name))


class TestStatistics(unittest.TestCase):
    def test_long_words_hand_checked(self):
        # genomförandet(13) av(2) direktivet(10) är(2) viktigt(7): 3 of 5 have > 6 letters
        d = Doc("Genomförandet av direktivet är viktigt.")
        vals = [v for v, _ in S.long_word_units(d)]
        self.assertEqual(vals, [1.0, 0.0, 1.0, 0.0, 1.0])
        # exactly 6 letters is NOT long (LIX: more than six)
        self.assertEqual([v for v, _ in S.long_word_units(Doc("Sverig fråga."))], [0.0, 0.0])
        # spans are the sentences
        d = Doc("Detta är viktigt. Vi röstar nej.")
        spans = {sp for _, sp in S.long_word_units(d)}
        self.assertEqual(spans, set(d.sent_spans))

    def test_av_split_is_disjoint_and_complete(self):
        t = ("Genomförandet av direktivet och användningen av medel är en del av lösningen. "
             "Vi talar om skapandet. Av detta följer att tillämpningen av reglerna, av alla, är viktig.")
        d = Doc(t)
        nom = [v for v, _ in S.av_nominal_units(d)]
        oth = [v for v, _ in S.av_other_units(d)]
        n_av = sum(1 for w, _, _, _ in S.word_items(d) if w == "av")
        self.assertEqual(sum(nom) + sum(oth), n_av)
        self.assertTrue(all(a + b <= 1 for a, b in zip(nom, oth)))
        # genomförandet av, användningen av, tillämpningen av; NOT 'skapandet. Av' (sentence boundary)
        self.assertEqual(sum(nom), 3)
        self.assertEqual(sum(oth), 3)
        spans = [d.text[a:b] for v, (a, b) in [(v, sp) for v, sp in S.av_nominal_units(d) if v]]
        self.assertEqual(spans, ["Genomförandet av", "användningen av", "tillämpningen av"])

    def test_mattr_hand_checked(self):
        # words: ett två ett tre fyra fyra; window 3 -> TTRs 2,3,3,2 (starts 0..3)
        d = Doc("Ett två ett tre fyra fyra.")
        u = S.mattr_units(d, window=3)
        self.assertEqual(len(u), 2)
        self.assertAlmostEqual(u[0][0], (2 + 3 + 3) / 9.0)
        self.assertAlmostEqual(u[1][0], 2 / 3.0)
        self.assertEqual(d.text[u[0][1][0]:u[0][1][1]], "Ett två ett")
        self.assertEqual(S.mattr_units(Doc("Ett två."), window=3), [])  # undefined below the window

    def test_mattr_equals_classic_when_blocks_are_full(self):
        # N = 2 * window + window - 1 -> every block has `window` windows: unit mean == MATTR
        ws = ("a b c a d e f a b g h i j k a".split())  # 15 words, window 5 -> 11 windows
        d = Doc("Ord " + " ".join("x" + w for w in ws[1:]) + ".")
        items = [w for w, _, _, _ in S.word_items(d)]
        win = 5
        classic = sum(len(set(items[i:i + win])) for i in range(len(items) - win + 1)) / (
            (len(items) - win + 1) * float(win))
        u = S.mattr_units(d, window=win)
        self.assertEqual(len(u), 3)
        # blocks hold 5, 5, 1 windows: weighted mean of units by #windows equals classic MATTR
        w = [5, 5, 1]
        self.assertAlmostEqual(sum(v * k for (v, _), k in zip(u, w)) / sum(w), classic)

    def test_mattr_extremes(self):
        all_diff = Doc(" ".join("ord%s" % chr(97 + i % 26) + ("x" * (i // 26)) for i in range(60)).capitalize() + ".")
        self.assertAlmostEqual(S.mattr_units(all_diff)[0][0], 1.0)
        same = Doc("Ja " + "ja " * 59 + "ja.")
        self.assertAlmostEqual(S.mattr_units(same)[0][0], 1 / 50.0)

    def test_ovix_hand_checked(self):
        d = Doc("Vi vill vi kan.")  # N = 4, V = 3
        x = S.ovix_units(d)[0][0]
        self.assertAlmostEqual(x, math.log(4) / math.log(2 - math.log(3) / math.log(4)))
        self.assertEqual(S.ovix_units(Doc("Alla ord olika.")), [])  # V == N: undefined


class TestLengthStat(unittest.TestCase):
    def mk(self, tau, **kw):
        return S.LengthStat("t.ls", units=S.long_word_units, mu_o=0.2, mu_t=0.4, sd=0.5, tau=tau,
                            cap=1000, **kw)

    def test_llr_is_gaussian_difference(self):
        r = self.mk(0.1)
        for x, n in ((0.3, 1), (0.5, 7), (0.0, 400)):
            s = math.sqrt(0.25 / n + 0.01)
            ref = B._gauss_logpdf(x, 0.4, s) - B._gauss_logpdf(x, 0.2, s)
            self.assertAlmostEqual(r._llr(x, n), ref)

    def test_tau_zero_equals_statrule(self):
        r0 = self.mk(0.0)
        st = B.StatRule("t.sr", units=S.long_word_units, mu_o=0.2, mu_t=0.4, sd=0.5, cap=100)
        d = Doc("Genomförandet av direktivet är viktigt. Vi röstar nej till förslaget.")
        self.assertAlmostEqual(r0.contribution(d), st.contribution(d))

    def test_hand_checked_value(self):
        # 3 long of 5 words: x = .6, n = 5, var = .25/5 + .01 = .06, slope = .2/.06
        d = Doc("Genomförandet av direktivet är viktigt.")
        self.assertAlmostEqual(self.mk(0.1).contribution(d), (0.2 / 0.06) * (0.6 - 0.3))

    def test_shares_sum_to_contribution(self):
        r = self.mk(0.05, span_min=0.0)
        d = Doc("Genomförandet av direktivet är viktigt. Vi röstar nej. Kommissionens förslag är bra.")
        hits, resid = r.evidence(d)
        self.assertEqual(len(hits), 3)  # one span per sentence (word units grouped)
        self.assertAlmostEqual(sum(h.share for h in hits) + resid, r.contribution(d))
        d = Doc(FILL + d.text)
        for rr in S.RULES:
            self.assertNotEqual(rr.contribution(d), 0.0)
            hits, resid = rr.evidence(d)
            self.assertAlmostEqual(sum(h.share for h in hits) + resid, rr.contribution(d))

    def test_bounded_in_length(self):
        # metamorphic: repeating a text k times multiplies an independent-units LLR by k,
        # but with tau > 0 the LLR converges to (mu_t-mu_o)/tau^2 * (x - mid)
        t = "Genomförandet av direktivet är viktigt. "
        r, r0 = self.mk(0.1), self.mk(0.0)
        c = [r.contribution(Doc(t * k)) for k in (1, 10, 100)]
        c0 = [r0.contribution(Doc(t * k)) for k in (1, 10, 100)]
        self.assertAlmostEqual(c0[2] / c0[0], 100.0, places=6)
        self.assertLess(c[2] / c[0], 6.0)
        self.assertLess(c[2], (0.2 / 0.01) * (0.6 - 0.3))
        self.assertTrue(c[0] < c[1] < c[2])

    def test_pooled_sd_required(self):
        with self.assertRaises(ValueError):
            S.LengthStat("t.bad", units=S.long_word_units, mu_o=0.2, mu_t=0.4, sd_o=0.4, sd_t=0.5)
        with self.assertRaises(ValueError):
            S.LengthStat("t.bad2", units=S.long_word_units, mu_o=0.2, mu_t=0.4, sd=0.5, tau=-1)


# neutral filler (short words, no 'av') to lift short examples over MIN_WORDS
FILL = "Vi har talat om det här i dag och vi ska tala om det i morgon igen. " * 4  # 64 words


class TestRules(unittest.TestCase):
    def test_discovered_and_named(self):
        rules, caps, _ = discover()
        mine = [r for r in rules if r.family == "stats"]
        self.assertEqual({r.name for r in mine}, {"stats.long_words", "stats.av_nominal", "stats.av_other"})
        self.assertEqual(caps["stats"], S.FAMILY_CAP)
        for r in mine:
            self.assertTrue(r.rationale and r.source)
            self.assertEqual(r.direction, "T")
            self.assertEqual(r.min_units, S.MIN_WORDS)

    def test_abstains_below_min_words(self):
        body = "Genomförandet av direktivet är en del av lösningen för kommissionens förslag."
        d = Doc(body)
        self.assertLess(len(S.word_items(d)), S.MIN_WORDS)
        for r in S.RULES:
            self.assertEqual(r.contribution(d), 0.0)
            self.assertEqual(r.evidence(d), ([], 0.0))
            self.assertIn("undefined", r.describe(d))
        # exactly at the threshold the rules fire
        words = ("ord " * (S.MIN_WORDS - 1)).split()
        at = Doc("Genomförandet " + " ".join(words) + ".")
        self.assertEqual(len(S.word_items(at)), S.MIN_WORDS)
        self.assertNotEqual(rule("stats.long_words").contribution(at), 0.0)

    def test_av_hit_worth_about_poisson(self):
        # one extra 'av' is worth ~ (slope(n)/n) * 1 ~ ln(rate_t / rate_o) when n is moderate
        r = rule("stats.av_other")
        with_av = Doc(FILL + "Det är en del av lösningen.")
        without = Doc(FILL + "Det är en bit på lösningen.")
        n = len(S.word_items(with_av))
        diff = r.contribution(with_av) - r.contribution(without)
        self.assertAlmostEqual(diff, r._slope(n) / n)
        self.assertGreater(diff, 0.5 * math.log(0.017 / 0.013))
        self.assertLess(diff, math.log(0.017 / 0.013))

    def test_metamorphic_av_periphrasis(self):
        # replacing a compound by an X-av-Y periphrasis moves the family toward T
        e = Families(only="stats")
        native = FILL + "Vi stöder genomförandet. Miljöskyddet är viktigt för oss alla i unionen."
        periph = FILL + "Vi stöder genomförandet. Skyddet av miljön är viktigt för oss alla i unionen."
        nominal = FILL + "Vi stöder genomförandet av direktivet. Miljöskyddet är viktigt för oss alla."
        self.assertGreater(e.score(periph), e.score(native))
        self.assertGreater(rule("stats.av_nominal").contribution(Doc(nominal)),
                           rule("stats.av_nominal").contribution(Doc(native)))

    def test_metamorphic_long_words(self):
        r = rule("stats.long_words")
        short = Doc("Vi vill ha en bra lag nu. Det är dags att gå. " * 6)
        long_ = Doc("Vi vill ha en bra lagstiftning. Det är nödvändigt att genomföra. " * 6)
        self.assertLess(r.contribution(short), 0)
        self.assertGreater(r.contribution(long_), r.contribution(short))

    def test_case_and_topic_invariance(self):
        t = FILL + "Genomförandet av direktivet om vattenkvalitet är viktigt för Sverige och Irland."
        masked = FILL + "Genomförandet av direktivet om vattenkvalitet är viktigt för XTOPIC och XTOPIC."
        no_topic = FILL + "Genomförandet av direktivet om vattenkvalitet är viktigt för och."
        for r in S.RULES:
            self.assertNotEqual(r.contribution(Doc(t)), 0.0)
            self.assertAlmostEqual(r.contribution(Doc(t)), r.contribution(Doc(t.upper().capitalize())))
            self.assertAlmostEqual(r.contribution(Doc(masked)), r.contribution(Doc(no_topic)))

    def test_vocative_ignored(self):
        body = FILL + "Genomförandet av direktivet är en del av lösningen."
        for r in S.RULES:
            self.assertAlmostEqual(r.contribution(Doc("Herr talman! " + body)), r.contribution(Doc(body)))

    def test_describe(self):
        d = Doc(FILL + "Genomförandet av direktivet är en del av lösningen.")
        for r in S.RULES:
            self.assertIn("toward", r.describe(d))
        self.assertIn("undefined", rule("stats.long_words").describe(Doc("Herr talman!")))


if __name__ == "__main__":
    unittest.main()

"""Rule-type maths on hand-checked examples; engine, discovery, caps, spans."""
import json
import math
import unittest

from tsw.doc import Doc
from tsw.engine import CAP_PREFIX, Default, Engine, Families
from tsw.rules import base as B
from tsw.rules import discover


def words(n, w="ord"):
    """A sentence of n words starting with a capital (so the splitter splits)."""
    return " ".join([w.capitalize()] + [w] * (n - 1)) + "."


class TestHitRule(unittest.TestCase):
    def test_per_hit_and_cap(self):
        r = B.HitRule("t.calque", pattern=r"\bi\s+termer\s+av\b", per_hit=1.5, cap=2.0)
        self.assertEqual(r.direction, "T")
        self.assertEqual(r.contribution(Doc("Inget här.")), 0.0)
        self.assertAlmostEqual(r.contribution(Doc("Vi talar i termer av pengar.")), 1.5)
        self.assertAlmostEqual(r.contribution(Doc("I termer av x och i  termer av y.")), 2.0)  # capped
        self.assertAlmostEqual(r.contribution(Doc("I termer av x."), weight=0.5), 0.75)

    def test_lexicon(self):
        r = B.HitRule("t.lex", lexicon={"a": (r"\bfoo\b", 1.0), "b": (r"\bbar\b", 0.5)}, cap=10)
        d = Doc("foo bar bar foo foo")
        self.assertAlmostEqual(r.contribution(d), 3 * 1.0 + 2 * 0.5)
        hits, resid = r.evidence(d)
        self.assertEqual(sorted(h.label for h in hits), ["a", "a", "a", "b", "b"])
        self.assertAlmostEqual(sum(h.share for h in hits) + resid, r.contribution(d))
        with self.assertRaises(ValueError):
            B.HitRule("t.mixed", lexicon={"a": ("x", 1.0), "b": ("y", -1.0)})

    def test_direction_check(self):
        with self.assertRaises(ValueError):
            B.HitRule("t.bad", pattern="x", per_hit=1.0, direction="O")
        r = B.HitRule("t.o", words=["ju"], per_hit=-0.5)
        self.assertEqual(r.direction, "O")
        self.assertAlmostEqual(r.contribution(Doc("Det är ju bra, ju.")), -1.0)
        # whole words only: not inside a word or a hyphenated compound
        self.assertEqual(r.contribution(Doc("juni och ju-ju")), 0.0)


class TestRateRule(unittest.TestCase):
    def test_poisson_llr(self):
        r = B.RateRule("t.rate", words=["ju"], rate_o=2.0, rate_t=0.5, cap=100)
        self.assertEqual(r.direction, "O")
        # 1 hit in 100 words: 1*ln(0.25) - 100*(0.5-2.0)/1000 = -1.386294 + 0.15
        d = Doc("ju " + words(99))
        self.assertEqual(d.n_words, 100)
        self.assertAlmostEqual(r.contribution(d), math.log(0.25) + 0.15)
        # 0 hits in 1000 words: absence is evidence, scaled by length: +1.5
        self.assertAlmostEqual(r.contribution(Doc(words(1000))), 1.5)
        # weight and cap
        r2 = B.RateRule("t.rate2", words=["ju"], rate_o=2.0, rate_t=0.5, weight=0.5, cap=0.6)
        self.assertAlmostEqual(r2.contribution(Doc(words(1000))), 0.6)
        self.assertAlmostEqual(r2.contribution(Doc(words(400))), 0.5 * 0.6)
        self.assertEqual(r.contribution(Doc("")), 0.0)

    def test_shares_add_up(self):
        r = B.RateRule("t.rate", words=["ju"], rate_o=2.0, rate_t=0.5, cap=0.5)
        d = Doc("ju ju ju " + words(50))
        hits, resid = r.evidence(d)
        self.assertEqual(len(hits), 3)
        self.assertAlmostEqual(sum(h.share for h in hits) + resid, r.contribution(d))
        self.assertAlmostEqual(r.contribution(d), -0.5)

    def test_min_words(self):
        r = B.RateRule("t.rate", words=["ju"], rate_o=2.0, rate_t=0.5, min_words=10)
        self.assertEqual(r.contribution(Doc("ju ju")), 0.0)


class TestChoiceRule(unittest.TestCase):
    def test_binomial_llr(self):
        r = B.ChoiceRule("t.choice", words_a=["skall"], words_b=["ska"], p_o=0.25, p_t=0.6, cap=100)
        self.assertEqual(r.direction, "T")
        self.assertEqual(r.contribution(Doc("Inget val här.")), 0.0)  # no choice, no evidence
        d = Doc("Vi skall och ni ska och de skall.")
        self.assertEqual(r.counts(d), (2, 1))
        expect = 2 * math.log(0.6 / 0.25) + 1 * math.log(0.4 / 0.75)
        self.assertAlmostEqual(r.contribution(d), expect)
        hits, resid = r.evidence(d)
        self.assertEqual([h.share > 0 for h in hits], [True, False, True])
        self.assertAlmostEqual(sum(h.share for h in hits) + resid, expect)

    def test_validation(self):
        with self.assertRaises(ValueError):
            B.ChoiceRule("t.c", words_a=["a"], words_b=["b"], p_o=0.5, p_t=0.5)
        with self.assertRaises(ValueError):
            B.ChoiceRule("t.c", words_a=["a"], words_b=["b"], p_o=0.0, p_t=0.5)
        with self.assertRaises(ValueError):
            B.ChoiceRule("t.c", words_a=["a"], words_b=["b"], p_o=0.2, p_t=0.5, direction="O")


class TestStatRule(unittest.TestCase):
    def test_gaussian_llr_pooled(self):
        r = B.StatRule("t.stat", units=B.sentence_lengths, mu_o=10, mu_t=20, sd=10, cap=100)
        self.assertEqual(r.direction, "T")
        # one sentence of 15 words: midway -> 0
        self.assertAlmostEqual(r.contribution(Doc(words(15))), 0.0)
        # one sentence of 20 words: LLR = (20-10)^2/(2*100) = 0.5
        self.assertAlmostEqual(r.contribution(Doc(words(20))), 0.5)
        # four sentences averaging 20: sd shrinks by 2 -> 4 * 0.5 = 2.0, = sum of per-unit LLRs
        d = Doc(" ".join([words(20)] * 4))
        self.assertEqual(len(B.sentence_lengths(d)), 4)
        self.assertAlmostEqual(r.contribution(d), 2.0)
        hits, resid = r.evidence(d)
        self.assertEqual(len(hits), 4)
        self.assertAlmostEqual(sum(h.share for h in hits) + resid, 2.0)

    def test_unequal_sd_and_clip(self):
        r = B.StatRule("t.stat", units=B.sentence_lengths, mu_o=10, mu_t=20, sd_o=5, sd_t=10,
                       unit_clip=(1, 30), cap=100)
        x = 30.0  # 100-word sentence clipped to 30
        expect = (-0.5 * ((x - 20) / 10) ** 2 - math.log(10)) - (-0.5 * ((x - 10) / 5) ** 2 - math.log(5))
        self.assertAlmostEqual(r.contribution(Doc(words(100))), expect)

    def test_skips_address(self):
        d = Doc("Herr talman! " + words(20))
        self.assertEqual([v for v, _ in B.sentence_lengths(d)], [20])


class TestPredicateRule(unittest.TestCase):
    def test_spans_hook(self):
        def fn(doc):
            return 0.5 * doc.first_words.count("vi")

        def spans(doc):
            return [(s, s + 2, 0.5, "subject-first 'vi'") for (s, e), w in zip(doc.sent_spans, doc.first_words)
                    if w == "vi"]

        r = B.PredicateRule("t.pred", fn=fn, spans_fn=spans, direction="T", cap=0.8)
        d = Doc("Vi går. Nu går vi. Vi stannar.")
        self.assertAlmostEqual(r.contribution(d), 0.8)
        hits, resid = r.evidence(d)
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(sum(h.share for h in hits) + resid, 0.8)
        with self.assertRaises(ValueError):
            B.PredicateRule("t.nodir", fn=fn)


class TestEngine(unittest.TestCase):
    def rules(self):
        return [
            B.HitRule("a.one", words=["foo"], per_hit=2.0, cap=10, family="a"),
            B.HitRule("a.two", words=["bar"], per_hit=2.0, cap=10, family="a"),
            B.HitRule("b.one", words=["baz"], per_hit=-1.0, cap=10, family="b"),
        ]

    def test_family_caps_and_explain(self):
        e = Engine(rules=self.rules(), family_caps={"a": 3.0})
        s, ex = e.score_and_explain("foo bar baz")
        self.assertAlmostEqual(s, 3.0 - 1.0)
        self.assertAlmostEqual(ex[CAP_PREFIX + "a"], -1.0)
        self.assertAlmostEqual(sum(ex.values()), s)
        self.assertNotIn(CAP_PREFIX + "b", ex)
        self.assertEqual(e.explain("nothing"), {})
        self.assertEqual(e.score("nothing"), 0.0)

    def test_selection(self):
        self.assertEqual([r.name for r in Engine(rules=self.rules(), families="b").rules], ["b.one"])
        self.assertEqual([r.name for r in Engine(rules=self.rules(), exclude="a.two,b").rules], ["a.one"])
        with self.assertRaises(ValueError):
            Engine(rules=self.rules(), exclude="nope")
        e = Engine(rules=self.rules(), weights={"a.one": 0.5})
        self.assertAlmostEqual(e.explain("foo")["a.one"], 1.0)

    def test_disabled(self):
        rs = self.rules() + [B.HitRule("c.off", words=["qux"], per_hit=1.0, family="c", enabled=False)]
        self.assertNotIn("c.off", [r.name for r in Engine(rules=rs).rules])
        self.assertIn("c.off", [r.name for r in Engine(rules=rs, include_disabled=True).rules])


class TestSpans(unittest.TestCase):
    """Offsets of explain_spans index the ORIGINAL text, also when normalisation
    changes lengths (curly quotes, dashes, double spaces, NBSP, soft hyphens, ...)."""

    def engine(self):
        return Engine(rules=[
            B.ChoiceRule("s.dem", a=r"\b(?:den|det)\s+här\b", b=r"\b(?:denna|detta)\b", p_o=0.1, p_t=0.3),
            B.HitRule("s.quote", pattern=r'"\w+"', per_hit=0.5),
            B.HitRule("s.dash", pattern="–", per_hit=0.1),
            B.RateRule("s.ju", words=["ju"], rate_o=2, rate_t=0.5),
            B.StatRule("s.len", units=B.sentence_lengths, mu_o=5, mu_t=10, sd=3, span_min=0.0),
        ])

    TEXTS = [
        "Detta  är ”viktigt” — och det  här är ju  bra.",
        "  “Den­na”  fråga – det\n\nhär … ju.  Herr talman!  Det här är bra.",
        "å det här, ju ‘x’ ”y”.",
    ]

    def test_offsets_round_trip(self):
        e = self.engine()
        for t in self.TEXTS:
            spans = e.explain_spans(t)
            self.assertTrue(spans, t)
            for s in spans:
                self.assertEqual(t[s["start"]:s["end"]], s["text"])

    def test_expected_surfaces(self):
        e = self.engine()
        t = self.TEXTS[0]
        got = {(s["rule"], s["text"]) for s in e.explain_spans(t)}
        self.assertIn(("s.dem", "Detta"), got)
        self.assertIn(("s.dem", "det  här"), got)              # double space kept in the original
        self.assertIn(("s.quote", "”viktigt”"), got)  # curly quotes in the original
        self.assertIn(("s.dash", "—"), got)                # em dash normalised to en dash
        self.assertIn(("s.ju", "ju"), got)
        t = self.TEXTS[1]
        got = {(s["rule"], s["text"]) for s in e.explain_spans(t)}
        self.assertIn(("s.dem", "det\n\nhär"), got)
        self.assertIn(("s.quote", "“Den­na”"), got)  # soft hyphen inside

    def test_annotate_json_and_consistency(self):
        e = self.engine()
        for t in self.TEXTS:
            a = e.annotate(t)
            json.dumps(a)
            self.assertAlmostEqual(a["score"], e.score(t))
            # per rule: span shares + doc-level residual = rule contribution
            tot = {}
            for s in a["spans"]:
                tot[s["rule"]] = tot.get(s["rule"], 0.0) + s["contribution_share"]
            for d in a["doc_level"]:
                tot[d["rule"]] = tot.get(d["rule"], 0.0) + d["contribution"]
            for rule, c in a["rules"].items():
                self.assertAlmostEqual(tot.get(rule, 0.0), c, places=9, msg=rule)
            for key in ("score", "prob", "label", "rules", "spans", "doc_level"):
                self.assertIn(key, a)


class TestDiscovery(unittest.TestCase):
    def test_discovers_starter(self):
        rules, caps, modules = discover()
        names = {r.name for r in rules}
        self.assertIn("starter.modal_particles", names)
        self.assertEqual(len(names), len(rules))  # unique names
        for r in rules:
            self.assertIn(r.direction, ("T", "O"), r.name)
            self.assertTrue(r.family, r.name)
            self.assertTrue(r.rationale and r.source, r.name)
        self.assertIn("starter", modules)
        self.assertNotIn("base", modules)

    def test_harness_constructors(self):
        self.assertTrue(Default().rules)
        f = Families(only="starter", exclude="starter.sent_len")
        self.assertNotIn("starter.sent_len", [r.name for r in f.rules])
        self.assertTrue(all(r.family == "starter" for r in f.rules))
        with self.assertRaises(ValueError):
            Families(only="no_such_family")

    def test_harness_loader(self):
        import os
        import sys
        evals = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "evals")
        sys.path.insert(0, evals)
        import run
        clf = run.load_clf('tsw.engine:Families(only="starter", exclude="starter.sent_len")')
        s, e = clf.score_and_explain("Det är ju så.")
        self.assertLess(s, 0)
        self.assertIn("starter.modal_particles", e)


if __name__ == "__main__":
    unittest.main()

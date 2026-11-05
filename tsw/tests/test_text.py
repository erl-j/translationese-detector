"""python3 -m unittest discover -s tsw/tests -t . -v"""
import unicodedata
import unittest

from tsw import text as T
from tsw.doc import Doc


def sents(s):
    n = T.normalise(s)
    return [n[a:b] for a, b in T.split_sentences(n)]


class TestNormalise(unittest.TestCase):
    CASES = [
        "Detta är ”viktigt” – och  det här är ‘bra’…",
        "  leading and trailing  ",
        "soft­hyphen och zero​width",
        "a ­ b",
        "decomposed å and ö",  # å, ö written as base + combining mark
        "tabs\tand\nnewlines\r\n\n here",
        "em—dash, minus − 5, «guillemets»",
        "",
        "   ",
    ]

    def test_fast_and_mapped_agree(self):
        for s in self.CASES:
            norm, starts, ends = T.normalise_with_map(s)
            self.assertEqual(norm, T.normalise(s), repr(s))
            self.assertEqual(len(starts), len(norm))
            self.assertEqual(len(ends), len(norm))

    def test_unifies(self):
        self.assertEqual(T.normalise("“x” ‘y’"), "\"x\" 'y'")
        self.assertEqual(T.normalise("a—b"), "a–b")
        self.assertEqual(T.normalise("a  \n b c"), "a b c")
        self.assertEqual(T.normalise("x…"), "x...")
        self.assertEqual(T.normalise("å"), "å")

    def test_map_points_back(self):
        s = "Detta  ”är” å…"
        d = Doc(s)
        i = d.text.index('"är"')
        a, b = d.to_orig(i, i + 4)
        self.assertEqual(s[a:b], "”är”")
        j = d.text.index("å")
        a, b = d.to_orig(j, j + 1)
        self.assertEqual(unicodedata.normalize("NFC", s[a:b]), "å")
        a, b = d.to_orig(len(d.text) - 3, len(d.text))
        self.assertEqual(s[a:b], "…")

    def test_lower_same_length(self):
        s = "İstanbul ÅSA"
        self.assertEqual(len(T.lower_same_length(s)), len(s))


class TestSplitter(unittest.TestCase):
    def test_abbreviations(self):
        self.assertEqual(sents("Vi har t.ex. fisket, bl.a. torsk, dvs. d.v.s. allt. Nu går vi vidare."),
                         ["Vi har t.ex. fisket, bl.a. torsk, dvs. d.v.s. allt.", "Nu går vi vidare."])
        self.assertEqual(sents("Se art. 5 och kap. 3, jfr. prop. 12. Fr.o.m. i dag gäller det."),
                         ["Se art. 5 och kap. 3, jfr. prop. 12.", "Fr.o.m. i dag gäller det."])
        self.assertEqual(sents("Det kostar ca. 5 kr, nr. 7 är dyrare. A. Smith höll med."),
                         ["Det kostar ca. 5 kr, nr. 7 är dyrare.", "A. Smith höll med."])

    def test_end_abbreviations(self):
        self.assertEqual(sents("Vi talade om fisk, jordbruk osv. Sedan röstade vi."),
                         ["Vi talade om fisk, jordbruk osv.", "Sedan röstade vi."])
        self.assertEqual(sents("Fisk, kött m.m. Det var allt."), ["Fisk, kött m.m.", "Det var allt."])

    def test_numbers(self):
        self.assertEqual(sents("Tillväxten var 1.5 procent och 2,3 procent i år. 2004 blev bättre."),
                         ["Tillväxten var 1.5 procent och 2,3 procent i år.", "2004 blev bättre."])

    def test_address_and_questions(self):
        self.assertEqual(sents("Herr talman! Jag vill tacka. Varför? Därför att det är bra!"),
                         ["Herr talman!", "Jag vill tacka.", "Varför?", "Därför att det är bra!"])
        d = Doc("Herr talman! Fru kommissionär, kära kolleger! Jag vill tacka.")
        self.assertEqual(d.is_address, [True, True, False])

    def test_no_split_before_lowercase_or_stranded(self):
        self.assertEqual(sents("Det är det vi lever i. Sen kom han."), ["Det är det vi lever i.", "Sen kom han."])
        self.assertEqual(sents("Han sa nej. och gick."), ["Han sa nej. och gick."])

    def test_quotes(self):
        self.assertEqual(sents("Hon sa: ”Nej.” Sedan gick hon."), ['Hon sa: "Nej."', "Sedan gick hon."])


class TestTokenizer(unittest.TestCase):
    def toks(self, s):
        return T.tokenize(T.normalise(s))[0]

    def test_words_compounds_numbers(self):
        self.assertEqual(self.toks("EU:s miljö- och klimatpolitik kostar 1 000 000,5 kr eller 2,3 %."),
                         ["EU:s", "miljö-", "och", "klimatpolitik", "kostar", "1 000 000,5", "kr", "eller",
                          "2,3", "%", "."])
        self.assertEqual(self.toks("2000-talet, G8-länderna och Côte d'Ivoire."),
                         ["2000-talet", ",", "G8-länderna", "och", "Côte", "d'Ivoire", "."])

    def test_abbreviation_tokens(self):
        self.assertEqual(self.toks("T.ex. dvs. bl.a. (12:30)..."),
                         ["T.ex.", "dvs.", "bl.a.", "(", "12:30", ")", "..."])

    def test_offsets(self):
        s = T.normalise("Det  här, t.ex. 2,3 procent!")
        toks, starts = T.tokenize(s)
        for t, a in zip(toks, starts):
            self.assertEqual(s[a:a + len(t)], t)

    def test_doc_views(self):
        d = Doc("Herr talman! Det här är, t.ex., bra. Vi röstar ja.")
        self.assertEqual(d.n_sentences, 3)
        self.assertEqual(d.first_words, ["herr", "det", "vi"])
        self.assertEqual(d.sent_words[1], ["det", "här", "är", "t.ex.", "bra"])
        self.assertEqual(d.n_words, 2 + 5 + 3)
        self.assertEqual(d.counts["det"], 1)
        self.assertEqual(d.punct[","], 2)


if __name__ == "__main__":
    unittest.main()

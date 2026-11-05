"""calques family: every lexicon entry fires on its calque and not on its native
alternative; exclusions; metamorphic tests (insert a calque -> score up,
replace it by the native alternative -> score down)."""
import unittest

from tsw.doc import Doc
from tsw.engine import Default, Families
from tsw.rules import calques as C
from tsw.rules import discover

RULES = {r.name: r for r in C.RULES}
LEX = RULES["calques.lexicon"]
NAT = RULES["calques.native_counterparts"]

# (calque sentence, native rewrite) for every lexicon entry, keyed by a label substring
PAIRS = {
    "det faktum att": ("Jag beklagar det faktum att rådet inte har agerat.",
                       "Jag beklagar att rådet inte har agerat."),
    "först av allt": ("Först av allt vill jag tacka föredraganden.",
                      "Först och främst vill jag tacka föredraganden."),
    "spendera": ("Vi måste spendera pengarna klokt.", "Vi måste använda pengarna klokt."),
    "på marken": ("Situationen på marken är fortfarande svår.", "Situationen på plats är fortfarande svår."),
    "på X basis": ("Uppgifterna samlas in på daglig basis.", "Uppgifterna samlas in dagligen."),
    "med detta sagt": ("Med detta sagt stöder vi förslaget.", "Trots det stöder vi förslaget."),
    "på samma gång": ("På samma gång måste vi vara realistiska.", "Vi måste ändå vara realistiska."),
    "det är tydligt att": ("Det är tydligt att reformen behövs.", "Det står fast att reformen behövs."),
    "de av oss": ("De av oss som deltog vet hur svårt det var.", "Vi som deltog vet hur svårt det var."),
    "i det förflutna": ("Så har det varit i det förflutna.", "Så har det varit tidigare."),
    "i det att": ("Förslaget är bra i det att det stärker kontrollen.",
                  "Förslaget är bra genom att det stärker kontrollen."),
    "med avseende på": ("Vi har farhågor med avseende på finansieringen.",
                        "Vi har farhågor när det gäller finansieringen."),
    "när det rör": ("Vi är eniga när det rör budgeten.", "Vi är eniga när det gäller budgeten."),
    "Rörande": ("Rörande budgeten har vi inga invändningar.",
                "När det gäller budgeten har vi inga invändningar."),
    "känner till att": ("Jag känner till att frågan är känslig.", "Jag vet att frågan är känslig."),
    "möjligt för X att": ("Det är inte möjligt för oss att godta detta.", "Vi kan inte godta detta."),
    "detta särskilda": ("Detta särskilda problem måste lösas.", "Just detta problem måste lösas."),
    "låt oss vara tydliga": ("Låt oss vara tydliga om målen.", "Vi måste säga klart vilka målen är."),
    "tycks mig": ("Det tycks mig som om förslaget är svagt.", "Jag tycker att förslaget är svagt."),
    "håller fullständigt med": ("Jag håller fullständigt med föregående talare.",
                                "Jag håller helt med föregående talare."),
    "inflytande på": ("Parlamentet har litet inflytande på besluten.",
                      "Parlamentet har litet inflytande över besluten."),
    "under dessa omständigheter": ("Under dessa omständigheter kan vi inte rösta ja.",
                                   "I det läget kan vi inte rösta ja."),
    "hårda arbete": ("Jag tackar föredraganden för hennes hårda arbete.",
                     "Jag tackar föredraganden för hennes stora arbete."),
    "misslyckande (med) att": ("Kommissionens misslyckande med att agera är allvarligt.",
                               "Att kommissionen inte har agerat är allvarligt."),
    "ta en titt": ("Vi bör ta en närmare titt på förslaget.", "Vi bör granska förslaget närmare."),
    "det slags": ("Det slags politik leder ingenstans.", "Den sortens politik leder ingenstans."),
    "det antal": ("Vi oroas över det antal människor som drabbas.",
                  "Vi oroas över antalet människor som drabbas."),
    "skälet till varför": ("Det är skälet till varför vi röstar nej.", "Det är skälet till att vi röstar nej."),
    "vid en tid då": ("Detta sker vid en tid då arbetslösheten ökar.",
                      "Detta sker i en tid då arbetslösheten ökar."),
    "i sig själv": ("Avtalet är i sig självt inte tillräckligt.", "Avtalet är i sig inte tillräckligt."),
    "i ett försök att": ("De ändrade reglerna i ett försök att spara pengar.",
                         "De ändrade reglerna för att försöka spara pengar."),
    "bara rätt": ("Det är bara rätt och riktigt att de får ersättning.",
                  "Det är rimligt att de får ersättning."),
    "argumentera att": ("Man kan argumentera att reglerna är för stränga.",
                        "Man kan hävda att reglerna är för stränga."),
    "röra oss framåt": ("Vi måste röra oss framåt i frågan.", "Vi måste gå vidare i frågan."),
    "jämn spelplan": ("Företagen behöver en jämn spelplan.", "Företagen behöver lika villkor."),
    "Så vi": ("Så vi måste agera nu.", "Så måste vi agera nu."),
    "de senaste två": ("Under de senaste två åren har mycket hänt.", "Under de två åren har mycket hänt."),
    "denna eftermiddag": ("Vi diskuterade frågan denna eftermiddag.", "Vi diskuterade frågan i eftermiddag."),
    "det är en fråga om": ("Det är en fråga om rättvisa.", "Det handlar om rättvisa."),
    "bästa praxis": ("Vi bör sprida bästa praxis mellan länderna.", "Vi bör sprida goda exempel mellan länderna."),
    "välkomna": ("Jag välkomnar betänkandet.", "Jag ser positivt på betänkandet."),
    "säkerställa": ("Vi måste säkerställa att reglerna följs.", "Vi måste se till att reglerna följs."),
    "se fram emot": ("Jag ser fram emot samarbetet.", "Jag hoppas på ett gott samarbete."),
}


def lex_labels(text):
    hits, _ = LEX.evidence(Doc(text))
    return [h.label for h in hits]


class TestLexiconEntries(unittest.TestCase):
    def test_every_entry_has_a_pair(self):
        for label in C.CALQUES:
            self.assertTrue(any(k in label for k in PAIRS), "no test pair for %r" % label)

    def test_calque_fires_native_does_not(self):
        for key, (calque, native) in PAIRS.items():
            with self.subTest(entry=key):
                labs = lex_labels(calque)
                self.assertTrue(any(key in l for l in labs), "%r: %r not hit in %r" % (key, labs, calque))
                self.assertEqual(lex_labels(native), [], "native rewrite fires: %r" % native)

    def test_values_rounded_and_bounded(self):
        for label, (_, v) in list(C.CALQUES.items()) + list(C.NATIVE.items()):
            self.assertLessEqual(abs(v), 1.5, label)
            self.assertEqual(float("%.2g" % v), v, label)       # <= 2 significant digits
            self.assertIn("EN '", label) if v > 0 else self.assertIn("vs '", label)

    def test_exclusions(self):
        none = [
            "Vad vill vi uppnå med detta?",                    # question, not pseudo-cleft
            "När det rör sig om barn måste vi vara försiktiga.",  # native "rör sig om"
            "Det sker på frivillig basis.",                    # accepted Swedish
            "Planet landade på marken vid flygplatsen.",       # literal (followed by 'vid')
            "Så att vi kan gå vidare.",                         # 'så att' is not discourse 'so'
            "Tunnorna låg på marken i hamnen.",
        ]
        for t in none:
            with self.subTest(text=t):
                self.assertEqual(lex_labels(t), [])

    def test_sentence_initial_only(self):
        self.assertTrue(lex_labels("Herr talman! Så vi måste agera."))
        self.assertEqual(lex_labels("Det är inte så vi arbetar."), [])
        self.assertEqual(lex_labels("Frågor rörande budgeten."), [])
        self.assertTrue(lex_labels("Vi är överens. Rörande budgeten har vi inga invändningar."))


class TestNativeAndChoices(unittest.TestCase):
    def test_native_counterparts(self):
        for t in ["Det är klart att reformen behövs.", "Parlamentet har inflytande över budgeten."]:
            with self.subTest(text=t):
                self.assertLess(NAT.contribution(Doc(t)), 0)
        # integration: sentence-initial 'Samtidigt' / 'Först vill jag' are wordorder.fronted_theme's
        for t in ["Samtidigt måste vi vara realistiska.", "Först vill jag tacka föredraganden."]:
            with self.subTest(text=t):
                self.assertEqual(NAT.contribution(Doc(t)), 0.0)
        self.assertEqual(NAT.contribution(Doc("Samtidigt som vi talar händer det.")), 0.0)

    def test_skulle_vilja(self):
        r = RULES["calques.skulle_vilja"]
        self.assertGreater(r.contribution(Doc("Jag skulle vilja tacka kommissionären.")), 0)
        self.assertLess(r.contribution(Doc("Jag vill tacka kommissionären.")), 0)
        self.assertGreater(r.contribution(Doc("Därför skulle jag också vilja säga något.")), 0)

    def test_forsoka_att(self):
        r = RULES["calques.forsoka_att"]
        self.assertGreater(r.contribution(Doc("Vi måste försöka att hitta en lösning.")), 0)
        self.assertLess(r.contribution(Doc("Vi måste försöka hitta en lösning.")), 0)
        self.assertEqual(r.contribution(Doc("Det var ett försök att hitta en lösning.")), 0.0)

    def test_pied_piped(self):
        r = RULES["calques.pied_piped_relative"]
        self.assertGreater(r.contribution(Doc("Det är en process i vilken alla deltar.")), 0)
        self.assertGreater(r.contribution(Doc("De länder från vilka de kommer.")), 0)
        # interrogative / determiner uses and indirect questions are native
        for t in ["I vilken utsträckning gäller detta?", "Det beror på vilket sätt man väljer.",
                  "Frågan om vilka åtgärder som behövs.", "I vilket fall som helst röstar vi nej."]:
            with self.subTest(text=t):
                self.assertEqual(r.contribution(Doc(t)), 0.0)
        self.assertEqual(r.contribution(Doc("Det är en process som alla deltar i.")), 0.0)


class TestMetamorphic(unittest.TestCase):
    BASE = ("Herr talman! Betänkandet tar upp viktiga frågor om det inre marknaden. "
            "Kommissionen har lagt fram ett förslag. {} Vi hoppas att rådet följer parlamentet.")

    def setUp(self):
        self.eng = Default()
        self.fam = Families(only="calques")

    def test_insert_raises_and_native_lowers(self):
        neutral = "Vi har diskuterat detta länge."
        for key, (calque, native) in PAIRS.items():
            with self.subTest(entry=key):
                s0 = self.fam.score(self.BASE.format(neutral))
                s_c = self.fam.score(self.BASE.format(calque))
                s_n = self.fam.score(self.BASE.format(native))
                self.assertGreater(s_c, s0, "inserting %r did not raise the score" % calque)
                self.assertLess(s_n, s_c, "native %r did not lower the score" % native)
                # in the whole engine, the calques family's share moves the same way
                fam_c = sum(v for k, v in self.eng.explain(self.BASE.format(calque)).items()
                            if k.startswith("calques.") or k == "cap:calques")
                self.assertGreater(fam_c, 0, key)

    def test_native_counterpart_below_neutral(self):
        s0 = self.fam.score(self.BASE.format("Vi har diskuterat detta länge."))
        s_n = self.fam.score(self.BASE.format("Det är klart att reformen behövs."))
        self.assertLess(s_n, s0)

    def test_family_cap(self):
        many = " ".join(c for c, _ in PAIRS.values())
        self.assertAlmostEqual(LEX.contribution(Doc(many)), LEX.cap)          # rule cap binds
        self.assertLessEqual(abs(self.fam.score(many)), C.FAMILY_CAP + 1e-9)
        lots = " ".join(["Vi försöker att hjälpa dem. Det är tydligt att det faktum att det slags "
                         "politik behövs."] * 5)
        self.assertAlmostEqual(self.fam.score(lots), C.FAMILY_CAP)
        self.assertIn("cap:calques", self.fam.explain(lots))

    def test_pseudo_cleft_left_to_wordorder(self):
        self.assertEqual(lex_labels("Vad vi behöver är en gemensam strategi."), [])

    def test_single_calque_bounded(self):
        for label, (_, v) in C.CALQUES.items():
            self.assertLessEqual(v, 1.5)


class TestDiscovery(unittest.TestCase):
    def test_registered_and_starter_calques_disabled(self):
        rules, caps, _ = discover()
        names = {r.name: r for r in rules}
        for n in RULES:
            self.assertIn(n, names)
            self.assertEqual(names[n].family, "calques")
        self.assertEqual(caps.get("calques"), 3.0)
        self.assertFalse(names["starter.calques"].enabled)
        self.assertNotIn("starter.calques", [r.name for r in Default().rules])


if __name__ == "__main__":
    unittest.main()

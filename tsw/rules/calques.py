"""calques: English interference in lexis and phraseology.

Word-for-word renderings of English multi-word expressions (*det faktum att*
<- "the fact that", *de av oss som* <- "those of us who"), English-patterned
constructions (pied-piped *i vilket* <- "in which", *försöka att* <- "try to",
non-V2 *Så vi ...* <- "So we ..."), English collocations and anglicisms where
Swedish has a native alternative. Every entry names its English source and a
native alternative; entries whose native alternative is itself clearly
O-leaning get a mirror entry in `calques.native_counterparts`.

Selection. Candidates came from docs/literature.md (A12-A18, C10, §4.6) and
from `python3 -m tsw.tools.collocations` (T-vs-O n-gram contrast). Each item
was then checked one by one with KWIC lines, --by-speaker and per-year rates
and kept only if (a) it is a recognisable English construction with a native
Swedish alternative, (b) it leans T, (c) it is produced by many speakers
(>= ~10 distinct translated-class speakers; Commission boilerplate and
idiolect dropped), (d) it contains no topic word. Literature items with no
lean on EP dev (i termer av, på basis av, i ljuset av, baserat på, i relation
till, sist men inte minst, i linje med, jag är rädd att, leverera,
inkluderande) or a reversed lean (implementera, det är därför) are NOT used.
Native idioms that translators prefer (när allt kommer omkring, uppriktigt
sagt, först och främst, ta itu med) are not calques and are left to other
families.

Strengths. Per-hit values are hand-rounded log rate ratios ln(r_T/r_O) read
off dev-train counts; for items (almost) absent from originals the ratio is
taken against an O count of 1-2 (a conservative bound, not infinity) and
capped at 1.5, and lowered further for time drift, few speakers or O usage
in 1997. No single calque should decide a short text on its own against
strong counter-evidence: per-hit <= 1.5, family cap 3.

Calibration: dev-train only (dev sessions < 2002-03-01), dev.sent.all, counts
O/T per entry in the comments as "O cO / T cT (rO/rT per 1000 words; sO/sT
distinct speakers)". O = 0.57M words, T = 2.4M words. 2026-11-05. An earlier
screening pass (before the dev-train/dev-val protocol) looked at full dev; every entry was
re-checked and its value re-read on dev-train only.
(Protocol since revised: trn = all of the old dev; dev-train is a subset of trn, so the
values stand.)
"""
from .base import ChoiceRule, HitRule

FAMILY = "calques"
FAMILY_CAP = 3.0

_B = r"(?<![\w-])"          # word start (not inside a word or hyphenated compound)
_E = r"(?![\w-])"           # word end
_S0 = r"(?:^|(?<=[.!?] )|(?<=[.!?]\" ))"   # sentence start (on doc.lower)
_PRON = r"(?:oss|dem|mig|er|dig|honom|henne)"
_NUM = r"(?:två|tre|fyra|fem|sex|sju|åtta|nio|tio|elva|tolv|femton|tjugo)"

CALQUES = {
    # "the fact that" -> native: att-clause, "det att", "att ... faktiskt".
    # Used by ~30 Swedish MEPs already in 1997 -> modest value.
    # dev-train: O 50 / T 927 (0.087/0.384 per 1000; 24/171 speakers; ln T/O +1.47)
    "calque 'det faktum att' (EN 'the fact that')":
        (_B + r"(?:det\s+faktum|faktumet)\s+att" + _E, 1.0),
    # "first of all" -> native: "först och främst", "för det första", "Först vill jag".
    # dev-train: O 2 / T 125 (0.004/0.052 per 1000; 2/61 speakers; ln T/O +2.48)
    "calque 'först av allt' (EN 'first of all')":
        (_B + r"först\s+av\s+allt" + _E, 1.2),
    # "spend (money/time)" -> native: "lägga (pengar) på", "använda", "tillbringa".
    # dev-train: O 3 / T 83 (0.005/0.034 per 1000; 3/44 speakers; ln T/O +1.73)
    "anglicism 'spendera' (EN 'spend')":
        (_B + r"spender(?:a|ar|ade|at|as|ades)" + _E, 0.9),
    # "on the ground" -> native: "på plats", "i praktiken", "i fält". Literal readings
    # (landing on the ground) are rare in EP.
    # dev-train: O 0 / T 14 (0.000/0.006 per 1000; 0/14 speakers; ln T/O +1.93)
    "calque 'på marken' (EN 'on the ground')":
        (_B + r"på\s+marken" + _E + r"(?!\s+(?:i|vid|nära)" + _E + ")", 1.0),
    # "on a daily/regular/permanent basis" -> native: adverb "dagligen", "regelbundet",
    # "permanent". "på frivillig basis" is accepted Swedish and excluded.
    # dev-train: O 0 / T 19 (0.000/0.008 per 1000; 0/13 speakers; ln T/O +2.22)
    "calque 'på X basis' (EN 'on a X basis')":
        (_B + r"på\s+(?:daglig|regelbunden|permanent|tillfällig|årlig|löpande|individuell|bilateral|"
              r"ad\s+hoc|kontinuerlig|systematisk)\s+basis" + _E, 1.0),
    # "having said that / that said" -> native: "samtidigt", "trots det", "ändå".
    # dev-train: O 1 / T 38 (0.002/0.016 per 1000; 1/27 speakers; ln T/O +1.81)
    "calque 'med detta sagt' (EN 'having said that')":
        (_B + r"med\s+(?:det|detta)\s+sagt" + _E, 1.2),
    # "at the same time" (discourse connective) -> native: "samtidigt".
    # dev-train: O 4 / T 136 (0.007/0.056 per 1000; 3/71 speakers; ln T/O +1.97)
    "calque 'på samma gång' (EN 'at the same time')":
        (_B + r"på\s+samma\s+gång" + _E, 1.5),
    # "it is clear that" -> native: "det är klart att", "det står klart att", "uppenbarligen".
    # dev-train: O 2 / T 165 (0.004/0.068 per 1000; 2/70 speakers; ln T/O +2.75)
    "calque 'det är tydligt att' (EN 'it is clear that')":
        (r"(?:" + _B + r"det\s+(?:är|var|blir)|" + _B + r"är\s+det)\s+(?:\w+\s+)?tydligt\s+att" + _E, 1.5),
    # "those of us who" -> native: "vi som".
    # dev-train: O 0 / T 44 (0.000/0.018 per 1000; 0/35 speakers; ln T/O +3.05)
    "calque 'de av oss (som)' (EN 'those of us (who)')":
        (_B + r"de\s+av\s+oss" + _E, 1.5),
    # "in the past" -> native: "tidigare", "förr", "förut".
    # dev-train: O 0 / T 54 (0.000/0.022 per 1000; 0/39 speakers; ln T/O +3.25)
    "calque 'i det förflutna' (EN 'in the past')":
        (_B + r"i\s+det\s+förflutna" + _E, 1.5),
    # "in that" (= because, insofar as) -> native: "genom att", "i och med att", "eftersom".
    # dev-train: O 0 / T 50 (0.000/0.021 per 1000; 0/33 speakers; ln T/O +3.18)
    "calque 'i det att' (EN 'in that')":
        (_B + r"i\s+det\s+att" + _E, 1.2),
    # "with regard/respect to" -> native: "när det gäller", "beträffande", "i fråga om".
    # dev-train: O 3 / T 321 (0.005/0.133 per 1000; 3/115 speakers; ln T/O +3.08)
    "calque 'med avseende på' (EN 'with regard to')":
        (_B + r"med\s+avseende\s+på" + _E, 1.5),
    # "as regards / concerning / when it comes to" (EN concern -> röra) -> native:
    # "när det gäller", "vad gäller". Not "när det rör sig om" (native: when it is a matter of).
    # dev-train: O 0 / T 317 (0.000/0.131 per 1000; 0/85 speakers; ln T/O +5.01)
    "calque 'när det rör' (EN 'as regards / when it concerns')":
        (_B + r"när\s+det\s+rör" + _E + r"(?!\s+sig" + _E + ")", 1.5),
    # sentence-initial "Regarding/Concerning X, ..." -> native: "När det gäller X", "Vad gäller X".
    # dev-train: O 0 / T 47 (0.000/0.019 per 1000; 0/34 speakers; ln T/O +3.11)
    "calque sentence-initial 'Rörande' (EN 'Regarding')":
        (_S0 + r"rörande" + _E, 1.2),
    # "be aware that / know that" -> native: "vet att", "är medveten om att"
    # ("känna till" takes a noun phrase).
    # dev-train: O 3 / T 105 (0.005/0.043 per 1000; 3/63 speakers; ln T/O +1.97)
    "calque 'känner till att' (EN 'be aware that')":
        (_B + r"kän(?:na|ner|de|t)\s+till\s+att" + _E, 1.0),
    # "make it possible for X to / it is possible for us to" (for-to infinitive) ->
    # native: "möjliggöra", "ge X möjlighet att", "vi kan".
    # dev-train: O 15 / T 337 (0.026/0.140 per 1000; 13/103 speakers; ln T/O +1.64)
    "calque 'möjligt för X att' (EN 'possible for X to')":
        (_B + r"(?:göra|gör|gjorde|gjort|görs)\s+det\s+möjligt\s+för" + _E + "|" +
         _B + r"möjligt\s+för\s+" + _PRON + r"\s+att" + _E, 1.2),
    # "this particular X" -> native: "just detta", "detta", "den här".
    # dev-train: O 1 / T 168 (0.002/0.070 per 1000; 1/77 speakers; ln T/O +3.28)
    "calque 'detta särskilda' (EN 'this particular')":
        (_B + r"(?:denna|detta|dessa|den\s+här|det\s+här|de\s+här)\s+(?:särskilda|speciella)" + _E, 1.5),
    # "let us be clear/honest" -> native: "låt mig säga klart", "ärligt talat", "vi ska vara medvetna om".
    # dev-train: O 0 / T 21 (0.000/0.009 per 1000; 0/18 speakers; ln T/O +2.32)
    "calque 'låt oss vara tydliga' (EN 'let us be clear')":
        (_B + r"låt\s+(?:oss|mig)\s+vara\s+(?:tydliga|tydlig|ärliga|ärlig|klara|klar|realistiska|"
              r"uppriktiga|uppriktig|helt)" + _E, 1.2),
    # "it seems to me" -> native: "jag tycker", "jag anser", "som jag ser det".
    # dev-train: O 1 / T 61 (0.002/0.025 per 1000; 1/46 speakers; ln T/O +2.27)
    "calque 'tycks mig' (EN 'it seems to me')":
        (_B + r"(?:tycks|förefaller|verkar)\s+mig" + _E, 1.5),
    # "I fully agree" -> native: "jag håller helt med", "jag instämmer helt".
    # dev-train: O 0 / T 28 (0.000/0.012 per 1000; 0/22 speakers; ln T/O +2.60)
    "calque 'håller fullständigt med' (EN 'fully agree')":
        (_B + r"(?:håller|höll|hålla)\s+(?:\w+\s+)?fullständigt\s+med" + _E, 1.2),
    # "influence on" -> native: "inflytande över" (O-leaning, see native_counterparts).
    # dev-train: O 0 / T 33 (0.000/0.014 per 1000; 0/21 speakers; ln T/O +2.77)
    "calque 'inflytande på' (EN 'influence on')":
        (_B + r"inflytande(?:t)?\s+på" + _E, 1.2),
    # "under/in these circumstances" -> native: "i det läget", "under sådana förhållanden".
    # dev-train: O 0 / T 67 (0.000/0.028 per 1000; 0/35 speakers; ln T/O +3.47)
    "calque 'under dessa omständigheter' (EN 'under these circumstances')":
        (_B + r"under\s+(?:dessa|sådana|de\s+här)\s+omständigheter" + _E, 1.0),
    # "(thank you for) the hard work" -> native: "det stora/omfattande arbete", "det arbete".
    # dev-train: O 2 / T 81 (0.004/0.034 per 1000; 2/46 speakers; ln T/O +2.05)
    "calque 'hårda arbete' (EN 'hard work')":
        (_B + r"(?:hårda|hårt)\s+arbete" + _E, 1.2),
    # "X's failure to" -> native: "att X inte har lyckats", "X:s oförmåga att".
    # dev-train: O 3 / T 80 (0.005/0.033 per 1000; 3/45 speakers; ln T/O +1.70)
    "calque 'misslyckande (med) att' (EN 'failure to')":
        (_B + r"misslyckande(?:t|n|na)?\s+(?:med\s+)?att" + _E, 1.2),
    # "have/take a look at" -> native: "titta på", "se på", "granska".
    # dev-train: O 0 / T 52 (0.000/0.021 per 1000; 0/35 speakers; ln T/O +3.21)
    "calque 'ta en titt' (EN 'take a look')":
        (_B + r"(?:ta|tar|tog|tagit)\s+en\s+(?:\w+\s+)?titt" + _E, 1.2),
    # "this/that kind of" -> native: "den sortens", "sådan(a)", "den typen av".
    # dev-train: O 0 / T 94 (0.000/0.039 per 1000; 0/48 speakers; ln T/O +3.80)
    "calque 'det slags' (EN 'that kind of')":
        (_B + r"(?:det|den|detta|denna)\s+slags" + _E, 1.5),
    # "the number of X who" -> native: "antalet X som".
    # dev-train: O 0 / T 55 (0.000/0.023 per 1000; 0/44 speakers; ln T/O +3.27)
    "calque 'det antal' (EN 'the number of')":
        (_B + r"det\s+antal" + _E, 1.5),
    # "the reason why" -> native: "skälet till att", "varför".
    # dev-train: O 7 / T 154 (0.012/0.064 per 1000; 6/81 speakers; ln T/O +1.59)
    "calque 'skälet till varför' (EN 'the reason why')":
        (_B + r"(?:skäl|skälet|skälen|anledning|anledningen|anledningar|orsak|orsaken|orsakerna|"
              r"förklaring|förklaringen|förklaringar)\s+till\s+varför" + _E, 1.0),
    # "at a time when" -> native: "i en tid då", "nu när".
    # dev-train: O 1 / T 46 (0.002/0.019 per 1000; 1/36 speakers; ln T/O +2.00)
    "calque 'vid en tid då' (EN 'at a time when')":
        (_B + r"vid\s+en\s+tid(?:punkt)?\s+(?:då|när)" + _E, 1.5),
    # "in itself" -> native: "i sig".
    # dev-train: O 1 / T 89 (0.002/0.037 per 1000; 1/58 speakers; ln T/O +2.65)
    "calque 'i sig själv' (EN 'in itself')":
        (_B + r"i\s+sig\s+själv(?:t|a)?" + _E, 1.0),
    # "in an attempt to" -> native: "för att försöka", "i syfte att".
    # dev-train: O 0 / T 20 (0.000/0.008 per 1000; 0/17 speakers; ln T/O +2.27)
    "calque 'i ett försök att' (EN 'in an attempt to')":
        (_B + r"i\s+ett\s+försök\s+att" + _E, 1.0),
    # "it is only right (and proper) that" -> native: "det är rimligt att", "det är självklart att".
    # dev-train: O 0 / T 44 (0.000/0.018 per 1000; 0/30 speakers; ln T/O +3.05)
    "calque 'bara rätt (och riktigt) att' (EN 'only right and proper')":
        (_B + r"(?:bara|endast)\s+(?:rätt|riktigt|rimligt|rättvist)(?:\s+och\s+(?:riktigt|rimligt|"
              r"korrekt|lämpligt))?\s+att" + _E + "|" + _B + r"rätt\s+och\s+riktigt" + _E, 1.2),
    # pseudo-cleft "What we need is ..." -> native: "Vi behöver ...", "Det vi behöver är ...".
    # Declarative order only (vad + subject), so questions "Vad vill vi?" do not match.
    # dev-train: O 5 / T 91 (0.009/0.038 per 1000; 5/53 speakers; ln T/O +1.37)
    "calque pseudo-cleft 'Vad vi behöver är' (EN 'What we need is')":
        (_S0 + r"vad\s+(?:vi|jag|ni|de|man)\s+(?:\w+\s+){0,2}?(?:behöver|behövde|vill|ville|måste|kräver|"
               r"önskar|säger|föreslår|ser|gör)\s+(?:\w+\s+){0,4}?är" + _E, 1.0),
    # "argue that" -> native: "hävda att", "anföra att" ("argumentera för").
    # dev-train: O 0 / T 9 (0.000/0.004 per 1000; 0/8 speakers; ln T/O +1.51)
    "calque 'argumentera att' (EN 'argue that')":
        (_B + r"argumenter(?:a|ar|ade|at)\s+att" + _E, 1.0),
    # "move forward" -> native: "gå vidare", "komma framåt".
    # dev-train: O 0 / T 8 (0.000/0.003 per 1000; 0/8 speakers; ln T/O +1.39)
    "calque 'röra oss framåt' (EN 'move forward')":
        (_B + r"(?:röra|rör)\s+(?:oss|sig)\s+framåt" + _E, 1.0),
    # "level playing field" -> native: "lika villkor", "lika konkurrensvillkor".
    # dev-train: O 0 / T 10 (0.000/0.004 per 1000; 0/9 speakers; ln T/O +1.61)
    "calque 'jämn spelplan' (EN 'level playing field')":
        (_B + r"(?:jämn|jämna|plan|lika)\s+spelplan", 1.0),
    # sentence-initial discourse "So we/I/it ..." without V2 -> native: "Så måste vi",
    # "Därför ...". Subject pronoun right after "så" = English word order.
    # dev-train: O 1 / T 299 (0.002/0.124 per 1000; 1/109 speakers; ln T/O +3.86)
    "calque sentence-initial 'Så vi/jag/det' (EN 'So we/I/it', no V2)":
        (_S0 + r"så\s+(?:vi|jag|det|de|ni|man|han|hon|den|detta)(?= )(?!\s+att" + _E + ")", 1.5),
    # "the last two years" -> native: "de två senaste åren" (both orders occur, the English
    # order is the T one).
    # dev-train: O 5 / T 212 (0.009/0.088 per 1000; 4/98 speakers; ln T/O +2.22)
    "calque 'de senaste två' (EN 'the last two')":
        (_B + r"de\s+senaste\s+" + _NUM + _E, 1.5),
    # "this afternoon/morning/evening" -> native: "i eftermiddag", "i morse", "i kväll".
    # dev-train: O 1 / T 243 (0.002/0.101 per 1000; 1/87 speakers; ln T/O +3.65)
    "calque 'denna eftermiddag' (EN 'this afternoon')":
        (_B + r"(?:denna|den\s+här)\s+(?:eftermiddag|förmiddag|morgon|kväll)" + _E, 1.5),
    # "it is a question of" -> native: "det är fråga om" (no article; see native_counterparts).
    # dev-train: O 4 / T 37 (0.007/0.015 per 1000; 4/33 speakers; ln T/O +0.68)
    "calque 'det är en fråga om' (EN 'it is a question of')":
        (_B + r"(?:är|var|blir)\s+en\s+fråga\s+om" + _E, 0.5),
    # "best practice" -> native: "goda exempel", "beprövade metoder".
    # dev-train: O 5 / T 59 (0.009/0.024 per 1000; 4/38 speakers; ln T/O +0.94)
    "calque 'bästa praxis' (EN 'best practice')":
        (_B + r"bästa\s+(?:praxis|metoder|metod|praxisen)" + _E, 0.6),
    # "welcome (a report/proposal)" -> native: "jag ser positivt på", "jag gläder mig åt".
    # Established EU Swedish; many Swedish MEPs use it too, hence small.
    # dev-train: O 170 / T 1878 (0.297/0.778 per 1000; 45/182 speakers; ln T/O +0.96)
    "anglicism 'välkomna' (EN 'welcome (a proposal)')":
        (_B + r"välkomna(?:r|de|t)?" + _E, 0.5),
    # "ensure" -> native: "se till att". Weak; T rate falls over time.
    # dev-train: O 54 / T 871 (0.094/0.361 per 1000; 21/149 speakers; ln T/O +1.33)
    "anglicism 'säkerställa' (EN 'ensure')":
        (_B + r"säkerställ(?:a|er|de|t|s|des)" + _E, 0.6),
    # "look forward to" -> native: "vänta på", "hoppas på", "glädja sig åt". Old, established calque.
    # dev-train: O 29 / T 259 (0.051/0.107 per 1000; 17/96 speakers; ln T/O +0.74)
    "calque 'se fram emot' (EN 'look forward to')":
        (_B + r"(?:ser|såg|se|sett)\s+fram\s+emot" + _E, 0.5),
}

NATIVE = {
    # native for "it is clear that" (calque: det är tydligt att)
    # dev-train: O 55 / T 19 (0.096/0.008 per 1000; 21/18 speakers; ln T/O -2.48)
    "native 'det är klart att' (vs 'det är tydligt att')":
        (r"(?:" + _B + r"det\s+(?:är|var)|" + _B + r"är\s+det)\s+klart\s+att" + _E, -1.5),
    # native for "influence on" (calque: inflytande på)
    # dev-train: O 13 / T 11 (0.023/0.005 per 1000; 7/10 speakers; ln T/O -1.60)
    "native 'inflytande över' (vs 'inflytande på')":
        (_B + r"inflytande(?:t)?\s+över" + _E, -1.0),
    # native connective for "at the same time" (calque: på samma gång); not "samtidigt som"
    # dev-train: O 171 / T 117 (0.299/0.049 per 1000; 37/57 speakers; ln T/O -1.82)
    "native sentence-initial 'Samtidigt' (vs 'På samma gång')":
        (_S0 + r"samtidigt" + _E + r"(?!\s+som" + _E + ")", -0.8),
    # native for "first of all I would like to" (calque: först av allt)
    # dev-train: O 31 / T 34 (0.054/0.014 per 1000; 15/26 speakers; ln T/O -1.35)
    "native 'Först vill jag' (vs 'Först av allt')":
        (_S0 + r"först\s+(?:vill|skulle)\s+jag" + _E, -0.8),
}

_PREP = (r"(?:i|på|av|för|med|genom|till|under|om|från|mot|enligt|inom|utan|vid|efter|över|bakom|"
         r"kring|bland|mellan|hos|åt|utanför|inför)")
# interrogative / determiner uses ("i vilken utsträckning", "på vilket sätt", "i vilket fall som
# helst"): a noun right after vilken/vilket/vilka; these are native
_NOT_REL_NOUN = (r"(?!\s+(?:utsträckning|mån|grad|sätt|form|omfattning|riktning|takt|anledning|skäl|syfte|"
                 r"grund|nivå|typ|slags|sorts|slag|del|åtgärder|åtgärd|tidpunkt|datum|rättslig|rättsliga|"
                 r"kriterier|villkor|länder|land|område|områden|frågor|fråga|medel|regler|mål|ändamål|"
                 r"ståndpunkt|hållning|status|roll|uppgifter|resultat|princip|principer|lösning|metod|"
                 r"metoder|fall)" + _E + ")")
# ... and indirect questions after verbs of asking/knowing ("frågan om vilka")
_NOT_ASK = (r"(?<!fråga )(?<!frågan )(?<!undrar )(?<!veta )(?<![\w-]vet )(?<!visa )(?<!klart )"
            r"(?<!beror )(?<!oavsett )")
_PIED_PIPED = _NOT_ASK + _B + _PREP + r"\s+vilk(?:en|et|a)" + _E + _NOT_REL_NOUN

RULES = [
    HitRule(
        "calques.lexicon",
        description="English calques and anglicisms",
        lexicon=CALQUES,
        direction="T",
        rationale="INT: word-for-word renderings of English multi-word expressions and constructions "
                  "where Swedish has a native alternative (named next to each entry). High precision, "
                  "low recall: a hit is evidence, absence is not. Values are conservative log rate "
                  "ratios from dev-train (<= 1.5 per hit), lowered for drift and O usage.",
        source="literature A12, A16/§4.6, A18, C10; [exp] collocations.py candidates",
        cap=3.0),
    HitRule(
        "calques.native_counterparts",
        description="native alternatives of calques",
        lexicon=NATIVE,
        direction="O",
        rationale="The native variant of a calque pair is itself O-leaning when the English source "
                  "pushes translators to the calque (det är klart att vs det är tydligt att; "
                  "inflytande över vs på; Samtidigt vs På samma gång; Först vill jag vs Först av allt).",
        source="literature §4.6 native counterparts; [exp]",
        cap=2.0),
    ChoiceRule(
        "calques.skulle_vilja",
        description="'jag/vi skulle vilja' vs 'jag/vi vill'",
        a=_B + r"(?:jag|vi)\s+skulle\s+(?:vilja|önska)" + _E + "|" + _B + r"skulle\s+(?:jag|vi)\s+(?:\w+\s+)?vilja" + _E,
        b=_B + r"(?:jag|vi)\s+vill" + _E + "|" + _B + r"vill\s+(?:jag|vi)" + _E,
        label_a="jag skulle vilja", label_b="jag vill",
        # dev-train: A 181 / B 1657 in O (0.098), A 2479 / B 5017 in T (0.33); A by 40 O / 226 T speakers
        p_o=0.10, p_t=0.33,
        direction="T",
        rationale="INT: EN 'I would like to (thank/say)' -> 'jag skulle vilja'; native 'jag vill'. "
                  "Broad (hundreds of T speakers, no speaker > 3%), but also used by Swedish MEPs, "
                  "so a ChoiceRule with a small cap.",
        source="literature A13/G1, c42",
        cap=1.5),
    ChoiceRule(
        "calques.forsoka_att",
        description="'försöka att + V' vs 'försöka + V'",
        a=_B + r"försök(?:a|er|te|t)\s+att" + _E,
        b=_B + r"försök(?:a|er|te|t)\s+(?:inte\s+|också\s+|verkligen\s+|nu\s+)?(?!att" + _E + r")\w+a" + _E,
        label_a="försöka att", label_b="försöka + inf",
        # dev-train: A 24 / B 237 in O (0.092), A 386 / B 1124 in T (0.26); B rate 0.41 vs 0.47 per 1000
        p_o=0.09, p_t=0.26,
        direction="T",
        rationale="INT: EN 'try to V' -> 'försöka att V'; native Swedish takes a bare infinitive "
                  "('försöka göra'). The B variant has the same rate in both classes, so the choice "
                  "model holds.",
        source="[exp] collocations.py ('försöka att', 'att försöka att')",
        cap=1.5),
    ChoiceRule(
        "calques.pied_piped_relative",
        description="pied-piped 'P + vilken/vilket/vilka' vs 'som'",
        a=_PIED_PIPED, b=_B + r"som" + _E,
        label_a="P + vilken/vilket/vilka", label_b="som",
        # dev-train: A 203 / B 12911 in O (0.0155), A 2006 / B 52765 in T (0.037); A by 45 O / 216 T speakers
        p_o=0.015, p_t=0.037,
        direction="T",
        rationale="INT: EN 'in which / for which / on whom' -> 'i vilket / för vilka'; native Swedish "
                  "uses 'som ... P' (stranding), 'där', 'vari', 'varav'. 'som' has the same rate in "
                  "both classes (22.5 vs 21.9 per 1000 on dev-train), so it is a fair denominator.",
        source="literature A18",
        cap=1.5),
]

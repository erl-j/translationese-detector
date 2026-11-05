"""Expert-knowledge feature extraction for the hybrid (ML) side track.

Every feature comes from docs/literature.md (IDs A1, B5, ... refer to its
catalogue) and is a plain count of a word list or regex: no tagger, no
learned vocabulary. `raw_counts(text)` returns the counts; `transform()` turns
a count matrix into model features (rates per 1000 words, sentence shares,
smoothed choice ratios, length statistics), so the cache stores counts once
and the representation can change without re-extracting.

Each COUNTS entry: name -> (literature id, predicted direction, description)
with direction +1 = T-up (more frequent in translations), -1 = T-down,
0 = literature says "calibrate on data".
"""
import math
import re

# --------------------------------------------------------------------------
# word lists (docs/literature.md section 4)
# --------------------------------------------------------------------------

FIN = """är var har hade kan kunde ska skall skulle vill ville måste bör borde får fick
blir blev kommer kom finns fanns gäller gällde anser ansåg tror trodde menar menade
vet visste tycker tyckte säger sa sade gör gjorde ser såg står stod går gick ger gav
tar tog handlar handlade innebär innebar behöver behövde verkar verkade föreslår föreslog
stöder stödjer stödde välkomnar välkomnade hoppas hoppades känner kände ligger låg
betyder betydde kräver krävde krävs krävdes behövs behövdes görs gjordes ges gavs
sägs anses ansågs brukar brukade tänker tänkte hör hörde talar talade pratar
lever levde arbetar arbetade leder ledde fortsätter fortsatte börjar började
uppmanar uppmanade beklagar beklagade understryker underströk betonar betonade
lär tycks syns slutar slutade utgör utgjorde omfattar omfattade
riskerar riskerade saknar saknade skapar skapade ämnar""".split()
FIN_MODAL = "kan kunde ska skall skulle vill ville måste bör borde får fick lär torde må".split()
SUBJ = "jag vi du ni han hon hen de dom man den det denna detta dessa denne".split()
SADV = """inte ej icke aldrig alltid också även bara endast enbart redan ofta sällan kanske troligen
troligtvis säkert säkerligen förmodligen antagligen verkligen faktiskt egentligen ju väl nog
ändå dock likväl heller fortfarande alltjämt åtminstone nästan knappt knappast gärna helst
snarare absolut definitivt naturligtvis givetvis självklart tyvärr lyckligtvis möjligen
eventuellt därför alltså således emellertid dessutom tydligen uppenbarligen visserligen""".split()
SUBJUNCTION = ["om", "eftersom", "när", "därför att", "för att", "så att", "medan", "innan",
               "tills", "även om", "fast", "fastän", "trots att", "ifall", "såvida", "emedan",
               "ehuru", "utan att", "i och med att", "genom att", "huruvida"]
INIT_ADV = """därför|dessutom|vidare|slutligen|emellertid|dock|således|alltså|följaktligen|därmed|
först|sedan|sen|nu|då|här|där|idag|i dag|igår|i går|imorgon|i morgon|i år|i fjol|
tyvärr|lyckligtvis|naturligtvis|givetvis|självklart|faktiskt|kanske|verkligen|egentligen|
samtidigt|tidigare|nyligen|ofta|sällan|aldrig|inte heller|bara|också|även|sammanfattningsvis|
avslutningsvis|för det första|för det andra|för det tredje|med andra ord|i själva verket|
å ena sidan|å andra sidan|trots detta|trots det|icke desto mindre|i synnerhet|framför allt|
enligt min mening|enligt min uppfattning|i detta sammanhang|i det här sammanhanget|
på samma sätt|på lång sikt|på sikt|i längden|i praktiken|i princip|i grund och botten"""
CONN_T = """dessutom|vidare|därtill|därutöver|likaså|följaktligen|därmed|därigenom|
dock|icke desto mindre|likväl|trots detta|trots det|med andra ord|
för det första|för det andra|för det tredje|slutligen|sammanfattningsvis|avslutningsvis|
å ena sidan|å andra sidan|i synnerhet|i allmänhet|framför allt|därför|sålunda|i och med detta|
med hänsyn till detta|mot bakgrund av detta|i enlighet med detta"""
ROLE = """föredragande ledamot parlamentariker ordförande kommissionär socialdemokrat liberal
konservativ moderat kristdemokrat miljöpartist vänsterpartist feminist ekonom jurist
läkare lärare bonde jordbrukare fiskare journalist advokat politiker minister
företrädare representant europé svensk britt""".split()
S_PASSIVE_STOP = set("""finns fanns funnits finnas hoppas hoppades lyckas lyckades lyckats känns kändes syns
syntes tycks andas minnas minns trivs trivdes låtsas umgås träffas ses möts slåss brottas kräks svettas
åldras skämmas skäms väsnas fattas fattades saknas saknades mötas enas enades kivas brås gas kaos ethos
patos dess oss ens vems hennes hans deras eders vars annars alls behövs behövdes krävs krävdes
alltjämt grekiska""".split())
ANDE_STOP = """betänkande yttrande förhållande förtroende beroende anseende uppträdande
ställningstagande tillkännagivande meddelande bemötande tänkande handlande leende ende ande
förfarande omfattande pågående gällande betydande avgörande nuvarande kommande följande
ovanstående närvarande frånvarande liknande motsvarande övervägande likalydande fortgående
framstående överensstämmande enastående utestående inflytande underhållande
uppseendeväckande""".split()
ANDE_STOP_LONG = tuple(w for w in ANDE_STOP if len(w) >= 7)  # also catches compounds

# calque lexicon, docs/literature.md 4.6 (entries that duplicate a dedicated
# feature below -- c20, c39, c40, c41, c42, c43, c44 -- are left out here)
CALQUES = {
    "c01": r"\b(ta|tar|tog|tagit|tas|togs)\s+plats\b",
    "c02": r"\bi\s+termer\s+av\b",
    "c03": r"\bpå\s+(daglig|regelbunden|permanent|tillfällig|årlig|löpande|individuell|bilateral|ad\s+hoc|kontinuerlig|systematisk)\s+basis\b",
    "c04": r"\bpå\s+basis\s+av\b",
    "c05": r"\badresser(a|ar|ade|at|as)\s+(frågan|frågor|problemet|problem|utmaningen|utmaningar|oro)\b",
    "c06": r"\bgör(a|)\s+(en\s+)?(stor\s+|verklig\s+)?skillnad\b(?!\s+(på|mellan))|\bgjorde\s+(en\s+)?skillnad\b(?!\s+(på|mellan))",
    "c07": r"\bi\s+slutet\s+av\s+dagen\b",
    "c08": r"\bdet\s+är\s+upp\s+till\b",
    "c09": r"\bspender(a|ar|ade|at|as)\b",
    "c10": r"\bbaser(ad|at|ade|as|ar)\s+på\b",
    "c11": r"\bså\s+långt\s+(som\s+)?\S+(\s+\S+){0,3}\s+(beträffar|angår|är\s+berörd|är\s+berörda)\b",
    "c12": r"\bvid\s+denna\s+punkt\b",
    "c13": r"\bmed\s+respekt\s+till\b",
    "c14": r"\bi\s+ljuset\s+av\b",
    "c15": r"\bnär\s+det\s+kommer\s+till\b",
    "c16": r"\brealiser(a|ar|ade|at|as)\b",
    "c17": r"\bimplementer(a|ar|ade|at|as|ing|ingen)\b",
    "c18": r"\bsist\s+men\s+inte\s+minst\b",
    "c19": r"\bförst\s+av\s+allt\b",
    "c21": r"\bgör(a|)\s+(ingen\s+|inte\s+)?mening\b|\bgjorde\s+(ingen\s+)?mening\b",
    "c22": r"\bi\s+kontexten\s+av\b|\bi\s+sammanhanget\s+av\b",
    "c23": r"\bpå\s+vägnar\s+av\b",
    "c24": r"\bi\s+relation\s+till\b",
    "c25": r"\bpå\s+marken\b(?!\s+(i|vid|nära))",
    "c26": r"\b(jämn|jämna|plan|lika)\s+spelplan\b",
    "c27": r"\b(sätta|sätter|satte|satt|satts|sätts)\s+på\s+plats\b",
    "c28": r"\b(vara|är|var)\s+i\s+en\s+position\s+att\b",
    "c29": r"\bmed\s+(det|detta)\s+sagt\b",
    "c30": r"\b(röra|rör)\s+(oss|sig)\s+framåt\b",
    "c31": r"\bkomm(a|er|it)\s+upp\s+med\b|\bkom\s+upp\s+med\b",
    "c32": r"\b(ta|tar|tog|tagit|tas)\s+(åtgärder|åtgärd)\b",
    "c33": r"\b(göra|gör|gjorde|gjort|görs)\s+(ett|något|detta|det)\s+beslut\b",
    "c34": r"\bhöj(a|er|de|t|s)\s+(frågan|en\s+fråga|frågor|problemet)\b",
    "c35": r"\bbetal(a|ar|ade)\s+uppmärksamhet\b",
    "c36": r"\bargumenter(a|ar|ade|at)\s+att\b",
    "c37": r"\bsom\s+(sådan|sådant|sådana)\s*[,.]",
    "c38": r"\bför\s+att\s+vara\s+ärlig\b",
    "c45": r"\bvälkomnar\s+(betänkandet|förslaget|initiativet|beslutet|rapporten|meddelandet|kommissionens)\b",
    "c46": r"\bsäkerställ(a|er|de|t|s)\b",
    "c47": r"\bpå\s+ett\s+\w+\s+sätt\b",
    "c48": r"\bpå\s+samma\s+gång\s+som\b",
    "c49": r"\bvid\s+(detta|det\s+här)\s+ögonblick\b",
    "c50": r"\bi\s+det\s+långa\s+loppet\b",
    "c52": r"\bi\s+ordning\s+att\b",
    "c53": r"\beventuellt\s+(kommer|blev|lyckades)\b",
}
# native counterparts / fixed expressions, 4.6-4.7 (T-down)
NATIVE_FIXED = [r"på\s+sikt", r"i\s+längden", r"vad\s+\w+(\s+\w+)?\s+beträffar", r"för\s+\w+\s+räkning",
                r"se\s+till\s+att", r"ta\s+itu\s+med", r"äga\s+rum", r"vidta\w*\s+åtgärder", r"fatta\w*\s+beslut",
                r"ta\s+upp\s+frågan", r"i\s+nuläget", r"det\s+gäller\s+att", r"det\s+handlar\s+om",
                r"i\s+och\s+för\s+sig", r"så\s+att\s+säga", r"när\s+allt\s+kommer\s+omkring",
                r"i\s+grund\s+och\s+botten", r"på\s+något\s+sätt"]
LATINATE = r"(presenter|fastställ|garanter|eliminer|initier|accepter|analyser|revider)\w*"
PARTICLE_VERBS = [r"(ta|tar|tog|tagit|tas)\s+(\w+\s+){0,2}?upp", r"(lägga|lägger|lade|la|lagt|läggs)\s+(\w+\s+){0,2}?fram",
                  r"(ta|tar|tog|tagit)\s+(\w+\s+){0,2}?fram", r"(slå|slår|slog|slagit)\s+(\w+\s+){0,2}?fast",
                  r"(se|ser|såg|sett)\s+till", r"(komma|kommer|kom|kommit)\s+fram\s+till",
                  r"(ställa|ställer|ställde|ställt)\s+sig\s+bakom", r"(gå|går|gick|gått)\s+igenom",
                  r"(ta|tar|tog|tagit)\s+ställning", r"(komma|kommer|kom|kommit)\s+överens",
                  r"(hålla|håller|höll|hållit)\s+med", r"(stå|står|stod|stått)\s+bakom",
                  r"(lyfta|lyfter|lyfte|lyft)\s+fram", r"(ta|tar|tog|tagit)\s+bort", r"(ta|tar|tog|tagit)\s+del\s+av"]


def _alt(words):
    return "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in sorted(words, key=len, reverse=True))


_FIN = _alt(FIN)
_SUBJ = _alt(SUBJ)
_SADV = _alt(SADV)
_SUBJN = _alt(SUBJUNCTION)
_INIT_ADV = _alt(w.strip() for w in INIT_ADV.split("|"))
_CONN_T = _alt(w.strip() for w in CONN_T.split("|"))
_ROLE = _alt(ROLE)

# --------------------------------------------------------------------------
# count definitions: name -> (lit id, predicted dir, description, regex)
# "text" counts run on the lower-cased text, "sinit" counts on each
# lower-cased sentence (vocative stripped), "raw" counts on the original case.
# --------------------------------------------------------------------------

TEXT_COUNTS = [
    ("dem_T", "A1", +1, "denna/detta/dessa/denne", r"\b(denna|detta|dessa|denne)\b"),
    ("dem_O", "A1", -1, "den/det/de här|där", r"\b(den|det|de|dom)\s+(här|där)\b"),
    ("skall", "A2", +1, "skall", r"\bskall\b"),
    ("ska", "A2", -1, "ska", r"\bska\b"),
    ("formal_core", "A3", +1, "emellertid/huruvida/således", r"\b(emellertid|huruvida|således)\b"),
    ("formal_ext", "A3", +1, "extended archaic/formal list (icke, ty, sålunda, beträffande...)",
     r"\b(icke|ty|ehuru|dylik[at]?|därest|varvid|varigenom|ävensom|ånyo|alltjämt|därjämte|sålunda|likaledes|"
     r"densamma|detsamma|följaktligen|ifråga|beträffande)\b"),
    ("colloq_core", "A4", -1, "väldigt/lite", r"\b(väldigt|lite)\b"),
    ("colloq_ext", "A4", -1, "jätte-, ganska, kul, sån(t), nåt/nån...",
     r"\b(jätte\w+|ganska|massa|grej|grejen|grejer|kul|sånt|sån|sånn|sådär|nåt|nån|nånting|nånsin)\b"),
    ("p_ju", "A5", -1, "modal particle ju", r"\bju\b(?!\s+\w+(re|are)\b)"),
    ("p_nog", "A5", -1, "modal particle nog", r"\bnog\b(?!\s+(med|för|att)\b)"),
    ("p_val", "A5", -1, "modal particle väl (after finite verb)",
     r"\b(%s)\s+(?:(%s)\s+)?väl\b(?!\s+\w+(ad|at|ade|en|et|na|d|t)\b)|\bväl\s*\?" % (_FIN, _SUBJ)),
    ("p_visst", "A5", -1, "visst (not 'ett/i visst')", r"(?<!ett )(?<!\bi )\bvisst\b"),
    ("p_anda", "A5", -1, "ändå", r"\bändå\b"),
    ("p_namligen", "A5", -1, "nämligen", r"\bnämligen\b"),
    ("p_alltsa", "A5", -1, "alltså", r"\balltså\b"),
    ("av", "A6", +1, "preposition av", r"\bav\b"),
    ("naturligtvis", "A7", +1, "naturligtvis/givetvis/självklart", r"\b(naturligtvis|givetvis|självklart)\b"),
    ("tycka", "A8", -1, "tycker/tyckte/tycka", r"\b(tycker|tyckte|tycka|tyckt)\b"),
    ("anse", "A8", +1, "anser/ansåg/anse", r"\b(anser|ansåg|anse|ansett)\b"),
    ("tro_anse_1p", "A8", +1, "jag/vi tror|anser att", r"\b(jag|vi)\s+(tror|anser)\s+att\b"),
    ("man", "A9", -1, "generic man", r"\bman\b"),
    ("ens_gen", "A10", +1, "generic en/ens", r"\bens\s+(egen|eget|egna)\b|(?:^|[.!?]\s+)en\s+(kan|måste|bör|får|skulle|vill)\b"),
    ("pr_jag", "A11", 0, "jag/mig/min", r"\b(jag|mig|min|mitt|mina)\b"),
    ("pr_vi", "A11", 0, "vi/oss/vår", r"\b(vi|oss|vår|vårt|våra)\b"),
    ("pr_ni", "A11", 0, "ni/er/du", r"\b(ni|er|era|ert|du|dig|din|ditt|dina)\b"),
    ("pr_de", "A11", 0, "de/dem/deras", r"\b(de|dem|deras)\b"),
    ("pr_hanhon", "A11", 0, "han/hon/honom/henne", r"\b(han|hon|honom|henne)\b"),
    ("faktum_att", "A12", +1, "det faktum att", r"\b(det\s+faktum|faktumet)\s+att\b"),
    ("skulle_vilja", "A13", +1, "jag/vi skulle vilja", r"\b(jag|vi)\s+skulle\s+(vilja|önska)\b|\bskulle\s+(jag|vi)\s+vilja\b"),
    ("vill_1p", "A13", -1, "jag/vi vill", r"\b(jag|vi)\s+vill\b|\bvill\s+(jag|vi)\b"),
    ("lat_mig", "A14", +1, "låt mig/oss", r"\blåt\s+(mig|oss)\b"),
    ("damer_herrar", "A15", 0, "mina damer och herrar", r"\bmina\s+damer\s+och\s+herrar\b"),
    ("kommer_att", "A17", +1, "kommer att", r"\bkommer\s+att\b"),
    ("kommer_inf", "A17", -1, "kommer + infinitive (att dropped)",
     r"\bkommer\s+(inte\s+|nog\s+|säkert\s+)?(bli|vara|ha|göra|ta|ge|få|kunna|behöva|leda|innebära|kräva|se)\b"),
    ("pied_rel", "A18", +1, "pied-piped i/på/för vilket|vilka; ', vilken/vilka'",
     r"\b(i|på|av|för|med|genom|till|under|om|från|mot|enligt|inom|utan|vid|efter)\s+(vilken|vilket|vilka)\b|,\s*(vilken|vilka)\s"),
    ("vars", "A19", +1, "vars", r"\bvars\b"),
    ("poss_refl", "A20", -1, "sin/sitt/sina", r"\b(sin|sitt|sina)\b"),
    ("poss_nonrefl", "A20", +1, "hans/hennes/deras", r"\b(hans|hennes|deras)\b"),
    ("mycket_int", "A21", +1, "mycket + adjective", r"\bmycket\s+(\w+(ig|lig|sam|isk|bar|ad|ande)|bra|stor\w*|viktig\w*|god\w*)\b"),
    ("i_sjalva_verket", "A22", +1, "i själva verket", r"\bi\s+själva\s+verket\b"),
    ("faktiskt", "A22", 0, "faktiskt", r"\bfaktiskt\b"),
    ("verkligen", "A22", 0, "verkligen", r"\bverkligen\b"),
    ("helt_enkelt", "A22", 0, "helt enkelt", r"\bhelt\s+enkelt\b"),
    ("egentligen", "A22", -1, "egentligen", r"\begentligen\b"),
    ("inte_minst", "A22", -1, "inte minst", r"\binte\s+minst\b"),
    ("sa_short", "A23", -1, "sa (said)", r"\bsa\b"),
    ("sade_long", "A23", +1, "sade", r"\bsade\b"),
    ("sen_short", "A23", -1, "sen (since/then)", r"\bsen\b"),
    ("sedan_long", "A23", +1, "sedan", r"\bsedan\b"),
    ("short_other", "A23", -1, "nåt/nån/sån/mej/dej/sej/dom", r"\b(nåt|nån|nåra|sån|sånt|såna|mej|dej|sej|dom)\b"),
    ("formal_misc", "A25", +1, "samt/respektive/i enlighet med/i syfte att",
     r"\b(samt|respektive)\b|\bi\s+enlighet\s+med\b|\bi\s+syfte\s+att\b"),
    ("calque", "A16", +1, "calque lexicon hits (4.6)", None),  # filled below
    ("native_fixed", "E10", -1, "native fixed expressions (4.6-4.7)",
     r"\b(%s)\b" % "|".join(NATIVE_FIXED)),
    ("particle_verb", "4.7", -1, "particle verbs (ta upp, lägga fram...)", r"\b(%s)\b" % "|".join(PARTICLE_VERBS)),
    ("latinate_verb", "4.7", +1, "Latinate verbs (presentera, garantera...)", r"\b%s\b" % LATINATE),
    ("det_ar_darfor", "B8", +1, "det är därför", r"\bdet\s+är\s+därför\b"),
    ("bli_pass", "C1", +1, "bli-passive", r"\b(bli|blir|blev|blivit|bliva)\s+\w+(ad|at|ade|ada|d|t|dd|tt|en|et|na|ne)\b"),
    ("vara_pass", "C1", +1, "vara + participle", r"\b(är|var|varit|vara)\s+\w+(erad|erat|erade|ad|at|ade)\b"),
    ("s_pass_modal", "C1", 0, "modal + s-passive (måste göras)", r"\b(%s)\s+\w+as\b" % _alt(FIN_MODAL)),
    ("andes", "C2", +1, "-andes participle", r"\b\w+andes\b"),
    ("role_pred", "C3", +1, "är/som en ROLE", r"\b(är|var|blir|blev|som)\s+(en|ett)\s+(%s)\b" % _ROLE),
    ("en_adj_sak", "C4", +1, "en ADJ sak", r"\b(en|ett)\s+(\w+\s+)?(sak|grej|ting)\b"),
    ("def_gen", "C5", -1, "definite s-genitive (kommissionens)", r"\b[a-zåäö]+(ens|ets|arnas|ernas|ornas|nas)\b"),
    ("kommer_kunna", "C7", +1, "kommer att kunna/behöva", r"\bkommer\s+att\s+(kunna|behöva)\b"),
    ("haller_pa", "C8", +1, "håller på att", r"\b(håller|höll)\s+på\s+att\b"),
    ("att_kept", "C9", +1, "cognition verb + att",
     r"\b(tror|tycker|anser|vet|hoppas|menar|säger|sa|tänker)\s+(jag\s+|vi\s+)?att\b"),
    ("att_dropped", "C9", -1, "cognition verb + clause without att",
     r"\b(tror|tycker|vet|hoppas|menar|säger|sa)\s+(jag\s+|vi\s+)?(det|den|vi|jag|man|ni|de|han|hon|detta)\s+(%s)\b" % _FIN),
    ("conn_T", "D1", +1, "T-leaning connectives (4.3a minus A3 core)", r"\b(%s)\b" % _CONN_T),
    ("conn_O", "D2", -1, "liksom/i alla fall/över huvud taget", r"\b(liksom|i\s+alla\s+fall|över\s+huvud\s+taget|överhuvudtaget)\b"),
    ("talman_addr", "G", 0, "Herr/Fru talman/ordförande address", r"\b(herr|fru)\s+(talman|ordförande|kommissionär|minister)\b"),
    ("valkomnar", "G4", +1, "välkomnar", r"\bvälkomna(r|de)\b"),
]
RAW_COUNTS = [  # punctuation / orthography on the original text
    ("comma", "E2", +1, "commas (not decimal)", r",(?!\d)"),
    ("paren", "D3", +1, "parentheses", r"\("),
    ("semicolon", "E8", 0, "semicolons", r";"),
    ("colon", "E8", 0, "colons", r":(?!\w)"),
    ("exclam", "E9", -1, "exclamation marks (incl. vocatives like Herr talman!)", r"!"),
    ("question", "E9", -1, "question marks", r"\?"),
    ("apos_gen", "F1", +1, "apostrophe genitive", r"\b\w+['’]s\b"),
    ("pct_nospace", "F7", +1, "% without space", r"\d%"),
    ("cap_month_nat", "F4", +1, "capitalised nationality/month mid-sentence",
     r"(?<=[a-zåäö,] )(Svensk|Engelsk|Brittisk|Europeisk|Fransk|Tysk|Januari|Februari|Mars|April|Juni|Juli|Augusti|"
     r"September|Oktober|November|December|Måndag|Tisdag|Onsdag|Torsdag|Fredag|Lördag|Söndag)\w*"),
]
SINIT_COUNTS = [  # per sentence, anchored at its start
    ("si_subj", "B1", +1, "subject-first: PRON (SADV) FIN", r"^(jag|vi|du|ni|han|hon|hen|de|dom|man|den|denna|detta|dessa)\s+((%s)\s+)?(%s)\b" % (_SADV, _FIN)),
    ("si_inverted", "B1", -1, "topicalised: X ... FIN SUBJ (V2 inversion)",
     r"^(?!(jag|vi|du|ni|han|hon|de|man|det|den)\b)(\S+\s+){0,5}?(%s)\s+(jag|vi|du|ni|han|hon|hen|de|dom|man|det|den|detta)\b" % _FIN),
    ("si_subclause", "B1", 0, "starts with a subjunction", r"^(%s)\b" % _SUBJN),
    ("si_det_expl", "B2", 0, "det är/finns/gäller...", r"^det\s+(är|var|finns|fanns|blir|blev|handlar|gäller|rör)\b"),
    ("si_det_obj", "B3", -1, "fronted object det (Det vet vi)",
     r"^det\s+(vet|tror|tycker|anser|menar|hoppas|håller|kan|vill|ska|måste|gör|gjorde|har|hade|säger|sa)\s+(jag|vi|ni|man|de|du|han|hon)\b"
     r"|^så\s+(är|var)\s+det\b|^det\s+(är|var)\s+(det|detta)\s+som\b"),
    ("si_resump_sa", "B4", -1, "resumptive så after fronted clause", r"^(%s)\b[^.!?]*?,\s*så\s+(%s)\s+(%s)\b" % (_SUBJN, _FIN, _SUBJ)),
    ("si_adv_comma", "B5", +1, "fronted adverbial + comma (Dessutom, ...)", r"^(%s)\s*," % _INIT_ADV),
    ("si_v3", "B6", +1, "V3: SUBJ SADV FIN (Vi också vill)",
     r"^(%s)\s+(också|även|bara|endast|alltid|aldrig|inte|verkligen|faktiskt|redan|ofta|fortfarande|definitivt|alla|båda|själva)\s+(%s)\b" % (_SUBJ, _FIN)),
    ("si_men_och", "B9", -1, "sentence-initial men/och/för", r"^(men|och|för)\b"),
    ("si_conn_T", "D1", +1, "sentence-initial T connective", r"^(%s|emellertid|således)\b" % _CONN_T),
    ("si_final_O", "B10", -1, "sentence-final också/heller/ju/då/alltså", r"\b(också|heller|ju|då|alltså)\s*[.!?]*\s*$"),
]

# cheap literal prefilters: a regex runs only if one of its keys occurs in the text
CALQUE_KEYS = {"c01": "plats", "c02": "termer", "c03": "basis", "c04": "basis", "c05": "adresser",
               "c06": "skillnad", "c07": "dagen", "c08": "upp till", "c09": "spender", "c10": "baser",
               "c11": "långt", "c12": "punkt", "c13": "respekt", "c14": "ljuset", "c15": "kommer till",
               "c16": "realiser", "c17": "implementer", "c18": "sist", "c19": "först av", "c21": "mening",
               "c22": "av", "c23": "vägnar", "c24": "relation", "c25": "marken", "c26": "spelplan",
               "c27": "plats", "c28": "position", "c29": "sagt", "c30": "framåt", "c31": "upp med",
               "c32": "åtgärd", "c33": "beslut", "c34": "höj", "c35": "uppmärksamhet", "c36": "argumenter",
               "c37": "som såd", "c38": "ärlig", "c45": "välkomnar", "c46": "säkerställ", "c47": "sätt",
               "c48": "gång som", "c49": "ögonblick", "c50": "loppet", "c52": "ordning att", "c53": "eventuellt"}
TEXT_KEYS = {"p_val": ("väl",), "particle_verb": ("upp", "fram", "fast", "till", "igenom", "ställning",
                                                  "överens", "med", "bakom", "bort", "del av"),
             "native_fixed": ("sikt", "längden", "beträffar", "räkning", "se till", "itu", "rum", "åtgärder",
                              "beslut", "frågan", "nuläget", "det gäller", "handlar", "för sig", "säga",
                              "omkring", "botten", "något sätt"),
             "att_dropped": ("tror", "tycker", "vet", "hoppas", "menar", "säger", "sa "),
             "p_visst": ("visst",), "ens_gen": ("ens ", "en "), "pied_rel": ("vilk",),
             "role_pred": ("en ", "ett "), "en_adj_sak": ("sak", "grej", "ting"), "s_pass_modal": ("as",),
             "andes": ("andes",), "latinate_verb": ("presenter", "fastställ", "garanter", "eliminer",
                                                    "initier", "accepter", "analyser", "revider")}
_text_res = [(n, re.compile(rx), TEXT_KEYS.get(n)) for n, _, _, _, rx in TEXT_COUNTS if rx]
_calque_re = [(re.compile(rx), CALQUE_KEYS[k]) for k, rx in CALQUES.items()]
_raw_res = [(n, re.compile(rx)) for n, _, _, _, rx in RAW_COUNTS]
_sinit_res = [(n, re.compile(rx)) for n, _, _, _, rx in SINIT_COUNTS]
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ\"“”(])")  # same rule as evals/baselines.py
_VOC = re.compile(r"^((herr|fru)\s+(talman|ordförande|kommissionär|minister)|mina\s+damer\s+och\s+herrar|ärade\s+\w+)"
                  r"[^!,.]{0,40}[!,]\s*", re.I)
_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_ANDE = re.compile(r"\b(\w{3,}(?:ande|ende))\b")

STAT_COUNTS = ["n_words", "n_sents", "n_chars", "n_long15", "ande_part", "s_pass", "n_types50", "n_win50"]
COUNT_NAMES = [c[0] for c in TEXT_COUNTS] + [c[0] for c in RAW_COUNTS] + [c[0] for c in SINIT_COUNTS] + STAT_COUNTS
META = {c[0]: (c[1], c[2], c[3]) for c in TEXT_COUNTS + RAW_COUNTS + SINIT_COUNTS}
META.update({"ande_part": ("C2", +1, "-ande/-ende participles minus lexicalised stop-list"),
             "s_pass": ("C1", 0, "-s verb forms (s-passive proxy) minus deponent stop-list"),
             "n_long15": ("C6", 0, "words >= 15 characters")})


def sentences(text):
    return [s for s in _SENT.split(text.strip()) if s.strip()]


def raw_counts(text, return_words=False):
    """Count vector (list of floats, order COUNT_NAMES) for one text
    (and its lower-cased word tokens if return_words)."""
    low = text.lower()
    out = []
    for name, rx, keys in _text_res:
        out.append(len(rx.findall(low)) if keys is None or any(k in low for k in keys) else 0)
        if name == "formal_misc":  # keep column order of TEXT_COUNTS: calque comes next
            out.append(sum(len(c.findall(low)) for c, k in _calque_re if k in low))
    for _, rx in _raw_res:
        out.append(len(rx.findall(text)))
    sents = []
    for s in sentences(text):
        s = _VOC.sub("", s.lstrip(" -–\"”“'(")).strip()
        if s:
            sents.append(s.lower())
    for _, rx in _sinit_res:
        out.append(sum(1 for s in sents if rx.search(s)))
    words = _WORD.findall(low)
    n = len(words)
    ande = sum(1 for w in words if (w.endswith("ande") or w.endswith("ende")) and len(w) >= 6
               and w not in ANDE_STOP and not w.endswith(ANDE_STOP_LONG))
    spass = sum(1 for w in words if len(w) >= 5 and (w.endswith("as") or w.endswith("es"))
                and w not in S_PASSIVE_STOP)
    # moving-average TTR over windows of 50 words (MATTR); short texts: one window
    if n >= 50:
        win = 0
        types = 0
        for i in range(0, n - 49, 10):  # stride 10 to keep it cheap
            types += len(set(words[i:i + 50]))
            win += 1
    else:
        types, win = len(set(words)), 1
    out += [n, max(1, len(sents)), sum(len(w) for w in words), sum(1 for w in words if len(w) >= 15),
            ande, spass, types, win if n >= 50 else max(n, 1) / 50.0]
    assert len(out) == len(COUNT_NAMES)
    return (out, words) if return_words else out


# order check: calque is inserted right after formal_misc in TEXT_COUNTS
assert [c[0] for c in TEXT_COUNTS].index("calque") == [c[0] for c in TEXT_COUNTS].index("formal_misc") + 1


# --------------------------------------------------------------------------
# count matrix -> model features
# --------------------------------------------------------------------------

RATIOS = [  # name, T-leaning count(s), O-leaning count(s), lit id, description
    ("r_dem", ["dem_T"], ["dem_O"], "A1", "share denna-type among demonstratives"),
    ("r_skall", ["skall"], ["ska"], "A2", "share skall among skall+ska"),
    ("r_anse_tycka", ["anse"], ["tycka"], "A8", "share anse among anse+tycka"),
    ("r_vilja", ["skulle_vilja"], ["vill_1p"], "A13", "share 'skulle vilja' among skulle vilja + vill"),
    ("r_kommer_att", ["kommer_att"], ["kommer_inf"], "A17", "share kommer att among kommer att + kommer INF"),
    ("r_poss", ["poss_nonrefl"], ["poss_refl"], "A20", "share hans/deras among possessives"),
    ("r_sade", ["sade_long"], ["sa_short"], "A23", "share sade among sade+sa"),
    ("r_natur_ju", ["naturligtvis"], ["p_ju"], "A7", "share naturligtvis among naturligtvis+ju"),
    ("r_subj_first", ["si_subj"], ["si_inverted"], "B1", "share subject-first among subject-first+inverted"),
    ("r_att_kept", ["att_kept"], ["att_dropped"], "C9", "share att kept after cognition verbs"),
    ("r_latinate", ["latinate_verb"], ["particle_verb"], "4.7", "share Latinate among Latinate+particle verbs"),
]
RATE_SKIP = {"n_words", "n_sents", "n_chars", "n_types50", "n_win50"}
SINIT_NAMES = {c[0] for c in SINIT_COUNTS}


def feature_names():
    names = []
    for c in COUNT_NAMES:
        if c in RATE_SKIP:
            continue
        names.append(("share_" if c in SINIT_NAMES else "rate_") + c)
    names += [r[0] for r in RATIOS]
    names += ["mean_sent_len", "mean_word_len", "mattr50", "commas_per_sent", "log_n_words"]
    return names


def feature_meta(name):
    """(lit id, predicted direction, plain description) for a model feature."""
    if name.startswith("rate_") or name.startswith("share_"):
        base = name.split("_", 1)[1]
        lit, d, desc = META.get(base, ("", 0, base))
        return lit, d, ("%s per 1000 words" % desc) if name.startswith("rate_") else ("share of sentences: %s" % desc)
    for r in RATIOS:
        if r[0] == name:
            return r[3], +1, r[4]
    return {"mean_sent_len": ("E1", +1, "mean sentence length (words)"),
            "mean_word_len": ("E5", 0, "mean word length (chars)"),
            "mattr50": ("E3", -1, "moving-average type/token ratio (50-word windows)"),
            "commas_per_sent": ("E2", +1, "commas per sentence"),
            "log_n_words": ("-", 0, "log number of words")}[name]


def transform(C, np):
    """Count matrix (n x len(COUNT_NAMES)) -> feature matrix (n x len(feature_names()))."""
    idx = {c: i for i, c in enumerate(COUNT_NAMES)}
    nw = np.maximum(C[:, idx["n_words"]], 1.0)
    ns = np.maximum(C[:, idx["n_sents"]], 1.0)
    cols = []
    for c in COUNT_NAMES:
        if c in RATE_SKIP:
            continue
        cols.append(C[:, idx[c]] / ns if c in SINIT_NAMES else 1000.0 * C[:, idx[c]] / nw)
    for _, t, o, _, _ in RATIOS:
        kt = sum(C[:, idx[x]] for x in t)
        ko = sum(C[:, idx[x]] for x in o)
        cols.append((kt + 0.5) / (kt + ko + 1.0))
    cols.append(nw / ns)
    cols.append(C[:, idx["n_chars"]] / nw)
    cols.append(C[:, idx["n_types50"]] / np.maximum(C[:, idx["n_win50"]], 1e-9) / 50.0)
    cols.append(C[:, idx["comma"]] / ns)
    cols.append(np.log(nw))
    return np.stack(cols, axis=1)


def word_tokens(text):
    """Lower-cased word tokens for the bag-of-words reference model."""
    return _WORD.findall(text.lower())


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else ("Herr talman! Det vet vi ju. Dessutom, vi skulle vilja säga att detta "
                                                "betänkande är mycket viktigt, i vilket kommissionen tar plats.")
    for n, v in zip(COUNT_NAMES, raw_counts(t)):
        if v:
            print("%-16s %s" % (n, v))

# Translationese in Swedish (translated from English): literature and feature catalogue

Purpose: background for a **fully rule-based** detector (word lists, regexes, counts, hand-set
weights; no trained models, ideally no tagger/parser) that separates **original Swedish (O)**
from **Swedish translated from English (T)**. Primary eval: Europarl (sentences, ~100-token and
~500-token chunks). Rules should generalise beyond parliament.

Conventions in this document

- **T↑** = over-represented in translations from English; **T↓** (= O↑) = under-represented.
- **Mechanism**: `INT` interference/source-language shining through; `UNI` a hypothesised
  universal (simplification, explicitation, normalisation/conservatism, levelling-out);
  `UNQ` unique-items hypothesis (target-language items with no source counterpart are
  under-used); `NORM` normalisation / conservatism (formal, old-fashioned, "safe" choices);
  `SPK` spoken-vs-written register difference (Europarl originals are edited transcripts of
  speech, translations are written).
- **Priority**: **P1** implement first; **P2** next; **P3** low-yield / risky / mainly useful
  outside Europarl.
- **Evidence**: `[lit]` direction attested in the cited literature (for Swedish or, where noted,
  by analogy for another language); `[exp]` expert / contrastive-grammar judgement, *not* found
  attested in a verified source; treat as a hypothesis to check on dev data.
- Citations marked **(unverified)** could not be confirmed online during this review. Everything
  else was confirmed at least at the bibliographic level (see the bibliography, which records
  *how* each source was checked).

---

## 1. Key literature

### 1.1 Origin of the term and theory of translation universals

| Work | Takeaway for us |
|---|---|
| **Gellerstam (1986)**, "Translationese in Swedish novels translated from English", in Wollin & Lindquist (eds.), *Translation Studies in Scandinavia* (SSOTT II), Lund: CWK Gleerup, pp. 88–95. | Coined *translationese* on exactly our language pair. Compared Swedish original novels with Swedish novels translated from English and found systematic, statistical (not error-based) differences: "fingerprints" of English. Known examples, reported by later authors: semantic drift of words through translation (e.g. *dramatisk* in "dramatiska förändringar", first in translations, later spreading to originals — per Danielsson 2003), and the "*en [ADJ] sak*" pattern where Swedish prefers a pronominal construction (*något fånigt hände* vs *en fånig sak hände*; per Kunz et al. 2026). Full text not obtained. |
| **Gellerstam (1996)**, "Translations as a source for cross-linguistic studies", in Aijmer, Altenberg & Johansson (eds.), *Languages in Contrast*, Lund Studies in English 88, pp. 53–62. | Larger statistical comparison of original vs translated Swedish fiction; features "overused" in translations; argues translations are a legitimate object of contrastive study. |
| **Gellerstam (2005)**, "Fingerprints in translation" (details unverified). | Translationese is not in itself a sign of poor quality (cited this way by Kunz et al. 2026). |
| **Baker (1993)**, "Corpus linguistics and translation studies: implications and applications", in Baker, Francis & Tognini-Bonelli (eds.), *Text and Technology*, Benjamins. (Verified via secondary sources only.) | Proposes source-language-independent "universals" of translation: **explicitation**, **simplification**, **normalisation/conservatism**, later **levelling out** (Baker 1996). Gives the feature families most computational work tests. |
| **Blum-Kulka (1986)**, "Shifts of cohesion and coherence in translation" (cited in Volansky et al. 2015). | Explicitation hypothesis: translators make cohesive relations more explicit → more connectives. Source of the "cohesive markers" feature. |
| **Toury (1995; 2nd ed. 2012)**, *Descriptive Translation Studies and Beyond*, Benjamins. (Secondary verification.) | Two "laws": **growing standardisation** (textual relations of the source are replaced by more habitual target-language options; matches NORM) and **interference** (source-text phenomena are transferred; negative transfer = deviant frequencies, positive transfer = over-use of target forms that resemble source forms). |
| **Laviosa (1998)**, "Core patterns of lexical use in a comparable corpus of English narrative prose", *Meta* 43(4) (cited in Volansky et al. 2015). | Simplification operationalised: lower lexical density, higher share of the most frequent words (list head), lower TTR in translated English. |
| **Tirkkonen-Condit (2002)**, "Translationese – a myth or an empirical fact?", *Target* 14(2); **(2004)** "Unique items – over- or under-represented in translated language?", in Mauranen & Kujamäki (eds.), *Translation Universals: Do they exist?*, Benjamins. | **Unique-items hypothesis**: target-language items with no straightforward counterpart in the source (Finnish clitics *-kin*, *-hAn*, verbs of sufficiency) are **under-represented** in translations. For us: Swedish modal particles (*ju, väl, nog*), generic *man*, the *sin/hans* split, *tycka/tro/anse*, verb particles, *på sikt*, etc. |
| **Kenny (2001)**, *Lexis and Creativity in Translation* (cited in Volansky et al. 2015). | Translators are lexically conservative, which supports NORM features (e.g. avoiding sentence-initial *But* in English; formal words in Swedish). |
| **Teich (2003)**, *Cross-linguistic Variation in System and Text* (cited in Volansky et al. 2015). | "Source language shining through", a positive-transfer view of interference. |

### 1.2 Computational detection of translationese

| Work | Takeaway |
|---|---|
| **Baroni & Bernardini (2006)**, "A new approach to the study of translationese: Machine-learning the difference between original and translated text", *Literary and Linguistic Computing* 21(3): 259–274. | SVMs on Italian geopolitical articles (~2M original, ~900K translated tokens). Knowledge-poor ML reached ~86.7% and **beat human judges**. Function words and POS n-grams carried the signal. Humans are poor detectors, so rules should rely on frequencies and not "sounds odd". |
| **Ilisei, Inkpen, Corpas Pastor & Mitkov (2010)**, "Identification of translationese: a machine learning approach", CICLing, LNCS 6008: 503–511. | Spanish medical and technical corpora. 21 language-independent features (proportions of grammatical words, nouns, finite verbs, auxiliaries, conjunctions; sentence length; simple vs complex sentences; syllables per word; lexical richness; information load). Up to 97.6%. Adding **simplification** features gave a significant gain. |
| **Koppel & Ordan (2011)**, "Translationese and its dialects", ACL, pp. 1318–1326. | Europarl English, 2,000-token chunks, ~300 **function words** → ~96–97% O-vs-T. Translations from different source languages form "dialects": a classifier trained on one source language degrades on another, sometimes to <60%. **Implication: build rules for EN→SV specifically.** |
| **van Halteren (2008)**, "Source language markers in EUROPARL translations", COLING, pp. 937–944. | N-gram "markers" identify the source language of Europarl speeches. Shows that Europarl translations carry lexical source-language fingerprints (e.g. formulaic openings). |
| **Volansky, Ordan & Wintner (2015)**, "On the features of translationese", *Digital Scholarship in the Humanities* (formerly LLC) 30(1): 98–118. | **32 theory-motivated feature sets** on Europarl (English O vs T from 10 languages, ~2,000-token chunks, SVM 10-fold). Accuracy by family (from Wintner's COLING 2016 tutorial slides, which reproduce the tables): **Interference** is best: POS trigrams 98, character trigrams 100, *contextual function words* 100, **positional token frequency 97** (first, second, antepenultimate, penultimate and last token of each sentence), prefixes/suffixes 80. **Miscellaneous**: function words 96, punctuation 81–85, pronouns 77, passives/all verbs 65. **Explicitation**: **cohesive markers 81** (*moreover* 17.5× more frequent in T, *thus* 4×, *besides* 3.8×); explicit naming 58. **Simplification**: TTR 72–76, mean word rank 77, top-N words 64, mean word length 66, syllable ratio 61, mean sentence length 65, lexical density 53 (chance). **Normalisation**: repetitions 55, contractions 50 (rare in Europarl), average PMI 52, threshold PMI 66 (direction *opposite* to hypothesis: O has more strongly associated bigrams). Punctuation detail: **',' strong T marker, '.' strong O marker** (T has longer sentences); '.' and ',' alone give 79%; **parentheses typical of T**. Subject pronouns (*I, he, she*) mark O; reflexives mark T. O opens sentences with *But* 2.25× more often (translators are conservative). |
| **Rabinovich & Wintner (2015)**, "Unsupervised identification of translationese", TACL 3: 419–432. | Supervised in-domain accuracy (Europarl): FW 96.3, char-trigrams 98.8, POS-trigrams 98.5, contextual FW 95.2, **cohesive markers 83.6**. **Cross-domain** (train Europarl, test Hansard or literature) drops to ~60%. Clustering works with as little as **1,000-token chunks (>70%)**. Europarl artefact: *ladies and gentlemen* appears 1,258 times in T vs 12 in O (English side). **Warning: domain features swamp translationese features.** |
| **Avner, Ordan & Wintner (2016)**, "Identifying translationese at the word and sub-word level", DSH; **Rabinovich, Ordan & Wintner (2017)**, "Found in translation: reconstructing phylogenetic language trees from translations", ACL. | Interference is strong enough that translations into English from different source languages cluster by source-language family. Supports writing EN-specific interference rules. |
| **Joelsson (2016)**, *Translationese and Swedish–English Statistical Machine Translation*, BA thesis, Uppsala (supervisor S. Stymne). | **Only verified Swedish-Europarl translationese classifier found.** Europarl v7 sv–en: **37,327 Swedish-original vs 137,732 English-original sentence pairs** (542k unknown, 893k other). Three-class SVM (orig-Swedish / from-English / from-other) on top-300 POS bi/trigrams + char bi/trigrams, ~2,000-token single-speaker chunks: 94.1% 10-fold CV but only **80% on a held-out test set, mostly because original Swedish was misclassified**. Swedish originals are scarce and heterogeneous, so expect speaker effects. |
| **Ustaszewski (2019)**, "Optimising the Europarl corpus for translation studies with the EuroparlExtract toolkit", *Perspectives* 27(1). | Europarl's `LANGUAGE` speaker attribute is the only source-language evidence. Only ~2/3 of statements are tagged, and tags can be inconsistent across language files. Use the toolkit's tag dissemination. Europarl v10 TSVs carry speaker and language metadata. **Label noise is real.** |
| **Kunz, Jarochenko & Bollmann (2026)**, "A dataset for probing translationese preferences in English-to-Swedish translation", LREC 2026 (arXiv 2603.08450). | 600 subtitle sentences with an MT output and a human idiomatic alternative. Concrete EN→SV translationese types: superfluous **indefinite article in predicatives** (*Jag är en översättare* → *Jag är översättare*); **participial constructions** (*tittandes in i kameran* → *när du tittar…*); **"en ADJ sak"**; floated-quantifier order (*Vi saknar dig alla* → *Vi alla saknar dig*); literal idioms. Builds on **Katourgi (2020)**, *Svenskan går bananer. En bok om översättningar som syns* (Lys förlag), a popular book on Swedish translationese (games, subtitles, ads). |

### 1.3 Swedish / Scandinavian contrastive and translation studies

| Work | Takeaway |
|---|---|
| **English–Swedish Parallel Corpus (ESPC)**: Altenberg, Aijmer & Svensson (2001), *The English-Swedish Parallel Corpus (ESPC): Manual of Enlarged Version*, Lund/Göteborg. | Bidirectional fiction and non-fiction, published 1980–2000, ~2.8M words. Basis of most EN↔SV translation-corpus findings below. Available via Språkbanken (on request). |
| **Nilsson (2002)**, "An assessment of the translation-specificity of over-represented multi-word patterns in Swedish fiction texts translated from English", *Linguistica Antverpiensia NS* 1. **Nilsson (2004)**, "Translation-specific lexicogrammar? Characteristic lexical and collocational patterning in Swedish texts translated from English", in Mauranen & Kujamäki (eds.), pp. 129–141. | **The preposition *av* ('of'/'by') is over-represented** in Swedish fiction translated from English, mainly through **source-language transfer** (English *of*-phrases → *X av Y* where Swedish prefers compounds or s-genitives). Over-represented multi-word patterns partly reflect **explicitation of clausal relations**. |
| **Danielsson (2003)**, "Units of meaning in translation – how to make real use of corpus evidence", *Translating and the Computer 25*, Aslib. | **Closest thing to a Swedish-Europarl translationese study.** Keyword comparison of Swedish EP debates translated into Swedish against original Swedish Riksdag debates (small sample). T-key non-topical words: ***skall*, *emellertid*, *huruvida*, *denna*, *detta*, *av***. O-key words: ***ska*, *lite*, *väldigt*, *här*** (as in *den här*, *det här*). O-clusters: *i den här frågan*, *det här är*, *den här typen av*, *tycker att det här*, *i det här fallet*, *det gäller inte minst*, *det händer för lite*. T: *i denna fråga*. Monitor corpora show that *skall*, *emellertid* and *således* declined 1965→1998, so T prefers **old-fashioned/formal** words (NORM). Caveat: the O side was the Riksdag, not EP originals, so the topic words (*Sverige*, party names) are artefacts. |
| **Aijmer (1996)**, "Swedish modal particles in a contrastive perspective", *Language Sciences* 18(1–2): 393–427. **Aijmer (2015)**, "The Swedish modal particle *väl* in a contrastive perspective", *NJES*. **Aijmer (2016)**, "The Swedish modal particle *nog*. A contrastive analysis", *NJES*. | *ju, nog, väl, visst* have no stable English equivalent; **omission** is a frequent translation strategy. ESPC: ***nog* is more frequent in original Swedish than in translations** (unique-items effect). Aijmer (2013) notes that English *of course* has particle-like uses resembling *ju*, so EN *of course* → SV *naturligtvis/givetvis*, not *ju*. |
| **Altenberg (1998)**, "Connectors and sentence openings in English and Swedish", in Johansson & Oksefjell (eds.), *Corpora and Cross-linguistic Research*, Rodopi, pp. 115–143; **Svensson (2000)**, sentence openings in English and Swedish (details unverified). | Sentence openings and connector placement differ between EN and SV (full text not obtained; no figures verified). |
| **Hasselgård (2004)**, "Thematic choice in English and Norwegian", *Functions of Language* 11(2). | ENPC: **Norwegian makes non-subject elements Theme more often than English**, and translators **preserve the source's topical Theme** in the great majority of cases. By analogy, Swedish translated from English should be **more subject-initial / less topicalised** than original Swedish (Swedish and Norwegian are both V2 and behave alike here). |
| **Johansson (2004)**, "Why change the subject? On changes in subject selection in translation from English into Norwegian", *Target* 16(1); **Johansson (2007)**, *Seeing through Multilingual Corpora*, Studies in Corpus Linguistics 26, Benjamins. | ENPC (English–Norwegian Parallel Corpus; Swedish sister corpus = ESPC). Systematic subject-selection and word-order shifts in EN→NO translation. Methodological model for bidirectional comparison. |
| **Bohnacker & Rosén (2008)**, "The clause-initial position in L2 German declaratives: transfer of information structure", *SSLA* 30(4): 511–538. | Swedish puts **given/low-information elements, including expletive *det***, in the prefield far more than German. Frequent Swedish openings: **subject-initial, expletive-initial, *så* constructions, object *det***. These are native-Swedish structures that translations from English may under-produce (e.g. *Det vet vi*, *Det tycker jag*). |
| **Fredriksson (2016)**, *A corpus-based contrastive study of the passive and related constructions in English and Swedish*, PhD thesis, Göteborg. | ESPC: **passives are most frequent in Swedish non-fiction**, then English non-fiction, English fiction, Swedish fiction. ***be*-passive and *s*-passive are the dominant, unmarked forms; *bli*- and *vara*-passives are marginal**. Translations tend to keep the source's thematic structure. So passive rate alone is a weak discriminator; the *bli*/*vara* share may carry the interference. |
| **Englund Dimitrova (2005)**, *Expertise and Explicitation in the Translation Process*, Benjamins (BTL 64). | Russian→Swedish process study: explicitation (more cohesive explicitness) varies with translator expertise and is partly norm-governed. Supports the cohesive-marker family. |
| **Ingo (1991)**, *Från källspråk till målspråk. Introduktion i översättningsvetenskap*, Studentlitteratur; **Ingo (2007)**, *Konsten att översätta*, Studentlitteratur. | Standard Swedish-language translation textbooks. Discuss interference at every level (lexis, word order, information structure, style) and Swedish target norms. Useful for curating calque lists (content not re-checked here). |
| **Bendegard (2014)**, *Begriplig EU-svenska? Klarspråksarbetets förutsättningar inom den interinstitutionella översättningsprocessen*, PhD thesis, Uppsala. | EU Swedish is produced under strong source–target parallelism requirements. Plain-language ("klarspråk") ideals clash with institutional constraints, so expect **formal, nominal, source-shaped** Swedish in EP translations. |
| **Språkrådet / Isof, *Myndigheternas skrivregler*** (various editions). | Swedish house norms: *ska* not *skall* outside legal text; space before %; decimal comma; space as thousands separator; lower-case months, weekdays and nationalities; *EU:s* (colon genitive). Deviations are anglicisms, but EP editorial staff normalise much of this (see pitfalls). |
| **Kunz et al. (2026)** and **Katourgi (2020)** (above). | Modern catalogues of EN→SV translationese (participles, article use, literal idioms). |
| Swedish reference grammar: Teleman, Hellberg & Andersson (1999), *Svenska Akademiens grammatik* (standard reference; not checked online). | Basis for the V2/BIFF, *sin*/*hans*, particle-verb and s-passive claims below. |

Not found despite searching: a peer-reviewed study of **Swedish Europarl** translationese with
feature-level results (beyond Danielsson 2003 and Joelsson 2016); a Språkrådet list naming
specific calques such as *ta plats*, *göra skillnad*, *i slutet av dagen*. The calque lexicon in
§3 is therefore `[exp]`. Catharina Nyström Höög and "Lindgren" were checked; no relevant
EU-translationese study by them was verified.

### 1.4 What the literature implies for a rule-based system

1. **Interference features dominate.** In Volansky et al. the best families are interference
   ones (positional tokens, function-word contexts, character n-grams), not the "universals".
   Hand-written EN→SV interference rules (pied-piped *i vilket*, *denna* for *this*, *av* for
   *of*, comma habits, subject-first openings, calques) are where the signal is.
2. **Function words and sentence position carry most of the signal.** A rule system can capture
   much of this through (a) a curated list of T-leaning and O-leaning function words and
   particles and (b) sentence-initial and second-position token profiles. Both are pure counting.
3. **Simplification features are weak** (TTR ~72%, lexical density at chance, mean sentence
   length 65% and in the "wrong" direction). Use them as low-weight tie-breakers only, and only
   on ≥500-token chunks.
4. **Cohesive markers are a mid-strength, interpretable family** (81–84% alone on 2k-token
   chunks).
5. **Unique items** (Swedish modal particles, *man*, *tycka*, object-*det* fronting) should be
   **under-represented** in T. Their *absence* is weak evidence on short texts and their
   *presence* is fairly strong O evidence. Model them asymmetrically.
6. **Domain shifts kill supervised classifiers.** Keep topical words (country and party
   names) out of the core rule set, or isolate them in a separately reported "Europarl
   artefact" module.

---

## 2. Feature catalogue

Each entry: **ID. Name**, then description; direction; mechanism; detection (regex/lists in §4);
strength/false-positive notes; source; priority. All regexes assume Python `re` on a **lower-cased**
string with Unicode `\w` (default for `str`), unless a feature needs case (marked). `FIN`, `SUBJ`,
`SADV`, etc. refer to lists in §4.

### A. Function words, particles and formal vs colloquial lexis

**A1. Demonstrative *denna/detta/dessa* vs *den/det/de här*** — P1 `[lit]`
- English *this/these* → *denna/detta/dessa* (+ bare noun, no definite suffix). Native speech and
  modern prose prefer *den här/det här/de här* + definite noun (*i den här frågan*).
- **T↑** *denna* family; **T↓** *den här* family. INT + NORM.
- Count `\b(denna|detta|dessa|denne)\b` vs `\b(den|det|de|dom)\s+(här|där)\b`. Use as a ratio
  (binomial evidence, see §5). Also bigrams: `i denna fråga`, `i detta avseende`, `detta är` (T)
  vs `i den här frågan`, `det här är` (O).
- Strong, frequent. FP: pronominal *detta* ("Detta är viktigt") is native written style; formal
  O speakers reading prepared speeches also use *denna*. Check the dev-set ratio.
- Source: Danielsson (2003). By analogy, Volansky et al.'s FW results.

**A2. *skall* vs *ska*** — P1 `[lit]` (with caveats)
- **T↑** *skall*; O↑ *ska*. NORM (old-fashioned/formal form; Språkrådet recommends *ska*).
- Ratio `\bskall\b` / (`\bskall\b` + `\bska\b`).
- Potentially very strong but **confounded by year** (*skall* → *ska* drift 1996–2011) and by EP
  house style (editors may normalise both O and T). Calibrate on data and per year if possible.
- Source: Danielsson (2003).

**A3. Formal/archaic connectives and words** — P1 `[lit]` for the core three, `[exp]` for the rest
- Core (verified as T-keywords): ***emellertid*, *huruvida*, *således***.
- Extended `[exp]`: *icke, ty, ehuru, dylik(a), därest, varvid, varigenom, ävensom, ånyo,
  alltjämt, därjämte, sålunda, likaledes, densamma/detsamma (as pronoun), icke desto mindre,
  följaktligen, ifråga, beträffande, avseende (prep.)*.
- **T↑**. NORM (translators lexically conservative).
- Count per 1000 tokens. Strong for the core three; extended list medium. FP: formal O
  speeches, legal quotes.
- Source: Danielsson (2003); Kenny (2001) on conservatism.

**A4. Colloquial/informal O markers** — P1 `[lit]` core, `[exp]` extension
- Core: ***väldigt*, *lite*** (+ *den/det här*, see A1).
- Extension `[exp]`: *jätte-* prefix (`\bjätte\w+`), *ganska*, *rätt* (as "quite": `\brätt\s+(så\s+)?(bra|stor|många|mycket|viktig\w*)`),
  *massa*, *grej(en|er)?*, *kul*, *bra* (as evaluative "det är bra att"), *typ* (colloquial),
  *liksom* (filler), *sånt|sån|sånn|sådär*, *nåt|nån|nånting|nånsin*.
- **T↓**. SPK + NORM (translators avoid informal register).
- Strong for speech-derived O, weak in written genres.
- Source: Danielsson (2003).

**A5. Modal particles *ju, väl, nog, visst, ändå, nämligen, då, nu, alltså*** — P1 `[lit]`
- Swedish modal particles have no English counterpart, so they are omitted in SV→EN translation
  and rarely *created* in EN→SV translation (*of course* → *naturligtvis*, not *ju*).
- **T↓** (presence = strong O evidence). UNQ + SPK.
- Detection (disambiguate the ambiguous ones):
  - `\bju\b(?!\s+\w+(re|are)\b)`: excludes comparative *ju … desto/ju större*.
  - *nog*: `\bnog\b(?!\s+(med|för|att)\b)` (drops "nog med", "tillräckligt nog").
  - *väl* as particle, after the finite verb or pronoun: `\b(FIN)\s+(?:(SUBJ)\s+)?väl\b(?!\s+\w+(ad|at|ade|en|et|na|d|t)\b)`.
    This drops *väl* + participle/adjective (*väl fungerande*, *väl medveten*). Also `\bväl\s*\?`.
  - *visst*: `\bvisst\b` minus determiner uses `\b(ett|i)\s+visst\b`. Also count `\bvisst\s+(är|var|finns|kan|ska)\b`.
  - *ändå*, *nämligen*, *alltså*: plain `\b…\b` counts (*alltså* is also used by translators for
    *thus*, so give it a lower weight).
  - *då*, *nu* as particles are too ambiguous without a parser: skip, or count only `\b(FIN)\s+(SUBJ)\s+(ju|väl|nog)\b`.
- Strong when present. *ju* is extremely common in Swedish political speech (*det är ju så att*).
- Source: Aijmer (1996, 2015, 2016); Tirkkonen-Condit (2004). Analogy: German *doch* under-represented
  in translation (Wintner tutorial).

**A6. *av* frequency (of-genitive / periphrasis)** — P1 `[lit]`
- English *X of Y* → *X av Y* where Swedish prefers a compound (*kommissionens förslag*,
  *miljöfrågor*) or s-genitive. *av*-agent passive from *by* adds to it.
- **T↑**. INT.
- `\bav\b` per 1000 tokens. Sub-features: `\b(en|ett)\s+(antal|del|rad|serie|mängd|typ|form)\s+av\b`
  (*a number of*, *a series of*: weak, native too); `\bav\s+(den|det|de)\s+\w+\s+(som)\b`.
- Medium-strong (high frequency, small effect). FP: phrasal verbs (*ta del av*, *bero av*).
- Source: Nilsson (2004); Danielsson (2003) lists *AV* among T keywords in EP.

**A7. *naturligtvis/givetvis/självklart* vs *ju*** — P2 `[lit]`(mechanism) / `[exp]`(direction)
- EN *of course* → *naturligtvis*. Native speakers express shared knowledge with *ju*.
- **T↑** *naturligtvis, givetvis*; ratio vs A5's *ju*.
- Medium. Source: Aijmer (2013) (*of course* ≈ *ju*); Danielsson's EP example
  "... naturligtvis också en gemensam grundval i denna vitbok".

**A8. *tycker* vs *anser/tror/menar*** — P1 `[exp]` (+ Danielsson's O-cluster *tycker att det här*)
- EN *think/believe/feel* map onto *tro/tycka/anse*. Translators pick the formal *anser* or literal
  *tror* (*I think* → *jag tror*). *tycka* (evaluative opinion) is a Swedish-specific distinction.
- **T↓** *tycker/tyckte/tycka*; **T↑** `\b(jag|vi)\s+(tror|anser)\s+att\b`.
- Ratio *tyck-* / (*tyck-* + *anse-* + *tro-*) on first-person contexts. Medium-strong in speeches.

**A9. Generic *man*** — P1 `[exp]` + analogy `[lit]`
- *man* is the Swedish generic pronoun. English uses *you/we/one/people/passive*, which
  translators render as *vi*, *du*, *en*, *människor*, or a passive.
- **T↓** *man* (UNQ).
- `\bman\b` per 1000 tokens (*man* = "man/husband" is rare in EP; in general text use
  `\bman\s+(FIN|kan|måste|bör|ska|skulle|har|är)\b` and `\b(FIN)\s+man\b`).
- Strong. Analogy: *one is* over-represented in English translated from German because of
  *man ist* (Wintner tutorial, cited as an interference example).

**A10. Generic *en/ens*** (*one*) — P3 `[exp]`
- **T↑**. `(^|[.!?]\s+)en\s+(kan|måste|bör|får|skulle|vill)\b`, `\bens\s+(egen|eget|egna)\b`. Rare.

**A11. Pronoun profile (*jag, vi, du, ni*)** — P2 `[lit]` (English analogue)
- Volansky et al.: subject pronouns (*I, he, she*) mark O in English. For SV, EP T may have more
  *vi* (EN *we*) and fewer *man*. Treat as a data-calibrated rate vector; direction uncertain.

**A12. *det faktum att*** (the fact that) — P1 `[exp]`
- **T↑** INT. `\b(det\s+faktum|faktumet)\s+att\b`. Native: plain *att*-clause or *det att*.
  High precision, moderate frequency in EP.

**A13. *jag/vi skulle vilja* (I/we would like to)** — P1 `[exp]`
- EP formula *I would like to thank/say/stress* → *Jag skulle vilja tacka*. Native: *Jag vill tacka*.
- **T↑** `\b(jag|vi)\s+skulle\s+(vilja|önska)\b`, and `\bskulle\s+jag\s+vilja\b`; ratio vs
  `\b(jag|vi)\s+vill\b`, `\bvill\s+(jag|vi)\b`.
- Strong in Europarl. FP: polite Swedish speakers use it too.

**A14. *låt mig / låt oss*** (let me / let us) — P2 `[exp]`
- **T↑** `\blåt\s+(mig|oss)\b`. Medium (native in formal oratory).

**A15. *mina damer och herrar*** — P3 `[exp]`/`[lit]` analogue
- Rabinovich & Wintner: *ladies and gentlemen* 1258 (T) vs 12 (O) in *English* EP, coming
  mostly from non-English sources. Unclear for EN→SV. Check on data.

**A16. Calque / English-collocation lexicon** — P1 `[exp]` (+ a few `[lit]`)
- One aggregated feature with per-entry weights. List in §4.6. **T↑** INT. High precision,
  low recall: each hit is strong evidence, absence is ~no evidence.

**A17. *kommer att* vs *kommer* + infinitive (att-drop)** — P2 `[exp]`
- Modern native (esp. spoken) Swedish often drops *att* (*det kommer bli*). Translators and
  editors keep it. EN *will* → *kommer att/ska*. Native Swedish also uses the present for future.
- **T↑** `\bkommer\s+att\s+\w+`; **T↓** `\bkommer\s+(inte\s+|nog\s+|säkert\s+|att\s+)?(bli|vara|ha|göra|ta|ge|få|kunna|behöva|leda|innebära|kräva|se)\b`
  without *att*. Also overall *kommer att* rate per 1000 (T↑). Medium. Diachronic drift.

**A18. Pied-piped relatives *i vilket / för vilka / på vilken*** — P1 `[exp]`
- English *in which / for which / on which* → *i vilket*. Native Swedish strands (*som … i*) or
  uses *där/dit/varifrån/varvid/varav/vari*.
- **T↑** `\b(i|på|av|för|med|genom|till|under|om|från|mot|enligt|inom|utan|vid|efter)\s+(vilken|vilket|vilka)\b`.
  Also bare relative `,\s*(vilken|vilka)\s` (not *vilket* alone, which is a native sentential relative).
- Strong, decent frequency in formal EP. FP: indirect questions (*frågan om vilka åtgärder*).
  Exclude `\bom\s+vilk` after *fråga|undra|vet|veta|diskutera*.

**A19. *vars*** (whose) — P2 `[exp]`
- **T↑** `\bvars\b`. Medium-weak (native written, but EN *whose/of which* inflates it).

**A20. Reflexive possessive *sin/sitt/sina* vs *hans/hennes/deras*** — P2 `[exp]`
- English has no reflexive possessive (*their* covers both). Interference leads to over-use of
  *deras* where *sina* is required, and translators may avoid *sin* in complex clauses.
- **T↑** share `\b(hans|hennes|deras)\b` / (that + `\b(sin|sitt|sina)\b`).
- Medium. FP: legitimate non-coreferential *deras* (very common in EP: *deras regeringar*).

**A21. Intensifiers *mycket* vs *väldigt/jätte-/otroligt*** — P2 `[lit]` (*väldigt*)
- EN *very* → *mycket* (formal); native speech prefers *väldigt*, *jätte-*.
- **T↑** share *mycket* among intensifiers (`\bmycket\s+(\w+(ig|lig|sam|isk|bar|ad|ande)|bra|stor\w*|viktig\w*|god\w*)\b`).

**A22. Hedges and stance adverbs** — P3 `[exp]`
- *i själva verket* (T↑, *in fact*), *faktiskt* (O?), *verkligen* (T↑? *really/indeed*),
  *helt enkelt* (T↑? *simply*), *egentligen* (O↑, Swedish-specific nuance), *inte minst*
  (O-cluster per Danielsson). Calibrate direction on data.

**A23. Short (spoken/modern) vs long (written/old) forms** — P2 `[lit]` (*ska*), `[exp]` rest
- *sa/sade, la/lade, sen/sedan, nåt/något, nån/någon, dom/de|dem, mej/mig, dej/dig, sån/sådan,
  stan/staden, ta/taga, ha/hava, bli/bliva, dra/draga*.
- **T↓** short forms. SPK + NORM. Ratio per pair (binomial). Rare in edited EP, strong in general.

**A24. *huruvida* vs *om*** — included in A3 `[lit]`.

**A25. *samt, såväl … som, respektive, i enlighet med, i syfte att*** — P3 `[exp]` formal-register
markers, likely T↑ in EP (NORM) but also native bureaucratese. Calibrate.

### B. Word order and information structure (V2 language vs SVO source)

**B1. Subject-initial vs topicalised declaratives** — P1 `[lit]` (by analogy, Norwegian)
- English is subject-first. Swedish V2 frequently fronts adverbials and objects (*I dag har vi…*,
  *Därför anser jag…*, *Det vet vi*). Translators preserve the source's Theme.
- **T↑** share of sentences starting with a subject. **T↓** share starting with
  adverbial/object + inverted finite verb.
- Detection (after stripping vocatives like `^(herr|fru)\s+talman\s*[!,]\s*`, *Tack*, and
  interjections):
  - `SUBJ_START`: `^(jag|vi|du|ni|han|hon|hen|de|dom|man|den|denna|detta|dessa)\s+(SADV\s+)?(FIN)\b`
    (exclude *det* here, see B2).
  - `INVERTED`: first token not in SUBJ, and within the first 1–6 tokens a `(FIN)\s+(SUBJ)\b`
    sequence: `^(?!(jag|vi|du|ni|han|hon|de|man|det|den)\b)(\S+\s+){0,5}?(FIN)\s+(jag|vi|du|ni|han|hon|hen|de|dom|man|det|den|detta|vi\s+alla)\b`.
  - `SUBCLAUSE_FIRST`: `^(SUBJUNCTION)\b` (*om, när, eftersom, även om…*).
  - Feature = INVERTED / (SUBJ_START + INVERTED). Report SUBCLAUSE_FIRST separately.
- Strong, frequent, genre-robust. FP: questions (drop sentences ending in `?`); full-NP subjects
  are missed by SUBJ_START, so they fall into "other". Use the ratio only between the two
  detectable classes.
- Sources: Hasselgård (2004); Johansson (2004); Bohnacker & Rosén (2008); positional-token
  results in Volansky et al. (97%).

**B2. Expletive / anticipatory *det*-initial sentences** — P2 `[lit]`(Swedish preference) / `[exp]`(direction)
- Swedish fills the prefield with low-information *det*. English also has *it/there*-initial
  sentences (*There is* → *Det finns*), so the net direction is uncertain. Separate:
  - `^det\s+(är|var|finns|fanns|blir|blev|handlar|gäller|rör)\b` (expletive and presentational)
  - **B3** below (object-*det*), the more Swedish-specific part.

**B3. Fronted object / resumptive *det*** (*Det vet vi.*, *Det tycker jag.*, *Det håller jag med om.*) — P1/P2 `[lit]`
- **T↓** (O↑). UNQ (English rarely fronts *that*).
- `^det\s+(vet|tror|tycker|anser|menar|hoppas|håller|kan|vill|ska|måste|gör|gjorde|har|hade|säger|sa)\s+(jag|vi|ni|man|de|du|han|hon)\b`.
  Also `^så\s+(är|var)\s+det\b` (*Så är det*), `^det\s+(är|var)\s+(det|detta)\s+som\b`.
- High precision for O; moderate frequency in speech.
- Source: Bohnacker & Rosén (2008) (object *det* and *så* constructions typical of Swedish).

**B4. Resumptive *så* after a fronted subordinate clause** (*Om vi inte agerar nu, så kommer…*) — P2 `[lit]`(Swedish) / `[exp]`
- **T↓** (O↑). SPK + UNQ. `,\s*så\s+(FIN)\s+(SUBJ)\b` following `^(SUBJUNCTION)`. Rare in written T.

**B5. Comma after a sentence-initial adverbial/connective** (*Därför, …*, *Dessutom, …*, *Slutligen, …*) — P1 `[exp]`
- English sets off fronted adverbials with a comma. Swedish V2 does not (*Därför anser jag…*).
- **T↑** `^(INIT_ADV)\s*,` (case-insensitive; the list is in §4.5). Also `^(INIT_ADV),\s+(SUBJ)\s+(FIN)`,
  which is *no inversion* after the adverbial and is an almost-certain anglicism (*Dessutom, vi måste…*).
- High precision. EP editors may remove some commas. Check the dev rate.

**B6. Pre-verbal sentence adverbial in a main clause (V3)** (*Vi också vill*, *Jag bara vill*, *Vi alla vet*) — P2 `[exp]` + `[lit]` (Kunz: *Vi alla*)
- Main-clause V2 requires SADV *after* the finite verb (*Vi vill också*).
- **T↑** `^(SUBJ)\s+(också|även|bara|endast|alltid|aldrig|inte|verkligen|faktiskt|redan|ofta|fortfarande|definitivt|alla|båda|själva)\s+(FIN)\b`.
- Rare but precise. FP: *vi alla* / *de flesta* / *ni själva* as subject NP (in Kunz this is flagged
  as translationese anyway); colloquial *Jag bara älskar…*.

**B7. Adverbial placement in subordinate clauses** — P3 `[exp]`
- Native (BIFF): `\b(som|om|eftersom|när|därför att|så att|även om|innan|medan|trots att)\s+(SUBJ)\s+(SADV)\s+(FIN)\b`.
- Main-clause order in non-*att* subordinate clauses: `\b(som|om|eftersom|när|även om|trots att)\s+(SUBJ)\s+(FIN)\s+(inte|aldrig|alltid|också|bara)\b` → T↑ / L2 error (rare).
- Main-clause order after *att* (*han sa att han kunde inte*): **embedded V2**, a native colloquial
  feature → T↓ (O↑ in speech). Two-sided and rare; low weight.
- Note on *som* clauses: `som inte` is the native order (*som inte kan*). `som kan inte` is non-native.

**B8. Clefts** — P3 `[exp]`
- `^det\s+(är|var)\s+.{1,60}?\s+som\b`: Scandinavian languages use clefts more than English
  (direction **T↓**, unverified).
- **T↑** sub-pattern: `\bdet\s+är\s+(därför|av\s+det\s+skälet|skälet\s+till\s+att|anledningen\s+till\s+att)\b`
  (*That is why…* → *Det är därför…*, where native is *Därför…*).

**B9. Sentence-initial *Men* / *Och*** — P2 `[lit]` (English analogue)
- In English, O starts with *But* 2.25× more often than T (translator conservatism). Predicted
  for Swedish: **T↓** `^(men|och|för)\b` share. NORM.

**B10. Positional token profile** — P2 `[lit]`
- The most successful single interference family in Volansky et al. (97%). In a rule system:
  hand-curated lists of first-token words with direction (from B1–B9, A1, A3, C-features)
  and of last-token patterns (e.g. sentence-final particles *också, heller, ju, då, alltså* → O;
  sentence-final *dock*, *emellertid* → T?). Derive candidates from dev counts and then fix them
  by hand.

### C. Morphosyntax

**C1. Passive type** — P2 `[lit]` (frequencies) / `[exp]` (direction)
- EN *be*-passive → SV *s*-passive (default) or periphrastic *bli/vara* + participle. The
  periphrastic forms are marginal in original Swedish (Fredriksson 2016), so **T↑** the
  periphrastic share is plausible (INT). The overall passive rate is weak (Volansky: 65%).
- *bli*-passive: `\b(bli|blir|blev|blivit|bliva)\s+(\w+(ad|at|ade|ada|d|t|dd|tt|en|et|na|ne))\b`.
- *vara*-passive/stative: `\b(är|var|varit|vara)\s+(\w+(erad|erat|erade|ad|at|ade))\b`.
- *s*-passive: `\b\w{3,}(as|es)\b` minus the deponent/reciprocal stop-list in §4.8, plus
  `\b(FIN_MODAL)\s+\w+as\b` (*måste göras*, *kan antas*) as a cleaner subset.
- Feature: periphrastic / (periphrastic + s-passive). Medium-weak.

**C2. Present participles *-ande/-ende*** — P2 `[lit]` (Kunz et al.)
- EN *-ing* → *-ande* (*baserat på*, *inkluderande*, *tittandes*, *innehållande*) and adverbial
  participles (*Tagande hänsyn till…*).
- **T↑** `\b\w{3,}(ande|ende)s?\b` per 1000 minus a stop-list of lexicalised nouns and adjectives
  (*betänkande, förslag?*, see §4.8), plus `\b\w+andes\b` (strong). Medium.

**C3. Indefinite article in predicative role nouns** (*Jag är en socialdemokrat*, *som en föredragande*) — P2 `[lit]` (Kunz et al.)
- Swedish omits the article with bare professions, roles and nationalities (*Jag är läkare*,
  *som föredragande*).
- **T↑** `\b(är|var|blir|blev|som)\s+(en|ett)\s+(ROLE_NOUN)\b` with §4.9 role list. Precise, low
  frequency. FP: modified NPs (*en erfaren föredragande*; adjectives intervene and are not matched).

**C4. *en [ADJ] sak*** (Gellerstam's example) — P3 `[lit]`
- **T↑** `\b(en|ett)\s+(\w+\s+)?(sak|grej|ting)\b` vs native *något ADJ-t*. Low frequency in EP.

**C5. s-genitive vs *av*-periphrasis** — P3 `[lit]` (Nilsson)
- Hard without POS. Proxy: `\b[a-zåäö]+(ens|ets|arnas|ernas|ornas|nas)\b` (definite-genitive
  forms, e.g. *kommissionens*, *ländernas*) per 1000 → T↓. Combine with A6.

**C6. Compound density / long words** — P2 `[exp]` (direction to be checked)
- Swedish closed compounds vs English open compounds and *of*-phrases. Interference predicts
  fewer compounds (**T↓** words ≥ 15 chars, more *X för Y*, *X av Y*). EU terminology works the
  other way (*medlemsstaternas*, *gemenskapslagstiftningen*). Use only after dev-set calibration.
  Mean word length got 66% in Volansky.

**C7. Future and modality** — see A17 (*kommer att*) and A13 (*skulle vilja*). Also *kommer att
kunna/behöva* (T↑?) `[exp]`.

**C8. *håller på att* progressive** — P3 `[exp]`. Possibly T↑ (EN progressive). Weak.

**C9. *att*-complementiser retention** — P2 `[exp]`
- Native speech drops *att* after *tror, tycker, vet, hoppas, säger, menar* (*Jag tror det är
  viktigt*). Translators and editors keep it (explicitation + NORM).
- **T↑** `P(att | V_COG)`: count `\b(tror|tycker|anser|vet|hoppas|menar|säger|sa|tänker)\s+(jag\s+|vi\s+)?att\b`
  vs `\b(tror|tycker|vet|hoppas|menar|säger|sa)\s+(jag\s+|vi\s+)?(det|den|vi|jag|man|ni|de|han|hon|detta)\s+(FIN)\b`.
- Medium.

**C10. Prepositional calques and preposition choice** — P3 `[exp]`
- *i termer av*, *på basis av*, *baserad på*: in the calque list. Prepositions are tricky without
  POS; keep to fixed phrases.

### D. Explicitation and cohesion

**D1. Cohesive markers (T-leaning)** — P1 `[lit]`
- Over-use of explicit connectives in translation. English *moreover/thus/besides/however/
  therefore/furthermore* → *dessutom, således, därtill, emellertid, dock, därför, vidare*.
- **T↑** rate per 1000 tokens of list §4.3a. Weight sentence-initial occurrences higher.
- Medium-strong (81–84% alone on 2k chunks). FP: argumentative O speeches.
- Source: Blum-Kulka (1986); Volansky et al. (2015); Rabinovich & Wintner (2015); Englund
  Dimitrova (2005); Nilsson (2002).

**D2. O-leaning connectives / discourse particles** — P1 `[lit]`/`[exp]`
- *ju, ändå, nämligen, alltså* (spoken use), *så* (resumptive), *liksom, typ, ba*.
- **T↓**. List §4.3b. Pair with D1 as a ratio (T-connectives / all connectives) for a more
  length-robust signal.

**D3. Parentheses** — P2 `[lit]` (English analogue)
- **T↑** (Volansky: parentheses typical of T, a form of explicitation). Remove Europarl
  stage directions first: `\((applåder|skratt|talmannen\s+avbröt\s+talaren|munterhet|protester|sorl|.*?föredragande.*?)\)`
  and any parenthesised text in which >50% of the tokens are capitalised.

**D4. Explicit full forms and spelled-out reference** — P3 `[lit]` analogue (weak: explicit naming 58% in Volansky). Skip.

### E. Simplification and normalisation statistics (use on ≥500 tokens only)

| ID | Feature | Expected | Strength | Notes |
|---|---|---|---|---|
| E1 | Mean sentence length (tokens) | **T↑** (longer; source sentence boundaries kept, editors split Swedish speech) | weak–medium (65%) | Contrary to the simplification hypothesis but what Volansky found ('.' = O marker). P2 |
| E2 | Commas per token and per sentence | **T↑** | medium–strong (',' strong T marker in Volansky; Swedish commas less frequent than English) | P1. Exclude decimal commas `\d,\d`. |
| E3 | Windowed TTR (MATTR, window 100) | T↓ | weak (72%) | P3. Never use raw TTR across lengths. |
| E4 | Hapax ratio in fixed windows | T↓ | weak | P3 |
| E5 | Mean word length (chars) | ? | weak (66%) | P3, calibrate |
| E6 | Share of tokens in the top-100 / top-1000 Swedish word list ("mean word rank") | T↑ (fewer rare words) | weak–medium (77%) | P2 if we ship a frequency list (a word list, so allowed) |
| E7 | Lexical density | — | chance (53%) | Skip |
| E8 | Semicolons, colons | ? | weak | P3 |
| E9 | Exclamation / rhetorical questions | O↑ (speech)? | weak | P3; strip *Herr talman!* first |
| E10 | Threshold-PMI / fixed-expression density | **T↓** (O has more strongly associated bigrams: Volansky 66%, opposite to hypothesis) | weak–medium | Rule proxy: rate of hits from a hand list of native Swedish fixed expressions (§4.7) → T↓. P2 |

### F. Orthography and punctuation (strong outside EP; EP editors normalise much of it)

| ID | Feature | Detection (case-sensitive where noted) | Dir | Priority |
|---|---|---|---|---|
| F1 | Apostrophe genitive (*EU's*, *Sveriges's*) | `\b\w+['’]s\b` | T↑ | P2 (very precise) |
| F2 | Serial (Oxford) comma | `\b\w+,\s+\w+(\s+\w+){0,2},\s+(och|eller)\b` | T↑ | P2 |
| F3 | Comma after fronted subordinate clause | `^(SUBJUNCTION)\b[^,.;:!?]{3,80},` vs no comma before the inverted `FIN SUBJ` | T↑ (weak; Swedish optional) | P3 |
| F4 | Capitalised nationality/language/month/weekday mid-sentence (case-sensitive) | `(?<=[a-zåäö,]\s)(Svensk|Engelsk|Brittisk|Europeisk|Fransk|Tysk|Januari|Februari|Mars|April|Maj|Juni|Juli|Augusti|September|Oktober|November|December|Måndag|Tisdag|Onsdag|Torsdag|Fredag|Lördag|Söndag)\w*` | T↑ | P3 |
| F5 | Decimal point | `\b\d+\.\d{1,2}\b(?!\.\d)` (exclude dates/times) | T↑ | P3 |
| F6 | Thousands separator comma | `\b\d{1,3},\d{3}\b` | T↑ | P3 |
| F7 | % without space | `\d%` vs `\d\s%` | T↑ | P3 |
| F8 | Straight or English quotes | `"…"` or `“…”` vs `”…”` (Swedish uses ” at both ends) | T↑ | P3 (tokeniser-dependent) |
| F9 | Split compounds (*särskrivning*) | Hand list of frequent EP compounds written apart: `\bmedlems\s+stat`, `\bmiljö\s+(fråg|politik)`, `\bjordbruks\s+politik` … | T↑ (mainly non-professional) | P3 |
| F10 | Title-case headings | Many capitalised content words in a line | T↑ | P3 |

### G. Europarl-specific (keep in a separate module; see pitfalls)

| ID | Feature | Dir | Notes | Priority |
|---|---|---|---|---|
| G1 | *Jag/vi skulle vilja* (A13) | T↑ | Classic EN *I would like to*. | P1 |
| G2 | *Låt mig / låt oss* (A14) | T↑ | | P2 |
| G3 | *på vägnar av* (on behalf of) vs *för … räkning*, *å … vägnar*, *på min grupps vägnar* | T↑ | Calque, precise. | P1 (in §4.6) |
| G4 | *Jag välkomnar* (I welcome the report) | T↑ | EU-English formula; native *Jag ser positivt på / Jag är glad över*. Also common in EU Swedish O, so low weight. | P2 |
| G5 | *Jag vill tacka föredraganden* | neutral/O | Contrast with G1. | — |
| G6 | *Förenade kungariket*, *brittisk*, *Storbritannien*, *Skottland*, *Wales*, *Irland*, *London*, *labour*, *de konservativa*, *tories* | T↑ | **Topic artefact**: UK and Irish speakers. Report separately, off by default. | artefact |
| G7 | *Sverige*, *svensk*, *riksdag*, *Norden*, *Finland*, *Stockholm*, party names *(s), (m), fp, kd, mp, v, moderaterna, socialdemokraterna, folkpartiet, vänsterpartiet, miljöpartiet, centerpartiet, junilistan* | T↓ | **Topic artefact** (Danielsson shows *Sverige* and party names as O keywords). Off by default. | artefact |
| G8 | Procedural sentences (*Debatten är avslutad.*, *Omröstningen kommer att äga rum…*, *(Applåder)*) | — | Spoken by the President in any language: **strip before scoring.** | preprocessing |
| G9 | *mina damer och herrar* (A15) | ? | Check. | P3 |
| G10 | *naturligtvis* (A7) | T↑ | | P2 |
| G11 | *Det är därför (som)* (B8) | T↑ | *That is why*. | P2 |

### H. Priority summary (implement first)

1. **A1** *denna/detta/dessa* vs *den/det här*: T↑ *denna*
2. **A5** modal particles *ju/nog/väl/visst/ändå/nämligen*: T↓
3. **A3** formal words *emellertid/huruvida/således* (+ extended list): T↑
4. **A2** *skall* vs *ska*: T↑ *skall* (calibrate per period)
5. **A4** colloquial *väldigt/lite/jätte-*: T↓
6. **B1** subject-initial vs inverted (topicalised) declaratives: T↑ subject-first, T↓ inversion
7. **B3** fronted object *det* (*Det vet vi*): T↓
8. **B5** comma after fronted adverbial / no inversion (*Dessutom, vi…*): T↑
9. **D1** T-leaning cohesive markers (*dessutom, därför, således, emellertid, dock, vidare…*): T↑
10. **A16** calque lexicon (§4.6): T↑ (high precision)
11. **A13/G1** *jag skulle vilja*: T↑ (vs *jag vill*)
12. **A18** pied-piped *i/på/för vilket/vilka*: T↑
13. **A9** generic *man*: T↓
14. **A8** *tycker* vs *anser/tror*: T↓ *tycker*
15. **A6** *av* rate: T↑
16. **A12** *det faktum att*: T↑
17. **E2** comma density: T↑
18. **A17/C9** *kommer att* and *att*-retention: T↑ (P2)

---

## 3. Notes on mechanisms per family (why the direction is what it is)

- **Interference (INT)**: word-for-word mapping of English function words (*this* → *denna*,
  *of* → *av*, *which* → *vilket*, *in which* → *i vilket*, *the fact that* → *det faktum att*,
  *I would like to* → *jag skulle vilja*), English punctuation habits (commas after fronted
  adverbials, Oxford comma), English Theme choice (subject-first).
- **Normalisation/conservatism (NORM)**: translators choose formal, written, older variants
  (*skall, emellertid, huruvida, således, denna, sade, något*) and avoid colloquial ones
  (*ska, väldigt, lite, nåt, dom*). In EP this is amplified by institutional norms (Bendegard 2014).
- **Unique items (UNQ)**: Swedish-only resources (modal particles, *man*, *tycka*, object-*det*
  fronting, resumptive *så*, particle verbs, *sin*) lack an English trigger and are under-produced.
- **Explicitation (UNI)**: more connectives, *att*/*som* retained, parentheses.
- **Spoken vs written (SPK)**: Europarl originals are edited transcripts of (often prepared)
  speech. Translations are written renderings of the English transcript, so spoken markers
  favour O. This is partly a corpus artefact; in non-parliamentary written text, SPK features
  will be weaker. Give them their own weight group.

---

## 4. Ready-to-use word lists

All lower-case. `|`-join for regex alternation. Multi-word entries use single spaces; match them
with `\s+`.

### 4.1 Finite verbs and auxiliaries (V2 / inversion detection)

```text
# FIN: frequent finite verb forms (present + past). Lower-case.
är var har hade kan kunde ska skall skulle vill ville måste bör borde får fick
blir blev kommer kom finns fanns gäller gällde anser ansåg tror trodde menar menade
vet visste tycker tyckte säger sa sade gör gjorde ser såg står stod går gick ger gav
tar tog handlar handlade innebär innebar behöver behövde verkar verkade föreslår föreslog
stöder stödjer stödde välkomnar välkomnade hoppas hoppades känner kände ligger låg
betyder betydde kräver krävde krävs krävdes behövs behövdes görs gjordes ges gavs
sägs anses ansågs brukar brukade tänker tänkte vet hör hörde talar talade pratar
gäller lever levde arbetar arbetade leder ledde fortsätter fortsatte börjar började
uppmanar uppmanade beklagar beklagade understryker underströk betonar betonade
hoppas måste lär tycks syns verkar slutar slutade utgör utgjorde omfattar omfattade
riskerar riskerade saknar saknade skapar skapade vill ämnar
```

```text
# FIN_MODAL (subset, for s-passive detection "måste göras")
kan kunde ska skall skulle vill ville måste bör borde får fick lär torde må
```

### 4.2 Subject pronouns, sentence adverbials, subjunctions

```text
# SUBJ: subject pronouns (det/den also expletive/demonstrative; handle separately where noted)
jag vi du ni han hon hen de dom man den det denna detta dessa denne en
```

```text
# SADV: sentence adverbials (pre-finite in subordinate clauses; post-finite in main clauses)
inte ej icke aldrig alltid också även bara endast enbart redan ofta sällan kanske troligen
troligtvis säkert säkerligen förmodligen antagligen verkligen faktiskt egentligen ju väl nog
ändå dock likväl heller fortfarande alltjämt åtminstone nästan knappt knappast gärna helst
snarare absolut definitivt naturligtvis givetvis självklart tyvärr lyckligtvis möjligen
eventuellt därför alltså således emellertid dessutom tydligen uppenbarligen visserligen
överhuvudtaget över huvud taget i och för sig trots allt
```

```text
# SUBJUNCTION: subordinating conjunctions (clause-initial)
att som om eftersom när då därför att för att så att medan innan sedan tills även om fast
fastän trots att ifall såvida om än emedan ehuru liksom utan att i och med att genom att
huruvida såsom
```

### 4.3 Connectives / cohesive markers

```text
# 4.3a T-leaning (explicitation + formal) — T↑
dessutom vidare därtill därutöver ytterligare likaså således följaktligen därmed därigenom
emellertid dock icke desto mindre likväl trots detta trots det i själva verket med andra ord
för det första för det andra för det tredje slutligen sammanfattningsvis avslutningsvis
å ena sidan å andra sidan i synnerhet i allmänhet framför allt
därför sålunda i och med detta i detta avseende
med hänsyn till detta mot bakgrund av detta i enlighet med detta
```

```text
# 4.3b O-leaning (unique items / spoken) — T↓
# ("så" only as resumptive ", så FIN"; "då"/"nu" only in particle contexts, see A5)
ju väl nog visst ändå nämligen alltså liksom typ ba så då nu
egentligen inte minst i alla fall iaf över huvud taget
```

```text
# 4.3c uncertain direction — calibrate on dev data before assigning a sign
faktiskt helt enkelt verkligen i stället istället
```

### 4.4 Modal particles (unique items)

```text
# MODAL_PARTICLES (T↓) — see A5 for disambiguation regexes
ju väl nog visst ändå nämligen alltså då nu ba liksom
```

### 4.5 Sentence-initial adverbials (for B1 inversion and B5 comma rule)

```text
# INIT_ADV (one- and multi-word). Match at sentence start, case-insensitive.
därför dessutom vidare slutligen emellertid dock således alltså följaktligen därmed
först sedan sen nu då här där idag i dag igår i går imorgon i morgon i år i fjol
tyvärr lyckligtvis naturligtvis givetvis självklart faktiskt kanske verkligen egentligen
samtidigt tidigare nyligen ofta sällan aldrig inte heller bara också även sammanfattningsvis
avslutningsvis för det första för det andra för det tredje med andra ord i själva verket
å ena sidan å andra sidan trots detta trots det icke desto mindre i synnerhet framför allt
enligt min mening enligt min uppfattning i detta sammanhang i det här sammanhanget
på samma sätt på lång sikt på sikt i längden i praktiken i princip i grund och botten
```

### 4.6 Calque lexicon (English source → calque regex → native alternative)

`[exp]` unless marked. Weight = suggested log-likelihood contribution per hit (hand-set,
initial guess). EP = likely to occur in Europarl (H/M/L).

```text
# id | english | calque_regex (lower-case) | native alternative | EP | weight | notes
c01 | take place | \b(ta|tar|tog|tagit|tas|togs)\s+plats\b | äga rum | L | 1.0 | FP: "ta plats" = take a seat / claim space
c02 | in terms of | \bi\s+termer\s+av\b | när det gäller, i fråga om, uttryckt i | M | 1.5 |
c03 | on a (daily/regular/...) basis | \bpå\s+(daglig|regelbunden|permanent|tillfällig|årlig|löpande|individuell|bilateral|ad\s+hoc|kontinuerlig|systematisk)\s+basis\b | dagligen, regelbundet, permanent... | M | 1.5 | "på frivillig basis" is accepted Swedish → excluded
c04 | on the basis of | \bpå\s+basis\s+av\b | på grundval av, utifrån | M | 1.2 |
c05 | address (an issue) | \badresser(a|ar|ade|at|as)\s+(frågan|frågor|problemet|problem|utmaningen|utmaningar|oro)\b | ta itu med, behandla, hantera | L | 1.5 |
c06 | make a difference | \bgör(a|)\s+(en\s+)?(stor\s+|verklig\s+)?skillnad\b(?!\s+(på|mellan))|\bgjorde\s+(en\s+)?skillnad\b(?!\s+(på|mellan)) | spela roll, göra nytta | M | 1.0 | "göra skillnad på/mellan" = distinguish (native)
c07 | at the end of the day | \bi\s+slutet\s+av\s+dagen\b | när allt kommer omkring, i slutändan | L | 1.0 | FP: literal
c08 | it is up to X | \bdet\s+är\s+upp\s+till\b | det är X:s sak, det får X avgöra | M | 0.5 | widely accepted now
c09 | spend (time/money) | \bspender(a|ar|ade|at|as)\b | tillbringa (tid), lägga (pengar), använda | M | 0.8 |
c10 | based on | \bbaser(ad|at|ade|as|ar)\s+på\b | grundad på, som bygger på, utifrån | H | 0.6 | common in modern O too
c11 | as far as X is concerned | \bså\s+långt\s+(som\s+)?\S+(\s+\S+){0,3}\s+(beträffar|angår|är\s+berörd|är\s+berörda)\b | vad X beträffar, när det gäller X | M | 2.0 |
c12 | at this point (in time) | \bvid\s+denna\s+punkt\b | i nuläget, just nu | M | 1.0 | "vid denna tidpunkt" is fine
c13 | with respect to | \bmed\s+respekt\s+till\b | med hänsyn till, när det gäller | L | 2.0 |
c14 | in (the) light of | \bi\s+ljuset\s+av\b | mot bakgrund av, med tanke på | H | 0.8 | established in EU Swedish
c15 | when it comes to | \bnär\s+det\s+kommer\s+till\b | när det gäller | M | 1.2 |
c16 | realise (achieve) | \brealiser(a|ar|ade|at|as)\b | förverkliga, genomföra | M | 0.5 |
c17 | implement | \bimplementer(a|ar|ade|at|as|ing|ingen)\b | genomföra, införa | M | 0.5 | EU Swedish uses "genomföra"; O also uses "implementera"
c18 | last but not least | \bsist\s+men\s+inte\s+minst\b | slutligen, till sist | H | 1.5 |
c19 | first of all | \bförst\s+av\s+allt\b | först och främst, för det första | H | 1.2 | "först och främst" is native (NOT a calque)
c20 | the fact that | \b(det\s+faktum|faktumet)\s+att\b | att (+ omskrivning), det att | H | 1.5 | = feature A12
c21 | make sense | \bgör(a|)\s+(ingen\s+|inte\s+)?mening\b|\bgjorde\s+(ingen\s+)?mening\b | vara vettigt / begripligt | L | 2.0 |
c22 | in the context of | \bi\s+kontexten\s+av\b|\bi\s+sammanhanget\s+av\b | inom ramen för, i samband med | M | 1.5 |
c23 | on behalf of | \bpå\s+vägnar\s+av\b | för X:s räkning, å/på X:s vägnar | H | 2.0 | = G3
c24 | in relation to | \bi\s+relation\s+till\b | i förhållande till, när det gäller | H | 0.6 |
c25 | on the ground | \bpå\s+marken\b(?!\s+(i|vid|nära))  | på plats, i praktiken, i fält | H | 0.8 | FP: literal ground
c26 | level playing field | \b(jämn|jämna|plan|lika)\s+spelplan\b | lika villkor | M | 1.5 |
c27 | put in place | \b(sätta|sätter|satte|satt|satts|sätts)\s+på\s+plats\b | införa, inrätta, upprätta | H | 1.2 |
c28 | be in a position to | \b(vara|är|var)\s+i\s+en\s+position\s+att\b | kunna, ha möjlighet att | M | 1.5 |
c29 | having said that / that being said | \bmed\s+(det|detta)\s+sagt\b | med detta sagt is spreading; native: "Samtidigt", "Ändå" | M | 1.0 |
c30 | move forward | \b(röra|rör)\s+(oss|sig)\s+framåt\b | gå vidare, komma framåt | M | 1.0 |
c31 | come up with | \bkomm(a|er|it)\s+upp\s+med\b|\bkom\s+upp\s+med\b | komma på, lägga fram | M | 1.5 |
c32 | take action/measures | \b(ta|tar|tog|tagit|tas)\s+(till\s+)?(åtgärder|åtgärd)\b | vidta åtgärder | H | 1.0 | "ta till åtgärder" is native-ish → lower weight if "till"
c33 | make a decision | \b(göra|gör|gjorde|gjort|görs)\s+(ett|något|detta|det)\s+beslut\b | fatta/ta ett beslut | M | 1.5 |
c34 | raise (a question) | \bhöj(a|er|de|t|s)\s+(frågan|en\s+fråga|frågor|problemet)\b | ta upp, väcka frågan | M | 2.0 |
c35 | pay attention | \bbetal(a|ar|ade)\s+uppmärksamhet\b | ägna uppmärksamhet, uppmärksamma | L | 2.0 |
c36 | argue that | \bargumenter(a|ar|ade|at)\s+att\b | hävda att, argumentera för att | M | 1.2 |
c37 | as such | \bsom\s+(sådan|sådant|sådana)\s*[,.] | i sig | M | 0.6 | weak; "som sådan" exists natively
c38 | to be honest | \bför\s+att\s+vara\s+ärlig\b | ärligt talat | M | 1.2 |
c39 | in this respect/regard | \bi\s+detta\s+avseende\b | i det här avseendet, på den punkten | H | 0.6 | overlaps A1
c40 | an [ADJ] thing (Gellerstam) [lit] | \b(en|ett)\s+\w+(a|ig|lig|sam|isk|bar)?\s+(sak|grej)\b | något ADJ-t | L | 0.8 |
c41 | I am a [role] [lit, Kunz] | \b(är|var|blir|blev)\s+(en|ett)\s+(ROLE_NOUN)\b | utan artikel | M | 1.0 | = C3
c42 | I would like to | \b(jag|vi)\s+skulle\s+(vilja|önska)\b | jag vill | H | 0.8 | = A13
c43 | let me / let us | \blåt\s+(mig|oss)\b | jag vill, vi bör | H | 0.4 | = A14
c44 | that is why | \bdet\s+är\s+därför\b | därför | H | 0.8 | = B8
c45 | welcome (a report) | \bvälkomnar\s+(betänkandet|förslaget|initiativet|beslutet|rapporten|meddelandet|kommissionens)\b | ser positivt på, är glad över | H | 0.3 | EU Swedish, also in O
c46 | ensure | \bsäkerställ(a|er|de|t|s)\b | se till att, garantera | H | 0.2 | very weak; pairs with "se till att" (O)
c47 | in an X manner/way | \bpå\s+ett\s+\w+\s+sätt\b | adverb in -t | H | 0.2 | native too; tiny weight
c48 | at the same time as | \bpå\s+samma\s+gång\s+som\b | samtidigt som | M | 0.8 |
c49 | at the moment (wrong prep) | \bvid\s+(detta|det\s+här)\s+ögonblick\b | just nu, för närvarande | M | 0.8 |
c50 | in the long run | \bi\s+det\s+långa\s+loppet\b | på sikt, i längden | M | 0.4 | older Swedish idiom also
c51 | a number of | \bett\s+antal\b | flera, ett par | H | 0.1 | native; negligible
c52 | in order to | \bi\s+ordning\s+att\b | för att | L | 2.0 | error-level calque
c53 | eventually (false friend) | \beventuellt\s+(kommer|blev|lyckades)\b | till slut, så småningom | L | 1.0 |
c54 | actually (false friend) | \baktuell(t|a)?\s+(faktum|sanning)\b | faktisk | L | 1.0 |
c55 | play a role → native; NOT a calque | — | — | — | 0 | listed to avoid adding by mistake
```

**Native counterparts (O-evidence when present; T↓)** `[exp]`: *på sikt, i längden, vad … beträffar, för … räkning,
se till att, ta itu med, äga rum, vidta åtgärder, fatta beslut, ta upp frågan, mot bakgrund av,
när det gäller, i nuläget, inte minst, det gäller att, det handlar om, i och för sig, så att säga,
över huvud taget, när allt kommer omkring, i grund och botten, i alla fall, på något sätt*.
These also serve as the "native fixed expressions" list for E10.

### 4.7 Native Swedish fixed expressions and particle verbs (UNQ, T↓) `[exp]`

```text
# particle verbs (lexicalised verb+particle; match verb lemma forms + particle)
ta upp | lägga fram | ta fram | slå fast | se till | komma fram till | ställa sig bakom
gå igenom | ta ställning | ta hänsyn | sätta igång | komma överens | ta itu med | lägga ner
gå ut på | dra igång | ta bort | ta till | ställa upp | hålla med | gå med på | sätta stopp för
ta tag i | se över | slå vakt om | ta del av | stå bakom | ställa krav | ta ansvar | lyfta fram
```

Detection: for each pair, regex `\b(VERB_FORMS)\s+(\w+\s+){0,2}PARTICLE\b` with a short gap for
pronoun objects or *inte*. Compare against Latinate single verbs (*presentera, fastställa,
garantera, eliminera, initiera, acceptera, analysera, revidera*) → T↑. Ratio = particle /
(particle + Latinate).

### 4.8 Stop-lists

```text
# S_PASSIVE_STOP: deponent / reciprocal / non-passive -s verb forms and -as/-es words
finns fanns funnits finnas hoppas hoppades lyckas lyckades lyckats känns kändes syns syntes
tycks andas minnas minns trivs trivdes låtsas umgås träffas ses möts
slåss brottas kräks svettas åldras skämmas skäms väsnas fattas fattades
saknas saknades mötas enas enades kivas brås gas kaos ethos patos
dess oss ens vems hennes hans deras eders vars annars alls
behövs behövdes krävs krävdes
```

(`behövs`/`krävs` are passive-like modals; they are in the stop-list so they count in neither class.)

```text
# ANDE_STOP: lexicalised -ande/-ende nouns/adjectives (not participle interference)
betänkande yttrande förhållande förtroende beroende anseende
uppträdande ställningstagande tillkännagivande meddelande bemötande tänkande handlande
leende ende ande förfarande omfattande pågående gällande betydande
avgörande nuvarande kommande följande ovanstående närvarande frånvarande
liknande motsvarande övervägande likalydande fortgående framstående
överensstämmande enastående utestående inflytande underhållande uppseendeväckande
```

(`gällande` as a preposition, "gällande X", is a modern Swedish trend, not English interference: keep it out.)

### 4.9 Role nouns (for C3 "är en ROLE")

```text
föredragande ledamot parlamentariker ordförande kommissionär socialdemokrat liberal
konservativ moderat kristdemokrat miljöpartist vänsterpartist feminist ekonom jurist
läkare lärare bonde jordbrukare fiskare journalist advokat politiker minister
företrädare representant europé svensk britt
```

(Leave out *medlem*: "en medlem av" is fine Swedish.)

### 4.10 Demonstratives, *skall/ska*, short/long form pairs (binomial ratio features)

```text
# pair_id | T-leaning form(s) | O-leaning form(s)
dem1 | denna detta dessa denne | den här det här de här dom här den där det där de där
aux1 | skall | ska
say1 | sade | sa
lay1 | lade | la
adv1 | sedan | sen
ind1 | något | nåt
ind2 | någon | nån
ind3 | några | nåra
ind4 | sådan sådant sådana | sån sånt såna
pro1 | mig dig sig | mej dej sej
pro2 | de dem | dom
ws1  | huruvida | om (only after fråga|undra|oklart|osäkert|avgöra)
int1 | mycket (as intensifier) | väldigt jätte-
fut1 | kommer att + INF | kommer + INF
opn1 | anser tror (1st person) | tycker
pos1 | hans hennes deras | sin sitt sina
gen1 | i vilket på vilket för vilka ... (pied-piped) | där som ... PREP (stranded) [counts only left side]
mod1 | naturligtvis givetvis | ju
fut2 | jag skulle vilja | jag vill
```

---

## 5. Abstractions for the engine

### 5.1 Pipeline

```
raw text
  → normalise (Unicode NFC, unify quotes/apostrophes but record originals for F-features,
     keep case in a parallel copy for case-sensitive rules)
  → Europarl preprocessing (strip vocatives "Herr/Fru talman!", stage directions "(Applåder)",
     procedural President sentences; optionally mask topical artefacts G6/G7)
  → sentence splitter (regex on [.!?] + capital/quote, with an abbreviation list:
     t.ex., bl.a., m.m., dvs., s.k., osv., ca., nr., art., kap., jfr., resp., p.g.a., fr.o.m., t.o.m.)
  → tokenizer (words incl. åäö, hyphenated compounds, numbers with , and space; punctuation tokens)
  → feature extractors → FeatureObservation(id, k_hits, n_opportunities, spans)
  → rules: map each observation to evidence (log-likelihood ratio)
  → aggregation → posterior / score → verdict + explanation (top-k contributing rules with spans)
```

### 5.2 Feature types (three kinds, all count-based)

1. **Rate features** (events per token): modal particles, *av*, cohesive markers, calques,
   commas. Observation = (k hits, n tokens).
2. **Choice/ratio features** (which variant was chosen when a choice was made): *skall* vs *ska*,
   *denna* vs *den här*, *tycker* vs *anser/tror*, inversion vs subject-first, *att* retained vs
   dropped, *kommer att* vs *kommer* + INF, *sin* vs *deras*. Observation = (k_T variant, k_O variant).
   **These are the most length-robust and topic-robust features.** Prefer them.
3. **Statistic features** (sentence length, MATTR, word-rank coverage): use only with ≥ ~300
   tokens, with small weights.

### 5.3 Evidence model (recommended: hand-set Poisson/binomial log-likelihood ratios)

For each rule the expert sets **reference values** (rate or proportion under T and under O, from
the literature direction plus a one-off look at a dev set) and a **trust weight** w ∈ [0,1]:

- Rate feature, k hits in n tokens, reference rates r_T, r_O per token:
  `LLR = w · [ k·ln(r_T/r_O) − n·(r_T − r_O) ]` (Poisson). A zero count on a short text gives
  only a small contribution, so length is handled automatically.
- Choice feature, k_T and k_O variant counts, reference proportions p_T, p_O (of the T variant):
  `LLR = w · [ k_T·ln(p_T/p_O) + k_O·ln((1−p_T)/(1−p_O)) ]` (binomial). This is zero when no
  choice was made.
- High-precision rare rules (calques, V3 order, apostrophe genitive, *på vägnar av*): a fixed
  `LLR_per_hit` (the "weight" column in §4.6), **capped** per text (e.g. max 3 hits counted)
  to stop one repeated phrase from dominating.
- Statistic features: z-score against hand-set (mean, sd) for T and O at the text's length
  bucket; LLR from two Gaussians; heavily down-weighted.

Aggregate with a prior (EP class balance, or 0) and **naive-Bayes summation** of LLRs, with
**group-level caps** (e.g. the sum of all SPK features is capped at ±X, the sum of all EP
artefacts is excluded by default) to limit double counting of correlated features (A1 *detta*
and c39 *i detta avseende*; A13 and c42). Verdict: `T if Σ > θ` (θ tuned on dev for accuracy or
equal error rate). Output the top contributing rules with matched spans as the explanation.

MYCIN certainty factors are an alternative: CF combination `cf = a + b(1−a)` for same-sign
evidence. The LLR sum is preferable because (i) it is calibrated in odds, (ii) it handles text
length through the k·ln − n·Δr term, and (iii) the hand-set numbers have a clear meaning
("*skall* is chosen 60% of the time in T vs 25% in O").

### 5.4 Data structures (suggested)

```python
@dataclass
class Rule:
    id: str                    # "A1_dem"
    family: str                # "lexis" | "order" | "morph" | "cohesion" | "stat" | "ortho" | "ep"
    kind: str                  # "rate" | "choice" | "rare" | "stat"
    mechanism: str             # INT | NORM | UNQ | UNI | SPK | ARTEFACT
    pattern_T: re.Pattern | None
    pattern_O: re.Pattern | None   # for choice features
    ref: dict                  # {"r_T":..., "r_O":...} or {"p_T":..., "p_O":...} or {"llr":..., "cap":...}
    weight: float              # trust 0..1
    group: str                 # for group caps
    enabled_by_default: bool   # False for ARTEFACT rules
    source: str                # citation key / "[exp]"
```

Keep rules in a declarative YAML/TOML file plus the word lists in plain text files (§4), so
linguists can edit them without touching code.

### 5.5 Pitfalls

1. **Short texts.** On single sentences most rules fire zero times. Expect sentence-level
   accuracy only modestly above chance. Literature numbers (96–100%) are for ~2,000-token chunks;
   clustering degrades to ~70% at 1,000 tokens (Rabinovich & Wintner 2015). Do not report
   rate-feature evidence without the Poisson length term. Choice features are the best bet for
   sentences.
2. **Topic confounds.** Country, party and nationality words (G6/G7) are highly predictive in
   Europarl and meaningless elsewhere; Danielsson's keyword list is dominated by them. Keep them
   off by default and report the "with artefacts" score separately.
3. **Europarl artefacts and house style.** EP Swedish editors transcribe and edit the originals,
   and EP translators produce the T side, so orthographic/punctuation conventions (decimal comma,
   %, quotes, capitalisation) may be normalised on **both** sides and carry no signal. Procedural
   sentences by the President occur in all source languages: strip them. *Herr talman!* is in
   both classes.
4. **Diachrony.** Europarl spans 1996–2011. *skall* → *ska*, *kommer att* → *kommer*, and the
   uptake of anglicisms (*fokusera, implementera, adressera, baserat på*) all change over time,
   and the two classes may have different year distributions. Check per-year rates before fixing
   reference values.
5. **Few Swedish speakers.** Swedish originals are ~37k sentences from a small number of MEPs
   (Joelsson 2016), so speaker idiolect can masquerade as "originalness" (one MEP who loves *ju*).
   Evaluate with speaker-disjoint splits and inspect per-speaker rates.
6. **Label noise.** Europarl language tags are missing for about a third of statements and are
   sometimes inconsistent across language files (Ustaszewski 2019). "Translated from English"
   may include relay translation. Expect a noise ceiling.
7. **Originals are not "pure" Swedish.** Swedish MEPs live in an English/EU-jargon environment
   and read prepared speeches. Many "T" anglicisms (*baserat på, fokusera på, implementera,
   välkomna*) occur in O too. Keep their weights low and rely on *relative* rates.
8. **Spoken vs written confound.** Many O-markers (modal particles, *väldigt*, short forms,
   att-drop, resumptive *så*) are really spoken-register markers. They will over-fire on
   informal written original Swedish (blogs, fiction dialogue) and under-fire on formal written
   originals (reports, legislation). Put them in their own capped group so the system still
   works on written genres.
9. **Professional translations rarely contain errors.** Rules that detect outright errors (V3
   order, *i ordning att*, *betala uppmärksamhet*, split compounds, apostrophe genitive) are
   precise but almost never fire on EP data. They matter for MT output and amateur translation.
   Frequency-shift rules carry the EP signal.
10. **Correlated rules.** *detta*/*denna* (A1), *i detta avseende* (c39), and *Detta är*-initial
    positional rules all fire on the same tokens. Use group caps or assign each token span to at
    most one rule (longest match wins).
11. **Ambiguity of short function words** (*då, nu, väl, visst, man, en, det, om*). Restrict them
    to the disambiguating contexts in §2 instead of raw counts.
12. **Tokenisation and casing.** Europarl text files are often tokenised (spaces before
    punctuation) and sometimes lower-cased, which breaks case-sensitive and punctuation-adjacent
    rules (F-family, B5 comma). Write regexes that tolerate `\s*` before punctuation.

---

## 6. Bibliography and verification status

Verification legend: **[read]** full text or slides read during this review; **[abs]** abstract
or publisher/bibliographic record seen; **[sec]** confirmed only through citing sources;
**(unverified)** could not confirm.

- Aijmer, K. (1996). Swedish modal particles in a contrastive perspective. *Language Sciences* 18(1–2): 393–427. [sec]
- Aijmer, K. (2015). The Swedish modal particle *väl* in a contrastive perspective. *Nordic Journal of English Studies*. [abs] (publicera.kb.se/njes/article/view/29995)
- Aijmer, K. (2016). The Swedish modal particle *nog*. A contrastive analysis. *Nordic Journal of English Studies*. [abs, year per search snippet] (publicera.kb.se/njes/article/view/26827)
- Aijmer, K. (2013). Analyzing modal adverbs as modal particles and discourse markers (on *of course* ≈ *ju*). Benjamins (pbns.234). [sec]
- Altenberg, B. (1998). Connectors and sentence openings in English and Swedish. In Johansson & Oksefjell (eds.), *Corpora and Cross-linguistic Research*, Rodopi, 115–143. [sec]
- Altenberg, B., Aijmer, K. & Svensson, M. (2001). *The English-Swedish Parallel Corpus (ESPC): Manual of Enlarged Version*. [abs] (Språkbanken / researchdata.se doi:10.23695/pmpb-ry83)
- Avner, E. A., Ordan, N. & Wintner, S. (2016). Identifying translationese at the word and sub-word level. *DSH*. doi:10.1093/llc/fqu047 [abs]
- Baker, M. (1993). Corpus linguistics and translation studies: implications and applications. In Baker, Francis & Tognini-Bonelli (eds.), *Text and Technology*, Benjamins, 233–250. [sec]
- Baroni, M. & Bernardini, S. (2006). A new approach to the study of translationese: Machine-learning the difference between original and translated text. *Literary and Linguistic Computing* 21(3): 259–274. doi:10.1093/llc/fqi039 [abs]
- Bendegard, S. (2014). *Begriplig EU-svenska? Klarspråksarbetets förutsättningar inom den interinstitutionella översättningsprocessen*. PhD thesis, Uppsala University. [sec] (via Landqvist 2015 slides, diva2:736377)
- Blum-Kulka, S. (1986). Shifts of cohesion and coherence in translation. In House & Blum-Kulka (eds.), *Interlingual and Intercultural Communication*. [sec]
- Bohnacker, U. & Rosén, C. (2008). The clause-initial position in L2 German declaratives: transfer of information structure. *Studies in Second Language Acquisition* 30(4): 511–538. [abs]
- Danielsson, P. (2003). Units of meaning in translation – how to make real use of corpus evidence. *Translating and the Computer 25*, Aslib. (aclanthology.org/2003.tc-1.11) [read]
- Englund Dimitrova, B. (2005). *Expertise and Explicitation in the Translation Process*. Benjamins Translation Library 64. [abs]
- Fredriksson, A.-L. (2016). *A corpus-based contrastive study of the passive and related constructions in English and Swedish*. PhD thesis, University of Gothenburg. [abs]
- Gellerstam, M. (1986). Translationese in Swedish novels translated from English. In Wollin & Lindquist (eds.), *Translation Studies in Scandinavia*, Lund: CWK Gleerup, 88–95. [sec] (Glottolog ref. 475831)
- Gellerstam, M. (1996). Translations as a source for cross-linguistic studies. In Aijmer, Altenberg & Johansson (eds.), *Languages in Contrast*, Lund Studies in English 88, 53–62. [sec]
- Gellerstam, M. (2005). Fingerprints in translation. (venue details unverified) [sec]
- Hasselgård, H. (2004). Thematic choice in English and Norwegian. *Functions of Language* 11(2). doi:10.1075/fol.11.2.03has [abs]
- Ilisei, I., Inkpen, D., Corpas Pastor, G. & Mitkov, R. (2010). Identification of translationese: a machine learning approach. CICLing 2010, LNCS 6008, 503–511. [read]
- Ingo, R. (1991). *Från källspråk till målspråk. Introduktion i översättningsvetenskap*. Lund: Studentlitteratur. [abs, catalogue] — Ingo, R. (2007). *Konsten att översätta*. Studentlitteratur. [abs, catalogue]
- Joelsson, J. (2016). *Translationese and Swedish-English Statistical Machine Translation*. BA thesis, Uppsala University. (diva2:1034684) [read]
- Johansson, S. (2004). Why change the subject? On changes in subject selection in translation from English into Norwegian. *Target* 16(1). [abs]
- Johansson, S. (2007). *Seeing through Multilingual Corpora*. Studies in Corpus Linguistics 26, Benjamins. [abs]
- Katourgi, A. (2020). *Svenskan går bananer. En bok om översättningar som syns*. Umeå: Lys förlag. [sec] (review in *Språk och stil*)
- Kenny, D. (2001). *Lexis and Creativity in Translation*. St. Jerome. [sec]
- Koehn, P. (2005). Europarl: A parallel corpus for statistical machine translation. MT Summit X. [sec]
- Koppel, M. & Ordan, N. (2011). Translationese and its dialects. ACL 2011, 1318–1326. (P11-1132) [abs]
- Kunz, J., Jarochenko, A. & Bollmann, M. (2026). A dataset for probing translationese preferences in English-to-Swedish translation. LREC 2026 (arXiv:2603.08450). [read]
- Kurokawa, D., Goutte, C. & Isabelle, P. (2009). Automatic detection of translated text and its impact on machine translation. MT Summit XII. [sec, cited in Rabinovich & Wintner 2015]
- Landqvist, H. (2015). Är EU-svenskan begriplig? (slides, Hieronymusdagen 2015, GUP). [read]
- Laviosa, S. (1998). Core patterns of lexical use in a comparable corpus of English narrative prose. *Meta* 43(4). [sec]
- Mauranen, A. & Kujamäki, P. (eds.) (2004). *Translation Universals: Do They Exist?* Benjamins. [abs]
- Nilsson, P.-O. (2002). An assessment of the translation-specificity of over-represented multi-word patterns in Swedish fiction texts translated from English. *Linguistica Antverpiensia NS* 1. [abs]
- Nilsson, P.-O. (2004). Translation-specific lexicogrammar? Characteristic lexical and collocational patterning in Swedish texts translated from English. In Mauranen & Kujamäki (eds.), 129–141. [abs]
- Rabinovich, E. & Wintner, S. (2015). Unsupervised identification of translationese. *TACL* 3: 419–432. [read]
- Rabinovich, E., Ordan, N. & Wintner, S. (2017). Found in translation: reconstructing phylogenetic language trees from translations. ACL 2017. [abs]
- Språkrådet / Institutet för språk och folkminnen. *Myndigheternas skrivregler*. [abs: existence and scope confirmed; specific rules (ska, %, decimal comma) are from general knowledge (unverified in this review)]
- Teich, E. (2003). *Cross-linguistic Variation in System and Text*. Mouton de Gruyter. [sec]
- Teleman, U., Hellberg, S. & Andersson, E. (1999). *Svenska Akademiens grammatik*. (standard reference; not checked online)
- Tirkkonen-Condit, S. (2002). Translationese – a myth or an empirical fact? *Target* 14(2). [abs]
- Tirkkonen-Condit, S. (2004). Unique items – over- or under-represented in translated language? In Mauranen & Kujamäki (eds.). [sec]
- Toury, G. (1995/2012). *Descriptive Translation Studies – and Beyond*. Benjamins. [sec]
- Ustaszewski, M. (2019). Optimising the Europarl corpus for translation studies with the EuroparlExtract toolkit. *Perspectives* 27(1). doi:10.1080/0907676X.2018.1485716 [abs; author inferred from the toolkit's GitHub handle]
- van Halteren, H. (2008). Source language markers in EUROPARL translations. COLING 2008, 937–944. (C08-1118) [abs]
- Volansky, V., Ordan, N. & Wintner, S. (2015). On the features of translationese. *Digital Scholarship in the Humanities* 30(1): 98–118. doi:10.1093/llc/fqt031 [abs + result tables read via Wintner, COLING 2016 tutorial slides "Translationese: between human and machine translation"]
- Wintner, S. (2016). Translationese: between human and machine translation. COLING 2016 tutorial slides. [read]

Not verified / not found: Gundel (2002) on clefts in English vs Norwegian (relied on for B8's
direction: **unverified**); Svensson (2000) on sentence openings (**unverified** details); any
Språkrådet page that names the specific calques in §4.6 (**not found**; the calque list is
expert judgement); studies on EU-Swedish translationese by Nyström Höög or Lindgren (**not found**).

# The rule engine (`tsw/`)

A fully rule-based expert system: tokenisation, regexes, word lists, counts and
hand-written rules with **hand-set** parameters. No trained models, no tagger or
parser, no automatic fitting or search over parameters. A knowledge engineer may
look at dev-train statistics (`tsw/tools/rates.py`) and type rounded values (at
most 2 significant digits) into a rule, like an expert calibrating it.

Protocol: **calibrate on dev-train, select/compare on dev-val, test once at the
end.** dev-train = dev sessions before 2002-03-01, dev-val = dev sessions from
2002-03-01 to 2004-06-30 (`evals/devsplit.py`), test = sessions after
2004-06-30 (locked).

```
raw text ─ normalise (+ offset map) ─ sentences ─ tokens ─ Doc views
        └─ rules (one module per family, auto-discovered) ─ contribution per rule
        └─ per-family sum, clipped at ±FAMILY_CAP ─ score = Σ families  (> 0 = translated)
        └─ explain / verdict / explain_spans / annotate
```

## Quick start

```bash
python3 -m tsw "Herr talman! Det här är ju viktigt. Först av allt vill jag tacka."
python3 -m tsw -f speech.txt --json            # Engine.annotate() as JSON
python3 -m tsw --list-rules --all-rules
python3 -m unittest discover -s tsw/tests -t .  # unit tests (fast, login node is fine)

# evaluation (dev-train + dev-val by default); the full run goes through Slurm:
srun -p berzelius-hopper-cpu --time=00:15:00 -c 1 --mem=8G \
    python3 evals/run_all.py --clf tsw.engine:Default --quiet
python3 evals/run.py --clf 'tsw.engine:Families(only="starter")' --split dev-val --gran 500 --limit 500 --mask-topic
python3 evals/run.py --clf 'tsw.engine:Engine(exclude="starter.sent_len")' --split dev-val --gran 100 --limit 500
```

Harness constructors (keyword literals only, parsed by `evals/run.py`):

| spec | meaning |
|---|---|
| `tsw.engine:Default` | all enabled rules of all families |
| `tsw.engine:Families(only="a,b", exclude="x")` | only families a and b |
| `tsw.engine:Engine(exclude="fam,rule.name")` | drop families and/or single rules |
| `tsw.engine:Engine(family_caps={"stats": 1.0})` | override family caps |
| `tsw.engine:Engine(weights={"starter.sent_len": 0.25})` | ablate/reweight without editing files |
| `tsw.engine:Engine(include_disabled=True)` | also rules defined with `enabled=False` |

## Files

| file | what |
|---|---|
| `tsw/text.py` | normalisation (+ offset map), sentence splitter, tokenizer, vocative regex |
| `tsw/doc.py` | `Doc(text)`: lazily cached views; rules read only from a Doc |
| `tsw/rules/base.py` | rule types `HitRule`, `RateRule`, `ChoiceRule`, `StatRule`, `PredicateRule`; `Matcher`; `words_rx`; `sentence_lengths` |
| `tsw/rules/__init__.py` | `discover()`: imports every `tsw/rules/*.py` except `base` and `_*` |
| `tsw/rules/<family>.py` | one module per family: `RULES = [...]`, optional `FAMILY`, `FAMILY_CAP` |
| `tsw/engine.py` | `Engine`, `Default`, `Families` |
| `tsw/__main__.py` | CLI |
| `tsw/tools/rates.py` | calibration helper (dev-train only; refuses dev-val, dev, test) |
| `tsw/tests/` | unit tests |

## Writing a rule family

Add **one file** `tsw/rules/<family>.py`. Do not edit any registry, the engine,
or another family's module: discovery picks the file up, and parallel branches
never conflict.

```python
"""<family>: one paragraph on what the family covers and why.

Calibration: python3 -m tsw.tools.rates ... on dev-train 500 bal, <date>.
"""
from .base import ChoiceRule, HitRule, PredicateRule, RateRule, StatRule, words_rx

FAMILY = "register"        # optional (default: module name); used by Families(only=...)
FAMILY_CAP = 4.0           # optional (default 4.0): cap on |sum of this family's rules|

RULES = [
    ChoiceRule(
        "register.sa_vs_sade",                      # unique across ALL families: prefix with family
        description="'sa' vs 'sade'",
        words_a=["sa"], words_b=["sade"], label_a="sa", label_b="sade",
        p_o=0.30, p_t=0.10,                         # P(A | choice) in O and T, typed from rates.py
        direction="O",                              # direction of variant A; checked against p_o/p_t
        rationale="NORM: translators prefer the long written form.",
        source="literature 4.10 say1",              # or "[exp]"
        cap=2.0),
    RateRule(
        "register.colloquial",
        description="colloquial väldigt/jätte-/lite",
        pattern={"'väldigt'": r"\bväldigt\b", "jätte- prefix '{m}'": r"\bjätte\w+"},
        rate_o=1.2, rate_t=0.3, direction="O",
        rationale="SPK + NORM", source="Danielsson (2003); literature A4"),
]
```

Checklist for every rule: a `rationale` (mechanism) and a `source` (key from
`docs/literature.md`, or `[exp]`), a direction that the constructor can check,
values from `rates.py` rounded to ≤ 2 significant digits, a `cap`, and a
`# dev-train 500 bal: ...` comment next to the numbers recording what you looked at.

## Rule types and their maths

All contributions are log-likelihood ratios ln P(text | T) / P(text | O) in
nats: > 0 points to "translated", < 0 to "original". Every rule multiplies its
raw LLR by `weight` (default 1.0) and clips it to ±`cap`. Patterns are given as
a regex string, a dict `{label: regex}` (each hit is labelled with the key of
the alternative that matched; `{m}` in the key is replaced by the matched text),
or `words=[...]` (whole words/phrases; `"den här"` matches `den\s+här`).

**HitRule** — rare, high-precision items (calques, errors).
`contribution = clip(w · Σ per_hit, cap)`. Either one `per_hit` for every match,
or `lexicon={label: (regex, per_hit)}` with a value per entry (all of one sign).
A sensible `per_hit` is ln(rate_T / rate_O) read off `rates.py`; for items that
never occur in O, use a judgement value (≤ 2) rather than infinity.

**RateRule** — frequency of an item relative to text length (Poisson).
`rate_o`, `rate_t` = expected hits per 1000 words in each class. With k hits in
n words (n = `doc.n_words`):

    LLR = k · ln(rate_t / rate_o) − n · (rate_t − rate_o) / 1000

The second term is the evidence of *absence*: an O-leaning particle that does not
occur in 500 words is evidence for T, and the term scales with length, so short
texts get little of it. `min_words` makes the rule abstain on very short texts.
Poisson assumes independent hits; bursty items (one speaker repeating a word)
make it overconfident on long texts: use `weight` < 1 and/or a cap.

**ChoiceRule** — which of two competing variants was chosen (binomial).
`p_o`, `p_t` = share of variant A among A+B occurrences in each class:

    LLR = kA · ln(p_t / p_o) + kB · ln((1 − p_t) / (1 − p_o))      (0 if kA + kB = 0)

Most robust to length and topic because it conditions on a choice having been
made; prefer it whenever a native alternative exists. Make A the *rarer* variant
so 2 significant digits keep the information (0.013 vs 0.060, not 0.99 vs 0.94).
A and B must not overlap. Caveat (see findings below): if the *rate* of B also
differs between classes, a ChoiceRule throws that information away.

**StatRule** — a per-unit statistic (e.g. words per sentence) modelled as
Gaussian per class. `units(doc)` returns `[(value, span)]` per unit; values are
clipped to `unit_clip`. With n units and mean x:

    LLR = log N(x; mu_t, sd_t/√n) − log N(x; mu_o, sd_o/√n)

`sd_o`, `sd_t` are SDs of a *single unit*. Pass a pooled `sd`: then the LLR is
linear in x and equals the sum of per-unit LLRs, so per-sentence spans are exact
and outliers cannot blow up through the quadratic term. Units are rarely
independent (speakers), so use `weight` < 1. `rates.py --stat` prints the
per-unit means/SDs, the pooled SD and the actual SD of text means (compare it to
sd/√n to choose the weight).

**PredicateRule** — escape hatch: `fn(doc) -> raw LLR` for logic that needs
code (V2/inversion detection, positional rules). Needs an explicit `direction`.
Give it `spans_fn(doc) -> [(start, end, raw_share, label)]` (offsets into
`doc.text`, raw shares in the same units as `fn`, ideally summing to it) so the
UI can highlight the evidence, e.g. the sentence-initial constituent + finite
verb for a word-order rule; and optionally `describe_fn(doc, contribution) -> str`.

## Engine, caps and explanations

    score = prior + Σ_family clip(Σ_rule contribution, ±FAMILY_CAP)

- `score(text)`, `explain(text)` → `{rule: contribution}` (non-zero only),
  `score_and_explain(text)` (one pass; what the harness uses).
- When a family cap binds, `explain` adds a pseudo-rule `cap:<family>` with the
  signed amount removed, so `sum(explain) == score`. It appears in the harness's
  per-rule table and tells you how often a family saturates.
- `verdict(text)` → label, score, `prob_translated` (logistic of the score),
  human-readable evidence sentences for each side (from each rule's `describe`).
- `explain_spans(text)` → `[{start, end, text, rule, family, direction,
  contribution_share, label}]`. **Offsets index the original input**, even
  though rules run on the normalised text: `Doc.to_orig` maps them back through
  the offset map of `text.normalise_with_map` (curly quotes, dashes, double
  spaces, newlines, soft hyphens, NFC changes are all handled; tested).
  `direction` is the direction this span points to (for a ChoiceRule, both
  variants are reported with opposite signs). `contribution_share` is the span's
  signed share of its rule's contribution *before* the family cap.
- `annotate(text)` → `{score, prob, label, n_words, n_sentences, rules,
  families: {f: {raw, clipped}}, spans, doc_level, evidence_translated,
  evidence_original}`, JSON-serialisable. `doc_level` holds what is not tied to
  a span: the Poisson absence term of RateRules, StatRule residuals (sentences
  whose share is below `span_min`), PredicateRule residuals and `cap:` entries.
  For every rule, span shares + its doc-level entry = its contribution.

How each type reports spans: HitRule/RateRule → every match (RateRule: the
per-hit term k·ln(r_t/r_o) is split over the hits, the length term is
doc-level); ChoiceRule → every A and B match with its own sign; StatRule → each
sentence whose share ≥ `span_min` (default 0.25); PredicateRule → `spans_fn`.

## The Doc API (what rules can read)

All offsets are into `doc.text` (normalised). Regex rules run on `doc.lower`
(same length as `doc.text`).

| view | |
|---|---|
| `raw`, `text`, `lower` | input; normalised; lower-cased normalised |
| `sent_spans`, `sentences`, `n_sentences` | sentence (start, end) / strings |
| `sent_tokens`, `sent_tok_starts` | tokens per sentence (case kept, punctuation included) and their start offsets |
| `sent_words`, `words`, `n_words`, `counts` | lower-cased word tokens per sentence / flat / count / Counter |
| `first_words` | lower-cased first word of each sentence |
| `punct` | Counter of punctuation tokens |
| `is_address` | per sentence: vocative only ("Herr talman!", "Fru kommissionär, kära kolleger!") |
| `to_orig(start, end)` | map a `text` span to a `raw` span |

Normalisation: `“ ” „ « »` → `"`, `‘ ’ ´ \`` → `'`, `— ― ‒` → `–` (en dash kept
distinct from hyphen `-`), `‐ ‑ −` → `-`, `…` → `...`, NBSP/tabs/newlines →
space, runs of spaces collapsed, soft hyphens and zero-width characters deleted,
NFC. Tokens: words (letters/digits, internal `-`, `:` and `'`: `EU-länderna`,
`EU:s`, `d'Ivoire`, trailing hyphen in `miljö- och`), numbers (`2,3`, `1.5`,
`1 000 000,5`, `12:30`), dotted abbreviations as one token (`t.ex.`, `dvs.`,
`bl.a.`), each punctuation character separately (`...` kept whole). Words =
non-punctuation tokens; `n_words` is the denominator of every rate.

Sentence splitter: a sentence ends at `.`, `!`, `?` or `...` (plus closing
quotes/brackets) when the next character (after opening quotes/brackets/dashes)
is upper-case or a digit, unless the `.` closes a known abbreviation
(`t.ex. bl.a. dvs. s.k. fr.o.m. t.o.m. resp. ca. nr. kap. art. prop. jfr. ...`)
or an initial (`A. Smith`, `E.U.`). `osv. etc. m.m. m.fl.` end a sentence when
a capital follows. `Herr talman! Jag ...` is two sentences.

## Calibration protocol

Calibrate on **dev-train**, select and compare on **dev-val**, score **test**
once at the end.

| part | sessions | used for |
|---|---|---|
| dev-train | dev, before 2002-03-01 | reading rule values off `rates.py` (and fitting hybrid models on `hybrid-ml`) |
| dev-val | dev, 2002-03-01 – 2004-06-30 | comparing rule sets, ablations, error reading, choosing what to keep |
| test | after 2004-06-30 | one final, pre-registered run (`--i-mean-it`) |

The cut is by session date (`evals/devsplit.py`, `VAL_START = "2002-03-01"`),
so a session never straddles it; the parts are not re-balanced (500-token bal:
train 975 O / 848 T, val 392 O / 519 T). The starter family was calibrated on
all of dev before the cut, so its dev-val numbers are slightly optimistic; new
families must not look at dev-val statistics.

1. Write the rule's pattern(s) with a rationale *before* looking at numbers.
2. Look at dev-train statistics (the tool refuses `--split dev-val`, `dev` and
   `test`; `--split dev-train` is the default):

   ```bash
   python3 -m tsw.tools.rates --rx '\bju\b(?!\s+\w+(re|are)\b)' --kwic 8 --by-speaker --by-year
   python3 -m tsw.tools.rates --choice '\bska\b' '\bskall\b'
   python3 -m tsw.tools.rates --words "emellertid,huruvida" --mask-topic
   python3 -m tsw.tools.rates --rule starter.modal_particles     # an existing rule's pattern(s)
   python3 -m tsw.tools.rates --stat sentlen --clip 1 60         # or --stat module:units_fn
   ```

   It prints, per class, hits, rate per 1000 words, share of texts with ≥ 1 hit,
   the rate rounded to 2 s.f., ln(T/O) per hit, the most frequent matched forms,
   P(A) for choices, random KWIC lines, speaker concentration (top-5 speakers'
   share of hits) and per-year rates. Default data: the dev-train part of
   dev.500.bal (~0.5M words per class); `--gran 100/sent`, `--unbalanced` (more
   T text for rare items),
   `--limit N`. A full pass takes a few seconds; anything heavier → `srun`.
3. Read the KWIC lines: are the hits what you meant? Fix the regex, not the numbers.
4. Type rounded values into the rule, with a comment of what you saw.
5. Evaluate (`evals/run_all.py` via `srun`; it scores dev-train and dev-val),
   check the per-rule table in the dev-val results JSON (`rules` → `auc`,
   `sign_acc`, `fire_O/T`) for **masked** conditions, and the family-level
   ablation (`Engine(exclude="yourfamily")`) on dev-val. Compare dev-train and
   dev-val AUC: a gap above ~0.03 at 500 tokens means the hand calibration is
   fitting dev-train. Read errors on dev-val (`--show-errors`), but take
   numbers only from dev-train.

## Anti-overfitting rules

- Every rule needs a linguistic rationale (cite `docs/literature.md` or say
  `[exp]`). No rule exists just because a dev statistic differs.
- **No topic words**: country/nationality/party/person/city/institution names
  are banned in patterns. Always check the `masked` numbers; a rule whose AUC
  drops under masking is keying on topic.
- **Speaker effects**: Swedish originals come from ~100 MEPs; the translated
  class is dominated by Commission speakers (Patten, Byrne, Nielson,
  Bolkestein, Liikanen, Barroso, ...). Use `--by-speaker`: if the top-5
  speakers carry most of the hits, the "effect" is idiolect or Commission
  boilerplate (e.g. "jag skulle vilja", "först av allt" at turn openings).
- **Boilerplate**: "Herr/Fru talman!" openings are 49.7% of original vs 65.8% of
  translated turns. That is an artefact of turn structure; do not build rules on
  it. Vocative-only sentences are flagged in `doc.is_address`; skip them in
  sentence-position and sentence-length rules (`sentence_lengths` does).
- **Diachrony**: dev = 1997–mid 2004, test = mid 2004–2011. House-style and
  language drift (skall → ska, kommer att → kommer, anglicisms) shifts rates
  between dev and test. Prefer ratios between native variants of the same era,
  keep caps small for drifting items, and mention the risk in the rationale.
- Europarl artefacts (procedural phrases, Commission formulae) belong in a
  separate family that can be excluded, not in the core families.
- Values: ≤ 2 significant digits, typed by hand. No scripts that write rule
  values, no grid search over weights/caps/thresholds, not even "just to see".
- Never look at the test split (no rates, no error inspection, no runs without a
  pre-registered reason; `evals/run.py` requires `--i-mean-it`).
- Never read rule values off dev-val: it is the selection set. Decisions such as
  "keep this family" are taken on dev-val; rates and probabilities come only
  from dev-train.

## Calibration findings so far (starter family, dev.500.bal)

The Europarl Swedish originals are *edited transcripts* written to formal norm,
and the EP translations are professional. Several textbook directions do not
hold, so check every literature claim:

| item | literature | dev (per 1000 words, O / T) | used as |
|---|---|---|---|
| *den/det/de här* share vs *denna/detta/dessa* | den här = O | 0.013 / 0.060 (share); den här rate 0.17 / 0.90 | reversed: den här = T |
| *ska* share vs *skall* | skall = T | 0.0022 / 0.014 (EP house style: skall ≥ 98% in both) | reversed, weight 0.5, cap 1 |
| *skall+ska* overall rate | – | 6.4 / 2.8 | not used yet (candidate O rule) |
| modal particles *ju* | O | 0.96 / 0.054 | O (strongest single item) |
| *nog / väl / visst* (disambiguated) | O | 0.18/0.12, 0.10/0.020, 0.066/0.017 | O |
| *ändå / nämligen / alltså* | O | 1.9 / 0.44 | not in starter (particles family) |
| *emellertid / huruvida / således* | T | 0.55/0.59, 0.089/0.13, 0.13/0.050 | no signal → rule disabled |
| *det faktum att* | T | 0.093 / 0.37 | calque, +1.3 per hit |
| *först av allt* | T | 0.0029 / 0.039 (all) | calque, +2.0 (often at turn openings) |
| *spendera* | T | 0.010 / 0.031 (all) | +1.1 |
| *jag/vi skulle vilja* | T | 0.21 / 0.70 (all) | left to the calques/register family |
| *i termer av, på basis av, i ljuset av, baserat på* | T | no difference | not used |
| *implementera* | T | 0.039 / 0.0003 | reversed (O MEPs use it) – not used |
| words per sentence (non-vocative) | T longer (weak) | mean 18.3 / 21.5, sd 9.7 / 11.6 | StatRule, weight 0.5 |

Lessons from the starter run (per-rule table, dev 500 masked):

- `starter.dem_har_vs_denna` has rule-alone AUC 0.505 although its mean
  contribution is −0.19 on O and +0.15 on T. The *rate* of *den/det här* differs
  5×, but *denna/detta/dessa* is also more frequent in T (14.0 vs 12.5 per 1000;
  pronominal *Detta är ...* has no *det här* alternative in edited EP prose), so
  the binomial "choice" assumption is violated and texts with many *detta* are
  pushed toward O. When the B variant's rate also differs, a ChoiceRule is the
  wrong model: use a RateRule on the informative variant, or restrict B to
  contexts where A is a real alternative (determiner + noun).
- `starter.ska_vs_skall` has AUC 0.75 but tiny contributions: it is really
  acting as a *skall*-rate detector (originals use *skall/ska* 2.3× as often).
- With only O-leaning *presence* rules, the Poisson absence term makes the
  default lean to T (without the StatRule, 75% of dev texts are predicted T).
  Judge rules by AUC and sign accuracy; accuracy becomes meaningful once both
  directions have presence evidence.

## Starter results (dev, balanced; commit bab4888)

| rules | sent | sent masked | 100 | 100 masked | 500 | 500 masked |
|---|---|---|---|---|---|---|
| Default (acc / AUC) | .560 / .587 | .560 / .587 | .649 / .711 | .649 / .710 | .771 / .847 | .771 / .847 |
| minus `starter.sent_len` | .484 / .539 | .484 / .539 | .519 / .568 | .519 / .567 | .634 / .708 | .634 / .708 |

A full `run_all` (6 conditions) takes ~55 s on one CPU, including Slurm start-up.

## Gotchas for family authors

- Regexes run on **lower-cased, normalised** text: write lower-case patterns
  with straight quotes `"` `'` and the en dash `–`; pass `case=True` to match
  `doc.text` instead. Use `\s+` between words (harmless; spaces are already
  collapsed).
- `\b` treats `-` as a boundary (`\bju\b` matches in `ju-ju`, `\bska\b` in
  `EU-ska`). `words=[...]` uses `(?<![\w-])…(?![\w-])`; do the same in
  hand-written regexes when it matters.
- Look-behinds must be fixed-width (`(?<!ett |i )` fails; chain them:
  `(?<!ett )(?<![\w-]i )`). Use non-capturing groups `(?:…)`; do not name your
  own groups `_m0`, `_m1`, … (used for dict-pattern labels).
- Rule names must be globally unique (discovery raises on duplicates): prefix
  with the family (`wordorder.inversion_after_adverbial`).
- Constructors check `direction` against the parameters: a sign typo raises at
  import time, which breaks discovery of all families. Run the unit tests.
- ChoiceRule: A and B must not overlap; `direction` refers to variant A.
- `contribution_share` is before the family cap; `explain` contains `cap:<fam>`
  entries (not real rules).
- Family caps (default 4.0) are your defence against correlated rules counting
  the same tokens twice (*detta* in a demonstrative rule and in *i detta
  avseende*); put correlated rules in one family, or assign each span to one rule.
- Sentence-level dev texts average ~20 words: RateRules contribute almost only
  through hits there; ChoiceRules and HitRules carry sentence-level accuracy.
- `doc.first_words` includes the vocative sentences; check `doc.is_address[i]`.
- The splitter needs an upper-case letter or digit after the terminal
  punctuation; lower-cased input will be one sentence.
- Speed: one `Doc` per text, views computed lazily and cached; a regex over
  `doc.lower` costs microseconds, a Python loop over all tokens is ~100× slower.
  Precompile at module import (rule constructors do). The whole `run_all`
  (6 conditions, ~10M words) should stay within a few minutes on 1 CPU; time
  your family with `--limit 500` before submitting a full run.

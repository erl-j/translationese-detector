# Integrating the rule families (2026-11-06)

Six family branches were developed in parallel against the starter engine and
merged into `integrate`: `feat/stats`, `feat/connpunct`, `feat/register`,
`feat/particles`, `feat/wordorder`, `feat/calques`. Each was calibrated on its
own, so evidence that two families read off the same tokens was counted twice
once they were combined. This file records how that double counting was found
and resolved, what the family caps do in the combined system, each family's
marginal contribution, and the final numbers.

Everything here was measured on **trn** (topic-masked, balanced). val was
scored once at the end, with metrics only; no decision below uses a val number.
No weight, cap or threshold was searched. Every change is a hand decision of
one of three kinds: exclude the overlapping pattern from one rule (preferred),
disable one rule, or scale one rule's weight by a round factor (×0.5).

## 1. Merge notes

- Order: stats, connpunct, register, particles, wordorder, calques (`--no-ff`,
  one merge commit each). Unit tests passed after every merge (62 → 152 tests).
- `tsw/rules/core_starter.py` merged without conflicts. All three branches'
  edits are kept. From register: `starter.dem_har_vs_denna` and
  `starter.ska_vs_skall` disabled. From particles: new `starter.modal_particles`
  regexes and rates. From calques: `starter.calques` disabled.
  (`starter.formal_connectives` was already disabled on main.)
- `results/LEADERBOARD.md` conflicted at every merge. Resolved as the union of
  rows, sorted by timestamp.
- stats and connpunct both committed a file named
  `results/20261105T230458Z_tsw.engine_Default_val_sent.json` (two different
  runs in the same second). Both are kept, renamed `..._sent__stats.json` and
  `..._sent__connpunct.json`.
- No docs conflicted: the branches kept their findings in module docstrings.
  The uncommitted LEADERBOARD edits in the calques worktree were ignored.
- Naive merge (all six branches, no resolution), trn masked: 100 tokens acc
  .818 / AUC .902, 500 tokens .947 / .989.

## 2. Double counting

### Method

`evals/integration/dump_contrib.py` scores every trn text (topic-masked) with
the merged engine and stores each rule's contribution, each family's raw and
clipped sum, and every pair of evidence spans (`Rule.evidence`) from two rules
that overlap in characters. `evals/integration/analyze_contrib.py corr`
computes the **within-class** Pearson correlation of every pair of rule
contributions, separately in O and in T texts (rules firing in ≥ 1% of texts).
Contributions are signed LLRs, so r > 0 means two rules move the score the
same way on the same texts (same-direction evidence counted twice), and r < 0
means they cancel. The analysis ran on trn 100-token chunks (14170) and was
repeated on 500-token chunks (2734); the 500-token level shows the slower,
text-level correlations (style, speaker).

Full matrices, before and after resolution:
`docs/integration/corr_trn{100,500}_{pre,post}.json` (`{"O"|"T": {rule: {rule: r}}}`).
Span overlaps: `docs/integration/span_overlap_trn100_{pre,post}.json`
(`"a|b": [n overlapping span pairs, same sign, opposite sign, Σ min |share|]`).
Overlaps with sentence-level spans (`starter.sent_len`, the stats rules) and
with wordorder's opening spans, which contain the finite verb and subject,
are mostly containment rather than shared evidence. They were read but are not
counted as double counting unless r also says so.

Flag rule: a same-direction pair with |r| > 0.3 in either class at 100 or 500
tokens, or a pair whose spans are the same tokens.

### Overlap table (r = within-class r on trn.100 masked, O / T; 500 in brackets where it matters)

| # | pair | r before | spans before → after | decision | why |
|---|---|---|---|---|---|
| 1 | `register.av_rate` × `stats.av_other` (and × `stats.av_nominal`) | .96 / .94 (.35 / .40 with av_nominal; 500: .40 / .45) | 22732 + 2742 → 0 | **disable `register.av_rate`** | Same token, same direction. The stats rules split 'av' into nominal and other uses and model text-level variance (tau), while the rate rule had an ad-hoc weight of 0.5. |
| 2 | `register.skulle_vilja_vs_vill` × `calques.skulle_vilja` | .99 / .98 | 4758 → 0 | **disable `calques.skulle_vilja`** | The same ChoiceRule twice (p .097/.32 vs .10/.33, rule AUC .559 vs .560). Kept in register: it is a broad lexical-choice rule that fires in 27% of texts, and the calques family is a precision lexicon whose cap of 3 already binds on 12% of T 500-token texts. |
| 3 | pseudo-cleft: `calques.lexicon` 'Vad vi behöver är' × `wordorder.subject_theme` PSEUDO_CLEFT | lexicon × subject_theme .02 / .00 (rare item) | 32 same-direction hits | **remove the lexicon entry** | wordorder owns the construction as an opening category (+0.72 per sentence), estimated jointly with the other openings. |
| 4 | 'Så + pronoun, no V2' (`calques.lexicon`) × `wordorder.v2_violation` | — | **0** | **keep both (no overlap)** | v2_violation only scores NOINV after a listed fronted adverbial/PP and V3. wordorder handles 'så' only as fronted object + finite verb, so 'Så vi måste' is never scored there. The calques entry is the only rule that sees it. |
| 5 | dvs: `particles.dvs` × `connpunct.written_out_abbrev` | .33 / .08 (500: .30 / .14) | 605 → 0 | **exclude dvs./d.v.s./det vill säga from `written_out_abbrev`** | particles.dvs counts both spellings as O (reformulation marker, a rate). In the abbreviation choice, 'dvs.' was a second O count and 'det vill säga' a contradicting T count. Choice re-read on trn: P(written out) O .28, T .71. |
| 6 | (found) bl.a.: `register.bland_annat` × `connpunct.written_out_abbrev` | −.03 / −.17 | 510 (237 same, 273 opposite) → 0 | **exclude bl.a./bland annat from `written_out_abbrev`** (same edit as 5) | Same structure as dvs: register counts both spellings as O, so 'bl.a.' was counted twice and 'bland annat' contradicted itself. |
| 7 | sentence-initial 'Då': `particles.da_anaphoric` × `wordorder.fronted_theme` (INV_ADV) | .14 / .02 (500: .20 / .03) | 162 → 11 | **drop the "'Då' + finite verb" alternative from `da_anaphoric`** | The prefield belongs to the opening model, where 'Då måste vi' is a topicalised adverb. da_anaphoric keeps the middle-field forms ('finns det då', 'det är då'). Re-read on trn: O .19 (2000-04 ~.15, typed .15 as the author did for drift), T .059. |
| 8 | därför/dessutom/…: `connpunct.native_connectives` × `wordorder.fronted_theme` | .20 / .19 (500: .22 / .18) | 1576 (Σ min share 790, the largest same-direction mass after #1/#2) → 45 | **exclude sentence-initial positions** (also after 'Men'/'Och') from `native_connectives` | Same token in the prefield: wordorder scores 'Därför anser jag' as INV_ADV. The connpunct rationale is the middle-field connective ('Vi måste därför …'), which it keeps. Re-read on trn: O 4.7, T 1.9 per 1000 words (was 6.4 / 2.2). |
| 9 | (found) 'i alla fall': `connpunct.native_connectives` × `particles.stance` | .04 / .02 | 70 → 0 | **remove 'i alla fall' from `native_connectives`** | Identical tokens in two O rules. Kept as a stance adverb, which is how particles calibrated it. |
| 10 | copular det: `particles.copular_det` × `wordorder.subject_theme` (S_DET 'Det är …', CLEFT) | **.37 / .30** (500: .42 / .30) | 5112 → 288 | **exclude sentence-initial 'Det är/var'** (also after Men/Och) from `copular_det` | The opening 'Det är …' is an S_DET/CLEFT category of the opening model. copular_det keeps middle-field 'att det är' and inverted 'är det'. Re-read on trn: O 3.8 (2000-02 ~3.5, typed 3.5), T 2.4 (typed 2.3). After the change r is .06 / .03 (500: .14 / .09). |
| 11 | också: `connpunct.additive_ocksa` × `register.aven_vs_ocksa` | **.61 / .49** (500: .61 / .55) | 6012 → 0 | **disable `register.aven_vs_ocksa`** | Every 'också' counted as O twice. The rate carries the information (rule-alone AUC .640 vs .570 on trn.100). register's own notes already said the 'också' rate carries more than the även share. 'även' (1.9 vs 1.6 per 1000 words) adds little on its own. |
| 12 | Samtidigt: `calques.native_counterparts` × `connpunct.native_connectives` (× `wordorder.fronted_theme`) | .13 / – | 278 (+310 with wordorder) → 0 / 7 | **remove sentence-initial 'Samtidigt' from `native_counterparts`** | It was counted three times: native counterpart, native connective and INV_ADV. After #8 the prefield 'Samtidigt' is wordorder's only. |
| 13 | (found) 'Först vill jag': `calques.native_counterparts` × `wordorder.fronted_theme` | (included in 12's span count) | → 0 | **remove the entry** | Same reason: 'Först' + finite verb is an INV_ADV opening. |
| 14 | (found) 'självklart': `register.sjalvklart` × `particles.of_course` | **.60 / .42** (500: .54 / .40) | 268 → 0 | **remove 'självklart' from the B variant of `of_course`** | Both rules counted 'självklart' as O. register models it as a rate, which the register author chose deliberately after a naturligtvis choice failed. of_course keeps naturligtvis vs givetvis/förstås, re-read on trn: P(naturligtvis) O .71, T .85. |
| 15 | (found) 'man': `particles.generic_man` × `particles.ni_vs_man` | **.68 / .44** (500: .65 / .49) | 5268 (unchanged) | **`generic_man` weight 0.5 → 0.25** (×0.5) | Every generic 'man' is O evidence in the rate rule and the B variant of the choice rule, so the overlap cannot be cut out of either pattern. The choice rule also carries the 'ni' side, so it stays in charge, and the drifting, bursty rate rule is halved. (r is scale-free, so it stays .68; the doubled weight per 'man' drops from ≈ −0.43 to ≈ −0.33 nats.) |
| 16 | (found, 500 only) `stats.long_words` × copular_det / tycker_vs_anser / generic_man / modal_particles / av_other / av_nominal | .29–.36 in O, .22–.34 for av (500); ≤ .22 at 100 | containment | **`stats.long_words` weight 1.0 → 0.5** | A global word-length share has the short register words (det, är, man, tycker, ju) in its denominator and nominal style ('genomförandet av') in its numerator, so at 500 words it re-counts the register and av rules. Halving the one global statistic is simpler than touching the five lexical rules. |
| 17 | (found) "… är skälet till varför": `connpunct.calque_connectives` × `calques.lexicon` | .02 / .01 | 19 → 0 | **exclude '(det\|detta) är skälet' + 'till varför' from `calque_connectives`** | Rare, but the same tokens. calques' 'skälet till varför' (EN 'the reason why') owns it. Values not re-read (a handful of hits). |

### Overlaps examined and kept (different evidence on the same stretch of text)

| pair | r (100; 500) | spans | why kept |
|---|---|---|---|
| `connpunct.initial_adverbial_comma` × `wordorder.v2_violation` | – ; T .22 | 19 | 'Dessutom, vi måste': the comma (punctuation habit) and the missing inversion (word order) are separate properties. connpunct deliberately scores only the comma. Below .3. |
| `connpunct.initial_adverbial_comma` × `wordorder.fronted_theme` | .16 / .03 | 4690 (mostly the small B term "no comma") | Mostly the "no comma" variant of the choice (−0.09 per sentence) against INV_ADV. Small per span. |
| `connpunct.calque_connectives` × `wordorder.fronted_theme` (INV_CONJ: 'Till sist', 'Först och främst') | .04 / .06 | 288 | Lexical choice of the English-equivalent marker vs. the fact that a structuring conjunct is fronted. 10× less mass than #8, r ≈ 0. |
| `connpunct.att_drop` × `register.tycker_vs_anser` | .20 / .15 (500: .26 / .07) | 110 | 'tycker jag är': the verb choice and the dropped complementiser are two native choices. Below .3. |
| `particles.copular_det` × `wordorder.fronted_theme` | .08 / .02 | 932 | Inverted 'är det' inside an INV opening ('Därför är det …'): the fronting and the expletive subject are different choices. |
| `register.dem_har_rate` × `wordorder.subject_theme` (S_DEM) | .04 / .01 | 89 | 'Det här är' opening. Rare, low r. |
| `register.tycker_vs_anser` × `wordorder.*` | ≤ .13 | 423 | The verb is inside the opening span; different evidence. |
| `calques.lexicon` × `wordorder.subject_theme` | .01 / −.02 | 538 | Containment: 'välkomnar', 'ser fram emot', 'känner till att' are the finite verb of a subject-first opening. |

### After resolution

The only pair left above |r| = .3 at 100 tokens is #15 (by construction,
handled by the weight). At 500 tokens it is #15 plus the `stats.long_words`
pairs from #16 (r unchanged, weight halved). Merged-and-resolved vs
naive merge, trn masked: 100 tokens AUC .902 → .907 (acc .818 → .822); 500
tokens AUC .989 → .990 (acc .947 → .945).

Rules: 60 defined, **49 enabled** (naive merge: 52).

## 3. Family caps

`analyze_contrib.py caps` / `uncap` on the final engine, trn masked. bind =
share of texts whose family sum hits the cap; *sign ok* = share of bound texts
where the removed amount points to the true class; *AUC(clipped)* = AUC of the
signed removed amount over all texts (0 where the cap does not bind); *wrong* =
bound texts the full engine misclassifies; Δ = effect of removing that cap alone.

| family | cap | 100: bind O / T | 500: bind O / T | 500: sign ok | 500: AUC(clipped) | 500: wrong | Δacc / ΔAUC (500, uncapped) |
|---|---|---|---|---|---|---|---|
| calques | 3 | 0.0% / 0.2% | 0.1% / 11.7% | .99 | .56 | 2 / 161 | .0000 / +.0002 |
| connpunct | 4 | 0.0% / 0.4% | 6.4% / 9.6% | 1.00 | .58 | 0 / 218 | .0000 / +.0001 |
| particles | 4 | 1.0% / 0.1% | 20.0% / 7.9% | .98 | .63 | 3 / 381 | .0000 / .0000 |
| register | 6 | 0.1% / 0.0% | 8.3% / 2.6% | 1.00 | .55 | 1 / 149 | .0000 / .0000 |
| stats | 2 | 0.0% / 0.2% | 0.4% / 3.3% | .90 | .52 | 2 / 50 | .0000 / .0000 |
| wordorder | 4 | 0 | 0.0% / 0.1% | – | – | 0 / 1 | .0000 / .0000 |
| starter | 6 | 0 | 0 | – | – | – | – |

Findings. Caps bind almost only at 500 tokens (particles on 20% of O texts,
calques on 12% of T texts). The clipped amount nearly always points the right
way (sign ok .90–1.00) and carries evidence (AUC .52–.63), but the texts where
a cap binds are ones the engine already gets right (≤ 3 errors per family), so
lifting any one cap changes trn accuracy by 0 and AUC by ≤ .0002. **No cap was
changed.** They cost nothing on trn and remain the safeguard against correlated
bursts we cannot see on trn (one speaker repeating a particle, a calque-heavy
speech). Before resolution the picture was the same (particles 24% O, calques
20% T at 500; Δ ≤ .0002).

## 4. Leave-one-family-out ablation (trn, masked, balanced)

Exact from the per-text family sums (score = Σ clipped family sums, so
dropping a family drops its term; identical to `Engine(exclude="<family>")`).

| without | sent acc / AUC | Δ | 100 acc / AUC | Δ | 500 acc / AUC | Δ |
|---|---|---|---|---|---|---|
| (full Default) | .653 / .723 | | .822 / .907 | | .945 / .990 | |
| connpunct | .632 / .695 | −.021 / −.028 | .788 / .877 | −.034 / −.030 | .921 / .980 | −.024 / −.010 |
| register | .633 / .696 | −.021 / −.027 | .800 / .882 | −.023 / −.025 | .935 / .985 | −.010 / −.006 |
| particles | .636 / .702 | −.018 / −.021 | .801 / .886 | −.021 / −.021 | .939 / .984 | −.006 / −.006 |
| starter (modal particles, sent_len) | .634 / .710 | −.019 / −.014 | .812 / .896 | −.010 / −.011 | .940 / .988 | −.005 / −.002 |
| calques | .647 / .713 | −.006 / −.010 | .810 / .896 | −.012 / −.011 | .941 / .988 | −.004 / −.003 |
| wordorder | .649 / .717 | −.004 / −.006 | .817 / .902 | −.005 / −.005 | .945 / .989 | +.000 / −.002 |
| stats | .653 / .723 | .000 / .000 | .821 / .905 | −.001 / −.002 | .943 / .990 | −.002 / .000 |

Every family helps, but stats only barely: it abstains below 50 words, so it
does nothing at sentence level, and in the combined system its av and
word-length evidence is largely carried by other families (at 500 tokens ΔAUC
−.00003). wordorder's marginal value is small too (it is spread over many weak
per-sentence opening terms). It is kept for the explanations and because its
categories now own the prefield evidence that #7, #8, #10, #12 and #13 took
away from other rules. connpunct, register and particles carry most of the
marginal signal.

## 5. Results

One `evals/run_all.py --clf tsw.engine:Default --splits trn,val` at commit
ef3cb36 (rule code as listed here; the runs are flagged `dirty` only because
the docs and dashboard notes were being edited in the worktree at the time).
acc / AUC, balanced:

| system | split | sent | sent masked | 100 | 100 masked | 500 | 500 masked |
|---|---|---|---|---|---|---|---|
| **Default (integrated)** | trn | .653 / .723 | .653 / .723 | .822 / .907 | .822 / .907 | .945 / .990 | .945 / .990 |
| **Default (integrated)** | val | .612 / .667 | .613 / .666 | .747 / .833 | .747 / .833 | .920 / .977 | .920 / .977 |
| starter Default (main 80d27ff) | trn | .560 / .587 | .560 / .587 | .649 / .711 | .649 / .710 | .771 / .847 | .771 / .847 |
| starter Default (main 80d27ff) | val | .560 / .577 | .560 / .577 | .611 / .640 | .611 / .640 | .700 / .791 | .700 / .790 |
| `baselines:MeanSentLen` | trn | .561 / .583 | .561 / .582 | .629 / .677 | .629 / .677 | .722 / .788 | .721 / .787 |
| `baselines:MeanSentLen` | val | .564 / .584 | .563 / .583 | .633 / .671 | .633 / .671 | .709 / .793 | .708 / .792 |

trn − val AUC gap (masked): sent .057, 100 .074 (starter .070), 500 .013.
Family-alone runs were not repeated on val.

## Reproduce

```bash
# per-text contributions + span overlaps on trn (one CPU, ~1.5 min per granularity)
srun -p berzelius-hopper-cpu --time=00:15:00 -c 1 --mem=8G \
    python3 evals/integration/dump_contrib.py . 100 /proj/.../trn100.jsonl --spans
python3 evals/integration/analyze_contrib.py corr  /proj/.../trn100.jsonl 0.3
python3 evals/integration/analyze_contrib.py caps  /proj/.../trn500.jsonl
python3 evals/integration/analyze_contrib.py uncap /proj/.../trn500.jsonl
python3 evals/integration/analyze_contrib.py lofo  /proj/.../trnsent.jsonl /proj/.../trn100.jsonl /proj/.../trn500.jsonl
```

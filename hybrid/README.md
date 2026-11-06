# hybrid/ — ML on top of the expert features (side track, branch `hybrid-ml`)

**Not part of the expert system.** `main` stays purely rule-based. This package fits
simple models (L2 logistic regression, gradient-boosted trees, shallow decision trees)
whose inputs are *expert* features, to measure how much signal the expert features
carry, re-weight the hand-set rules, and suggest rules back to the rule writers.

Inputs (all expert features):
1. **Literature word-list features** (`features.py`, Phase A): ~105 counts from
   docs/literature.md (IDs A1..G) turned into rates per 1000 words, sentence-initial
   shares, smoothed choice ratios and length statistics (110 model features).
2. **Engine outputs** (`engine_cache.py`, Phase B): each rule's contribution from
   `tsw.engine:Default.explain()`, per-family subscores, and the raw statistic each
   rule computes (hits/1000 words, choice share, unit mean).
3. Reference only: `bow_lr`, LR on the 300 most frequent words (no expert input).

Protocol (main's trn/val/test split, see README): every model is fitted on **all of
trn** (sessions ≤ 2004-06-30); hyper-parameters (LR's C) by 5-fold CV *inside* trn,
folds grouped by session date. Each fitted model is scored **once** on **val**
(2004-07 .. 2006) through the harness, unmasked and topic-masked, next to
`tsw.engine:Default` and MeanSentLen on the same records. Val is metrics-only: the
models are wrapped score-only (no per-feature val statistics) and nothing reads val
texts. Test (2007-2011) is never touched.

## Run (Slurm, 1 CPU; ~10 min for everything)

```bash
sbatch hybrid/slurm/retrain_val.sh                                  # Phase A + B, val eval, reports
sbatch --export=ALL,STEPS=phase_b hybrid/slurm/retrain_val.sh       # after merging rule families
```

Steps (all `python -m hybrid.<step>` with the venv python
`.venv-hybrid/bin/python`, numpy/scipy/scikit-learn):
`cache` (trn feature cache) → `train` (Phase A fits) → `engine_cache` (engine contributions +
rule statistics on trn) → `stack` (Phase B fits) → `eval_val` (harness on val) → `report`.

Harness (live scoring from raw text):

```bash
$PY evals/run.py --clf "hybrid.clf:Hybrid(model='lr_expert')" --split val --gran 500 --mask-topic
$PY evals/run.py --clf "hybrid.clf:Stacked(model='stack_rules')" --split trn --gran 100
```

`Hybrid` / `Stacked` pick the per-granularity model by text length (< 90 tokens →
sentence model, < 300 → 100-token model, else 500), or `gran='500'` forces one.

## Files

| file | what |
|---|---|
| `features.py` | literature features: word lists, regexes, `raw_counts`, `transform`, `feature_meta` |
| `cache.py` | trn feature/BoW cache → `data/cache/hybrid/trn.<gran>.npz` (gitignored) |
| `train.py` | Phase A fits on trn → `models/*.json`, `reports/training.json` |
| `engine_cache.py` | engine contributions + rule statistics on trn → `data/cache/hybrid/engine.default.<gran>.npz` |
| `eval_val.py` | scores every fitted model once on val via `evals/run.py:run_one` (score-only) |
| `slurm/retrain_val.sh` | the whole pipeline as one 1-CPU Slurm job |
| `stack.py` | Phase B models (`engine_calib`, `stack_rules`, `stack_families`, `lr_engine`, `gbt_engine`, `tree_engine`, `hybrid_plus`, `lr_all`, `gbt_all`) |
| `clf.py` | harness classifiers `Hybrid`, `Stacked` |
| `report.py` | `reports/METRICS.md` (val), `coefficients.md`, `literature_check.md`, `tree_rules.md` (trn) |
| `models/` | fitted LR/tree models as JSON (GBT pickles live in the gitignored cache) |
| `reports/` | generated reports; `stacking_<gran>.md` = learned vs hand-set rule weights |

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

Protocol: fit on **dev-train**, report on **dev-val** (date split, `SPLIT.md`), unmasked
and topic-masked, plus an unseen-speaker subset; the pure engine and MeanSentLen are
scored on the same records. Test is never touched.

## Run (all steps fit the login-node budget with the 5-rule starter engine)

```bash
PY=.venv-hybrid/bin/python   # numpy, scipy, scikit-learn
export OMP_NUM_THREADS=1                                                       # be nice on the login node
$PY -m hybrid.cache --gran 500 100 sent       # Phase A feature cache       ~90 CPU-s, once per features.py
$PY -m hybrid.train                           # lr_expert, tree_d3/4, gbt, bow_lr   ~45 CPU-s
$PY -m hybrid.engine_cache --gran 500 100 sent  # engine contributions + stats  ~60 CPU-s (5 rules; grows with rules)
$PY -m hybrid.stack                           # Phase B models               ~40 CPU-s
$PY -m hybrid.report                          # reports/*.md                 ~10 CPU-s
$PY -m unittest discover -s hybrid/tests -t .
```

After a rule family is merged into main: `git merge main`, re-run `engine_cache`
(use Slurm if it exceeds ~1 CPU-minute, see its docstring), `stack`, `report`.
`engine_cache` also writes the official pure-engine dev-val results.

Harness (live scoring from raw text, same numbers as the cached evaluation; checked):

```bash
$PY evals/run.py --clf "hybrid.clf:Hybrid(model='lr_expert')" --split dev-val --gran 500 --mask-topic
$PY evals/run.py --clf "hybrid.clf:Stacked(model='stack_rules')" --split dev-val --gran 100
```

`Hybrid` / `Stacked` pick the per-granularity model by text length (< 90 tokens →
sentence model, < 300 → 100-token model, else 500), or `gran='500'` forces one.

## Files

| file | what |
|---|---|
| `features.py` | literature features: word lists, regexes, `raw_counts`, `transform`, `feature_meta` |
| `cache.py` | feature/BoW cache → `data/cache/hybrid/dev.<gran>.npz` (gitignored) |
| `train.py` | Phase A models; writes `models/*.json`, results JSON (split=dev-val) |
| `engine_cache.py` | engine contributions + rule statistics → `data/cache/hybrid/engine.default.<gran>.npz` |
| `stack.py` | Phase B models (`engine_calib`, `stack_rules`, `stack_families`, `lr_engine`, `gbt_engine`, `tree_engine`, `hybrid_plus`, `lr_all`, `gbt_all`) |
| `clf.py` | harness classifiers `Hybrid`, `Stacked` |
| `report.py` | `reports/METRICS.md`, `coefficients.md`, `literature_check.md`, `tree_rules.md` |
| `split_stats.py` | split sizes + speaker overlap (`SPLIT.md`) |
| `models/` | fitted LR/tree models as JSON (GBT pickles live in the gitignored cache) |
| `reports/` | generated reports; `stacking_<gran>.md` = learned vs hand-set rule weights |

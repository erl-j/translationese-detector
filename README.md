# translationese-swe

A fully rule-based expert system that tells Swedish translated from English
(translationese) apart from Swedish written in Swedish to begin with.

- No trained models anywhere in the pipeline: hand-written lexicons, regexes and
  heuristics, combined with hand-set weights.
- Evals come first: `evals/` holds the harness, and `data/` holds the scripts
  that build labelled splits.

Layout
- `data/`   scripts that build the corpora (raw downloads are gitignored)
- `evals/`  eval harness and metrics
- `tsw/`    the rule engine (features → rules → verdict)
- `docs/`   literature notes and the rule catalogue

## Data

Europarl v7 (Koehn 2005), Swedish side. Label 0 = speech given in Swedish,
label 1 = speech given in English and translated into Swedish by the EP.

```bash
# one deterministic command; downloads ~1.5 GB to data/raw/ on first run.
# Takes ~1-3 CPU-minutes, so run it through Slurm, not on the login node:
srun -p berzelius-hopper-cpu --time=00:20:00 -c 2 --mem=8G python3 -I data/build.py
python3 -I data/build.py --stats   # (also via srun) only print label/year counts
```

Outputs `data/processed/{dev,test}.{sent,100,500}.{bal,all}.jsonl` (gitignored,
~420 MB) and the committed `data/MANIFEST.md` / `data/manifest.json` (counts,
sha256 of each file, labelling statistics). Each record has `id, text, label,
src_lang, date, speaker, speakers, speaker_id, n_turns, granularity, n_tokens`.

- Labels: Europarl tags a turn's source language mostly in the *other* language
  versions, so labels come from the `LANGUAGE` attribute in `txt/sv` and in
  `txt/en` (same file, same `SPEAKER ID`) plus inline `(XX)` markers; turns
  whose evidence disagrees, whose sv/en speaker names disagree, or that do not
  look Swedish are dropped. Details in the `data/build.py` docstring.
- Splits by session date: dev = sessions up to 2004-06-30 (1997-mid 2004),
  test = later sessions (mid 2004-2011). `bal` files downsample the majority
  class (translated) with a fixed seed; `all` files keep everything.
- dev is cut once more by session date (`evals/devsplit.py`, no extra files):
  **dev-train** = dev sessions before 2002-03-01, **dev-val** = dev sessions
  from 2002-03-01 to 2004-06-30. Every unit carries one session date, so a
  session never straddles the cut. dev-train/dev-val are not re-balanced
  (500-token bal: train 975 O / 848 T, val 392 O / 519 T).

### Protocol: calibrate on dev-train, select/compare on dev-val, test once at the end

1. Rule values are read off **dev-train** only (`tsw/tools/rates.py` refuses
   dev-val, dev and test).
2. Rule sets, families and ablations are compared and selected on **dev-val**
   (`evals/run_all.py` scores dev-train and dev-val; a large train-minus-val
   AUC gap means the hand calibration is fitting dev-train).
3. **test** is scored once, for the final pre-registered system
   (`--i-mean-it`). The hybrid ML branch (`hybrid-ml`) uses the same cut: models
   are fitted on dev-train and reported on dev-val.
- Granularities: `sent` (sentences of >= 3 tokens), `100` and `500`
  (consecutive sentences of one class within one session date, cut as soon as
  they reach 100/500 whitespace tokens).

## Engine

`tsw/` is the expert system; how it works, how to write a rule family (one file
in `tsw/rules/`, auto-discovered), the rule maths, the calibration protocol and
anti-overfitting rules are in [docs/ENGINE.md](docs/ENGINE.md).

```bash
python3 -m tsw "Herr talman! Det här är ju viktigt."     # verdict + evidence + highlighted spans
python3 -m tsw -f speech.txt --json                      # Engine.annotate() as JSON
python3 -m tsw.tools.rates --words "ju,nog" --kwic 5     # dev statistics for calibrating a rule
python3 -m unittest discover -s tsw/tests -t .
```

## Evals

A classifier is any object with `score(text) -> float` (> 0 means
"translated"), optionally `explain(text) -> {rule_name: contribution}` (or a
fast `score_and_explain(text) -> (score, dict)`). Load it by dotted path:

```bash
python3 evals/run.py --clf baselines:Constant --split dev-val --gran 100
python3 evals/run.py --clf baselines:MeanSentLen --split dev-val --gran 500 --mask-topic --show-errors 5
python3 evals/run.py --clf "baselines:Random(seed=3)" --split dev-train --gran sent --limit 2000
python3 evals/run_all.py --clf tsw.engine:Default  # {dev-train,dev-val} x {sent,100,500} x {unmasked,masked}
python3 evals/run_all.py --clf tsw.engine:Default --split dev   # all of dev (old protocol)
python3 -m unittest discover -s evals/tests -v
```

- Metrics: accuracy, macro-F1, ROC-AUC, confusion matrix; with `explain`, a
  per-rule table (firing rate per class, mean contribution per class, smoothed
  log-odds of firing translated-vs-original, rule-alone AUC, sign accuracy).
- `--mask-topic` replaces country/nationality/city/party/politician tokens
  (`evals/topicmask.py`) with `XTOPIC` before scoring and also reports the
  unmasked numbers. A rule set whose AUC drops a lot under masking is keying on
  topic (see `baselines:TopicCheat`, which is a deliberate cheat).
- `--split` is `dev-train`, `dev-val`, `dev` (default of run.py) or `test`.
  `--split test` is refused without `--i-mean-it`; `--show-errors` is not
  allowed on test.
- Every run writes `results/<ts>_<clf>_<split>_<gran>[_all][_masked].json`
  (metrics, rule table, split (`dev-train`/`dev-val` runs also carry
  `subset_rule`), git commit/dirty/branch, worktree path, mask flag, data
  sha256) in the worktree it runs from, a copy in the shared
  `./results/all/` (gitignored;
  override with `$TSW_SHARED_RESULTS`), and one line in `results/LEADERBOARD.md`.
- Worktrees: `data/processed/` is gitignored, so a worktree without its own
  copy reads the main checkout's (or `$TSW_DATA`).
- Scoring the full sentence-level dev split (~88k sentences) with a real rule
  set can take minutes: use `srun -p berzelius-hopper-cpu --time=... -c 1`.
- Dashboard data: `evals/export_dashboard.py` (runs of every split, rule
  tables, corpus counts incl. dev-train/dev-val) and `evals/export_examples.py`
  (annotated TP/TN/FP/FN examples sampled from dev-val by default).

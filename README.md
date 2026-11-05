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

Outputs `data/processed/{trn,val,test}.{sent,100,500}.{bal,all}.jsonl`
(gitignored) and the committed `data/MANIFEST.md` / `data/manifest.json`
(counts, sha256 of each file, session-date ranges, labelling statistics). Each
record has `id, text, label, src_lang, date, speaker, speakers, speaker_id,
n_turns, granularity, n_tokens`.

- Labels: Europarl tags a turn's source language mostly in the *other* language
  versions, so labels come from the `LANGUAGE` attribute in `txt/sv` and in
  `txt/en` (same file, same `SPEAKER ID`) plus inline `(XX)` markers; turns
  whose evidence disagrees, whose sv/en speaker names disagree, or that do not
  look Swedish are dropped. Details in the `data/build.py` docstring.
- Splits by session date (a session never straddles a cut):

  | split | sessions | 500-token bal (O + T) | use |
  |---|---|---|---|
  | **trn** | 1997-01 .. 2004-06-30 | 1367 + 1367 | calibrate rules, read examples and errors |
  | **val** | 2004-07-01 .. 2006-12-31 | 318 + 318 | aggregate metrics only, sparingly |
  | **test** | 2007-01-01 .. 2011 | 589 + 589 | locked: one final run (`--i-mean-it`) |

  trn is the former `dev` split, byte for byte (same records, same balancing
  seed; sha256 checked), so results recorded on `dev` are trn results and
  `--split dev` is an alias of trn. `bal` files downsample the majority class
  (translated) with a fixed seed; `all` files keep everything.
- Granularities: `sent` (sentences of >= 3 tokens), `100` and `500`
  (consecutive sentences of one class within one session date, cut as soon as
  they reach 100/500 whitespace tokens).

### Protocol: calibrate and inspect on trn, compare on val, test once at the end

1. **trn**: read rule values off `tsw/tools/rates.py` (it refuses val and
   test), read KWIC lines, examples and `--show-errors` output.
2. **val**: look at it only as aggregate metrics (`evals/run.py` /
   `run_all.py`; `--show-errors` is refused on val), and sparingly: a handful
   of runs per branch to compare and select rule sets. Never read val texts,
   never tune a value against a val number. A large trn-minus-val AUC gap
   means the hand calibration fits trn.
3. **test**: scored once, for the final pre-registered system (`--i-mean-it`).

## Engine

`tsw/` is the expert system; how it works, how to write a rule family (one file
in `tsw/rules/`, auto-discovered), the rule maths, the calibration protocol and
anti-overfitting rules are in [docs/ENGINE.md](docs/ENGINE.md).

```bash
python3 -m tsw "Herr talman! Det här är ju viktigt."     # verdict + evidence + highlighted spans
python3 -m tsw -f speech.txt --json                      # Engine.annotate() as JSON
python3 -m tsw.tools.rates --words "ju,nog" --kwic 5     # trn statistics for calibrating a rule
python3 -m unittest discover -s tsw/tests -t .
```

## Evals

A classifier is any object with `score(text) -> float` (> 0 means
"translated"), optionally `explain(text) -> {rule_name: contribution}` (or a
fast `score_and_explain(text) -> (score, dict)`). Load it by dotted path:

```bash
python3 evals/run.py --clf baselines:Constant --split val --gran 100
python3 evals/run.py --clf baselines:MeanSentLen --gran 500 --mask-topic --show-errors 5   # trn
python3 evals/run.py --clf "baselines:Random(seed=3)" --split trn --gran sent --limit 2000
python3 evals/run_all.py --clf tsw.engine:Default  # {trn,val} x {sent,100,500} x {unmasked,masked}
python3 evals/run_all.py --clf tsw.engine:Default --splits trn  # trn only
python3 -m unittest discover -s evals/tests -v
```

- Metrics: accuracy, macro-F1, ROC-AUC, confusion matrix; with `explain`, a
  per-rule table (firing rate per class, mean contribution per class, smoothed
  log-odds of firing translated-vs-original, rule-alone AUC, sign accuracy).
- `--mask-topic` replaces country/nationality/city/party/politician tokens
  (`evals/topicmask.py`) with `XTOPIC` before scoring and also reports the
  unmasked numbers. A rule set whose AUC drops a lot under masking is keying on
  topic (see `baselines:TopicCheat`, which is a deliberate cheat).
- `--split` is `trn` (default; `dev` is an alias and is recorded as trn),
  `val` or `test`. `--split test` is refused without `--i-mean-it`;
  `--show-errors` is refused on val and test.
- Every run writes `results/<ts>_<clf>_<split>_<gran>[_all][_masked].json`
  (metrics, rule table, split, git commit/dirty/branch, worktree path, mask
  flag, data sha256) in the worktree it runs from, a copy in the shared
  `./results/all/` (gitignored;
  override with `$TSW_SHARED_RESULTS`), and one line in `results/LEADERBOARD.md`.
- Worktrees: `data/processed/` is gitignored, so a worktree without its own
  copy reads the main checkout's (or `$TSW_DATA`).
- Scoring the full sentence-level trn split (~88k sentences) with a real rule
  set can take minutes: use `srun -p berzelius-hopper-cpu --time=... -c 1`.
- Dashboard data: `evals/export_dashboard.py` (runs of every split, `dev` runs
  exported as trn, runs on the retired dev-train/dev-val halves marked
  `legacy`; rule tables; corpus counts) and `evals/export_examples.py`
  (annotated TP/TN/FP/FN examples, sampled from trn only).

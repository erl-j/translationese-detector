# dev-train / dev-val split (hybrid-ml branch only)

ML models need training data and the test split is locked, so the hybrid side
track cuts **dev** (sessions 1997-01-14 .. 2004-05-05) by session date:

- **dev-train** = dev records with session date `< 2002-03-01`
- **dev-val**   = dev records with session date `>= 2002-03-01`

Rule (`evals/devsplit.py`, `VAL_START = "2002-03-01"`): the 70% point of the dev
time span is 2002-02-24, rounded up to the next month start, so dev-val is the
last ~30% of the span (~33% of the balanced records). Every record carries one
session date and chunks never cross sessions, so all units of a session fall
on the same side (checked by `hybrid/split_stats.py`). The split is the same for
all three granularities and uses the balanced files (`dev.<gran>.bal.jsonl`), which is
what `evals/run.py` scores by default.

Usage: `python3 evals/run.py --split dev-val ...` (or `dev-train`). Results JSON
carry `split: "dev-val"` and `subset_rule`, so they never mix with full-dev numbers.
Models are fitted on dev-train only (hyper-parameters chosen by 5-fold CV inside
dev-train, folds grouped by session date) and reported on dev-val.

## Sizes (from `python3 hybrid/split_stats.py`)

| gran | subset | n | orig (0) | trans (1) | sessions | first date | last date |
|---|---|---:|---:|---:|---:|---|---|
| sent | dev-train | 58721 | 30881 | 27840 | 346 | 1997-01-14 | 2002-02-28 |
| sent | dev-val | 29183 | 13071 | 16112 | 137 | 2002-03-11 | 2004-05-05 |
| 100 | dev-train | 9560 | 5035 | 4525 | 342 | 1997-01-14 | 2002-02-28 |
| 100 | dev-val | 4610 | 2050 | 2560 | 136 | 2002-03-11 | 2004-05-05 |
| 500 | dev-train | 1823 | 975 | 848 | 312 | 1997-01-15 | 2002-02-28 |
| 500 | dev-val | 911 | 392 | 519 | 135 | 2002-03-11 | 2004-05-05 |

Speaker overlap (distinct speaker names in the `speakers` field):

| gran | class | dev-train speakers | dev-val speakers | in both | share of dev-val speakers also in dev-train |
|---|---|---:|---:|---:|---:|
| sent | orig | 114 | 75 | 38 | 51% |
| sent | trans | 311 | 209 | 147 | 70% |
| 100 | orig | 114 | 75 | 38 | 51% |
| 100 | trans | 267 | 186 | 134 | 72% |
| 500 | orig | 109 | 72 | 37 | 51% |
| 500 | trans | 234 | 153 | 120 | 78% |

Notes
- dev-val is not class-balanced (the `bal` files were balanced over all of dev):
  e.g. 500-token chunks are 392 orig / 519 trans. AUC is unaffected; accuracy is
  measured at the model's own threshold (score > 0).
- **Speaker overlap is large** (same parliamentary term, 1999-2004): half of the
  dev-val original speakers and ~3/4 of the translated speakers also speak in
  dev-train. A model can partly learn speakers. Every hybrid result therefore also
  reports `unseen_speaker_metrics`: AUC on the dev-val units none of whose speakers
  occur in dev-train (2,139 sentences and 256 100-token chunks; only 23 at 500
  tokens, too few to use). The test split (2004-2011) has much less overlap with dev.

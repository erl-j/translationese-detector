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

# Technical report

`translationese-swe-report.pdf` describes the rule engine, its calibration, the
Europarl evaluation and the findings. It is built from `report.qmd` (Quarto, LuaLaTeX).

```bash
make -C report      # = python build_assets.py && quarto render report.qmd
```

- `build_assets.py` writes everything in `assets/`: the data, feature, results, ablation and
  hybrid tables, the AUC figure, the appendix rule list and the annotated example boxes. It
  reads the numbers from `docs/INTEGRATION.md`, `hybrid/reports/*.md`, `data/MANIFEST.md`
  and the rule metadata in `tsw/rules/`, so the tables cannot drift from those sources.
- The examples are trn units whose texts are stored in `examples_trn.json` (re-extract
  with `build_assets.py --refresh-examples`, which needs `data/processed/`). Their
  highlights and contributions are recomputed with `Engine().annotate()` at every build;
  the build fails if an analyst note names a rule that no longer contributes.
- Requirements: Quarto with TinyTeX (`quarto install tinytex`; missing LaTeX packages are
  installed on the first render) and a Python with matplotlib (default
  `../.venv-hybrid/bin/python`; override with `make PYTHON=...`). No data is needed for a
  rebuild.

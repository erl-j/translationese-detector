# Build the technical report: assets (tables, figure, example boxes), then the PDF.
#   make -C report            # from the repo root
# Needs: quarto (with TinyTeX: `quarto install tinytex`) and a Python with matplotlib
# (default: the hybrid venv; override with `make PYTHON=...`).

PYTHON ?= ../.venv-hybrid/bin/python
PDF = translationese-swe-report.pdf

all: $(PDF)

assets:
	timeout 300 $(PYTHON) build_assets.py

$(PDF): assets report.qmd preamble.tex references.bib
	timeout 900 quarto render report.qmd

clean:
	rm -rf .quarto report_files report.tex *.log

.PHONY: all assets clean

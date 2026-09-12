#!/usr/bin/env bash
# Local build: pdflatex -> bibtex -> pdflatex x2. Mirrors the Overleaf toolchain.
set -e
export PATH="/c/Users/munif/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep -E "Output written|Warning: (Citation|Reference)" || true
echo "pages: $(grep -oP 'Output written on main\.pdf \(\K[0-9]+' main.log)"
echo "nonws chars: $(pdftotext main.pdf - 2>/dev/null | tr -d '[:space:]' | wc -c)"

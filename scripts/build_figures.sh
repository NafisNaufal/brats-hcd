#!/usr/bin/env bash
# Compile TikZ figure sources in paper/figures/ to PDFs.
# Run from repo root:  bash scripts/build_figures.sh
set -euo pipefail

FIGURES_DIR="paper/figures"
cd "$(dirname "$0")/.."

if ! command -v pdflatex &>/dev/null; then
  echo "ERROR: pdflatex not found. Install TeX Live: sudo apt-get install texlive-full"
  exit 1
fi

for tex in "$FIGURES_DIR"/*.tex; do
  name=$(basename "$tex" .tex)
  echo "Compiling $tex ..."
  (cd "$FIGURES_DIR" && pdflatex -interaction=nonstopmode "$name.tex" > /dev/null 2>&1)
  echo "  → $FIGURES_DIR/$name.pdf"
done

echo "Done. All figures compiled."

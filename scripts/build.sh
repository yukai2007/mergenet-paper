#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
mkdir -p build
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build mergenet-main.tex
elif command -v "${TECTONIC_BIN:-tectonic}" >/dev/null 2>&1; then
  "${TECTONIC_BIN:-tectonic}" --keep-logs --keep-intermediates --outdir build mergenet-main.tex
else
  echo 'Install TeX Live + latexmk or Tectonic, or compile mergenet-main.tex in Overleaf.' >&2
  exit 1
fi
cp build/mergenet-main.pdf mergenet-main.pdf
cmp -s build/mergenet-main.pdf mergenet-main.pdf
if rg -n 'LaTeX Error|Undefined control sequence|There were undefined references|Citation.*undefined|Overfull' build/mergenet-main.log; then
  echo 'Review and resolve the build-log findings above.' >&2
  exit 1
fi

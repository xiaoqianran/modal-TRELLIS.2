#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v codegraph >/dev/null; then
  echo "installing CodeGraph into ~/.local"
  npm config set prefix "${HOME}/.local"
  npm i -g @colbymchenry/codegraph
fi

for name in TRELLIS.2 fast-trellis2 meshii; do
  dir="$ROOT/vendor/$name"
  if [[ ! -d "$dir" ]]; then
    echo "missing $dir — run scripts/fetch-upstream.sh first"
    exit 1
  fi
  if [[ -d "$dir/.codegraph" ]]; then
    codegraph index "$dir"
  else
    codegraph init "$dir"
  fi
  codegraph status "$dir"
done

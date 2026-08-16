#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/vendor"
mkdir -p "$DEST"

clone() {
  local url="$1"
  local name="$2"
  if [[ -d "$DEST/$name/.git" ]]; then
    git -C "$DEST/$name" pull --ff-only
  else
    git clone --depth 1 "$url" "$DEST/$name"
  fi
}

clone https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
clone https://github.com/Archerkattri/fast-trellis2.git fast-trellis2
clone https://github.com/sciences44/meshii.git meshii

echo "upstream ready in $DEST"

#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv
fi

uv pip install -e ".[dev]"
.venv/bin/python -c "import modal_trellis2; print(modal_trellis2.__version__)"

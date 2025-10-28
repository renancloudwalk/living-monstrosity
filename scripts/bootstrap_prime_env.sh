#!/usr/bin/env bash
set -euo pipefail

# Bootstrap dependencies inside a Prime Intellect pod.
# Assumes the repo has already been cloned into the working directory.

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] Installing uv..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

echo "[bootstrap] Syncing project dependencies..."
UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu124} \
  uv sync --all-extras

echo "[bootstrap] Installing vf-function-caller environment (editable)..."
uv pip install -e environments/vf_function_caller

echo "[bootstrap] Done. Activate the venv with:"
echo "  source .venv/bin/activate"

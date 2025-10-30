#!/usr/bin/env bash
set -euo pipefail

# Bootstrap dependencies inside a Prime Intellect pod.
# Assumes the repo has already been cloned into the working directory.

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] Installing uv..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

DEFAULT_GPU_INDEX="https://download.pytorch.org/whl/cu124"
DEFAULT_CPU_INDEX="https://download.pytorch.org/whl/cpu"

if [[ -n "${UV_EXTRA_INDEX_URL:-}" ]]; then
  EXTRA_INDEX="$UV_EXTRA_INDEX_URL"
  FALLBACK_INDEX=""
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  EXTRA_INDEX="$DEFAULT_GPU_INDEX"
  FALLBACK_INDEX="$DEFAULT_CPU_INDEX"
else
  EXTRA_INDEX="$DEFAULT_CPU_INDEX"
  FALLBACK_INDEX=""
fi

INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"

echo "[bootstrap] Syncing project dependencies..."
sync_with_index() {
  UV_EXTRA_INDEX_URL="$1" \
  UV_INDEX_STRATEGY="$INDEX_STRATEGY" \
    uv sync --all-extras
}

if ! sync_with_index "$EXTRA_INDEX"; then
  if [[ -n "$FALLBACK_INDEX" ]]; then
    echo "[bootstrap] CUDA wheels unavailable, retrying with CPU index..." >&2
    sync_with_index "$FALLBACK_INDEX"
    EXTRA_INDEX="$FALLBACK_INDEX"
  else
    exit 1
  fi
fi

echo "[bootstrap] Installing vf-function-caller environment (editable)..."
uv pip install -e environments/vf_function_caller

echo "[bootstrap] Installing prime-rl (editable, vendored)..."
UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-$EXTRA_INDEX}" \
  uv pip install -e external/prime-rl

echo "[bootstrap] Done. Activate the venv with:"
echo "  source .venv/bin/activate"

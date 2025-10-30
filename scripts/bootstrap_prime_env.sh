#!/usr/bin/env bash
set -euo pipefail

# Bootstrap dependencies inside a Prime Intellect pod.
# Assumes the repo has already been cloned into the working directory.

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] Installing uv..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

REPO_ROOT="$(pwd)"
STUB_WHEEL_DIR="$REPO_ROOT/vendor/cuda_stubs"

if command -v apt-get >/dev/null 2>&1; then
  echo "[bootstrap] Ensuring CUDA runtime libraries via apt (best effort)..." >&2
  set +e
  apt-get update >/dev/null 2>&1
  apt-get install -y --no-install-recommends \
    cuda-runtime-12-8 \
    cuda-toolkit-12-8 \
    cuda-nvrtc-12-8 \
    cuda-nvtx-12-8 \
    cuda-nvjitlink-12-8 \
    cuda-cupti-12-8 \
    cuda-nvprof-12-8 \
    cuda-nvshmem-12-8 \
    libcublas-12-8 \
    libcufft-12-8 \
    libcurand-12-8 \
    libcusolver-12-8 \
    libcusparse-12-8 \
    libcufile-12-8 \
    libnccl2 \
    >/dev/null 2>&1
  APT_STATUS=$?
  set -e
  if [[ $APT_STATUS -ne 0 ]]; then
    echo "[bootstrap] Warning: CUDA apt packages could not be installed (continuing anyway)." >&2
  fi
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

FIND_LINKS_VALUE=""
if [[ -d "$STUB_WHEEL_DIR" ]]; then
  FIND_LINKS_VALUE="$STUB_WHEEL_DIR"
fi
if [[ -n "${UV_FIND_LINKS:-}" ]]; then
  if [[ -n "$FIND_LINKS_VALUE" ]]; then
    FIND_LINKS_VALUE+=" ${UV_FIND_LINKS}"
  else
    FIND_LINKS_VALUE="$UV_FIND_LINKS"
  fi
fi

echo "[bootstrap] Syncing project dependencies..."
sync_with_index() {
  UV_EXTRA_INDEX_URL="$1" \
  UV_INDEX_STRATEGY="$INDEX_STRATEGY" \
  UV_FIND_LINKS="$FIND_LINKS_VALUE" \
    uv sync --all-extras
}

if ! sync_with_index "$EXTRA_INDEX"; then
  if [[ -n "$FALLBACK_INDEX" ]]; then
    echo "[bootstrap] CUDA wheels unavailable, retrying with CPU index..." >&2
    rm -f uv.lock
    sync_with_index "$FALLBACK_INDEX"
    EXTRA_INDEX="$FALLBACK_INDEX"
  else
    exit 1
  fi
fi

echo "[bootstrap] Installing vf-function-caller environment (editable)..."
if [[ -n "$FIND_LINKS_VALUE" ]]; then
  uv pip install --find-links "$FIND_LINKS_VALUE" --index-strategy "$INDEX_STRATEGY" -e environments/vf_function_caller
else
  uv pip install --index-strategy "$INDEX_STRATEGY" -e environments/vf_function_caller
endif

echo "[bootstrap] Installing prime-rl (editable, vendored)..."
EXTRA_ARG=()
if [[ -n "${UV_EXTRA_INDEX_URL:-$EXTRA_INDEX}" ]]; then
  EXTRA_ARG+=(--extra-index-url "${UV_EXTRA_INDEX_URL:-$EXTRA_INDEX}")
fi
if [[ -n "$FIND_LINKS_VALUE" ]]; then
  EXTRA_ARG+=(--find-links "$FIND_LINKS_VALUE")
fi
uv pip install --index-strategy "$INDEX_STRATEGY" "${EXTRA_ARG[@]}" -e external/prime-rl

echo "[bootstrap] Done. Activate the venv with:"
echo "  source .venv/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

# One-shot bootstrap + rollout + training pipeline for Prime pods.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/renancloudwalk/living-monstrosity/main/scripts/prime_autorun.sh | bash
# Optional env vars (export before running or prefix the curl command):
#   LM_REPO   – Git repo URL (default: https://github.com/renancloudwalk/living-monstrosity.git)
#   LM_BRANCH – Git branch/tag to check out (default: main)
#   LM_ROOT   – Working directory for the repo (default: $HOME/living-monstrosity)
#   LM_PORT   – Port for the inference server (default: 8000)
#   LM_OUTPUT – Output directory for rollouts/checkpoints (default: outputs/manual-run)

REPO="${LM_REPO:-https://github.com/renancloudwalk/living-monstrosity.git}"
BRANCH="${LM_BRANCH:-main}"
ROOT="${LM_ROOT:-$HOME/living-monstrosity}"
PORT="${LM_PORT:-8000}"
OUTPUT_DIR="${LM_OUTPUT:-outputs/manual-run}"

echo "[autorun] repo: $REPO" >&2
echo "[autorun] branch: $BRANCH" >&2
echo "[autorun] root: $ROOT" >&2

PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[autorun] installing uv" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$ROOT" ]]; then
  echo "[autorun] cloning repository" >&2
  git clone --branch "$BRANCH" "$REPO" "$ROOT"
else
  echo "[autorun] updating repository" >&2
  git -C "$ROOT" fetch origin "$BRANCH"
  git -C "$ROOT" checkout "$BRANCH"
  git -C "$ROOT" pull --ff-only origin "$BRANCH"
fi

cd "$ROOT"

echo "[autorun] bootstrapping environment" >&2
./scripts/bootstrap_prime_env.sh

mkdir -p "$OUTPUT_DIR"

echo "[autorun] wiring NVIDIA Python wheels into library path (best effort)" >&2
NVIDIA_LIB_DIRS="$(
  uv run python - <<'PY' 2>/dev/null || true
import importlib
import pkgutil
from pathlib import Path

try:
    import nvidia  # type: ignore
except Exception:
    raise SystemExit

seen = set()
for module in pkgutil.iter_modules(nvidia.__path__):
    try:
        pkg = importlib.import_module(f"nvidia.{module.name}")
    except Exception:
        continue
    for candidate in ("lib", "lib64"):
        lib_dir = Path(pkg.__path__[0]) / candidate
        if lib_dir.is_dir():
            seen.add(str(lib_dir))

for path in sorted(seen):
    print(path)
PY
)"

if [[ -n "$NVIDIA_LIB_DIRS" ]]; then
  while IFS= read -r lib_dir; do
    if [[ -d "$lib_dir" ]]; then
      if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH}"
      else
        LD_LIBRARY_PATH="${lib_dir}"
      fi
      if [[ -n "${LIBRARY_PATH:-}" ]]; then
        LIBRARY_PATH="${lib_dir}:${LIBRARY_PATH}"
      else
        LIBRARY_PATH="${lib_dir}"
      fi
    fi
  done <<<"$NVIDIA_LIB_DIRS"
  export LD_LIBRARY_PATH
  export LIBRARY_PATH
  echo "[autorun] LD_LIBRARY_PATH updated: $LD_LIBRARY_PATH" >&2
else
  echo "[autorun] no NVIDIA Python wheels detected; continuing" >&2
fi

INFER_LOG="$OUTPUT_DIR/inference.log"
if pgrep -f "uv run inference" >/dev/null 2>&1; then
  echo "[autorun] vLLM inference already running" >&2
else
  echo "[autorun] starting vLLM inference (mirroring logs -> $INFER_LOG)" >&2
  VLLM_USE_V1=0 \
    nohup uv run inference @ configs/debug/rl/inference.toml \
      --server.port "$PORT" \
      > >(tee "$INFER_LOG") 2>&1 &
  SERVER_PID=$!
  sleep 3
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[autorun] inference process exited immediately; see logs above for details" >&2
    wait "$SERVER_PID" || true
    exit 1
  fi
  echo "[autorun] inference pid: $SERVER_PID" >&2
  echo "[autorun] waiting for inference server to accept connections" >&2
  for attempt in {1..30}; do
    if curl -sSf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "[autorun] inference server is live" >&2
      break
    fi
    sleep 2
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "[autorun] inference server died before becoming ready; see logs above" >&2
      wait "$SERVER_PID" || true
      exit 1
    fi
    if [[ $attempt -eq 30 ]]; then
      echo "[autorun] failed to reach inference server" >&2
      exit 1
    fi
  done
fi

echo "[autorun] generating rollouts" >&2
uv run orchestrator @ configs/debug/rl/orchestrator.toml \
  --run.output-dir "$OUTPUT_DIR" \
  --client.base-url "http://127.0.0.1:$PORT/v1"

echo "[autorun] training on collected rollouts" >&2
uv run trainer @ configs/debug/rl/train.toml \
  --run.output-dir "$OUTPUT_DIR"

echo "[autorun] pipeline complete" >&2
echo "[autorun] inference server still running on port $PORT (logs in $INFER_LOG)" >&2

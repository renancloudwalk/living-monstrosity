#!/usr/bin/env bash
set -euo pipefail

# Launch Prime-RL training with the vf-function-caller config.
# Usage:
#   ./scripts/run_prime_training.sh configs/prime/rl/train.toml
# Environment variables:
#   WANDB_API_KEY (optional) – enables W&B logging when set.
#   UV_EXTRA_INDEX_URL – override CUDA wheel index if needed.

CONFIG_PATH="${1:-configs/prime/rl/train.toml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[prime-train] Using config ${CONFIG_PATH}"
echo "[prime-train] Activating venv"
source .venv/bin/activate

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "$WANDB_API_KEY"
fi

echo "[prime-train] Starting trainer..."
UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu124} \
  uv run trainer @ "${CONFIG_PATH}" "$@"

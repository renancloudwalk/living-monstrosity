#!/usr/bin/env bash
set -euo pipefail

# Run a quick Verifiers evaluation against the vf-function-caller environment.
# Usage: ./scripts/run_local_eval.sh <model>
# Example: ./scripts/run_local_eval.sh gpt-4.1-mini

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model-id> [extra vf-eval args]" >&2
  exit 1
fi

MODEL="$1"
shift || true

uv run vf-eval vf-function-caller -m "${MODEL}" -n 3 --show-tool-calls "$@"

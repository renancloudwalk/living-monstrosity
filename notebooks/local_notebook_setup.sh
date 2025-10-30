#!/usr/bin/env bash
set -euo pipefail

# Spin up a local Jupyter notebook backed by the repo's uv-managed virtualenv.
# Usage: ./notebooks/local_notebook_setup.sh [--port 8888]

PORT=8888
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d .venv ]]; then
  echo "[notebook] Bootstrapping virtualenv via uv..." >&2
  uv sync --all-extras
fi

source .venv/bin/activate

python -m pip install --quiet jupyterlab ipykernel
python -m ipykernel install --user --name "living-monstrosity" --display-name "Living Monstrosity"

echo "[notebook] Launching JupyterLab on port ${PORT}" >&2
jupyter lab --no-browser --port "${PORT}" --NotebookApp.token='' --NotebookApp.password=''

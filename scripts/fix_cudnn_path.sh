#!/usr/bin/env bash
set -euo pipefail

# Fix script to locate nvidia-cudnn-cu12 libraries and set LD_LIBRARY_PATH
# Run this if you get "libcudnn.so.9: cannot open shared object file" error

echo "[fix] Searching for nvidia-cudnn-cu12 package libraries..." >&2

CUDNN_LIB_DIR=$(uv run python -c "
import importlib.metadata as md
from pathlib import Path
import sys

try:
    dist = md.distribution('nvidia-cudnn-cu12')
    if not dist.files:
        print('[fix] ERROR: nvidia-cudnn-cu12 has no files', file=sys.stderr)
        sys.exit(1)

    for file in dist.files:
        if file.name.startswith('libcudnn.so'):
            lib_path = Path(dist.locate_file(file)).parent.resolve()
            print(lib_path)
            sys.exit(0)

    print('[fix] ERROR: libcudnn.so not found in nvidia-cudnn-cu12 package', file=sys.stderr)
    sys.exit(1)
except md.PackageNotFoundError:
    print('[fix] ERROR: nvidia-cudnn-cu12 not installed', file=sys.stderr)
    sys.exit(1)
" 2>&1) || {
    echo "[fix] Failed to locate nvidia-cudnn-cu12. Installing it now..." >&2
    uv pip install --upgrade nvidia-cudnn-cu12
    CUDNN_LIB_DIR=$(uv run python -c "
import importlib.metadata as md
from pathlib import Path
dist = md.distribution('nvidia-cudnn-cu12')
for file in dist.files:
    if file.name.startswith('libcudnn.so'):
        print(Path(dist.locate_file(file)).parent.resolve())
        break
")
}

if [[ -z "$CUDNN_LIB_DIR" ]]; then
    echo "[fix] ERROR: Could not find cudnn library directory" >&2
    exit 1
fi

echo "[fix] Found cudnn libraries at: $CUDNN_LIB_DIR" >&2

# Export the library path
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="${CUDNN_LIB_DIR}${LIBRARY_PATH:+:$LIBRARY_PATH}"

echo "[fix] Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH" >&2
echo "[fix] Testing torch import..." >&2

if uv run python -c "import torch; print(f'torch {torch.__version__} loaded successfully')" 2>&1; then
    echo "[fix] Success! Torch can now load CUDA libraries." >&2
    echo "" >&2
    echo "[fix] To use these settings, run:" >&2
    echo "  export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"" >&2
    echo "  export LIBRARY_PATH=\"$LIBRARY_PATH\"" >&2
else
    echo "[fix] ERROR: Torch still cannot load. Check the error above." >&2
    exit 1
fi

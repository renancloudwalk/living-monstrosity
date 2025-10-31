#!/usr/bin/env bash
set -euo pipefail

# Fix script to locate nvidia-cudnn-cu12 libraries and set LD_LIBRARY_PATH
# Run this if you get "libcudnn.so.9: cannot open shared object file" error

echo "[fix] Searching for cudnn libraries in site-packages..." >&2

CUDNN_LIB_DIR=$(uv run python -c "
from pathlib import Path
import sys
import site

# Get all site-packages directories
site_packages = site.getsitepackages()
if hasattr(site, 'getusersitepackages'):
    site_packages.append(site.getusersitepackages())

for sp_dir in site_packages:
    sp_path = Path(sp_dir)
    if not sp_path.exists():
        continue

    # Check nvidia/cudnn/lib directory
    cudnn_lib = sp_path / 'nvidia' / 'cudnn' / 'lib'
    if cudnn_lib.exists() and any(cudnn_lib.glob('libcudnn.so*')):
        print(cudnn_lib.resolve())
        sys.exit(0)

    # Search for libcudnn.so* files anywhere
    for match in sp_path.glob('**/libcudnn.so*'):
        if match.is_file():
            print(match.parent.resolve())
            sys.exit(0)

print('[fix] ERROR: libcudnn.so not found in site-packages', file=sys.stderr)
sys.exit(1)
" 2>&1) || {
    echo "[fix] cudnn libraries not found. Trying to install nvidia-cudnn-cu12..." >&2
    if uv pip install --upgrade nvidia-cudnn-cu12; then
        CUDNN_LIB_DIR=$(uv run python -c "
from pathlib import Path
import site
for sp_dir in site.getsitepackages():
    sp_path = Path(sp_dir)
    cudnn_lib = sp_path / 'nvidia' / 'cudnn' / 'lib'
    if cudnn_lib.exists():
        print(cudnn_lib.resolve())
        break
")
    else
        echo "[fix] ERROR: Failed to install nvidia-cudnn-cu12" >&2
        exit 1
    fi
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

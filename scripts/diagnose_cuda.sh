#!/usr/bin/env bash
# Diagnostic script to find ALL CUDA libraries on the system

echo "=== CUDA Diagnostic ==="
echo ""

echo "1. Checking system paths for libcudnn:"
for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda*/lib64 /usr/lib64 /opt/cuda/lib64; do
  if [[ -d "$dir" ]]; then
    echo "  Checking: $dir"
    find "$dir" -name "libcudnn.so*" 2>/dev/null || echo "    (no libcudnn found)"
  fi
done

echo ""
echo "2. Checking LD_LIBRARY_PATH:"
echo "  $LD_LIBRARY_PATH"

echo ""
echo "3. Finding ALL libcudnn files on system:"
find /usr -name "libcudnn.so*" 2>/dev/null

echo ""
echo "4. Checking Python environment:"
if command -v python >/dev/null 2>&1; then
  python -c "import torch; print(f'Torch version: {torch.__version__}')" 2>&1 || echo "Torch import failed"
fi

echo ""
echo "5. CUDA toolkit locations:"
ls -la /usr/local/ | grep cuda

echo ""
echo "6. ldconfig cache:"
ldconfig -p 2>/dev/null | grep cudnn || echo "ldconfig not available or no cudnn in cache"

echo ""
echo "Done."

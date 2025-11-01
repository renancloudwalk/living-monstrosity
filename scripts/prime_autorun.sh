#!/usr/bin/env bash
set -euo pipefail

# One-shot bootstrap + rollout + training pipeline for Prime pods.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/renancloudwalk/living-monstrosity/main/scripts/prime_autorun.sh | bash
# Optional env vars (export before running or prefix the curl command):
#   LM_REPO        – Git repo URL (default: https://github.com/renancloudwalk/living-monstrosity.git)
#   LM_BRANCH      – Git branch/tag to check out (default: main)
#   LM_ROOT        – Working directory for the repo (default: $HOME/living-monstrosity)
#   LM_PORT        – Port for the inference server (default: 8000)
#   LM_OUTPUT      – Output directory for rollouts/checkpoints (default: outputs/manual-run)
#   WANDB_API_KEY  – Weights & Biases API key for tracking (optional, get from https://wandb.ai/authorize)

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

# Set up Weights & Biases if API key is provided
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[autorun] Setting up Weights & Biases..." >&2
  if uv run wandb login "$WANDB_API_KEY" >&2 2>&1; then
    echo "[autorun] ✓ W&B login successful" >&2
    WANDB_ENABLED=true
  else
    echo "[autorun] ✗ W&B login failed, continuing without wandb" >&2
    WANDB_ENABLED=false
  fi
else
  echo "[autorun] No WANDB_API_KEY set, skipping W&B setup" >&2
  echo "[autorun] To enable: export WANDB_API_KEY=your-key before running" >&2
  echo "[autorun] Get your key at: https://wandb.ai/authorize" >&2
  WANDB_ENABLED=false
fi

find_cuda_lib_dirs() {
  uv run python - <<'PY'
from pathlib import Path
import sys
import site

lib_dirs = set()

# Get all site-packages directories
site_packages = site.getsitepackages()
if hasattr(site, 'getusersitepackages'):
    site_packages.append(site.getusersitepackages())

cuda_prefixes = (
    "libcudnn",
    "libcublas",
    "libcublasLt",
    "libcuda",
    "libnvrtc",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libcufft",
)

for sp_dir in site_packages:
    sp_path = Path(sp_dir)
    if not sp_path.exists():
        continue

    # Search nvidia/*/lib directories (e.g., nvidia/cudnn/lib, nvidia/cublas/lib)
    for lib_dir in sp_path.glob("nvidia/*/lib"):
        if lib_dir.is_dir() and any(lib_dir.glob("*.so*")):
            lib_dirs.add(str(lib_dir.resolve()))

    # Search torch/lib directory
    torch_lib = sp_path / "torch" / "lib"
    if torch_lib.exists() and any(torch_lib.glob("*.so*")):
        lib_dirs.add(str(torch_lib.resolve()))

    # Search torchvision.libs
    torch_vision_libs = sp_path / "torchvision.libs"
    if torch_vision_libs.exists() and any(torch_vision_libs.glob("*.so*")):
        lib_dirs.add(str(torch_vision_libs.resolve()))

    # Direct search for CUDA .so files
    for prefix in cuda_prefixes:
        for match in sp_path.glob(f"**/{prefix}.so*"):
            lib_dirs.add(str(match.parent.resolve()))

# Check vendor stubs
vendor_stubs = Path("vendor/cuda_stubs")
if vendor_stubs.exists():
    lib_dirs.add(str(vendor_stubs.resolve()))

for path in sorted(lib_dirs):
    print(path)
PY
}

wire_cuda_libs() {
  local dirs="$1"
  while IFS= read -r lib_dir; do
    [[ -z "$lib_dir" ]] && continue
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
  done <<<"$dirs"
  export LD_LIBRARY_PATH
  export LIBRARY_PATH
}

verify_torch_loads() {
  uv run python -c "import torch; print(f'torch {torch.__version__}')" 2>&1
}

echo "[autorun] wiring CUDA libraries into library path" >&2

# First, check system CUDA paths (common in Docker images with CUDA)
SYSTEM_LIB_DIRS=""
for sys_lib in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda-12/lib64 /opt/cuda/lib64; do
  if [[ -d "$sys_lib" ]] && ls "$sys_lib"/libcudnn.so* >/dev/null 2>&1; then
    SYSTEM_LIB_DIRS="${SYSTEM_LIB_DIRS}${sys_lib}"$'\n'
  fi
done

if [[ -n "$SYSTEM_LIB_DIRS" ]]; then
  echo "[autorun] Found system CUDA libraries:" >&2
  echo "$SYSTEM_LIB_DIRS" | while IFS= read -r dir; do
    [[ -z "$dir" ]] && continue
    echo "  - $dir" >&2
  done
  wire_cuda_libs "$SYSTEM_LIB_DIRS"
  if verify_torch_loads >/dev/null 2>&1; then
    echo "[autorun] ✓ Torch loads successfully with system CUDA" >&2
    NVIDIA_LIB_DIRS="$SYSTEM_LIB_DIRS"
  else
    echo "[autorun] ✗ System CUDA found but torch cannot load" >&2
    NVIDIA_LIB_DIRS=""
  fi
else
  echo "[autorun] No system CUDA libraries found, checking Python environment..." >&2
  NVIDIA_LIB_DIRS="$(find_cuda_lib_dirs)"

  if [[ -n "$NVIDIA_LIB_DIRS" ]]; then
    echo "[autorun] Candidate CUDA library directories:" >&2
    echo "$NVIDIA_LIB_DIRS" | while IFS= read -r dir; do
      echo "  - $dir" >&2
    done

    # Check if libcudnn.so.9 actually exists in any of the directories
    FOUND_CUDNN=0
    while IFS= read -r dir; do
      if [[ -f "$dir/libcudnn.so.9" ]] || ls "$dir"/libcudnn.so.* >/dev/null 2>&1; then
        FOUND_CUDNN=1
        break
      fi
    done <<<"$NVIDIA_LIB_DIRS"

    if [[ $FOUND_CUDNN -eq 0 ]]; then
      echo "[autorun] ✗ None of the directories contain libcudnn.so.9" >&2
      NVIDIA_LIB_DIRS=""
    else
      # Wire up the paths and verify torch loads
      wire_cuda_libs "$NVIDIA_LIB_DIRS"
      echo "[autorun] Testing if torch can load with these paths..." >&2
      if ! verify_torch_loads >/dev/null 2>&1; then
        echo "[autorun] ✗ Torch still cannot load CUDA libraries" >&2
        NVIDIA_LIB_DIRS=""
      else
        echo "[autorun] ✓ Torch loads successfully" >&2
      fi
    fi
  fi
fi

# If no libs found or verification failed, try multiple recovery strategies
if [[ -z "$NVIDIA_LIB_DIRS" ]]; then
  echo "[autorun] No CUDA libraries found in Python environment" >&2

  # Strategy 1: Try reinstalling nvidia packages
  echo "[autorun] Strategy 1: Reinstalling nvidia-cudnn-cu12..." >&2
  if uv pip install --force-reinstall --no-deps nvidia-cudnn-cu12 nvidia-cublas-cu12 >&2 2>/dev/null; then
    NVIDIA_LIB_DIRS="$(find_cuda_lib_dirs)"
    if [[ -n "$NVIDIA_LIB_DIRS" ]]; then
      # Check for libcudnn.so.9
      FOUND_CUDNN=0
      while IFS= read -r dir; do
        if [[ -f "$dir/libcudnn.so.9" ]] || ls "$dir"/libcudnn.so.* >/dev/null 2>&1; then
          FOUND_CUDNN=1
          break
        fi
      done <<<"$NVIDIA_LIB_DIRS"

      if [[ $FOUND_CUDNN -eq 1 ]]; then
        wire_cuda_libs "$NVIDIA_LIB_DIRS"
        if verify_torch_loads >/dev/null 2>&1; then
          echo "[autorun] ✓ Found and verified libraries after reinstall" >&2
        else
          NVIDIA_LIB_DIRS=""
        fi
      else
        NVIDIA_LIB_DIRS=""
      fi
    fi
  fi

  # Strategy 2: Install system CUDA libraries if we have root
  if [[ -z "$NVIDIA_LIB_DIRS" ]] && [[ "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    echo "[autorun] Strategy 2: Installing system CUDA libraries via apt..." >&2
    export DEBIAN_FRONTEND=noninteractive
    if apt-get update >&2 2>/dev/null && \
       apt-get install -y --no-install-recommends libcudnn9 libcudnn9-cuda-12 >&2 2>/dev/null; then
      # Add system lib paths
      for sys_lib in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda-12/lib64; do
        if [[ -d "$sys_lib" ]] && ls "$sys_lib"/libcudnn.so* >/dev/null 2>&1; then
          echo "$sys_lib" >> /tmp/cuda_libs.txt
        fi
      done
      if [[ -f /tmp/cuda_libs.txt ]]; then
        NVIDIA_LIB_DIRS=$(cat /tmp/cuda_libs.txt)
        rm /tmp/cuda_libs.txt
        wire_cuda_libs "$NVIDIA_LIB_DIRS"
        if verify_torch_loads >/dev/null 2>&1; then
          echo "[autorun] ✓ Found and verified system CUDA libraries" >&2
        else
          NVIDIA_LIB_DIRS=""
        fi
      fi
    fi
  fi

  # Strategy 3: Try installing torch with CUDA bundles
  if [[ -z "$NVIDIA_LIB_DIRS" ]]; then
    echo "[autorun] Strategy 3: Installing torch with bundled CUDA..." >&2
    if uv pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu121 'torch>=2.2.0' >&2 2>/dev/null; then
      NVIDIA_LIB_DIRS="$(find_cuda_lib_dirs)"
      if [[ -n "$NVIDIA_LIB_DIRS" ]]; then
        # Check for libcudnn.so.9
        FOUND_CUDNN=0
        while IFS= read -r dir; do
          if [[ -f "$dir/libcudnn.so.9" ]] || ls "$dir"/libcudnn.so.* >/dev/null 2>&1; then
            FOUND_CUDNN=1
            break
          fi
        done <<<"$NVIDIA_LIB_DIRS"

        if [[ $FOUND_CUDNN -eq 1 ]]; then
          wire_cuda_libs "$NVIDIA_LIB_DIRS"
          if verify_torch_loads >/dev/null 2>&1; then
            echo "[autorun] ✓ Found and verified libraries in torch package" >&2
          else
            NVIDIA_LIB_DIRS=""
          fi
        else
          NVIDIA_LIB_DIRS=""
        fi
      fi
    fi
  fi
fi

if [[ -n "$NVIDIA_LIB_DIRS" ]]; then
  echo "[autorun] ✓ LD_LIBRARY_PATH configured: $LD_LIBRARY_PATH" >&2
  echo "[autorun] ✓ Torch CUDA verification passed" >&2
else
  echo "[autorun] ✗ FATAL: Could not find or install CUDA libraries" >&2
  echo "[autorun] Attempted:" >&2
  echo "[autorun]   1. Reinstall nvidia-cudnn-cu12 from pip" >&2
  echo "[autorun]   2. Install system libcudnn9 via apt (requires root)" >&2
  echo "[autorun]   3. Reinstall torch with CUDA bundles" >&2
  echo "[autorun]" >&2
  echo "[autorun] Manual fix: Install CUDA libraries on your system, then re-run." >&2
  exit 1
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
if [[ "$WANDB_ENABLED" == "true" ]]; then
  echo "[autorun] Training with W&B enabled" >&2
  uv run trainer @ configs/debug/rl/train.toml \
    --run.output-dir "$OUTPUT_DIR" \
    --logging.wandb-enabled true \
    --logging.wandb-project "vf-function-caller" \
    --logging.wandb-run-name "autorun-$(date +%Y%m%d-%H%M%S)"
else
  echo "[autorun] Training without W&B" >&2
  uv run trainer @ configs/debug/rl/train.toml \
    --run.output-dir "$OUTPUT_DIR"
fi

echo "[autorun] pipeline complete" >&2
echo "[autorun] inference server still running on port $PORT (logs in $INFER_LOG)" >&2

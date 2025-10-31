# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Living Monstrosity is a reinforcement learning playground for training open-source LLMs (primarily Qwen-based models) to call real tools using Prime-RL and Verifiers. The core focus is teaching models to safely make HTTP requests via a constrained `http_fetch` tool, with the long-term goal of scaling from local workstations to massive Prime Intellect GPU clusters.

## Core Architecture

### Verifiers Environment System
- **`environments/vf_function_caller/`** - Verifiers ToolEnv package that defines the multi-turn chat environment, tool schemas, scoring rubrics, and sandboxes
- Tool execution flow: model requests tool → tool runs in sandbox → result fed back to model
- Each environment exposes a `load_environment()` function via entry points
- Environment packages use naming pattern `vf_<task>` and install as `vf-<task>`
- Constrained allowlist controls which domains the `http_fetch` tool can access (example.com, httpbin.org, primeintellect.ai)
- Includes mock search index for web search simulation

### Prime-RL Integration
- Lives in `external/prime-rl/` as a git submodule
- Async GRPO/PPO trainer supporting FSDP2 + vLLM
- Orchestrator generates rollouts using vLLM inference backend
- Trainer consumes rollout data for policy updates
- Scales from 1 GPU locally to 1000+ GPUs on Prime pods

### Configuration System
- **`configs/debug/rl/`** - Local development configs (small models, short runs)
  - `train.toml` - GRPO trainer with Qwen2.5-0.5B, 50 steps
  - `orchestrator.toml` - Rollout generation settings
  - `inference.toml` - vLLM server config
- **`configs/prime/rl/`** - Production Prime pod configs
  - `train_qwen_1p8b.toml` - 2×GPU Qwen2.5-1.8B (POC mode)
  - `train_gpt_oss.toml` - 16×GPU openai/gpt-oss-20b (scaling)
  - `train.toml` - Default Prime config

## Development Commands

### Local Development
```bash
# Install dependencies (uses uv package manager)
uv sync --all-extras

# Install the vf-function-caller environment package
uv pip install -e environments/vf_function_caller

# Quick evaluation with any OpenAI-compatible model
./scripts/run_local_eval.sh gpt-4.1-mini

# Run minimal RL training loop locally (debug config)
uv run trainer @ configs/debug/rl/train.toml

# Launch Jupyter notebook for interactive experiments
./notebooks/local_notebook_setup.sh --port 8888
```

### Prime Pod Deployment

#### Automated End-to-End Setup
```bash
# Single command to bootstrap, serve, generate rollouts, and train
curl -fsSL https://raw.githubusercontent.com/renancloudwalk/living-monstrosity/main/scripts/prime_autorun.sh | bash

# Override defaults with environment variables
LM_REPO=your-fork LM_BRANCH=feature-branch LM_PORT=8001 LM_OUTPUT=outputs/custom bash prime_autorun.sh
```

#### Manual Step-by-Step (Makefile)
```bash
# 1. Bootstrap pod dependencies (installs uv, CUDA libs, Python deps)
make prime-bootstrap

# 2. Start vLLM inference server (keep running in separate shell)
make serve-qwen HOST=0.0.0.0 PORT=8000 TP=4 DP=1 DTYPE=float16

# 3. Generate tool-using rollouts
make orchestrate OUTPUT_DIR=outputs/manual-run CLIENT_BASE_URL=http://127.0.0.1:8000/v1 ORCH_MAX_STEPS=200

# 4. Train on collected rollouts
make train OUTPUT_DIR=outputs/manual-run TRAIN_MAX_STEPS=200 TRAIN_LR=3e-5

# 5. Inspect rollout data for debugging
make inspect-rollout OUTPUT_DIR=outputs/manual-run INSPECT_STEP=step_00000123
```

### Testing and Linting
```bash
# Run tests for environment packages
uv run pytest environments/vf_function_caller/ -v

# Verifiers evaluation (5 examples with gpt-4.1-mini)
uv run vf-eval vf-function-caller -m gpt-4.1-mini -n 5
```

### Key Makefile Variables
- `MODEL` - Model to use (default: Qwen/Qwen2.5-0.5B)
- `OUTPUT_DIR` - Output directory for rollouts/checkpoints
- `TP`/`DP` - Tensor/data parallelism for vLLM
- `ORCH_MAX_STEPS` - Orchestrator training steps
- `TRAIN_MAX_STEPS` - Trainer optimization steps
- `TRAIN_IMPL` - Model implementation (hf, liger_kernel)
- `ENVIRONMENT_ID` - Verifiers environment to use (vf-function-caller)

## Important Patterns

### Environment Package Structure
Each Verifiers environment follows this pattern:
```
environments/vf_<name>/
├── __init__.py          # Exposes load_environment()
├── vf_<name>.py         # Core ToolEnv implementation
├── pyproject.toml       # Package metadata with entry points
├── README.md            # Environment documentation
└── data/
    └── examples.jsonl   # Dataset for evaluation/training
```

### Tool Sandbox Constraints
- `http_fetch` tool enforces allowlist of domains (ALLOWED_NETLOCS)
- Responses truncated to MAX_RESPONSE_BYTES (1024 bytes)
- Request timeout DEFAULT_TIMEOUT (8 seconds)
- Only http/https schemes permitted
- Tool schemas must be JSON-serializable for model consumption

### Prime-RL Async Training Flow
1. **Orchestrator** calls vLLM `/v1/chat/completions` to generate model responses
2. Environment executes tools and scores outcomes
3. Rollout data saved as `rank_N.pt` files under `outputs/*/rollouts/step_*`
4. **Trainer** loads rollouts, computes GRPO/PPO updates
5. Checkpoints saved to `outputs/*/checkpoints/step_*`
6. Async level determines pipeline parallelism (2 = 2 batches in flight)

### Rollout Inspection Format
Rollout .pt files contain:
- `messages` or `dialogue` - List of conversation turns
- Each message has `role`, `content`, and optionally `tool`, `input`, `output` fields
- Use `make inspect-rollout` to pretty-print tool calls and results

## Code Style

- Python: Black-compatible, 4-space indentation, type hints (PEP 484)
- File naming: `snake_case.py` for modules, `UPPERCASE.md` for docs, `kebab-case.sh` for scripts
- Environment packages must expose `load_environment()` at package root
- Tool functions should include detailed docstrings (models use these for context)
- Commit messages: conventional commit style (`feat:`, `fix:`, `docs:`)

## Repository Layout
```
.
├── AGENTS.md              # Contribution and development guidelines
├── README.md              # Vision, roadmap, quick start
├── Makefile               # Prime pod deployment shortcuts
├── pyproject.toml         # Root package dependencies
├── uv.lock                # Locked dependency versions
├── configs/               # Training/eval configurations
│   ├── debug/rl/         # Local dev configs (small, fast)
│   └── prime/rl/         # Production Prime pod configs
├── environments/          # Verifiers environment packages
│   └── vf_function_caller/  # HTTP tool-calling environment
├── external/              # Git submodules
│   └── prime-rl/         # Prime-RL trainer/orchestrator
├── scripts/               # Helper shell scripts
│   ├── bootstrap_prime_env.sh    # Pod dependency setup
│   ├── prime_autorun.sh          # End-to-end automation
│   ├── run_local_eval.sh         # Quick Verifiers eval
│   └── run_prime_training.sh     # Training launcher
└── notebooks/             # Jupyter analysis notebooks
```

## Critical Notes

- **PYTHONPATH must include `external/prime-rl/src`** when running orchestrator/trainer directly (Makefile handles this)
- **VLLM_USE_V1=0** is required for compatibility with current Prime pod builds
- Prime-RL uses `.toml` configs with `@` syntax: `uv run trainer @ path/to/config.toml`
- Rollouts directory structure is critical: `outputs/*/rollouts/step_*/rank_*.pt`
- vLLM server must stay running in a separate shell during orchestration
- Environment packages must be installed (`uv pip install -e environments/vf_function_caller`) before use
- Bootstrap script attempts CUDA library installation but continues on failure (Prime pods usually pre-configured)

## Prime Pod Best Practices

- Start with debug configs locally before scaling to Prime
- Use `train_qwen_1p8b.toml` (2 GPUs) for proof-of-concept runs
- Scale to `train_gpt_oss.toml` (16 GPUs) once pipeline is validated
- Keep inference server alive between orchestration steps (restarting is slow)
- Override UV_EXTRA_INDEX_URL if Prime pod uses custom CUDA wheel index
- Monitor via Weights & Biases (set `wandb_enabled = true` in config)
- Checkpoint frequently (`save_interval` in trainer config)

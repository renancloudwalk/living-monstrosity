# Living Monstrosity — Tool-Calling RL Playground

## Quick Start (Prime Pod)

```bash
curl -fsSL https://raw.githubusercontent.com/renancloudwalk/living-monstrosity/main/scripts/prime_autorun.sh | bash
```

That one liner clones the repo on your Prime pod, installs dependencies, launches vLLM, generates rollouts, runs the GRPO trainer, and leaves inference running.

**Optional: Track with Weights & Biases**
```bash
export WANDB_API_KEY="your-key-here"  # Get from https://wandb.ai/authorize
curl -fsSL https://raw.githubusercontent.com/renancloudwalk/living-monstrosity/main/scripts/prime_autorun.sh | bash
```

Override repo/branch/port/output via `LM_REPO`, `LM_BRANCH`, `LM_PORT`, or `LM_OUTPUT` if you need custom settings.

## Why this exists
I want a reliable way to teach open-source LLMs (likely Qwen-based) to call real tools—starting with `curl`—and eventually scale that training from a single workstation to massive Prime Intellect clusters. The long-term goal is industrial-strength reinforcement learning for tool-using agents; this repo is the scratchpad, docs, and scripts to make that happen.

## Core stack
- **Verifiers (`ToolEnv`)** – used to define the multi-turn chat environment, tool schemas, scoring rubrics, and optional sandboxes. This handles “model asks to call tool → tool runs → result is fed back” natively.
- **Prime-RL** – async GRPO/PPO trainer that already speaks Verifiers environments, supports FSDP2 + vLLM, and scales from 1 GPU to 1000+. It will run locally first, then on Prime pods.
- **Qwen family models** – initial policy/reference models; easy to swap once the plumbing is solid.
- **`curl` tool wrapper** – starts tiny: a constrained HTTP client that the model can invoke safely; later it can fan out to more capable sandboxes.

## Immediate plan of attack
1. **Bootstrap the environment package**
   - Install the package locally with `uv pip install -e environments/vf_function_caller`.
   - `ToolEnv` already exposes a constrained `http_fetch` tool; extend its allowlist or rubric as new tasks demand.
   - Keep the dataset in `data/examples.jsonl` in sync with the curriculum you care about.

2. **Prototype locally**
   - Install Verifiers + Prime-RL with `uv` (CPU-only evals are fine even if training is not).
   - Run `./scripts/run_local_eval.sh gpt-4.1-mini` (or any OpenAI-compatible model) to sanity-check tool usage.
   - Fire up a notebook dev shell with `./notebooks/local_notebook_setup.sh --port 8888` if you prefer interactive experiments. Credentials are disabled for convenience on localhost—forward a tunnel or set a token if exposing elsewhere.
   - Iterate on prompts, parser, and rewards until success rate is measurable.

3. **Scale out on Prime Intellect**
   - `prime pods create ...` to spin up the GPU cluster you need.
   - `make prime-bootstrap` (or `./scripts/bootstrap_prime_env.sh`) inside the pod to sync dependencies.
   - Use the new Makefile shortcuts to drive the full loop (see **Prime end-to-end loop** below) or fall back to the shell scripts in `scripts/` if you prefer.
   - POC mode? Use `configs/prime/rl/train_qwen_1p8b.toml` (2×GPU Qwen2.5-1.8B).
   - Going bigger? `configs/prime/rl/train_gpt_oss.toml` targets `openai/gpt-oss-20b` with 16 GPUs.
   - Track metrics through Weights & Biases (or the Prime dashboards) and checkpoint frequently.

## Prime end-to-end loop
A single command can wire everything up on a fresh Prime pod:

```bash
curl -fsSL https://raw.githubusercontent.com/renancloudwalk/living-monstrosity/main/scripts/prime_autorun.sh | bash
```

That script installs dependencies, launches the vLLM endpoint, generates rollouts, runs the GRPO trainer, and leaves inference running for follow-up evaluation. Override repo/branch/port/output paths via `LM_REPO`, `LM_BRANCH`, `LM_PORT`, or `LM_OUTPUT` environment variables before running if you need custom settings.

A single Makefile now wires up the minimal orchestration needed to watch Qwen call the real `http_fetch` tool. Run each step from separate shells on your Prime pod (keep the inference server alive in its own session):

1. **Bootstrap the pod**
   ```bash
   make prime-bootstrap
   ```
   Override `UV_EXTRA_INDEX_URL` if you need a different CUDA wheel index.

2. **Serve Qwen through vLLM**
   ```bash
   make serve-qwen HOST=0.0.0.0 PORT=8000 TP=4 DP=1 DTYPE=float16
   ```
   - Defaults match a single-GPU run; bump `TP`/`DP` to match the pod layout.
   - `VLLM_USE_V1=0` is set automatically for compatibility with current Prime builds.

3. **Generate tool-using rollouts**
   ```bash
   make orchestrate OUTPUT_DIR=outputs/manual-run CLIENT_BASE_URL=http://127.0.0.1:8000/v1 ORCH_MAX_STEPS=200
   ```
   - Customize `ORCH_BATCH_SIZE`, `ORCH_ROLLOUTS`, or `ENVIRONMENT_ARGS='{"seed": 7}'` as needed.
   - Rollouts land under `$(OUTPUT_DIR)/rollouts/step_*`.

4. **Train on the collected data**
   ```bash
   make train OUTPUT_DIR=outputs/manual-run TRAIN_MAX_STEPS=200 TRAIN_LR=3e-5
   ```
   - Switch `TRAIN_IMPL=liger_kernel` or adjust optimizer knobs to mirror production configs.
   - All Prime-RL flags from the earlier manual command are exposed as Make variables.

5. **Inspect tool calls**
   ```bash
   make inspect-rollout OUTPUT_DIR=outputs/manual-run INSPECT_STEP=step_00000123
   ```
   - Omitting `INSPECT_STEP` shows the newest rollout automatically.
   - Use `RANK=1` to peek at other ranks in multi-GPU jobs.

These targets keep everything in one place and map 1:1 onto the raw CLI invocations described in the Prime-RL docs. Feel free to layer them into `prime pods exec …` workflows or adapt them for CI.

## Repository structure
```
.
├── AGENTS.md
├── README.md
├── configs/               # Prime-RL training/eval configs and notes
├── environments/          # Verifiers packages (vf-function-caller)
└── scripts/               # Helper scripts (vf-eval wrappers, etc.)
```
Everything new should live inside that layout—no stray files at repo root.

## Contributing (a.k.a. future me, pay attention)
1. Create or update files.
2. Run lint/tests if the change touches code.
3. `git add`, commit with a descriptive message, and push. I promised myself “always commit and push things,” so no local-only hacks.

## Next notes for myself
- Add coverage for negative/errored HTTP cases to the dataset to stress tool robustness.
- Tune the Prime config in `configs/prime/rl/train.toml` based on first cluster runs (start with the 4×GPU baseline).
- Stress-test the GPT-OSS config (`configs/prime/rl/train_gpt_oss.toml`) and record the required pod specs.
- Keep `docs/prime-playbook.md` updated as the deployment process evolves.

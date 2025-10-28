# Living Monstrosity — Tool-Calling RL Playground

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
   - Iterate on prompts, parser, and rewards until success rate is measurable.

3. **Scale out on Prime Intellect**
   - `prime pods create ...` to spin up the GPU cluster you need.
   - `./scripts/bootstrap_prime_env.sh` on the pod to sync dependencies.
   - `uv run trainer @ configs/prime/rl/train.toml` to launch GRPO at scale. The full pod workflow lives in `docs/prime-playbook.md`.
   - Track metrics through Weights & Biases (or the Prime dashboards) and checkpoint frequently.

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
- Expand the dataset beyond the three starter examples.
- Tune the Prime config in `configs/prime/rl/train.toml` based on first cluster runs.
- Keep `docs/prime-playbook.md` updated as the deployment process evolves.

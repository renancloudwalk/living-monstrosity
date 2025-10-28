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
   - Use `prime env init vf-function-caller` (or `vf-init`) to scaffold a Verifiers environment.
   - Implement `ToolEnv` with a `curl`-like Python function (restrict domains, redact headers, sanitize inputs).
   - Add a rubric that rewards correct final answers and penalizes useless tool spam.

2. **Prototype locally**
   - Install Verifiers + Prime-RL with `uv`.
   - Run the debug RL config against a small Qwen checkpoint (`configs/debug/rl/train.toml`) until the tool loop behaves.
   - Iterate on prompts, parser, and rewards until success rate is measurable.

3. **Scale out**
   - Containerize the working setup.
   - Launch Prime pods with the same environment package and dial up node/GPU counts using Prime-RL’s multi-node flags.
   - Track metrics through Weights & Biases (or the Prime dashboards) and checkpoint frequently.

## Repository structure (incoming)
```
.
├── environments/          # Verifiers packages live here (to be added)
├── configs/               # Prime-RL training/eval configs
├── scripts/               # Helper scripts (setup, rollout, analysis)
└── README.md              # You are here
```
Nothing besides this README is checked in yet—every new piece of work should show up under the layout above.

## Contributing (a.k.a. future me, pay attention)
1. Create or update files.
2. Run lint/tests if the change touches code.
3. `git add`, commit with a descriptive message, and push. I promised myself “always commit and push things,” so no local-only hacks.

## Next notes for myself
- Write the actual tool wrapper with tight security controls.
- Decide on dataset format (Parquet vs. on-the-fly generation) and hook it into Prime-RL.
- Document the exact Prime pod recipes once the first successful run lands.


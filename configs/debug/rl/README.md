# Debug RL Configs

This directory collects human-editable Prime-RL `.toml` configs tailored for the `vf-function-caller` environment. The included `train.toml` is a starting point for local GRPO experiments; update the following before running large jobs:

- `policy.model_id` / `reference.model_id` — pick checkpoints that exist in your environment (Qwen works well).
- `rollout_batch_size` / `update_batch_size` — scale them to leverage your GPU count.
- Logging settings — enable Weights & Biases or Prime telemetry once you move beyond quick smoke tests.

Run with:

```bash
uv run trainer @ configs/debug/rl/train.toml
```

Prime clusters can use the same file once you add the appropriate `launcher` / `cluster` blocks.

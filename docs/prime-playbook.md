# Prime Deployment Playbook

## 1. Prep locally
- Install the Prime CLI once: `uv tool install prime` then `prime login`.
- Push the latest repository state to GitHub (Prime pods will pull from there).
- Pick a config:
  - `configs/prime/rl/train.toml` — 4×GPU baseline with Qwen2.5-3B.
  - `configs/prime/rl/train_gpt_oss.toml` — 16×GPU recipe for `openai/gpt-oss-20b` (requires vLLM tensor parallelism).

## 2. Provision a pod
```bash
prime pods create gpux8-a100 \
  --image primeintellect/verifiers:latest \
  --name vf-function-caller-dev \
  --start
prime pods ssh vf-function-caller-dev
```
Pick the machine type and image that matches your needs (e.g., `gpux8-h100`, custom Docker image, etc.). Add `--team <slug>` if you’re using a shared account.

## 3. Bootstrap dependencies inside the pod
```bash
git clone git@github.com:renancloudwalk/living-monstrosity.git
cd living-monstrosity
./scripts/bootstrap_prime_env.sh
source .venv/bin/activate
```
The bootstrap script installs `uv`, syncs dependencies (including PyTorch ≥2.8), and attaches the `vf-function-caller` package in editable mode.

## 4. Run the training job
```bash
./scripts/run_prime_training.sh configs/prime/rl/train.toml
# or for GPT-OSS
./scripts/run_prime_training.sh configs/prime/rl/train_gpt_oss.toml
```
The wrapper ensures the virtualenv is active, sets the CUDA wheel index (`UV_EXTRA_INDEX_URL` defaults to `cu124`), and reuses `WANDB_API_KEY` if present. Append extra trainer flags after the config path (e.g., `--trainer.total_epochs=5`).

## 5. Monitor & iterate
- Check `artifacts/vf-function-caller-prime/` for checkpoints, logs, and metrics dumps.
- Use `prime pods logs <name>` from your local machine for real-time streaming.
- Adjust `rollout_batch_size`, `kl_coefficient`, or dataset contents between runs; commit changes so pods can pull them.

## 6. Post-run verification
- Load a checkpoint locally: `uv run vf-eval vf-function-caller -m path/to/checkpoint --show-tool-calls`.
- Inspect W&B curves (reward, KL, tool usage) to ensure the policy is learning rather than collapsing.
- Spot-check raw rollouts in `artifacts/.../rollouts/` to confirm the model respects the no-tool prompt (`42`).

## 7. Tear down
When finished, shut down resources to avoid extra cost:
```bash
prime pods delete vf-function-caller-dev
```
Capture key metrics and link the run in issues/PRs so future iterations have a history trail.

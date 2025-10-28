# Prime Deployment Playbook

## 1. Prep locally
- Install the Prime CLI once: `uv tool install prime` then `prime login`.
- Push the latest repository state to GitHub (Prime pods will pull from there).
- Note the config you intend to run (`configs/prime/rl/train.toml`) and update it with your model IDs, batch sizes, and launcher parameters.

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
uv run trainer @ configs/prime/rl/train.toml
```
Tune `UV_EXTRA_INDEX_URL` if you need to point at a different CUDA wheel index (e.g., `cu121`). To stream logs to Weights & Biases, ensure `wandb_enabled = true` and run `uv run wandb login` first.

## 5. Monitor & iterate
- Check `artifacts/vf-function-caller-prime/` for checkpoints, logs, and metrics dumps.
- Use `prime pods logs <name>` from your local machine for real-time streaming.
- Adjust `rollout_batch_size`, `kl_coefficient`, or dataset contents between runs; commit changes so pods can pull them.

## 6. Tear down
When finished, shut down resources to avoid extra cost:
```bash
prime pods delete vf-function-caller-dev
```
Capture key metrics and link the run in issues/PRs so future iterations have a history trail.

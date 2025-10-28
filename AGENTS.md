# Repository Guidelines

## Project Structure & Module Organization
- `README.md` — vision, roadmap, and high-level plan.
- `environments/` — houses Verifiers environment packages (each with `pyproject.toml`, `load_environment` implementation, and README). Use `prime env init <name>` to scaffold new ones.
- `configs/` — Prime-RL `.toml` configs for SFT/RL/eval pipelines.
- `scripts/` — helper CLI tooling (dataset prep, rollout inspection, deployment).
- `notebooks/` (optional) — exploratory analysis; keep heavy experiments documented.

## Build, Test, and Development Commands
- `uv sync --all-extras` — install Python dependencies pinned by the repo.
- `uv run sft @ configs/debug/sft/train.toml` — smoke-test SFT trainer with a tiny config.
- `uv run trainer @ configs/debug/rl/train.toml` — run the minimal RL loop locally.
- `uv run vf-eval vf-function-caller -m gpt-4.1-mini -n 5` — quick Verifiers evaluation to check tool plumbing.
- `prime pods create ...` — spin up Prime Intellect pods (document exact flags in `scripts/` when finalized).

## Coding Style & Naming Conventions
- Python: Black-compatible formatting, 4-space indentation. Prefer type hints (PEP 484).
- File names: `snake_case.py` for modules, `UPPERCASE.md` for docs, `kebab-case.sh` for shell scripts.
- Environment packages should expose `load_environment()` and reside in directories named `vf_<task>`.
- Use docstrings to describe tool behaviors so models have context.

## Testing Guidelines
- Unit tests live under `tests/` mirroring the module structure; name files `test_<module>.py`.
- Use `pytest` (`uv run pytest -v`) for functional validation.
- For Verifiers environments, add smoke tests that execute a single rollout with a mock model to ensure tool schemas are valid.
- Document manual test steps (if any) in the environment README.

## Commit & Pull Request Guidelines
- Follow conventional commit prefixes when possible (`feat:`, `fix:`, `docs:`). Keep messages imperative: `feat: add curl tool wrapper`.
- Every change should be committed and pushed—no long-lived local branches.
- PRs (even internal ones) should include: summary, testing evidence (`uv run pytest`, etc.), screenshots or logs when touching tooling, and references to issues/notes in this repo.
- Rebase onto main before merging to avoid noisy merge commits.


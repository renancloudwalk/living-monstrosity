# Convenience targets for Prime deployments.
.PHONY: prime-bootstrap serve-qwen orchestrate train inspect-rollout

UV ?= uv
MODEL ?= Qwen/Qwen2.5-0.5B
OUTPUT_DIR ?= outputs/manual-run
HOST ?= 0.0.0.0
PORT ?= 8000
TP ?= 1
DP ?= 1
DTYPE ?= bfloat16
MAX_MODEL_LEN ?= 32768
CLIENT_BASE_URL ?= http://127.0.0.1:$(PORT)/v1
ENVIRONMENT_ID ?= vf-function-caller
ENVIRONMENT_ARGS ?= {}
ORCH_MAX_STEPS ?= 20
ORCH_ASYNC_LEVEL ?= 2
ORCH_BATCH_SIZE ?= 16
ORCH_ROLLOUTS ?= 1
TRAIN_MAX_STEPS ?= 20
TRAIN_ASYNC_LEVEL ?= 2
TRAIN_OPTIM ?= adamw
TRAIN_LR ?= 5e-5
TRAIN_WEIGHT_DECAY ?= 0.01
TRAIN_BETAS1 ?= 0.9
TRAIN_BETAS2 ?= 0.95
TRAIN_IMPL ?= hf
TRAIN_ATTN ?= sdpa
RANK ?= 0
INSPECT_STEP ?=

prime-bootstrap:
	./scripts/bootstrap_prime_env.sh

serve-qwen:
	VLLM_USE_V1=0 $(UV) run inference \
		--server.host $(HOST) \
		--server.port $(PORT) \
		--model.name $(MODEL) \
		--model.dtype $(DTYPE) \
		--model.max-model-len $(MAX_MODEL_LEN) \
		--model.trust-remote-code \
		--parallel.tp $(TP) \
		--parallel.dp $(DP)

orchestrate:
	PYTHONPATH=external/prime-rl/src $(UV) run orchestrator \
		--output-dir $(OUTPUT_DIR) \
		--max-steps $(ORCH_MAX_STEPS) \
		--async-level $(ORCH_ASYNC_LEVEL) \
		--batch-size $(ORCH_BATCH_SIZE) \
		--rollouts-per-example $(ORCH_ROLLOUTS) \
		--model.name $(MODEL) \
		--model.trust-remote-code \
		--client.base-url $(CLIENT_BASE_URL) \
		--environment.id $(ENVIRONMENT_ID) \
		--environment.args '$(ENVIRONMENT_ARGS)'

train:
	PYTHONPATH=external/prime-rl/src $(UV) run trainer \
		--no-log.file \
		--output-dir $(OUTPUT_DIR) \
		--max-steps $(TRAIN_MAX_STEPS) \
		--async-level $(TRAIN_ASYNC_LEVEL) \
		--model.name $(MODEL) \
		--model.trust-remote-code \
		--model.impl $(TRAIN_IMPL) \
		--model.attn $(TRAIN_ATTN) \
		--optim.type $(TRAIN_OPTIM) \
		--optim.lr $(TRAIN_LR) \
		--optim.weight-decay $(TRAIN_WEIGHT_DECAY) \
		--optim.betas1 $(TRAIN_BETAS1) \
		--optim.betas2 $(TRAIN_BETAS2)

inspect-rollout:
	@OUTPUT_DIR="$(OUTPUT_DIR)" INSPECT_STEP="$(INSPECT_STEP)" RANK="$(RANK)" $(UV) run python - <<'PY'
import json
import os
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("[inspect] torch is required inside the uv environment.", file=sys.stderr)
    sys.exit(1)

output_dir = Path(os.environ["OUTPUT_DIR"])
if not output_dir.exists():
    print(f"[inspect] output dir not found: {output_dir}", file=sys.stderr)
    sys.exit(1)

rollouts_root = output_dir / "rollouts"
if not rollouts_root.exists():
    print(f"[inspect] no rollouts found under {rollouts_root}", file=sys.stderr)
    sys.exit(1)

step_override = os.environ.get("INSPECT_STEP") or None
if step_override:
    step_dir = rollouts_root / step_override
    if not step_dir.exists():
        print(f"[inspect] step directory not found: {step_dir}", file=sys.stderr)
        sys.exit(1)
else:
    step_dirs = sorted(rollouts_root.glob("step_*"))
    if not step_dirs:
        print(f"[inspect] no step_* directories in {rollouts_root}", file=sys.stderr)
        sys.exit(1)
    step_dir = step_dirs[-1]

rank = os.environ.get("RANK", "0")
rank_path = step_dir / f"rank_{rank}.pt"
if not rank_path.exists():
    print(f"[inspect] rank file missing: {rank_path}", file=sys.stderr)
    sys.exit(1)

payload = torch.load(rank_path, map_location="cpu")
print(f"[inspect] Loaded {rank_path}")
print(f"[inspect] Keys: {sorted(payload.keys())}")

messages = payload.get("messages") or payload.get("dialogue") or []
if not messages:
    print("[inspect] no messages payload to display; done.")
    sys.exit(0)

for idx, message in enumerate(messages, start=1):
    role = message.get("role") or message.get("sender") or "unknown"
    print(f"\n--- message {idx} ({role}) ---")
    if message.get("type") == "tool":
        tool_name = message.get("tool") or message.get("tool_name") or "unknown"
        print(f"tool call → {tool_name}")
        params = message.get("input") or message.get("parameters") or {}
        if params:
            print("params:", json.dumps(params, indent=2, ensure_ascii=False))
        result = message.get("output") or message.get("content") or message.get("result")
        if result:
            if isinstance(result, (list, dict)):
                print("result:", json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print("result:", result)
    else:
        content = message.get("content")
        if isinstance(content, list):
            print("content:", json.dumps(content, indent=2, ensure_ascii=False))
        else:
            print("content:", content)
PY

# vf-function-caller Environment

This Verifiers package defines a `ToolEnv` that lets models practice making constrained HTTP requests via a `curl`-style tool. The environment ships with a small synthetic dataset that rewards the model for:

1. Invoking the tool when the prompt requires external data.
2. Parsing the tool output.
3. Returning the requested final answer.

## Quick start

```bash
uv pip install -e environments/vf_function_caller
uv run python -c "import verifiers as vf; env = vf.load_environment('vf-function-caller'); print(env)"
uv run vf-eval vf-function-caller -m gpt-4.1-mini -n 3 --show-tool-calls
```

## Tool contract

- Function name: `http_fetch`
- Allowed schemes: `http`, `https`
- Allowed hosts: `example.com`, `www.primeintellect.ai`, `httpbin.org`
- The tool truncates responses to 4 KB and returns status code plus body.

Extend the `ALLOWED_NETLOCS` list before exposing the tool to untrusted prompts. The default configuration is intentionally restrictive.

## Dataset layout

The dataset lives inside `vf_function_caller.py` for now. Add richer tasks by expanding the `DEFAULT_DATASET` list or loading data from disk (recommended as the project evolves).

## Development tips

- Keep tool docstrings precise—the model reads them to decide when to call the tool.
- Adjust the rubric weights in `load_environment` to encourage or discourage tool usage.
- Use `vf.ToolEnv(max_turns=6, ...)` to guard against runaway conversations.

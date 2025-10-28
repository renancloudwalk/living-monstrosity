from __future__ import annotations

import json
from pathlib import Path
import textwrap
import urllib.parse
from typing import Any, Dict, Iterable, List

import httpx
from datasets import Dataset
import verifiers as vf
from verifiers.parsers import ToolParser

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_NETLOCS = {
    "example.com",
    "www.example.com",
    "primeintellect.ai",
    "www.primeintellect.ai",
    "httpbin.org",
    "www.httpbin.org",
}
MAX_RESPONSE_BYTES = 4_096
DEFAULT_TIMEOUT = 8.0
DATASET_PATH = Path(__file__).resolve().parent / "data" / "examples.jsonl"


def http_fetch(
    url: str,
    method: str = "GET",
    body: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT,
) -> str:
    """Perform a constrained HTTP request and return status plus trimmed body."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        return f"curl error: scheme '{parsed.scheme}' is not allowed."
    if parsed.netloc not in ALLOWED_NETLOCS:
        return f"curl error: domain '{parsed.netloc}' is not on the allowlist."

    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            response = client.request(
                method=method.upper(),
                url=url,
                content=(body.encode("utf-8") if body else None),
                headers={"User-Agent": "vf-function-caller/0.1"},
            )
    except httpx.RequestError as exc:
        return f"curl error: request failed ({exc})."

    truncated_body = response.text[:MAX_RESPONSE_BYTES]
    return textwrap.dedent(
        f"""\
        Status: {response.status_code}
        URL: {response.url}

        {truncated_body}
        """
    ).strip()


DEFAULT_DATASET = [
    {
        "prompt": (
            "Use the provided curl tool to fetch https://example.com. "
            "Return only the page title."
        ),
        "answer": "Example Domain",
        "metadata": {"require_tool": True},
    },
    {
        "prompt": (
            "Call the curl tool on https://httpbin.org/get and report the value of "
            "the url field from the JSON response."
        ),
        "answer": "https://httpbin.org/get",
        "metadata": {"require_tool": True},
    },
    {
        "prompt": (
            "Query https://www.primeintellect.ai/ with the curl tool. "
            "Reply with the main headline found on the page."
        ),
        "answer": "Prime Intellect - Commoditizing Compute & Intelligence",
        "metadata": {"require_tool": True},
    },
]

SYSTEM_PROMPT = (
    "You are an assistant with access to a single HTTP tool named http_fetch. "
    "Think briefly before invoking tools, and respond with concise answers once you "
    "have the required information."
)


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def _reward_answer(
    parser: ToolParser,
    completion: Iterable[Dict[str, Any]],
    answer: str,
    **_: Any,
) -> float:
    parsed_answer = parser.parse_answer(list(completion))
    if parsed_answer is None:
        # Fallback: look at last assistant message
        assistant_messages = [
            msg.get("content", "")
            for msg in completion
            if msg.get("role") == "assistant"
        ]
        parsed_answer = assistant_messages[-1] if assistant_messages else ""

    return 1.0 if _normalize(parsed_answer) == _normalize(answer) else 0.0


def _reward_tool_usage(
    completion: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
    **_: Any,
) -> float:
    used_tool = any(msg.get("role") == "tool" for msg in completion)
    require_tool = metadata.get("require_tool", False)
    if require_tool and used_tool:
        return 1.0
    if require_tool and not used_tool:
        return 0.0
    # Mild penalty for unnecessary tool calls.
    if not require_tool and used_tool:
        return 0.5
    return 1.0


def _load_records_from_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _build_dataset() -> Dataset:
    records = _load_records_from_file(DATASET_PATH)
    if not records:
        records = DEFAULT_DATASET
    return Dataset.from_list(records)


def load_environment(**kwargs: Any) -> vf.ToolEnv:
    """Return a ToolEnv that teaches curl-style tool usage."""
    parser = ToolParser()
    dataset = _build_dataset()
    rubric = vf.Rubric(
        funcs=[
            lambda completion, answer, **context: _reward_answer(
                parser, completion, answer, **context
            ),
            _reward_tool_usage,
        ],
        weights=[0.8, 0.2],
    )

    env = vf.ToolEnv(
        dataset=dataset,
        tools=[http_fetch],
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        max_turns=6,
        **kwargs,
    )
    return env

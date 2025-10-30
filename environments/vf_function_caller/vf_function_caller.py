from __future__ import annotations

import json
import re
from pathlib import Path
import textwrap
import urllib.parse
from typing import Any, Dict, Iterable, List

import httpx
from datasets import Dataset
import verifiers as vf

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_NETLOCS = {
    "example.com",
    "www.example.com",
    "primeintellect.ai",
    "www.primeintellect.ai",
    "httpbin.org",
    "www.httpbin.org",
}
MAX_RESPONSE_BYTES = 1_024
DEFAULT_TIMEOUT = 8.0
DATASET_PATH = Path(__file__).resolve().parent / "data" / "examples.jsonl"
SEARCH_INDEX = [
    {
        "title": "Example Domain refreshes its educational homepage",
        "url": "https://example.com/",
        "snippet": (
            "Example Domain keeps a minimalist landing page that explains its role "
            "in documentation and training exercises."
        ),
        "keywords": ("example domain", "documentation", "training"),
    },
    {
        "title": "HTTPBin publishes a sample slideshow JSON payload",
        "url": "https://httpbin.org/json",
        "snippet": (
            "The JSON endpoint demonstrates nested content, including a slideshow "
            "object with a descriptive title field."
        ),
        "keywords": ("httpbin", "json", "slideshow", "api"),
    },
    {
        "title": "HTTPBin showcases its HTML demo page",
        "url": "https://httpbin.org/html",
        "snippet": (
            "HTTPBin's HTML sample renders a simple document with a bold H1 title "
            "useful for validating curl output handling."
        ),
        "keywords": ("httpbin", "html", "demo"),
    },
    {
        "title": "Prime Intellect highlights marketplace tooling",
        "url": "https://www.primeintellect.ai/",
        "snippet": (
            "Prime Intellect's homepage advertises commoditized compute services "
            "and training infrastructure for RL teams."
        ),
        "keywords": ("prime intellect", "marketplace", "compute"),
    },
]


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


def news_search(query: str, top_k: int = 3) -> str:
    """Return curated news-style results that the agent can follow with `http_fetch`."""
    if not query or not query.strip():
        return "news_search error: query must not be empty."

    terms = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", query) if len(token) > 2]
    scored: List[tuple[int, Dict[str, Any]]] = []
    for entry in SEARCH_INDEX:
        haystack = " ".join(
            [entry["title"], entry["snippet"], " ".join(entry.get("keywords", ()))]
        ).lower()
        score = sum(1 for term in terms if term in haystack) if terms else 0
        scored.append((score, entry))

    scored.sort(key=lambda item: (item[0], item[1]["title"]), reverse=True)
    results = [entry for score, entry in scored if score > 0][:top_k]
    header = "Top matches:"
    if not results:
        results = [entry for _, entry in scored][: min(top_k, len(scored))]
        header = "No direct matches found. Showing recent items:"

    lines = [header]
    for idx, entry in enumerate(results, start=1):
        lines.append(
            textwrap.dedent(
                f"""\
                {idx}. {entry["title"]}
                   URL: {entry["url"]}
                   Summary: {entry["snippet"]}
                """
            ).rstrip()
        )

    lines.append("Follow up with http_fetch on the chosen URL to read full details.")
    return "\n".join(lines)


DEFAULT_DATASET = [
    {
        "question": (
            "Find coverage about Example Domain refreshing its educational homepage. "
            "Use news_search to pick the most relevant result, then call http_fetch on the suggested URL "
            "and report the exact H1 title you observe."
        ),
        "answer": "Example Domain",
        "required_tools": ["news_search", "http_fetch"],
        "info": {"expected_url": "https://example.com/"},
    },
    {
        "question": (
            "Research the HTTPBin announcement about a sample slideshow JSON payload. "
            "First, call news_search to identify the item, then fetch the linked JSON with http_fetch and "
            "return the value of slideshow.title."
        ),
        "answer": "Sample Slide Show",
        "required_tools": ["news_search", "http_fetch"],
        "info": {"expected_url": "https://httpbin.org/json"},
    },
    {
        "question": (
            "Look up HttpBin's HTML demo page via news_search. After selecting the best hit, use http_fetch "
            "to load the page and quote the full H1 heading."
        ),
        "answer": "Herman Melville - Moby-Dick",
        "required_tools": ["news_search", "http_fetch"],
        "info": {"expected_url": "https://httpbin.org/html"},
    },
    {
        "question": (
            "Investigate the Prime Intellect homepage update mentioned in search results. "
            "Run news_search, pick the top Prime Intellect item, fetch it with http_fetch, and summarize the "
            "main headline (the first prominent heading) from the page."
        ),
        "answer": "Prime Intellect - Commoditizing Compute & Intelligence",
        "required_tools": ["news_search", "http_fetch"],
        "info": {"expected_url": "https://www.primeintellect.ai/"},
    },
]

SYSTEM_PROMPT = (
    "You can call two tools: `news_search` surfaces likely URLs for recent updates, "
    "and `http_fetch` makes the follow-up HTTP request. "
    "Use news_search first to pick the best link, then fetch the page and answer concisely."
)


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def _extract_text(completion: Iterable[Dict[str, Any]]) -> str:
    messages = list(completion)
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    parts.append(chunk.get("text", ""))
            content = "\n".join(parts)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _reward_answer(
    completion: Iterable[Dict[str, Any]],
    answer: str,
    **_: Any,
) -> float:
    parsed_answer = _extract_text(completion)
    return 1.0 if _normalize(parsed_answer) == _normalize(answer) else 0.0


def _reward_tool_usage(
    completion: Iterable[Dict[str, Any]],
    info: Dict[str, Any],
    **_: Any,
) -> float:
    messages = list(completion)
    tool_messages = [
        msg
        for msg in messages
        if msg.get("role") == "tool"
    ]
    tools_used: List[str] = []
    for msg in tool_messages:
        name = msg.get("tool") or msg.get("tool_name") or msg.get("name")
        if isinstance(name, str):
            tools_used.append(name)

    required_raw = info.get("required_tools") or []
    if isinstance(required_raw, str):
        required_tools = [required_raw]
    else:
        required_tools = list(required_raw)
    if not required_tools:
        # Reward skipping tools when none are required, mild penalty otherwise.
        return 1.0 if not tools_used else 0.5

    last_index = -1
    for tool in required_tools:
        try:
            next_index = next(
                idx for idx, used in enumerate(tools_used) if idx > last_index and used == tool
            )
        except StopIteration:
            return 0.0
        last_index = next_index

    # Encourage the minimal tool chain—penalize extra unused tools slightly.
    extra_tools = len(tools_used) - len(required_tools)
    if extra_tools > 0:
        return 0.8
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
    normalised: List[Dict[str, Any]] = []
    for record in records:
        question = record.get("question") or record.get("prompt")
        if question is None:
            raise ValueError("Dataset record missing 'question' field.")
        answer = (record.get("answer") or "").strip()
        info: Dict[str, Any] = {}
        info.update(record.get("metadata", {}))
        info.update(record.get("info", {}))
        if "required_tools" in record:
            required = record["required_tools"]
        else:
            required = info.get("required_tools", [])
        if isinstance(required, str):
            required_list: List[str] = [required]
        else:
            required_list = list(required)
        info["required_tools"] = required_list
        if "expected_url" in record:
            info.setdefault("expected_url", record["expected_url"])
        info.setdefault("question", question)
        normalised.append(
            {
                "question": question,
                "answer": answer,
                "info": info,
            }
        )
    return Dataset.from_list(normalised)


def load_environment(**kwargs: Any) -> vf.ToolEnv:
    """Return a ToolEnv that teaches curl-style tool usage."""
    dataset = _build_dataset()
    rubric = vf.Rubric(
        funcs=[_reward_answer, _reward_tool_usage],
        weights=[0.8, 0.2],
    )

    env = vf.ToolEnv(
        dataset=dataset,
        tools=[news_search, http_fetch],
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=6,
        **kwargs,
    )
    return env

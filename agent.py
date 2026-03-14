"""Task 1 agent CLI: call an LLM and return structured JSON."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ENV_FILE = Path(".env.agent.secret")
REQUIRED_ENV_VARS = ("LLM_API_KEY", "LLM_API_BASE", "LLM_MODEL")
REQUEST_TIMEOUT_SECONDS = 45.0


def _load_env_file(path: Path) -> None:
    """Load key=value pairs from an env file without overriding existing vars."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _extract_answer(data: dict[str, Any]) -> str:
    """Extract assistant text from OpenAI-compatible chat completion response."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response is missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("LLM response choice has invalid format")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response is missing assistant message")

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(part.strip() for part in text_parts if part.strip())

    raise ValueError("LLM response has no text content")


def _call_llm(question: str, *, api_key: str, api_base: str, model: str) -> str:
    """Send a single chat completion request and return assistant text."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user question clearly.",
            },
            {"role": "user", "content": question},
        ],
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")

    return _extract_answer(data)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: uv run agent.py \"<question>\"", file=sys.stderr)
        return 1

    question = sys.argv[1].strip()
    if not question:
        print("Question cannot be empty.", file=sys.stderr)
        return 1

    _load_env_file(ENV_FILE)

    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        print(f"Missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        answer = _call_llm(
            question,
            api_key=os.environ["LLM_API_KEY"],
            api_base=os.environ["LLM_API_BASE"],
            model=os.environ["LLM_MODEL"],
        )
    except httpx.TimeoutException:
        print("LLM request timed out.", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"LLM request failed with status {exc.response.status_code}.", file=sys.stderr)
        return 1
    except httpx.RequestError as exc:
        print(f"LLM request error: {exc}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Invalid LLM response: {exc}", file=sys.stderr)
        return 1

    output = {"answer": answer, "tool_calls": []}
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Task 2 documentation agent CLI with tool-calling loop."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

ENV_FILE = Path(".env.agent.secret")
PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_ENV_VARS = ("LLM_API_KEY", "LLM_API_BASE", "LLM_MODEL")
REQUEST_TIMEOUT_SECONDS = 45.0
MAX_TOOL_CALLS = 10

SYSTEM_PROMPT = (
    "You are a documentation agent for this repository. "
    "Use tools to find answers in the wiki. "
    "First call list_files on 'wiki', then call read_file on relevant files. "
    "When you have the final answer, respond ONLY as JSON with keys "
    "'answer' and 'source'. Source must be a wiki reference like "
    "wiki/git-workflow.md#resolving-merge-conflicts."
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path from project root.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path from project root.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


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


def _extract_text(content: Any) -> str:
    """Extract text content from OpenAI-compatible message content."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts)

    return ""


def _extract_message(data: dict[str, Any]) -> dict[str, Any]:
    """Extract first assistant message from chat completion response."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response is missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("LLM response choice has invalid format")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response is missing assistant message")

    return message


def _call_llm(
    messages: list[dict[str, Any]],
    *,
    api_key: str,
    api_base: str,
    model: str,
) -> dict[str, Any]:
    """Send a chat completion request with tool schemas."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "tool_choice": "auto",
    }
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")

    return _extract_message(data)


def _resolve_project_path(relative_path: str) -> Path:
    """Resolve and validate a path against project root."""
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise ValueError("Path must be relative to project root")

    resolved = (PROJECT_ROOT / candidate_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Access denied: path is outside project root") from exc

    return resolved


def _tool_read_file(path_value: str) -> str:
    """Read file content from a safe path inside project root."""
    try:
        resolved = _resolve_project_path(path_value)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not resolved.exists():
        return f"ERROR: File not found: {path_value}"
    if not resolved.is_file():
        return f"ERROR: Not a file: {path_value}"

    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: File is not UTF-8 text: {path_value}"


def _tool_list_files(path_value: str) -> str:
    """List directory entries from a safe path inside project root."""
    try:
        resolved = _resolve_project_path(path_value)
    except ValueError as exc:
        return f"ERROR: {exc}"

    if not resolved.exists():
        return f"ERROR: Directory not found: {path_value}"
    if not resolved.is_dir():
        return f"ERROR: Not a directory: {path_value}"

    entries = sorted(resolved.iterdir(), key=lambda item: item.name.lower())
    if not entries:
        return "(empty)"

    lines: list[str] = []
    for entry in entries:
        relative = entry.relative_to(PROJECT_ROOT).as_posix()
        if entry.is_dir():
            lines.append(f"{relative}/")
        else:
            lines.append(relative)

    return "\n".join(lines)


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict[str, Any], str | None]:
    """Parse function-call arguments from JSON string or object."""
    if isinstance(raw_arguments, dict):
        return raw_arguments, None

    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}, None
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return {}, str(exc)
        if not isinstance(decoded, dict):
            return {}, "Tool arguments must decode to a JSON object"
        return decoded, None

    return {}, "Tool arguments must be a JSON string or object"


def _execute_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Execute a supported tool and return its result string."""
    path_value = args.get("path")
    if not isinstance(path_value, str):
        return "ERROR: Missing required string argument 'path'"

    if tool_name == "read_file":
        return _tool_read_file(path_value)
    if tool_name == "list_files":
        return _tool_list_files(path_value)

    return f"ERROR: Unknown tool: {tool_name}"


def _parse_final_answer(content_text: str) -> tuple[str, str]:
    """Parse final answer and source from assistant text."""
    stripped = content_text.strip()
    if not stripped:
        return "I could not determine an answer.", "unknown"

    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        decoded = None

    if isinstance(decoded, dict):
        answer = decoded.get("answer")
        source = decoded.get("source")
        if isinstance(answer, str) and answer.strip() and isinstance(source, str) and source.strip():
            return answer.strip(), source.strip()

    source_match = re.search(r"(wiki/[A-Za-z0-9._/-]+#[A-Za-z0-9._-]+)", stripped)
    source = source_match.group(1) if source_match else "unknown"
    return stripped, source


def _run_agent(question: str, *, api_key: str, api_base: str, model: str) -> dict[str, Any]:
    """Run the task-2 agentic loop and return structured output."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    trace: list[dict[str, Any]] = []
    tool_count = 0
    last_assistant_text = ""

    while True:
        assistant_message = _call_llm(
            messages,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )

        assistant_text = _extract_text(assistant_message.get("content"))
        if assistant_text:
            last_assistant_text = assistant_text

        raw_tool_calls = assistant_message.get("tool_calls")
        has_tool_calls = isinstance(raw_tool_calls, list) and len(raw_tool_calls) > 0

        if not has_tool_calls:
            answer, source = _parse_final_answer(last_assistant_text)
            return {
                "answer": answer,
                "source": source,
                "tool_calls": trace,
            }

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content") or "",
                "tool_calls": raw_tool_calls,
            }
        )

        for raw_tool_call in raw_tool_calls:
            if tool_count >= MAX_TOOL_CALLS:
                break

            tool_count += 1
            tool_name = "unknown"
            call_id = f"call-{tool_count}"
            args: dict[str, Any] = {}
            result = "ERROR: Invalid tool call payload"

            if isinstance(raw_tool_call, dict):
                raw_call_id = raw_tool_call.get("id")
                if isinstance(raw_call_id, str) and raw_call_id:
                    call_id = raw_call_id

                function_payload = raw_tool_call.get("function")
                if isinstance(function_payload, dict):
                    raw_name = function_payload.get("name")
                    if isinstance(raw_name, str) and raw_name:
                        tool_name = raw_name

                    raw_arguments = function_payload.get("arguments", "")
                    args, parse_error = _parse_tool_arguments(raw_arguments)
                    if parse_error is not None:
                        result = f"ERROR: Invalid tool arguments: {parse_error}"
                    else:
                        result = _execute_tool(tool_name, args)

            trace.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": result,
                }
            )

        if tool_count >= MAX_TOOL_CALLS:
            answer, source = _parse_final_answer(last_assistant_text)
            return {
                "answer": answer,
                "source": source,
                "tool_calls": trace,
            }


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
        output = _run_agent(
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

    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

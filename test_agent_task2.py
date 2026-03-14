"""Regression tests for Task 2 documentation-agent behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent


def _run_agent_with_scripted_llm(
    *,
    question: str,
    scripted_messages: list[dict[str, Any]],
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    """Run agent.py against a local fake LLM server with scripted responses."""

    class _ScriptedHandler(BaseHTTPRequestHandler):
        requests: list[dict[str, Any]] = []
        step = 0

        def do_POST(self) -> None:  # noqa: N802 (required by BaseHTTPRequestHandler API)
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode("utf-8"))

            handler = type(self)
            handler.requests.append(body)

            if handler.step >= len(scripted_messages):
                self.send_error(500, "No scripted response left")
                return

            message = scripted_messages[handler.step]
            handler.step += 1

            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message}],
            }
            payload = json.dumps(response).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        env = os.environ.copy()
        env["LLM_API_KEY"] = "test-key"
        env["LLM_API_BASE"] = f"http://127.0.0.1:{server.server_port}/v1"
        env["LLM_MODEL"] = "test-model"

        result = subprocess.run(
            [sys.executable, "agent.py", question],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
            check=False,
        )

        requests = _ScriptedHandler.requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    return result, requests


def test_task2_uses_read_file_and_returns_source() -> None:
    scripted_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"wiki/git-workflow.md"}',
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": (
                '{"answer":"Edit conflicting files, stage, then commit.",' 
                '"source":"wiki/git-workflow.md#resolving-merge-conflicts"}'
            ),
        },
    ]

    result, requests = _run_agent_with_scripted_llm(
        question="How do you resolve a merge conflict?",
        scripted_messages=scripted_messages,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())

    assert output["source"] == "wiki/git-workflow.md#resolving-merge-conflicts"
    assert any(call["tool"] == "read_file" for call in output["tool_calls"])
    assert len(requests) >= 2

    second_request_messages = requests[1]["messages"]
    assert any(message.get("role") == "tool" for message in second_request_messages)


def test_task2_uses_list_files_for_wiki_listing_question() -> None:
    scripted_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "arguments": '{"path":"wiki"}',
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": (
                '{"answer":"The wiki includes git-workflow and task guides.",' 
                '"source":"wiki/index.md#table-of-contents"}'
            ),
        },
    ]

    result, requests = _run_agent_with_scripted_llm(
        question="What files are in the wiki?",
        scripted_messages=scripted_messages,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())

    assert output["source"] == "wiki/index.md#table-of-contents"
    assert any(call["tool"] == "list_files" for call in output["tool_calls"])
    assert len(requests) >= 2

    listed_results = [call["result"] for call in output["tool_calls"] if call["tool"] == "list_files"]
    assert listed_results
    assert any("wiki/" in result_text for result_text in listed_results)

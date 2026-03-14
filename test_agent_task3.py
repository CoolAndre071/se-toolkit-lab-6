"""Regression tests for Task 3 system-agent behavior."""

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


def _run_agent_with_servers(
    *,
    question: str,
    scripted_llm_messages: list[dict[str, Any]],
    api_handler: type[BaseHTTPRequestHandler],
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    """Run agent.py against a scripted LLM server and fake backend API server."""

    class _ScriptedLLMHandler(BaseHTTPRequestHandler):
        requests: list[dict[str, Any]] = []
        step = 0

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode("utf-8"))

            handler = type(self)
            handler.requests.append(body)

            if handler.step >= len(scripted_llm_messages):
                self.send_error(500, "No scripted response left")
                return

            message = scripted_llm_messages[handler.step]
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

    llm_server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedLLMHandler)
    api_server = ThreadingHTTPServer(("127.0.0.1", 0), api_handler)

    llm_thread = Thread(target=llm_server.serve_forever, daemon=True)
    api_thread = Thread(target=api_server.serve_forever, daemon=True)
    llm_thread.start()
    api_thread.start()

    try:
        env = os.environ.copy()
        env["LLM_API_KEY"] = "test-key"
        env["LLM_API_BASE"] = f"http://127.0.0.1:{llm_server.server_port}/v1"
        env["LLM_MODEL"] = "test-model"
        env["LMS_API_KEY"] = "test-lms-key"
        env["AGENT_API_BASE_URL"] = f"http://127.0.0.1:{api_server.server_port}"

        result = subprocess.run(
            [sys.executable, "agent.py", question],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
            check=False,
        )

        requests = _ScriptedLLMHandler.requests
    finally:
        llm_server.shutdown()
        api_server.shutdown()
        llm_server.server_close()
        api_server.server_close()
        llm_thread.join(timeout=2)
        api_thread.join(timeout=2)

    return result, requests


class _NoopAPIHandler(BaseHTTPRequestHandler):
    """API server handler for tests that do not use query_api."""

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _ItemsAPIHandler(BaseHTTPRequestHandler):
    """Fake LMS API that validates auth and returns item data."""

    last_authorization: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        type(self).last_authorization = self.headers.get("Authorization")

        if self.path != "/items/":
            self.send_error(404)
            return

        if self.headers.get("Authorization") != "Bearer test-lms-key":
            payload = json.dumps({"detail": "Invalid API key"}).encode("utf-8")
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        payload = json.dumps(
            [
                {"id": 1, "title": "Item A"},
                {"id": 2, "title": "Item B"},
                {"id": 3, "title": "Item C"},
            ]
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def test_task3_uses_read_file_for_backend_framework_question() -> None:
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
                        "arguments": '{"path":"backend/app/main.py"}',
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": (
                '{"answer":"The backend uses FastAPI.",'
                '"source":"backend/app/main.py"}'
            ),
        },
    ]

    result, _requests = _run_agent_with_servers(
        question="What framework does the backend use?",
        scripted_llm_messages=scripted_messages,
        api_handler=_NoopAPIHandler,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())

    assert "fastapi" in output["answer"].lower()
    assert output.get("source") == "backend/app/main.py"
    assert any(call["tool"] == "read_file" for call in output["tool_calls"])


def test_task3_uses_query_api_for_database_count_question() -> None:
    scripted_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "query_api",
                        "arguments": '{"method":"GET","path":"/items/"}',
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": '{"answer":"There are 3 items in the database."}',
        },
    ]

    result, _requests = _run_agent_with_servers(
        question="How many items are in the database?",
        scripted_llm_messages=scripted_messages,
        api_handler=_ItemsAPIHandler,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip())

    assert "3" in output["answer"]
    query_calls = [call for call in output["tool_calls"] if call["tool"] == "query_api"]
    assert query_calls

    parsed_result = json.loads(query_calls[0]["result"])
    assert parsed_result["status_code"] == 200
    assert isinstance(parsed_result["body"], list)
    assert len(parsed_result["body"]) == 3
    assert _ItemsAPIHandler.last_authorization == "Bearer test-lms-key"
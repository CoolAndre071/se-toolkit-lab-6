"""Regression tests for Task 1 agent CLI."""

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


class _FakeChatHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible chat completions handler for tests."""

    response_body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Representational State Transfer.",
                },
            }
        ],
    }

    def do_POST(self) -> None:  # noqa: N802 (required by BaseHTTPRequestHandler API)
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)

        payload = json.dumps(self.response_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        """Silence test server logs."""
        return


def test_agent_outputs_required_json_fields() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeChatHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        env = os.environ.copy()
        env["LLM_API_KEY"] = "test-key"
        env["LLM_API_BASE"] = f"http://127.0.0.1:{server.server_port}/v1"
        env["LLM_MODEL"] = "test-model"

        result = subprocess.run(
            [sys.executable, "agent.py", "What does REST stand for?"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "agent.py should print JSON to stdout"

    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1, "stdout should contain exactly one JSON line"

    parsed = json.loads(lines[0])
    assert "answer" in parsed
    assert "tool_calls" in parsed
    assert isinstance(parsed["answer"], str)
    assert isinstance(parsed["tool_calls"], list)
    assert parsed["tool_calls"] == []

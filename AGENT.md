# Agent Architecture (Task 1)

## Overview

`agent.py` is a minimal CLI agent for Task 1.

Flow:

1. Read user question from the first CLI argument.
2. Load LLM settings from `.env.agent.secret`.
3. Call an OpenAI-compatible `chat/completions` endpoint.
4. Print one JSON line to stdout:
   - `answer` (string)
   - `tool_calls` (empty array in Task 1)

No agentic loop and no tools are used yet (added in later tasks).

## LLM provider and model

Current plan uses:

- Provider: Qwen Code API
- Model: `qwen3-coder-plus`

Environment variables required:

- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_MODEL`

`agent.py` loads these from `.env.agent.secret`. Secrets are not hardcoded.

## Output and logging rules

- Stdout: only valid JSON output.
- Stderr: errors and diagnostics.

On success, process exits with code `0`.

## Run

```bash
uv run agent.py "What does REST stand for?"
```

Example output:

```json
{"answer":"Representational State Transfer.","tool_calls":[]}
```

## Testing

Task 1 regression test:

- `backend/tests/unit/test_agent_cli.py`

The test starts a local fake chat-completions server, runs `agent.py` as a subprocess, parses stdout JSON, and verifies `answer` and `tool_calls` exist.

# Task 1 Plan: Call an LLM from Code

## Goal

Build a minimal CLI agent (`agent.py`) that:

1. Accepts a user question from the first command-line argument.
2. Calls an OpenAI-compatible chat completions API.
3. Prints exactly one JSON object to stdout:
   `{"answer":"...","tool_calls":[]}`

## Provider and model

- Provider: Qwen Code API
- Model: `qwen3-coder-plus`
- Why: generous free quota, reliable for this lab, and supported OpenAI-compatible API.

The agent will read configuration from `.env.agent.secret`:

- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_MODEL`

No secrets will be hardcoded.

## Data flow

1. Parse CLI args and read the question (`sys.argv[1]`).
2. Load env vars from `.env.agent.secret`.
3. Validate required env vars.
4. Send one chat-completions request to the configured model.
5. Extract the assistant text response.
6. Emit one JSON line to stdout with required fields:
   - `answer`: string
   - `tool_calls`: empty array
7. Send logs and debug output to stderr only.

## Error handling plan

- Missing question: write a clear error to stderr and exit non-zero.
- Missing env var: write which variable is missing to stderr and exit non-zero.
- API/network failure or timeout: write a short error to stderr and exit non-zero.
- Success path exits with code `0`.

To satisfy the 60-second requirement, set a request timeout below 60 seconds.

## Implementation steps

1. Create this plan before coding.
2. Implement `agent.py` with minimal prompt + single chat request.
3. Ensure stdout contains only valid JSON, with no extra text.
4. Create one regression test that runs `agent.py` as a subprocess and verifies:
   - stdout is valid JSON
   - `answer` exists
   - `tool_calls` exists and is an array
5. Update `AGENT.md` with architecture, provider choice, and run instructions.

## Verification checklist

- `uv run agent.py "What does REST stand for?"` returns valid JSON.
- `tool_calls` is always `[]` for Task 1.
- Debug/progress output appears only on stderr.
- API key remains only in `.env.agent.secret`.

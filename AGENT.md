# Agent Architecture

## Overview

`agent.py` is a CLI documentation agent that answers questions using an OpenAI-compatible chat API and local wiki tools.

Current flow:

1. Parse question from CLI (`sys.argv[1]`).
2. Load LLM settings from `.env.agent.secret`.
3. Send system + user messages with tool schemas.
4. Run an agentic loop:
   - if model requests tools, execute tools and feed results back,
   - if model returns a final response, parse `answer` and `source`.
5. Print one JSON line to stdout.

## LLM configuration

Environment variables:

- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_MODEL`

`agent.py` reads `.env.agent.secret` and does not hardcode secrets.

## Tooling

The agent defines two tools as function-calling schemas:

1. `list_files(path)`
2. `read_file(path)`

### Path security

All tool paths are resolved against project root.

- Absolute paths are rejected.
- `..` traversal outside project root is rejected.
- Tools return readable error strings instead of crashing.

## Agentic loop

- The model can call tools up to 10 times per question.
- Every executed tool call is stored in output `tool_calls` as:
  - `tool`
  - `args`
  - `result`
- Tool results are sent back to the model as `tool` role messages.
- Loop ends when the model returns a normal assistant response (no tool calls), or when the tool-call cap is reached.

## Prompt strategy

The system prompt asks the model to:

- discover docs with `list_files("wiki")`,
- read relevant docs with `read_file(...)`,
- return final output as JSON containing `answer` and `source`.

## Output contract

Stdout always contains one valid JSON object:

- `answer` (string)
- `source` (string)
- `tool_calls` (array)

All errors/diagnostics go to stderr.

## Run

```bash
uv run agent.py "How do you resolve a merge conflict?"
```

## Tests (root folder)

Regression tests are kept in the project root:

- `test_agent_cli.py` (Task 1 contract)
- `test_agent_task2.py` (Task 2 tool-calling behavior)

Task 2 tests use a local fake LLM server to validate loop and tool usage without external API dependency.

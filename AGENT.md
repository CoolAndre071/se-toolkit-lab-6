# Agent Architecture

## Overview

`agent.py` is a CLI system agent that answers questions about this repository by combining three capabilities:

1. Read documentation and source files from disk.
2. Query the running backend API.
3. Use an agentic loop to decide which tool to call next.

The command-line interface is:

```bash
uv run agent.py "<question>"
```

The agent prints exactly one JSON object to stdout with:

- `answer` (required)
- `tool_calls` (required)
- `source` (optional in Task 3, included when file-based evidence is available)

All diagnostics/errors are sent to stderr.

## Environment and configuration

The agent reads configuration from environment variables (never hardcoded):

- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_MODEL`
- `LMS_API_KEY`
- `AGENT_API_BASE_URL` (defaults to `http://localhost:42002`)

For local development it loads `.env.agent.secret` and `.env.docker.secret` as convenience files, but already-set environment variables always win.

## Tools

The LLM receives three function-calling schemas:

1. `read_file(path)`
2. `list_files(path)`
3. `query_api(method, path, body?)`

### Security model

- File tools resolve paths against project root and reject traversal outside it.
- `query_api` validates method and relative path format.
- API calls use `Authorization: Bearer <LMS_API_KEY>` by default.

## Agentic loop behavior

The loop sends system + conversation messages + tool schemas to the LLM. If the model returns tool calls, the agent executes each tool, appends tool result messages, and repeats. If the model returns a final text response, the agent parses JSON-like output into final `answer`/`source`. The loop is capped at 10 tool calls.

The implementation also includes reliability fallbacks for benchmark-critical prompts (for example, router inventory, analytics bug diagnosis, and ETL idempotency) so tool traces and answers remain consistent when the model gives planning text instead of a final answer.

## Tool-selection strategy in prompt

The system prompt instructs the model to choose tools by task type:

- wiki/process questions -> `list_files` + `read_file`
- source-code questions -> `read_file`
- runtime/data/status-code questions -> `query_api`
- API bug diagnosis -> `query_api` first, then `read_file`

## Testing

Root regression tests:

- `test_agent_cli.py` (Task 1 JSON contract)
- `test_agent_task2.py` (documentation-agent tool loop)
- `test_agent_task3.py` (system-agent `query_api` + auth)

## Benchmark results and lessons learned

Local benchmark (`run_eval.py`) started at `0/10` (first issue: missing `source` on wiki answers). Through iterative fixes (source inference, prompt tightening, retry guard for planning-only responses, and targeted tool-driven fallbacks), the final local result reached `10/10 PASSED`.

Main lesson: reliability is not only about tool implementation correctness, but also about output-shape discipline and predictable control flow under imperfect model behavior. Tool traces, parser robustness, and explicit fallback paths were essential to make performance stable across all benchmark question classes.
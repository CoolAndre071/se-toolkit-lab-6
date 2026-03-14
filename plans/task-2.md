# Task 2 Plan: The Documentation Agent

## Goal

Extend `agent.py` from a single LLM call to an agentic loop that can use wiki tools and return:

- `answer` (string)
- `source` (string, wiki file + section anchor)
- `tool_calls` (array of executed tool calls)

## LLM strategy

- Keep OpenAI-compatible API calls via `httpx`.
- Send tool schemas in the chat completion request.
- Use a system prompt that instructs the model to:
  - discover files with `list_files`,
  - read relevant docs with `read_file`,
  - provide final output with a source reference.

## Tool schemas

Define two function-calling schemas:

1. `list_files(path: string)`
2. `read_file(path: string)`

Each schema accepts an object with required `path` and no extra properties.

## Tool implementations and security

Project root is the only allowed boundary.

- Resolve every user/model-provided path against project root.
- Reject traversal attempts and absolute paths outside root.
- `read_file`:
  - returns text content,
  - returns clear error message if file does not exist/is not a file.
- `list_files`:
  - returns newline-separated entries,
  - returns clear error message if path does not exist/is not a directory.

## Agentic loop

1. Build initial messages: system + user.
2. Send request with tools.
3. If assistant returns `tool_calls`:
   - execute each tool,
   - append assistant tool-call message + tool result messages,
   - record each call in output `tool_calls`.
4. If assistant returns text without tool calls:
   - parse final `answer` and `source`,
   - return JSON and exit.
5. Stop after at most 10 tool calls and return best available answer.

## Output contract

- Stdout: only valid JSON.
- Stderr: diagnostics/errors only.
- Required fields in success output:
  - `answer`
  - `source`
  - `tool_calls`

## Tests (root folder)

Keep tests in project root as requested.

Add 2 Task 2 regression tests using a fake local LLM server:

1. Merge-conflict question:
   - scripted LLM calls `read_file` on `wiki/git-workflow.md`,
   - final response contains source anchor,
   - assert `read_file` appears in `tool_calls` and source path matches.
2. Wiki listing question:
   - scripted LLM calls `list_files` on `wiki`,
   - assert `list_files` appears in `tool_calls` and output includes source.

## Documentation updates

Update `AGENT.md` with:

- new tools,
- loop behavior,
- security boundary,
- source extraction/output format.

# angularai-mcp

AngularAI's cortex as MCP tools. Any MCP client can remember, recall, and forget, on disk, in the cortex `angularai` already uses. The server name is `AngularAI`.

No model is called. A key is never read for a tool, and never returned.

## Run

```bash
uvx --from "git+https://github.com/angularai-hq/AngularAI#subdirectory=packages/angularai-mcp" angularai-mcp
```

After a release to PyPI, `uvx angularai-mcp`. Not published yet.

From a checkout: `uv sync --all-packages`, then `uv run angularai-mcp`.

The cortex is the one `angularai` would use: `--cortex PATH`, else `--name NAME`, else `ANGULARAI_CORTEX`, else `~/.angularai/default`. A name follows the rules of `angularai --name`. A slash, a dot, `..`, an upper-case letter, or a link in place of the directory is refused before anything is created. One server serves one cortex. Another cortex is another entry with `--name`.

A missing cortex is an error that names `angularai init`. `--init` creates it, once.

Nothing but the protocol is written to stdout. Notices go to stderr.

## Tools

| tool | does | writes |
| --- | --- | --- |
| `remember` | keeps one fact, at most 240 characters and 4 tags | one pass |
| `recall` | finds facts for a query, best first, inside a token cap, with id, tags, source, time, and score | nothing |
| `forget` | removes one fact by its exact id | rewrites the log |
| `status` | name, budget, memory against the budget, counts, last pass, embeddings mode | nothing |
| `cortexes` | the named cortexes in the home, and the active one | nothing |

A fact passes the gates a model's trace passes. A secret is refused: a known key shape, a `password=` value, a random run, or the value of an environment variable whose name says secret. A value-lock phrase holds it. The same fact twice is kept once. Its source is `mcp`.

`recall` matches words. With `ANGULARAI_EMBEDDINGS` set in the server's environment it also matches by meaning, and sends the query and unembedded traces, redacted, to the embeddings API. `lexical_only` turns that off for one call. Results stop at `limit`, 1 to 50, and at `max_tokens` estimated tokens of text, by default and at most the cortex's workspace budget. Four ASCII characters are one token. A query that looks like a secret is refused.

`forget` takes `t_` and 12 lowercase hex digits, exactly as `recall` shows. No prefix, pattern, or text. An unknown id removes nothing. The trace leaves the line of the pass that wrote it, a pass that only noted it goes whole, a supersede that names it goes, and its vectors leave `embeddings.jsonl`. Both files are rewritten atomically under the cortex lock. Every other line stays, unreadable ones too. An episode that mentions the fact in its own words is kept. The tool is marked destructive, and Claude Code asks you on every call.

Resources: `angularai://status`, and `angularai://trace/{id}`.

There is no tool to run or edit files. The agent's tools stay in `angularai agent`.

## Errors

An error is a tool result marked `isError`, as `[code] sentence`. No traceback, no argument value, no environment value.

| code | when |
| --- | --- |
| `invalid_input` | empty, a bad tag, a limit out of range, arguments that do not fit the schema |
| `too_long` | a fact over 240 characters, a query over 2,000 |
| `secret_refused` | it looks like a secret |
| `locked` | a value-lock phrase is in it |
| `not_found` | no trace has that id |
| `no_cortex` | no cortex at the path |
| `config_error` | `cortex.json` or an `ANGULARAI_` setting is not valid |
| `storage_error` | the disk refused |
| `internal` | anything else. Only its type is logged |

## HTTP

```bash
angularai-mcp --http
```

Streamable HTTP at `http://127.0.0.1:8765/mcp`. `--port` moves it. On loopback, a request whose `Host` or `Origin` is not loopback is refused.

There is no auth unless `ANGULARAI_MCP_TOKEN` is set: then every request needs `Authorization: Bearer <token>`, and anything else gets 401. The token is 16 or more printable ASCII characters. It is compared in constant time and never printed.

A host beyond loopback needs `--allow-remote` and the token. There is no TLS. Put it behind a proxy that has one.

## Setup

Each snippet starts the server over stdio for the default cortex. Add `"--name", "bo"` for another. `uvx` must be on the client's `PATH`. A desktop app may not see your shell's: give the full path from `which uvx`. The first start downloads and builds, and can take a while.

### Claude Desktop

`claude_desktop_config.json`, from Settings, Developer, Edit Config:

```json
{
  "mcpServers": {
    "angularai": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/angularai-hq/AngularAI#subdirectory=packages/angularai-mcp",
        "angularai-mcp"
      ]
    }
  }
}
```

Quit and reopen Claude Desktop.

### Claude Code

```bash
claude mcp add --scope user angularai -- uvx --from "git+https://github.com/angularai-hq/AngularAI#subdirectory=packages/angularai-mcp" angularai-mcp
```

Over HTTP, with the server already running:

```bash
claude mcp add --transport http --scope user angularai http://127.0.0.1:8765/mcp --header "Authorization: Bearer $ANGULARAI_MCP_TOKEN"
```

### Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.angularai]
command = "uvx"
args = ["--from", "git+https://github.com/angularai-hq/AngularAI#subdirectory=packages/angularai-mcp", "angularai-mcp"]
startup_timeout_sec = 60

[mcp_servers.angularai.tools.forget]
approval_mode = "prompt"
```

Or `codex mcp add angularai -- uvx --from "git+https://github.com/angularai-hq/AngularAI#subdirectory=packages/angularai-mcp" angularai-mcp`. Over HTTP:

```toml
[mcp_servers.angularai]
url = "http://127.0.0.1:8765/mcp"
bearer_token_env_var = "ANGULARAI_MCP_TOKEN"
```

### Cursor

`.cursor/mcp.json` in a project, or `~/.cursor/mcp.json` for every project:

```json
{
  "mcpServers": {
    "angularai": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/angularai-hq/AngularAI#subdirectory=packages/angularai-mcp",
        "angularai-mcp"
      ]
    }
  }
}
```

Over HTTP:

```json
{
  "mcpServers": {
    "angularai": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": { "Authorization": "Bearer ${env:ANGULARAI_MCP_TOKEN}" }
    }
  }
}
```

After a release, each `uvx --from ... angularai-mcp` is `uvx angularai-mcp`.

## Verify

From `packages/angularai-mcp`:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest --cov
uv build
```

Tests run the tools in process, over a child process on stdio, and over HTTP on a free loopback port. No real API is called. An autouse fixture removes every key and `ANGULARAI_` variable without reading it, and a child server gets only `PATH`, `HOME`, and `ANGULARAI_HOME`.

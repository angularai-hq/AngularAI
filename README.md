# AngularAI

A cortex in front of the model you already call.

The call ends. The file does not. AngularAI sits in front of an API you already wire, and each pass sends only what fits a fixed budget. What the agent learns stays on disk. The vendor’s weights do not change.

The layer names come from Thameleon's CLASP model. The hardware stays hardware.

## Install

```bash
uv tool install git+https://github.com/baiehclaca/AngularAI.git
angularai init
```

The cortex is created at `~/.angularai/default`, readable only by you. The key is read from the environment when you speak. It is not written into the cortex.

```bash
export OPENAI_API_KEY=...
angularai wire openai --model <model>

export ANTHROPIC_API_KEY=...
angularai wire anthropic --model <model>

export ANGULARAI_API_KEY=local
angularai wire compatible --model <model> --base-url http://127.0.0.1:11434/v1
```

From a checkout, `uv sync` then `uv run angularai` does the same. `uv tool install --editable .` puts the command on your PATH.

The site is [baiehclaca.github.io/AngularAI](https://baiehclaca.github.io/AngularAI/). The paper is `docs/angularai.pdf`.

## Speak

```bash
angularai say "I have a blue notebook."
angularai say "What color is the notebook?"
angularai recall "notebook"
```

`angularai` in a terminal is the session: a transcript, a composer, and a status line. `/` opens the commands. `/clear` clears the screen. The cortex stays on disk. `angularai say` prints only the reply.

"What color is the notebook?" finds "The notebook is blue." "What shade is it?" does not, until a later version embeds traces.

## Work

```bash
angularai agent "make the failing test pass"
angularai agent --root ~/src/app --steps 12 "rename the command"
```

The agent works inside one directory. One tool a pass: `list`, `read`, `search`, `edit`, `write`, or `run`. A path that leaves the directory is refused. Every link is followed first, so a symlink that points out is refused by `list`, `read`, `search`, `edit`, and `write`. Inside a directory, `list` and `search` skip it. A secret in a file that matches a known key pattern, or equals `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `ANGULARAI_API_KEY`, is redacted before the next pass. The step budget is 8, and 24 is the most. The reply is the only line on stdout. The steps are on stderr.

`run` is off. `--allow-run` turns it on for that command.

```bash
angularai agent --allow-run "count the lines in the notes"
```

With `--allow-run`, `run` starts one program from an allowlist: `basename cat cmp cp cut dirname echo false grep head ls mkdir mv pwd rm rmdir sort stat tail touch true uniq wc`. There is no shell, no interpreter, no `find`, `xargs`, `env`, `git`, or `make`. The program comes from `/usr/bin:/bin`, never from the directory. Every argument that names a path must resolve inside the directory. Options that run a program or follow a link out are refused: `cp -L -H --dereference`, `grep -R --dereference-recursive`, `sort --compress-program`, `--files0-from`. The program starts in the directory with three variables: `PATH=/usr/bin:/bin`, `HOME` set to the directory, and `LANG`. No API key is passed. Output stops at 64 KiB. After 20 seconds the whole process group is killed.

This is a filter on what starts, not an operating-system sandbox. No allowlisted program opens a network connection, but nothing at the OS level blocks one. `rm` and `mv` can delete or move anything inside the directory. `run` needs Linux or macOS.

The membrane still holds. The model does not receive the tree. It receives the task, the last tool, and the traces that fit.

## The membrane

The estimate is four characters to one token. AngularAI enforces it.

1. A secret in the input is refused. Nothing is written.
2. Attention ranks traces and episodes against the input by word overlap.
3. The workspace takes the matches that fit, then the latest episode if it still fits.
4. That text, and only that text, is sent.
5. The value lock judges the reply. A forbidden phrase, or a secret, holds it.
6. A completed pass may write traces. A failed call writes none. A superseded trace stays on disk and leaves the workspace.

`--cortex PATH` can be placed anywhere on the command. `ANGULARAI_CORTEX` does the same.

## What a failure looks like

- No cortex yet: `angularai init`.
- No key: the missing variable is named. The value is never printed.
- The API is down, or the reply is not the JSON object AngularAI asked for: nothing new is remembered.
- The input is over the budget: it is not sent.
- The reply contains a lock phrase or a secret: you see `Value lock held this reply.` The held text is not stored. A held reply does not run a tool.
- The model names a trace it was not shown: that trace is not superseded.
- The step budget is spent: the last reply is printed, and the cortex keeps that pass.

## What waits

Paraphrase recall, and more than one cortex talking.

## Verify

```bash
uv run pytest
uv run mypy
uv run ruff check .
```

The invariant is `tests/test_membrane.py`, and for the agent `tests/test_agent.py`. A fact from the start of a long life is in the workspace, an unrelated episode that would have fit is not, and every call stays inside the budget. The agent reads the file it asked for. The file it did not ask for is not sent.

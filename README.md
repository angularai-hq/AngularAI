# AngularAI

A cortex in front of a language model.

A model is helpful inside its window and gone outside it. AngularAI wires the API you already call and keeps the agent in a cortex on disk. Each pass, the model sees a bounded workspace. What does not fit stays on disk and can return later. The window does not grow with the life of the agent.

The layer names come from Thameleon's CLASP model. The hardware stays hardware. This is that cognition, running on ordinary model APIs. The vendor's weights do not change. Plasticity is the cortex.

## The membrane

Every call is packed to a fixed budget. The estimate is four characters to one token, and it is the budget AngularAI enforces.

1. The input is checked. A secret is refused before it is sent, and nothing is written.
2. Attention ranks traces and past episodes against the input by word overlap.
3. The workspace takes the matches that fit, then the latest episode if it still fits.
4. That text, and only that text, is sent to the model.
5. The reply is judged by the value lock. The Thinker drops anything L4 did not judge.
6. A completed pass may write traces. A failed call writes none. A superseded trace stays on disk and leaves the workspace.

Recall is lexical. "What color is the notebook?" finds "The notebook is blue." It will not find that fact from "What shade is it?" until a later version embeds traces.

## Install

```bash
uv sync
uv run angularai init
```

The cortex is created at `~/.angularai/default` with permissions limited to you. Keys are read from the environment when you speak. They are not written into the cortex.

```bash
export OPENAI_API_KEY=...
uv run angularai wire openai --model <model>

export ANTHROPIC_API_KEY=...
uv run angularai wire anthropic --model <model>

export ANGULARAI_API_KEY=local
uv run angularai wire compatible --model <model> --base-url http://127.0.0.1:11434/v1
```

To put the command on your PATH:

```bash
uv tool install --editable .
```

The short paper is `docs/angularai.pdf`. Rebuild it with `uv run python docs/build_paper.py`.

## Use

```bash
angularai say "I have a blue notebook."
angularai say "What color is the notebook?"
angularai cortex
angularai recall "notebook"
angularai status
angularai budget 2000
angularai lock add "dosing chart"
angularai
```

`angularai` in a terminal is the session: a transcript, a composer, and a status line with the substrate, the model, and how much of the budget the last pass used. `/` opens the commands. Up and down move through them, and tab completes one. `/clear` clears the screen. The cortex stays on disk. `/cortex`, `/recall`, `/status`, `/locks`, `/budget`, and `/quit` read or leave without a model. Away from a terminal the same commands stay on a line. The reply is the only thing printed by `angularai say`, so it can be piped.

`--cortex PATH` can be placed anywhere on the command. `ANGULARAI_CORTEX` does the same.

## What a failure looks like

- No cortex yet: `angularai init`.
- No key: the missing variable is named. The value is never printed.
- The API is down, or the reply is not the JSON object AngularAI asked for: nothing new is remembered.
- The input is over the budget: it is not sent.
- The reply contains a lock phrase or a secret: you see `Value lock held this reply.` The held text is not stored.
- The model names a trace it was not shown: that trace is not superseded.

## What waits

Paraphrase recall, tools, and more than one cortex talking. The membrane is the product: a bounded workspace, a cortex that outlives it, and a lock that is a layer rather than a hope.

## Verify

```bash
uv run pytest
uv run mypy
uv run ruff check .
```

The invariant is `tests/test_membrane.py`. A fact from the start of a long life is in the workspace, an unrelated episode that would have fit is not, and every call stays inside the budget.

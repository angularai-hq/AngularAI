# Evals

v2 phase P0. Benchmark A: LongMemEval-S, reduced to fit a local model on a CPU. The v1 baseline.

Free models only. No paid key, no paid service. Every command removes `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY` and the other keys in `angularai/evals/client.py` from its environment before it runs. A base URL off this machine is refused. A frontier comparison waits until Teodor approves a paid key; until then `--allow-remote` is not used.

## Arms

| arm | what the model sees |
| --- | --- |
| `truncated` | The newest turns that fit the budget, with the system prompt counted. The small model alone. |
| `v1` | Every turn written to a real v1 cortex as episodes, then the question as one v1 pass (`pilot.pass_turn`) at the budget. Lexical recall, the v1 prompt, JSON reply. |
| `full` | The whole reduced history. LongMemEval's reading prompt. |
| `oracle` | Only the evidence sessions. An upper bound for reading, not for recall. |

The v1 arm writes no traces: on a CPU, a model call per history turn does not finish. Recall is over episodes, clipped as v1 clips them (700 / 400 characters). Episode dates are not shown, as in v1. Embeddings are off.

## The reduction

LongMemEval-S histories are about 115k tokens. A 3B model on 8 CPU cores reads about 36 tokens a second at 10k tokens of context. The subset is cut so the full history fits a 16k window.

- Source: `xiaowu0162/longmemeval-cleaned`, revision `98d7416c`, `longmemeval_s_cleaned.json`, SHA-256 `d6f21ea9…c3a442`. Checked on every load.
- Haystack: every evidence session, then distractor sessions from the same haystack in a seeded random order, while the rendered history stays within 12,000 estimated tokens (4 characters = 1 token, the membrane's estimate). Kept sessions stay in their original order.
- Eligible: a question whose evidence alone fits. 457 of 500.
- Sample: stratified by `question_type`, largest-remainder allocation, seed 0. Within a type, shuffled by seed and the first taken.
- 50 questions: knowledge-update 9, multi-session 11, single-session-assistant 6, single-session-preference 3, single-session-user 8, temporal-reasoning 13. 5 are abstention (`_abs`).
- The budget for `truncated` and `v1` is 2,000 tokens, v1's default. The history is 6x the window.

The subset file is written by `reduce` and rebuilt from the pinned file; the manifest in `results/` lists every kept session id.

## Scoring

- `judge`: LongMemEval's `evaluate_qa.py` prompts, word for word, answered by a local model (`aa-qwen3-4b`). The paper uses GPT-4o; a local judge is weaker. This is the headline number.
- `contains`: the normalized gold answer is in the normalized prediction. Model-free and strict. Not defined for preference rubrics or abstention.

## Models

Ollama 0.40.6, CPU only, flash attention on, KV cache q8_0, one request at a time. `models/*.Modelfile` pin the window (16,384), threads (6), temperature 0 and seed 0.

| name | base | Ollama digest | quant |
| --- | --- | --- | --- |
| `aa-qwen2.5-3b` | `qwen2.5:3b-instruct` | `49c0f343adef` | Q4_K_M |
| `aa-qwen3-4b` | `qwen3:4b-instruct-2507` | `8073292f4627` | Q4_K_M |

```bash
ollama create aa-qwen2.5-3b -f evals/models/aa-qwen2.5-3b.Modelfile
ollama create aa-qwen3-4b -f evals/models/aa-qwen3-4b.Modelfile
```

## Run

```bash
uv run angularai-eval fetch
uv run angularai-eval reduce --count 50 --seed 0 --haystack-tokens 12000 --out evals/runs/b50/subset.json
uv run angularai-eval run --subset evals/runs/b50/subset.json --arm v1 --model aa-qwen2.5-3b --budget 2000 --out evals/runs/b50/v1.jsonl
uv run angularai-eval judge --subset evals/runs/b50/subset.json --predictions evals/runs/b50/v1.jsonl --model aa-qwen3-4b --out evals/runs/b50/v1.judged.jsonl
uv run angularai-eval report --subset evals/runs/b50/subset.json --judged evals/runs/b50/*.judged.jsonl --out evals/runs/b50/results.json
```

Or every arm: `evals/baseline.sh 50`. Every step resumes: a rerun skips what is already in its file.

`uv run pytest` runs no model. `ANGULARAI_EVAL_LIVE=1 uv run pytest -m live` runs one live check against the local server.

## Kill rule

No "beats frontier" claim unless benchmark A shows at least +10 points over the small model alone.

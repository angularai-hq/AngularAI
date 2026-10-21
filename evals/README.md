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

## Results: v1 baseline, 50 questions

Tue 20 Oct 2026. `results/b50/`. Same subset, same model, same 2,000-token budget.

| arm | model | n | judge % | contains % | prompt tokens | p50 s |
| --- | --- | --- | --- | --- | --- | --- |
| `truncated` | `aa-qwen2.5-3b` | 50 | 22.0 | 7.1 | 1,479 | 13.1 |
| `v1` | `aa-qwen2.5-3b` | 50 | 54.0 | 31.0 | ≤ 2,000 (est. 1,904) | 21.8 |
| `full` | `aa-qwen2.5-3b` | 50 | 54.0 | 40.5 | 10,332 | 307.7 |

By type, judge %, `truncated` / `v1` / `full`:

| type | n | truncated | v1 | full |
| --- | --- | --- | --- | --- |
| knowledge-update | 9 | 22.2 | 55.6 | 44.4 |
| multi-session | 11 | 18.2 | 36.4 | 36.4 |
| single-session-assistant | 6 | 33.3 | 50.0 | 100.0 |
| single-session-preference | 3 | 66.7 | 100.0 | 66.7 |
| single-session-user | 8 | 25.0 | 75.0 | 62.5 |
| temporal-reasoning | 13 | 7.7 | 46.2 | 46.2 |
| abstention (`_abs`) | 5 | 100.0 | 80.0 | 80.0 |

- v1 is +32 points over the small model alone at the same budget. The kill rule's bar is +10. This is v1 against truncation, not against a frontier model.
- v1 ties the full history on the judge (54.0) at 18% of the tokens and 7% of the time. On `contains` it trails: 31.0 against 40.5. Most of the gap is single-session-assistant (16.7 against 83.3): v1 clips an assistant reply to 400 characters.
- The judge is a 4B local model. Spot checks find it lenient on some numeric answers. `contains` moves the same way: 7.1 → 31.0.
- 50 questions: one question is 2 points. Read differences under 10 points as noise.
- Prompt tokens for `truncated` and `full` are the server's count; for `v1`, the membrane's estimate (4 characters = 1 token), which runs about 15% above the server's.

Speed on this box (8 vCPU Xeon, AVX-512/AMX, no GPU, about 5 GB free RAM, other work running), from the server's timings:

| model | prompt tok/s | at ≥ 8k context | generation tok/s |
| --- | --- | --- | --- |
| `aa-qwen2.5-3b` | 97 | 36 | 14.7 |
| `aa-qwen3-4b` (judge) | 125 (short prompts) | not measured | 10 (1–2 token replies) |

The full 457 eligible questions, three arms and the judge: about 330 s a question, about 42 hours. `full` is 85% of it. The 50-question `full` arm took 5 h 8 min of model time. `qwen2.5:7b` does not fit a 16k window in the free RAM.

## Results: cortex v2, first run

Tue 20 Oct 2026. `v2` arm: `aa-qwen2.5-3b`, budget 2,000, local `nomic-embed-text` v1.5 (Q8_0 GGUF, Hugging Face) for vectors, no neighbor chunks. Same 50 questions.

| arm | judge % | contains % | est. tokens | p50 s |
| --- | --- | --- | --- | --- |
| `v1` | 54.0 | 31.0 | 1,904 | 21.8 |
| `v2` | 44.0 | 26.2 | 1,981 | 26.4 |
| `full` | 54.0 | 40.5 | 11,937 | 307.7 |

- v2 is below v1: -10 judge, -4.8 contains. Most of the loss is temporal-reasoning (46.2 → 15.4) and multi-session (36.4 → 27.3). Sentence chunks lose the surrounding turn that v1's episode lines keep. One question errored.
- Retrieval itself is not the gap: on a dev subset disjoint from b50, the gold answer is in v2's workspace as often as or more often than in v1's.
- The strict gap to `full` is not closed. v2 stays off by default (`ANGULARAI_INDEX=off`).

## Results: cortex v2, second run

Wed 21 Oct 2026. Revision `c520315`. Same arm, model, budget and questions. Sentence chunks, picked on a 40-question dev subset disjoint from b50 over turn and episode chunks (dev: sentence 57.5, episode 55.0, episode plus word rank 52.5, v1 62.5). File: `results/b50/v2-c520315-3b-2000.judged.jsonl`.

| arm | judge % | contains % | est. tokens | p50 s |
| --- | --- | --- | --- | --- |
| `v1` | 54.0 | 31.0 | 1,904 | 21.8 |
| `v2` | 52.0 | 38.1 | 1,981 | 30.8 |
| `full` | 54.0 | 40.5 | 11,937 | 307.7 |

- v2 is -2.0 judge and +7.1 contains against v1. It is -2.0 judge and -2.4 contains against `full`, at a sixth of the tokens. No errors.
- Sentence chunks scored 44.0 / 26.2 in the first run, one revision earlier. A swing that size means 50 questions on a 3B model cannot separate v1 from v2 on judge.
- v2 does not beat v1. v2 stays off by default (`ANGULARAI_INDEX=off`).

## Kill rule

No "beats frontier" claim unless benchmark A shows at least +10 points over the small model alone.

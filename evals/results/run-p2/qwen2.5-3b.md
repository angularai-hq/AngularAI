# P2 smoke: angularai run

Code: feat/v2-run @ ad2080f (PR #22). Model: aa-qwen2.5-3b (qwen2.5:3b-instruct Q4, num_ctx 16384, temp 0) on local Ollama, CPU. Recall: v2 index (ANGULARAI_INDEX=on, aa-nomic-embed). $0, no paid key.

Run r_5543be3209c3, budget 2,000 tokens, max 40 steps, native tool calls. Task: replace the TODO docstring in 8 modules under src/, then write SUMMARY.md.

| phase | what | step after |
| --- | --- | --- |
| A | start, `--steps 12` | 12, paused |
| B | `--resume`, SIGKILL after 60 s, mid step 16 | 15 |
| C | `--resume` to the end | 40, budget |

- Steps completed: 40/40. steps.jsonl numbers 1..40 contiguous: the killed step was repeated once, nothing lost or doubled.
- Resume: worked after a clean pause and after kill -9.
- Tokens per step (workspace estimate, 4 chars = 1 token): mean 976, median 1,027, min 266, max 1,412. Over budget: 0. Total 39,045.
- Seconds per step: mean 19.0, median 20.0 (2.4 to 32.1 s; 3B on 6 CPU threads).
- Tools: list 12, read 7, edit 13, remember 8. Idle steps: 0.
- Task: not finished. The 3B model loops on src/alpha.py (edits it repeatedly, and broke its return line); it never reached beta.py. The loop held its window and state; the model is the limit.

First attempt (attempt1/, before the final code: 4,000 budget, 30 steps): 30/30, resume after pause and SIGKILL, mean 1,101 tokens/step, same alpha.py loop.

Files: smoke.sh, smoke.log, state.json, steps.jsonl, summary.json, summarize.py, fixture/ (the task tree), work/ (after the run).

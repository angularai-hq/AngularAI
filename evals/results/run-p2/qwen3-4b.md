# P2 smoke: angularai run with aa-qwen3-4b

Same task and harness as the aa-qwen2.5-3b smoke: 8 modules under src/, replace each TODO docstring, then write SUMMARY.md and finish. Budget 2,000 tokens, max 40 steps, v2 index on (aa-nomic-embed), native tool calls, local Ollama on CPU, $0. Each run: 12 steps, pause; resume with SIGKILL after 60 s; resume to the end.

| run | code | status | steps | files done | SUMMARY.md | tokens/step mean / median / max | s/step mean / median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run1 | ad2080f | budget | 40/40 | 8/8 | no | 881 / 919 / 1,267 | 45.7 / 43.4 |
| run2 | ef3b267 | done | 21/40 | 3/8 | yes, wrong | 668 / 718 / 999 | 7.0 / 6.9 |

- run1: steady read, edit, search TODO per file. 9 steps lost on src/zeta.py: `edit` missed on whitespace, "That text is not in the file." Out of steps before SUMMARY.md. Its seconds are inflated: an idle aa-qwen2.5-3b held an Ollama slot, so the chat and embedding models swapped every step (21 to 87 s).
- ef3b267 (general, tested): working memory lists the files read and changed; a repeat is the exact same call; a fourth step in six on one file is named; a missed `edit` quotes the closest line.
- run2: alpha, beta, gamma done; then it wrote SUMMARY.md for those three, with wrong descriptions for alpha and beta ("square"), and called finish. It stopped early: the FILES list may read as the whole set. One run each at temperature 0; not a measured regression or gain.
- Resume: worked in both runs after pause and SIGKILL (steps contiguous, 0 over budget). Also after a box reboot at about 12:50 mid run (reboot/: 2 steps resumed from step 2, before run2 was restarted clean).

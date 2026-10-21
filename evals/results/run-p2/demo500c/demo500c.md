# P2 demo500c: pinned summary batch, write append, syntax check

Code 91d6c95. Same 130-module task as `../demo500` and `../demo500b`. gpt-5.4-mini,
`--budget 2000 --max-steps 600 --max-cost 1.8`, v2 index on with local embeddings,
finish gate with `--summary SUMMARY.md --summary-of "src/*.py"`.

New in this run:
- A summary refusal pins its batch in working memory as ready-made summary lines,
  with the exact next call (`write`, `append: true`), until SUMMARY.md changes.
- `write` takes `append`.
- An edit or write to a `.py` file reports its first syntax error.

The smoke (8 modules, `--max-steps 60`, `--max-cost 0.15`) had all 8 modules correct
by step 26. It then looped list/read to step 60 without calling finish, so it never
reached the gate. The main run went ahead.

## Result

| | |
|---|---|
| Steps | 453 of 600; status `done` |
| Modules done / correct | 130 / 130; the last module edit was at step 395 |
| SUMMARY.md | 130 listed, **130 correct** (and the generic summary check finds no problem) |
| Finish gate | header written at 431, then 10 rounds of refuse (13 lines pinned) → append; finish check at 452, done at 453 |
| Syntax errors reported | 0 (none made) |
| Tokens per step | mean 1,942, median 1,983, max 1,992; 0 over 2,000 |
| Window | held |
| Time | 17:13:48 to 17:23:32 CEST, 9.7 min; 1.3 s a step |

Resume:
- The pause at step 150 resumed cleanly.
- SIGKILL 150 s into the resume left the checkpoint at step 261. The next resume went
  on from 261 and ran to the end.
- Steps 1–453 are contiguous, with none lost or repeated.

By third:

| Steps | Module edits | Mean tokens | Cost |
|---|---|---|---|
| 1–151 | 49 | 1,859 | $0.3426 |
| 152–302 | 50 | 1,983 | $0.3665 |
| 303–453 | 32 (to step 395) | 1,982 | $0.3523 |

## Spend

| Run | Steps | Input | Cached | Output | Cost |
|---|---|---|---|---|---|
| smoke | 60 | 139,723 | 0 | 5,356 | $0.128894 |
| demo | 453 | 1,202,001 | 0 | 35,554 | $1.061494 |
| **This work** | | 1,341,724 | 0 | 40,910 | **$1.190388** (exactly $1.19038800) |

Priced from each response's `usage`: $0.75 input, $0.075 cached, $4.50 output per 1M
tokens. The call in flight at the SIGKILL was not recorded; one step is about $0.0024.
Running total of the approved $10: $2.805487 before, plus $1.190388, makes $3.995875.

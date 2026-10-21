# P2 demo500: gpt-5.4-mini, 130 modules + SUMMARY.md, finish gate on

Code 81f5e91 (finish gate on work left, final summary check). gpt-5.4-mini, paid,
`--budget 2000 --max-steps 500 --max-cost 3`, v2 index on with local embeddings
(aa-nomic-embed), markers TODO,FIXME counted in full, `--summary SUMMARY.md
--summary-of "src/*.py"`, `--finish-refusals 5`. One run.

Task: 130 one-function modules (`make_fixture.py`, seed 500), each `x * k` or `x + k`
with a TODO docstring. Describe each in its docstring and keep SUMMARY.md listing
every module with its description, changed last. Why 130: the 12-step smoke on the
150-module fixture (`spec-150.json`) took 3.3 steps a module, so 150 modules would need
about 550 steps with the summary, more than 500. Scored by `score.py` (ast docstrings;
"doubles x" counts for `x * 2`, "incremented" for `+`).

## Result

| | |
|---|---|
| Steps | 466 of 500; status `stalled` (fifth refused finish in a row, no change between) |
| Modules done / correct | 130 / 130; the last module edit was at step 406 |
| SUMMARY.md | 130 listed, 1 correct: the other 129 lines read `[description]` |
| Finish gate | 8 finishes, all refused (steps 417–466): no marker left, but the summary "is not specific about 129" |
| Tokens per step | mean 1,945, median 1,984, max 1,995; 0 over 2,000 |
| Window | held: full by step ~20 and steady to the end |
| Time | 16:08:10 to 16:19:08 CEST, 11.0 min; 1.4 s a step |

Resume:
- The pause at step 150 (`--steps 150`) resumed from 150 cleanly.
- SIGKILL 150 s into the resume left the checkpoint at step 252. The next resume went
  on from 252.
- One OpenAI call at step 461 came back empty ("returned no reply"). The driver
  resumed from 461.
- Steps 1–466 are contiguous, with none lost or repeated.

By third:

| Steps | Module edits | Mean tokens | Cost | s/step |
|---|---|---|---|---|
| 1–155 | 47 | 1,866 | $0.3745 | 1.37 |
| 156–310 | 51 | 1,982 | $0.3980 | 1.42 |
| 311–466 | 32 (to step 406) | 1,984 | $0.3708 | 1.28 |

Module quality did not drop: 3.0 steps a module in both halves, and every docstring is
correct in both halves.

## Spend

| Run | Steps | Input | Cached | Output | Cost |
|---|---|---|---|---|---|
| smoke (150-module fixture, `--max-cost 0.15`) | 12 | 18,083 | 0 | 982 | $0.017981 |
| demo (`--max-cost 3`) | 466 | 1,227,914 | 0 | 49,858 | $1.145296 |
| **This work** | | 1,245,997 | 0 | 50,840 | **$1.163278** |

Priced from each response's `usage`: $0.75 input, $0.075 cached, $4.50 output per 1M
tokens. The call in flight at the SIGKILL was not recorded; one step is about $0.0024.
Running total of the approved $10: $0.457874 before, plus $1.163278 (exactly $1.16327775), makes $1.621152.

## What it shows

The gate did its job. It refused a finish while the summary was a template, so the run
did not end as `done` with a wrong summary. It stalled after five refusals in a row
instead.

The model did not do the summary part. It ignored "add lines as you go", and at step
416 wrote a 130-line skeleton with `[description]`. With a 2,000-token window it could
not recall 129 descriptions, and `search` shows 30 hits at most, so it never filled the
skeleton in. The task needs a way to gather what was written. Two options:
- the refusal quotes each vague module's own docstring;
- ask for the summary line in the same step as the docstring edit.

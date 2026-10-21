# P2 long run: gpt-5.4-mini, 46 modules + SUMMARY.md

Paid run, approved 2026-10-21 15:14 CEST with a $10 hard cap. Same task, fixture,
scorer and window as `../long`: `--budget 2000`, v2 index on with local embeddings
(aa-nomic-embed on Ollama), finish check on (2 checks), files view `list`,
`--max-steps 500 --max-cost 8`. Default temperature (the API's). The key came from
`OPENAI_API_KEY_ANGULARAI` into the driver's process only; logs went through a
redaction filter and were scanned before commit.

Prices, OpenAI standard tier, read 2026-10-21 from platform.openai.com/docs/pricing:
gpt-5.4-mini $0.75 input, $0.075 cached input, $4.50 output per 1M tokens.

## Cost

| Run | Steps | Input tokens | Cached | Output tokens | Cost |
|---|---|---|---|---|---|
| smoke (`--max-cost 0.25`) | 6 | 6,523 | 0 | 878 | $0.008843 |
| long (`--max-cost 8`) | 169 | 419,919 | 0 | 29,799 | $0.449031 |
| **Total billed** | | 426,442 | 0 | 30,677 | **$0.457874** |

Spend is priced from the `usage` block of every response. The guard's own ledger
reads $0.511242: four failed resumes at step 164 (below) were charged the worst case,
$0.013341 each, though they failed before any call. 848d165 charges a failed step
only what it was billed. Of the $10 cap, $0.46 was spent and $9.54 is left.

## Result

| | |
|---|---|
| Steps | 169 of 500; status `done` (third finish, after checks at 80 and 168) |
| Files done / correct | 46 / 45 (quill's "Triple the input value." is right, but the strict scorer misses it); 46 / 46 by reading |
| SUMMARY.md | 25 modules listed, 0 correct: vague lines ("Applies a small arithmetic adjustment to the input.") |
| Tokens per step | mean 1,845, median 1,976, max 1,988; 0 over 2,000 |
| Window | held; it fills by step ~60 and stays at ~1,978 |
| Time | 1.5 s a step, 4.3 min of steps (15:36:09 to 15:43:49 CEST with the fix) |

By third:

| Steps | Changes | Mean tokens | Cost |
|---|---|---|---|
| 1–56 | 18 | 1,576 | $0.1213 |
| 57–112 | 17 | 1,978 | $0.1678 |
| 113–169 | 14 | 1,978 | $0.1599 |

Quality held through the run: 3.2 then 3.3 steps a file, docstrings correct in both
halves (1.00, 0.96 strict). The weak part is SUMMARY.md. The model wrote it early with
generic lines and spent its last steps rewriting it with long edits.

## Failure and resume

At step 165, the previous step was an edit of SUMMARY.md with 3,218 characters of
arguments. The native assistant/tool exchange alone outgrew half the window, and
every resume failed with "That input does not fit the workspace" (4 tries, no call
made). Fix in 848d165: a long last call's text fields are clipped, still as valid
JSON, and the exchange falls back to the inline record if that is not enough. There
is a test. The same run resumed from 164 and finished; it did not start over.

Files: `run.sh` (driver), `run.log`, `score-{smoke,long}.json`,
`state-{smoke,long}.json`, `steps-{smoke,long}.jsonl` (per-step `cost` and
`total_cost`), `SUMMARY.out.md`.

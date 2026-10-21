# P2 demo500b: rerun with summary quotes

Code 67a0167. This is the same 130-module task as `../demo500`, with gpt-5.4-mini,
`--budget 2000 --max-cost 2`, v2 index on with local embeddings, and the finish gate
(`--summary SUMMARY.md --summary-of "src/*.py"`, 5 refusals). New in this run: when
the summary check refuses, the refusal quotes the docstrings of the next batch of
modules the summary misses or is vague about. The batch is 13 at this budget,
`quote_batch` = budget // 150, from 3 to 15. `--search-hits` is now configurable; this
run kept the default of 30.

The smoke ran 8 modules to the end. It took 50 steps for the modules, at 6.3 a module,
then one refusal with quotes, one write of the summary, and done at step 54. The
summary was 8/8 correct. Its per-module rate was above the 3.0 of demo500, so the main
run got `--max-steps 600`.

## Result

| | |
|---|---|
| Steps | 452 of 600; status `stalled`, after 5 refused finishes in a row with no change |
| Modules done / correct | 130 / 130; the last module edit was at step 398 |
| SUMMARY.md | a header only (written at step 403); 0 of 130 lines |
| Finish gate | refused at steps 404, 418, 423, 436 and 452 |
| Tokens per step | mean 1,944, median 1,985, max 1,994; 0 over 2,000 |
| Window | held: full by step ~20 and steady |
| Time | 16:45:17 to 16:55:05 CEST, 9.8 min; 1.3 s a step |

Resume:
- The pause at step 150 resumed cleanly.
- SIGKILL 150 s into the resume left step 267. The next resume went on from 267.
- Steps 1–452 are contiguous, with none lost or repeated.

By third:

| Steps | Module edits | Mean tokens | Cost |
|---|---|---|---|
| 1–150 | 47 | 1,864 | $0.3543 |
| 151–300 | 50 | 1,983 | $0.3577 |
| 301–452 | 33 (to step 398) | 1,984 | $0.3507 |

## Why the summary failed again

The refusal fit the window: about 320 tokens, with all 13 quotes visible in working
memory (checked by rebuilding the step's input). The model did not act on the quotes.
After each refusal it read SUMMARY.md or listed src/. That replaced the refusal as the
last output, so the quotes left the window. It went on alternating read and list, and
made no edit between refusals.

In the smoke, the 8 quotes were enough for one write. With 130 modules, the model
would need to edit right after each refusal, and it did not.

Next fix, generic: keep the pending batch pinned in working memory (like the note)
until the summary changes, instead of only in the last output. Also, the refusal could
name the exact call to make: an edit appending the batch.

The smoke also showed a model edit that left a broken docstring (`"""..."".`, a syntax
error) that no check caught. A parse check after editing a `.py` file would be cheap.

## Spend

| Run | Steps | Input | Cached | Output | Cost |
|---|---|---|---|---|---|
| smoke (8 modules, `--max-cost 0.15`) | 54 | 123,483 | 0 | 6,447 | $0.121624 |
| demo (`--max-cost 2`) | 452 | 1,189,326 | 0 | 37,937 | $1.062711 |
| **This work** | | 1,312,809 | 0 | 44,384 | **$1.184335** (exactly $1.18433475) |

Priced from each response's `usage`: $0.75 input, $0.075 cached, $4.50 output per 1M
tokens. The call in flight at the SIGKILL was not recorded; one step is about $0.0024.
Running total of the approved $10: $1.621152 before, plus $1.184335, makes $2.805487.

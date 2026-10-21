# P2 finish check and change recall

These runs use 323255d with its defaults: `--files list --finish-check on --change-recall on`. The baseline is 856ad6d, using the `list` rows from files-view.md (run at fe0ad3c, which has the same code). Same smoke task, budget 2,000 tokens, max 40 steps, v2 index, native tool calls, local Ollama on CPU, $0. Each setup runs once with aa-qwen3-4b at temperature 0 (t0), and once each with copies at temperature 0.7 and seeds 1 to 3. One chat model was loaded at a time. Scored by check.py.

| code | run | files done (correct) | SUMMARY.md lines correct of 8 | steps | finish checks | end |
| --- | --- | --- | --- | --- | --- | --- |
| 856ad6d | t0 | 8 (8) | no file | 28 | - | done |
| 856ad6d | s1 | 1 (1) | no file | 4 | - | done |
| 856ad6d | s2 | 8 (8) | 2 | 26 | - | done |
| 856ad6d | s3 | 8 (8) | no file | 33 | - | done |
| 323255d | t0 | 7 (7) | 5 | 40 | 1 | budget |
| 323255d | s1 | 8 (8) | 8 | 29 | 3 | done |
| 323255d | s2 | 8 (8) | 8 | 40 | 3 | done |
| 323255d | s3 | 7 (7) | 5 | 40 | 3 | budget |

| code | mean files | runs with 8/8 | SUMMARY.md written | SUMMARY.md 8/8 correct | mean SUMMARY lines correct | mean steps |
| --- | --- | --- | --- | --- | --- | --- |
| 856ad6d | 6.25 | 3/4 | 1/4 | 0/4 | 0.5 | 22.8 |
| 323255d | 7.50 | 2/4 | 4/4 | 2/4 | 6.5 | 37.3 |

- The finish check stopped the early finish (s1 at 856ad6d: 4 steps, 1 file). Each first `finish` became a check that listed the remaining TODO lines, and the model went back to work. The check re-arms after a change, so most runs got 3 checks.
- Change recall: SUMMARY.md descriptions now come from the model's own edits, recalled from the log. Lines correct went from 0.5 to 6.5 of 8; no "square" or "factorial".
- Cost: more steps. Two runs hit the 40-step limit with 7 of 8 files done.
- n=4 per setup.

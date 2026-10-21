# P2 files view: list vs partial vs off

`angularai run --files list|partial|off` at fe0ad3c. Same smoke task (8 modules, replace each TODO docstring, write SUMMARY.md, finish), budget 2,000 tokens, max 40 steps, v2 index on, native tool calls, local Ollama on CPU, $0. All three variants include the closest-line hint on a failed `edit`. Four runs per variant: aa-qwen3-4b at temperature 0 (t0), and copies at temperature 0.7 with seeds 1 to 3 (s1 to s3; same weights). One chat model was loaded at a time. Scored by check.py: a file counts as done when the TODO is gone and the code is intact, and as correct when the docstring names its multiplier. A SUMMARY.md line is correct when it names the module's multiplier.

- list: FILES (read, changed). This is the ef3b267 behaviour.
- partial: the same list, labelled "FILES TOUCHED SO FAR ... Not the full set".
- off: no list (edit hint only).

| variant | run | files done (correct) | SUMMARY.md (lines correct of 8) | steps | end |
| --- | --- | --- | --- | --- | --- |
| list | t0 | 8 (8) | no | 28 | done |
| list | s1 | 1 (1) | no | 4 | done |
| list | s2 | 8 (8) | yes (2) | 26 | done |
| list | s3 | 8 (8) | no | 33 | done |
| partial | t0 | 8 (8) | yes (1) | 31 | done |
| partial | s1 | 0 (0) | no | 3 | done |
| partial | s2 | 7 (7) | yes (1) | 40 | budget |
| partial | s3 | 7 (7) | yes (1) | 26 | done |
| off | t0 | 6 (6) | yes (1) | 33 | done |
| off | s1 | 5 (5) | yes (3) | 28 | done |
| off | s2 | 8 (8) | no | 32 | done |
| off | s3 | 4 (4) | yes (1) | 20 | done |

| variant | mean files | runs with 8/8 | SUMMARY.md written | SUMMARY.md correct | mean steps |
| --- | --- | --- | --- | --- | --- |
| list | 6.25 | 3/4 | 1/4 | 0/4 | 22.8 |
| partial | 5.50 | 1/4 | 3/4 | 0/4 | 25.0 |
| off | 5.75 | 1/4 | 3/4 | 0/4 | 28.3 |

- Kept: list, the default. It finished the most files, and had the most complete runs.
- Labelling the list, or dropping it, gets SUMMARY.md written more often, but no variant got SUMMARY.md right. The model writes it from memory and invents descriptions ("square", "factorial") for files whose output has left the window.
- The main failure is finishing early: a call to `finish` after 3 or 4 steps (s1), or before every file is done.
- n=4 per variant. The differences are within noise.
- Ollama: the box has no cron or systemd (PID 1 is tini), so nothing starts at boot. /workspace/angularai/v2/ollama-up.sh starts the server if it does not answer; the run scripts call it first. After a reboot, run it by hand.

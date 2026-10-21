# P2 long run: aa-qwen3-4b, 46 modules + SUMMARY.md

Code d8853ac. Local Ollama on CPU, $0. aa-qwen3-4b at temperature 0, seed 0;
`--max-steps 300 --budget 2000`, v2 index on (aa-nomic-embed), finish check on with
2 checks, change recall on, files view `list`. One run, not repeated.

Task: 46 modules in `src/` (`make_fixture.py`, seed 13), each `return x * k` or
`x + k` with a TODO docstring. Replace every docstring with a one-line description,
then write SUMMARY.md listing every module. Expected cost: about 3 steps a module, so
150–300 steps. Scored by `score_long.py`: a file is done when its docstring has no
TODO and its code is intact; correct when the docstring names the operation and k.

## Result

| | |
|---|---|
| Steps | 71 of 300; status `done` (the model's third finish) |
| Files done / correct | 21 / 21 of 46 |
| SUMMARY.md | written at step 7 with 2 lines, both correct; never updated |
| Tokens per step | mean 1,428, median 1,565, max 1,978; 0 over 2,000 |
| Window | held: every step fits; the window fills to ~1,850 and stays |
| Time | 14:56:47 to 15:25:06 CEST, 28.3 min; 20.4 s a step on average |
| Finish checks | at steps 8 and 41, both premature; the third finish (step 71) ended the run |

Resume:
- Pause at step 60 (`--steps 60`), resume from 60: clean.
- SIGKILL 300 s into the resume (during step 70): the checkpoint held step 69,
  the lock released with the process, and the next resume went on from 69 with no
  lost or repeated step and contiguous step numbers (1–71).

By third of the run:

| Steps | Changes | Mean tokens | Mean s/step |
|---|---|---|---|
| 1–24 | 7 | 874 | 8.2 |
| 25–48 | 8 | 1,576 | 23.2 |
| 49–71 | 7 | 1,850 | 30.1 |

Quality did not drop: all 21 docstrings are correct, 3.1 then 3.2 steps a file
across the two halves, and the read/edit/search rhythm at step 70 is the one at
step 10. Steps got slower as the window filled (CPU prompt processing), not worse.
Late docstrings switched from `"""` to `"` quoting (6 files from step 52 on); still
valid docstrings. The scorer first missed them and was fixed to read docstrings with
`ast`.

## What went wrong

The run did not need 150 steps; it needed the model to keep going. It stopped at 21
of 46 because a capped finish check ends the run on the third finish, and the model
believed it was done each time: the check named the task and the files changed, but
a `search TODO` limited to 20 hits and a FILES view of the newest 16 do not show 25
untouched modules. SUMMARY.md was written once, early, and never revisited.

## For the 13 Nov 500-step demo

- Don't let a finish end the run on a count alone. Make the check concrete (count of
  remaining TODOs or unvisited files in the root) and end on finish only when the
  check finds nothing, or set `--finish-checks` high (10–20) for long tasks.
- Ask for SUMMARY.md last, or have the check verify it lists every module.
- On CPU, plan ~30 s a step once the window is full: 500 steps is ~4 h. Keep
  `--budget 2000`; the window held at 1,978 max.
- Pause/resume and SIGKILL/resume are safe to show live.

Files: `long.sh` (driver), `make_fixture.py`, `spec.json`, `score_long.py`,
`score.json`, `state.json`, `steps.jsonl`, `long.log`, `SUMMARY.out.md` (the model's
SUMMARY.md).

#!/usr/bin/env bash
# P2 smoke: angularai run, 40 steps at a 2,000-token window with aa-qwen2.5-3b on local Ollama, paused,
# hard-killed and resumed. $0: no paid key; ANGULARAI_API_KEY is a placeholder.
set -u
cd /workspace/angularai/v2/runs/p2
. ./env.sh
rm -rf home work && cp -r fixture work
AA="uv run --project /workspace/aa-v2-run angularai"
TASK="For each module in src/, read it and replace its TODO docstring with a one-line description of what the function does. Then write SUMMARY.md listing every module with its description. Finish when all 8 modules are done."
$AA init && $AA wire compatible --model aa-qwen2.5-3b --base-url http://127.0.0.1:11434/v1
echo "== $(date '+%F %T %Z') A: start, 12 steps then pause"
$AA run --root "$PWD/work" --max-steps 40 --steps 12 --budget 2000 "$TASK"
ID=$(ls home/runs | head -1)
echo "== run $ID after A: $(python3 -c "import json;s=json.load(open('home/runs/$ID/state.json'));print(s['step'],s['status'])")"
echo "== $(date '+%F %T %Z') B: resume, SIGKILL after 60 s"
timeout -s KILL 60 $AA run --resume "$ID"
echo "== exit $? after B: $(python3 -c "import json;s=json.load(open('home/runs/$ID/state.json'));print(s['step'],s['status'])")"
echo "== $(date '+%F %T %Z') C: resume to the end"
$AA run --resume "$ID"
echo "== after C: $(python3 -c "import json;s=json.load(open('home/runs/$ID/state.json'));print(s['step'],s['status'])")"
python3 summarize.py "home/runs/$ID" > summary.json
cp "home/runs/$ID/state.json" "home/runs/$ID/steps.jsonl" .
echo "== $(date '+%F %T %Z') DONE"

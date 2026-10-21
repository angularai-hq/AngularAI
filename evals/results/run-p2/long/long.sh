#!/usr/bin/env bash
# P2 long run: 46 modules + SUMMARY.md, aa-qwen3-4b at temperature 0, --max-steps 300,
# 2,000-token window, v2 index on. Pause at 60, SIGKILL 5 min into the resume, resume
# to the end. $0, local only; ANGULARAI_API_KEY is a placeholder.
set -u
L=/workspace/angularai/v2/runs/p2/long
cd $L
/workspace/angularai/v2/ollama-up.sh || exit 1
. /workspace/angularai/v2/runs/p2/env.sh
export ANGULARAI_HOME=$L/home
AA="uv run --project /workspace/aa-v2-run angularai"
TASK="For each module in src/, read it and replace its TODO docstring with a one-line description of what the function does. Then write SUMMARY.md listing every module with its description. Finish when all modules are done."
for m in $(curl -s http://127.0.0.1:11434/api/ps | python3 -c "import json,sys;[print(m['name']) for m in json.load(sys.stdin)['models'] if 'embed' not in m['name']]"); do
  [ "$m" = "aa-qwen3-4b:latest" ] || curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
done
st() { python3 -c "import json;s=json.load(open('home/runs/$ID/state.json'));print(s['step'],s['status'])"; }
if [ ! -d home/runs ]; then
  rm -rf work && cp -r fixture work
  $AA init && $AA wire compatible --model aa-qwen3-4b --base-url http://127.0.0.1:11434/v1
  echo "== $(date '+%F %T %Z') code $(git -C /workspace/aa-v2-run rev-parse --short HEAD) A: start, pause at 60"
  $AA run --root "$L/work" --max-steps 300 --steps 60 --budget 2000 "$TASK"
  ID=$(ls home/runs | head -1)
  echo "== $(date '+%F %T %Z') after A: $(st)"
  if [ "$(st | cut -d' ' -f2)" = running ]; then
    echo "== $(date '+%F %T %Z') B: resume, SIGKILL after 300 s"
    timeout -s KILL 300 $AA run --resume "$ID"
    echo "== $(date '+%F %T %Z') exit $? after B: $(st)"
  fi
fi
ID=$(ls home/runs | head -1)
while [ "$(st | cut -d' ' -f2)" = running ]; do
  echo "== $(date '+%F %T %Z') C: resume from $(st)"
  $AA run --resume "$ID" || { /workspace/angularai/v2/ollama-up.sh; sleep 5; }
done
echo "== $(date '+%F %T %Z') after C: $(st)"
python3 score_long.py work home/runs/$ID spec.json > score.json
cp home/runs/$ID/state.json home/runs/$ID/steps.jsonl .
echo "== $(date '+%F %T %Z') DONE"

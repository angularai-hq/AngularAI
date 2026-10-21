#!/usr/bin/env bash
# P2 demo500: gpt-5.4-mini on 150 modules + SUMMARY.md. --budget 2000, --max-steps 500,
# --max-cost 3, v2 index on with local embeddings, the finish gate on work left and a
# final summary check. Pause at 150, SIGKILL 150 s into the resume, resume to the end.
# The key comes from OPENAI_API_KEY_ANGULARAI into this process only; never echoed,
# written or logged. No set -x.
set -u
L=/workspace/angularai/v2/runs/p2/demo500
cd $L
/workspace/angularai/v2/ollama-up.sh || exit 1
. /workspace/angularai/v2/runs/p2/env.sh
[ -n "${OPENAI_API_KEY_ANGULARAI:-}" ] || { echo "no key"; exit 1; }
export OPENAI_API_KEY="$OPENAI_API_KEY_ANGULARAI"
unset ANGULARAI_API_KEY
AA="uv run --project /workspace/aa-v2-run angularai"
TASK="For each module in src/, read it and replace its TODO docstring with a one-line description of what the function does. Keep SUMMARY.md listing every module (by name) with its description: add lines as you go, and make SUMMARY.md the last file you change. Finish when every module is done."
GATE=(--summary SUMMARY.md --summary-of "src/*.py")
spent() { python3 - <<'PY'
import json, pathlib
print(f'{sum(json.loads(p.read_text()).get("cost", 0.0) for p in pathlib.Path(".").glob("home-*/runs/*/state.json")):.6f}')
PY
}
st() { python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print(s['step'],s['status'],round(s.get('cost',0),6))" "$1"; }
fresh() {  # fresh NAME: a new home and a copy of the fixture
  export ANGULARAI_HOME=$L/home-$1
  rm -rf work-$1 && cp -r fixture work-$1
  $AA init && $AA wire openai --model gpt-5.4-mini
}
finish() {  # finish NAME ID
  local s=$L/home-$1/runs/$2
  echo "== $(date '+%F %T %Z') $1: $(st $s/state.json); spent here \$$(spent)"
  python3 score.py work-$1 $s $([ $1 = smoke ] && echo spec-150.json || echo spec.json) > score-$1.json
  cp $s/state.json state-$1.json; cp $s/steps.jsonl steps-$1.jsonl; cp work-$1/SUMMARY.md summary-$1.md 2>/dev/null
}
case "${1:-}" in
smoke)
  fresh smoke
  echo "== $(date '+%F %T %Z') smoke: 12 steps, max-cost 0.15"
  $AA run --root "$L/work-smoke" --max-steps 12 --budget 2000 --max-cost 0.15 "${GATE[@]}" "$TASK"
  finish smoke $(ls home-smoke/runs | head -1) ;;
demo)
  export ANGULARAI_HOME=$L/home-demo
  if [ ! -d home-demo/runs ]; then
    fresh demo
    echo "== $(date '+%F %T %Z') code $(git -C /workspace/aa-v2-run rev-parse --short HEAD) A: start, pause at 150"
    $AA run --root "$L/work-demo" --max-steps 500 --steps 150 --budget 2000 --max-cost 3 "${GATE[@]}" "$TASK"
    ID=$(ls home-demo/runs | head -1)
    echo "== $(date '+%F %T %Z') after A: $(st home-demo/runs/$ID/state.json)"
    if [ "$(st home-demo/runs/$ID/state.json | cut -d' ' -f2)" = running ]; then
      echo "== $(date '+%F %T %Z') B: resume, SIGKILL after 150 s"
      timeout -s KILL 150 $AA run --resume "$ID"
      echo "== $(date '+%F %T %Z') exit $? after B: $(st home-demo/runs/$ID/state.json)"
    fi
  fi
  ID=$(ls home-demo/runs | head -1)
  tries=0
  while [ "$(st home-demo/runs/$ID/state.json | cut -d' ' -f2)" = running ] && [ $tries -lt 3 ]; do
    tries=$((tries+1))
    echo "== $(date '+%F %T %Z') C: resume $tries from $(st home-demo/runs/$ID/state.json)"
    $AA run --resume "$ID" || { /workspace/angularai/v2/ollama-up.sh; sleep 20; }
  done
  finish demo $ID ;;
*) echo "run.sh smoke|demo"; exit 2 ;;
esac
echo "== DONE"

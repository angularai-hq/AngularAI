#!/usr/bin/env bash
# P2 long run on gpt-5.4-mini (paid; approved 2026-10-21 15:14 CEST, $10 hard cap).
# Same task and fixture as ../long, 2,000-token window, v2 index on with local
# embeddings, --max-steps 500, --max-cost 8. A 6-step smoke (--max-cost 0.25) first.
# The key comes from OPENAI_API_KEY_ANGULARAI into this process only; it is never
# echoed, written, or logged. No set -x.
set -u
L=/workspace/angularai/v2/runs/p2/long-openai
cd $L
/workspace/angularai/v2/ollama-up.sh || exit 1
. /workspace/angularai/v2/runs/p2/env.sh
[ -n "${OPENAI_API_KEY_ANGULARAI:-}" ] || { echo "no key"; exit 1; }
export OPENAI_API_KEY="$OPENAI_API_KEY_ANGULARAI"
unset ANGULARAI_API_KEY
AA="uv run --project /workspace/aa-v2-run angularai"
TASK="For each module in src/, read it and replace its TODO docstring with a one-line description of what the function does. Then write SUMMARY.md listing every module with its description. Finish when all modules are done."
CAP=10
spent() { python3 - "$@" <<'PY'
import json, pathlib
total = 0.0
for state in pathlib.Path(".").glob("home*/runs/*/state.json"):
    total += json.loads(state.read_text()).get("cost", 0.0)
print(f"{total:.6f}")
PY
}
st() { python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print(s['step'],s['status'],round(s.get('cost',0),6))" "$1"; }
phase() {  # phase NAME STEPS MAXSTEPS MAXCOST
  local name=$1 home=$L/home-$1
  export ANGULARAI_HOME=$home
  if [ ! -d $home/runs ]; then
    rm -rf work-$name && cp -r ../long/fixture work-$name
    $AA init && $AA wire openai --model gpt-5.4-mini
    echo "== $(date '+%F %T %Z') $name: start, max-steps $3, max-cost $4, spent so far \$$(spent)"
    $AA run --root "$L/work-$name" --max-steps $3 --steps $2 --budget 2000 --max-cost $4 "$TASK"
  fi
  local id; id=$(ls $home/runs | head -1)
  local tries=0
  while [ "$(st $home/runs/$id/state.json | cut -d' ' -f2)" = running ] && [ $2 -ge $3 ] && [ $tries -lt 3 ]; do
    tries=$((tries+1))
    echo "== $(date '+%F %T %Z') $name: resume $tries from $(st $home/runs/$id/state.json)"
    $AA run --resume "$id" || { /workspace/angularai/v2/ollama-up.sh; sleep 20; }
  done
  echo "== $(date '+%F %T %Z') $name: $(st $home/runs/$id/state.json); total spent \$$(spent)"
  python3 ../long/score_long.py work-$name $home/runs/$id ../long/spec.json > score-$name.json
  cp $home/runs/$id/state.json state-$name.json; cp $home/runs/$id/steps.jsonl steps-$name.jsonl
}
case "${1:-}" in
  smoke) phase smoke 6 6 0.25 ;;
  long)
    python3 -c "import json,sys;sys.exit(0 if json.load(open('state-smoke.json'))['cost'] < 0.25 else 1)" || { echo "== smoke over budget; stop"; exit 1; }
    phase long 500 500 8 ;;
  *) echo "run.sh smoke|long"; exit 2 ;;
esac
echo "== $(date '+%F %T %Z') total spent \$$(spent) of \$$CAP"
echo "== DONE"

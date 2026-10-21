#!/usr/bin/env bash
# P2 files-view comparison: --files list|partial|off x aa-qwen3-4b at temperature 0 and
# three seeded copies at temperature 0.7 (aa-qwen3-4b-s1..s3, same weights). One chat
# model loaded at a time, plus the embedding model for the v2 index. $0, local only.
set -u
V=/workspace/angularai/v2/runs/p2/variants
/workspace/angularai/v2/ollama-up.sh || exit 1
. /workspace/angularai/v2/runs/p2/env.sh
AA="uv run --project /workspace/aa-v2-run angularai"
TASK="For each module in src/, read it and replace its TODO docstring with a one-line description of what the function does. Then write SUMMARY.md listing every module with its description. Finish when all 8 modules are done."
echo "== $(date '+%F %T %Z') code $(git -C /workspace/aa-v2-run rev-parse --short HEAD)"
for rep in t0 s1 s2 s3; do
  model=aa-qwen3-4b; [ $rep = t0 ] || model=aa-qwen3-4b-$rep
  for view in list partial off; do
    out=$V/$view-$rep
    [ -f $out/score.json ] && { echo "== skip $view-$rep"; continue; }
    rm -rf $out && mkdir -p $out && cp -r /workspace/angularai/v2/runs/p2/fixture $out/work
    for m in $(curl -s http://127.0.0.1:11434/api/ps | python3 -c "import json,sys;[print(m['name']) for m in json.load(sys.stdin)['models'] if 'embed' not in m['name']]"); do
      [ "$m" = "$model:latest" ] || curl -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
    done
    export ANGULARAI_HOME=$out/home
    $AA init >/dev/null && $AA wire compatible --model $model --base-url http://127.0.0.1:11434/v1 >/dev/null
    echo "== $(date '+%F %T %Z') $view-$rep ($model)"
    $AA run --root $out/work --max-steps 40 --budget 2000 --files $view "$TASK" > $out/run.log 2>&1
    id=$(ls $out/home/runs | head -1)
    python3 $V/check.py $out/work $out/home/runs/$id > $out/score.json
    python3 -c "import json;d=json.load(open('$out/score.json'));d.update(view='$view',rep='$rep');print(json.dumps(d))" | tee -a $V/results.jsonl
  done
done
echo "== $(date '+%F %T %Z') DONE"

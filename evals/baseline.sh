#!/usr/bin/env bash
# Benchmark A, v1 baseline. Local models only. No paid key reaches any step.
#   evals/baseline.sh [count] [out]
set -euo pipefail
unset OPENAI_API_KEY ANTHROPIC_API_KEY ANGULARAI_API_KEY XAI_API_KEY ANGULARAI_EMBEDDINGS
COUNT=${1:-50}
OUT=${2:-evals/runs/b$COUNT}
SMALL=${SMALL:-aa-qwen2.5-3b}
JUDGE=${JUDGE:-aa-qwen3-4b}
BUDGET=${BUDGET:-2000}
E="uv run angularai-eval"
S=$OUT/subset.json
mkdir -p "$OUT"
$E reduce --count "$COUNT" --seed 0 --haystack-tokens 12000 --out "$S"
for arm in truncated v1; do
  $E run --subset "$S" --arm $arm --model "$SMALL" --budget "$BUDGET" --out "$OUT/$arm.jsonl"
done
$E run --subset "$S" --arm full --model "$SMALL" --out "$OUT/full.jsonl"
for arm in truncated v1 full; do
  $E judge --subset "$S" --predictions "$OUT/$arm.jsonl" --model "$JUDGE" --out "$OUT/$arm.judged.jsonl"
done
$E report --subset "$S" --judged "$OUT"/truncated.judged.jsonl "$OUT"/v1.judged.jsonl \
  "$OUT"/full.judged.jsonl --out "$OUT/results.json" --markdown "$OUT/results.md"

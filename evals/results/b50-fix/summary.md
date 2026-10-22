# b50, placeholder-reply fix: repeats

LongMemEval-S b50, aa-qwen2.5-3b, 2,000-token budget, judge aa-qwen3-4b, local CPU, $0.

| arm | runs | judge % per run | judge mean | spread (max-min) | contains mean |
| --- | --- | --- | --- | --- | --- |
| v1 | 1 | 48.0 | 48.0 | 0.0 | 31.0 |
| v2 | 1 | 54.0 | 54.0 | 0.0 | 38.1 |

v2 - v1 judge mean: 6.0; largest per-arm spread: 0.0; v2 wins beyond spread: False.

# b50, placeholder-reply fix: repeats

LongMemEval-S b50, aa-qwen2.5-3b, 2,000-token budget, judge aa-qwen3-4b, local CPU, $0.

| arm | runs | judge % per run | judge mean | spread (max-min) | contains mean |
| --- | --- | --- | --- | --- | --- |
| v1 | 2 | 48.0, 52.0 | 50.0 | 4.0 | 32.1 |
| v2 | 2 | 54.0, 56.0 | 55.0 | 2.0 | 36.9 |

v2 - v1 judge mean: 5.0; largest per-arm spread: 4.0; v2 wins beyond spread: True.

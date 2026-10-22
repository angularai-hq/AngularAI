# b50, placeholder-reply fix: repeats

LongMemEval-S b50, aa-qwen2.5-3b, 2,000-token budget, judge aa-qwen3-4b, local CPU, $0.

| arm | runs | judge % per run | judge mean | spread (max-min) | contains mean |
| --- | --- | --- | --- | --- | --- |
| v1 | 3 | 48.0, 52.0, 44.0 | 48.0 | 8.0 | 32.5 |
| v2 | 3 | 54.0, 56.0, 54.0 | 54.7 | 2.0 | 38.1 |

v2 - v1 judge mean: 6.7; largest per-arm spread: 8.0; v2 wins beyond spread: False.
Notes:
- The two ranges do not overlap: the lowest v2 run (54.0) is above the highest v1 run (52.0). But the gap (6.7) is smaller than v1's own range (8.0), so by the strict rule (gap above the largest per-arm spread) v2 has not won beyond noise.
- Before this fix, v1 scored 54.0 twice on the same subset (`results/b50/`, and a noise repeat). Counting those runs too, v1's mean is 50.4 over 5 runs, with a range of 10.0.
- Cause fixed: in 4-6 of 50 answers, v2 replied with the format's `"..."` placeholder. After the fix that happens 0-1 times per run.

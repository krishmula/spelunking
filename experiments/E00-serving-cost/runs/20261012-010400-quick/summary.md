# E00 run 20261012-010400-quick

STATUS: OK

KV cache: 12416 tokens

## Part 1: totals

Medians over the recorded passes (drift passes excluded; flagged passes, where other apps used more than 8.0% of the GPU, are left out unless fewer than two are clean). Spread is (max - min) / median of output tokens per second.

| Workload | Passes used | Time per token (ms) | Inter-token median (ms) | Time to first token (ms) | Output tok/s | Spread | Flagged passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| in1k_c1 | 1 | 24.17 | 23.86 | 1009 | 31.33 | 0.0% | 1 |
| in8k_c1 | 1 | 42.65 | 40.31 | 14684 | 6.35 | 0.0% | 1 |
| in1k_c8 | 1 | 110.90 | 82.60 | 5398 | 50.99 | 0.0% | 1 |

Drift (1k x1, first vs last pass): 31.76 -> 21.92 tok/s (-31.0%). A big drop means the chip got hotter over the run.

Swap at start / end: total = 3072.00M  used = 1948.75M  free = 1123.25M  (encrypted) / total = 3072.00M  used = 1948.75M  free = 1123.25M  (encrypted)

## Part 2: GPU working vs idle, by context length

| Context | Reps | Wall ms/step | GPU ms/step | Idle ms/step | GPU busy |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 29.04 | 22.64 | 6.40 | 78.0% |
| 1024 | 1 | 32.68 | 25.47 | 7.21 | 77.9% |
| 2048 | 1 | 35.80 | 27.62 | 8.18 | 77.1% |
| 4096 | 1 | 37.86 | 28.65 | 9.21 | 75.7% |
| 8192 | 1 | 45.54 | 34.34 | 11.20 | 75.4% |

Fit: GPU ms per step = 23.64 + 1.325 x (context / 1000). The slope is attention; the intercept is everything that doesn't grow with context.

## Part 3: weights alone

Memory bandwidth tonight: 104.7 GB/s median, 132.4 best. Weight bytes: 2262.5 MB. Floor at the best bandwidth: 17.09 ms.

| Matmul | Count | MB | ms (median) |
|---|---:|---:|---:|
| down_proj | 36 | 504.3 | 4.26 |
| gate_proj | 36 | 504.3 | 4.12 |
| k_proj | 36 | 53.1 | 0.67 |
| lm_head | 1 | 218.8 | 1.88 |
| o_proj | 36 | 212.3 | 1.88 |
| q_proj | 36 | 212.3 | 1.86 |
| up_proj | 36 | 504.3 | 4.12 |
| v_proj | 36 | 53.1 | 0.68 |
| all, one pass | 253 | 2262.5 | 20.29 |

## The bill

Weights from part 3; attention = slope x context; KV write + other = intercept - weights; idle from part 2.

| | 1k step | 8k step |
|---|---:|---:|
| Weights | 20.29 ms | 20.29 ms |
| Attention | 1.36 ms | 10.85 ms |
| KV write + other | 3.35 ms | 3.35 ms |
| GPU idle | 7.21 ms | 11.20 ms |
| Sum | 32.21 ms | 45.69 ms |

Check: the Sum should match part 2's wall ms/step at 1k and 8k. A negative 'KV write + other' means weights run faster alone than inside the engine.

# E00 run 20261012-011313-full

STATUS: OK

KV cache: 12416 tokens

## Part 1: totals

Medians over the recorded passes (drift passes excluded; flagged passes, where other apps used more than 8.0% of the GPU, are left out unless fewer than two are clean). Spread is (max - min) / median of output tokens per second.

| Workload | Passes used | Time per token (ms) | Inter-token median (ms) | Time to first token (ms) | Output tok/s | Spread | Flagged passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| in1k_c1 | 5 | 23.03 | 22.64 | 1309 | 30.09 | 4.6% | 0 |
| in8k_c1 | 5 | 35.89 | 34.58 | 13381 | 7.16 | 8.4% | 0 |
| in1k_c8 | 5 | 119.77 | 87.40 | 6151 | 47.77 | 13.7% | 0 |

Drift (1k x1, first vs last pass): 26.44 -> 29.37 tok/s (+11.1%). A big drop means the chip got hotter over the run.

Swap at start / end: total = 3072.00M  used = 1916.75M  free = 1155.25M  (encrypted) / total = 3072.00M  used = 1908.75M  free = 1163.25M  (encrypted)

## Part 2: GPU working vs idle, by context length

Idle here is time our process had no GPU work running. Other apps' GPU work also shows up as our idle, so read it next to the last column.

| Context | Reps used | Wall ms/step | GPU ms/step | Idle ms/step | GPU busy | Other apps' GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5 | 21.65 | 21.61 | 0.04 | 99.8% | 0.0% |
| 1024 | 5 | 24.22 | 24.18 | 0.03 | 99.9% | 0.0% |
| 2048 | 5 | 25.69 | 25.62 | 0.05 | 99.8% | 0.0% |
| 4096 | 5 | 29.78 | 29.71 | 0.07 | 99.8% | 0.0% |
| 8192 | 5 | 33.22 | 33.19 | 0.03 | 99.9% | 0.0% |

Fit: GPU ms per step = 22.70 + 1.419 x (context / 1000). The slope is attention; the intercept is everything that doesn't grow with context.

## Part 3: weights alone

Memory bandwidth tonight: 132.2 GB/s median, 133.4 best. Weight bytes: 2262.5 MB. Floor at the best bandwidth: 16.96 ms.

| Matmul | Count | MB | ms (median) |
|---|---:|---:|---:|
| down_proj | 36 | 504.3 | 4.20 |
| gate_proj | 36 | 504.3 | 4.04 |
| k_proj | 36 | 53.1 | 0.61 |
| lm_head | 1 | 218.8 | 1.85 |
| o_proj | 36 | 212.3 | 1.80 |
| q_proj | 36 | 212.3 | 1.81 |
| up_proj | 36 | 504.3 | 4.07 |
| v_proj | 36 | 53.1 | 0.63 |
| all, one pass | 253 | 2262.5 | 17.64 |

## The bill

Weights from part 3; attention = slope x context; KV write + other = intercept - weights; idle from part 2.

| | 1k step | 8k step |
|---|---:|---:|
| Weights | 17.64 ms | 17.64 ms |
| Attention | 1.45 ms | 11.62 ms |
| KV write + other | 5.07 ms | 5.07 ms |
| GPU idle | 0.03 ms | 0.03 ms |
| Sum | 24.19 ms | 34.36 ms |

Check: the Sum should match part 2's wall ms/step at 1k and 8k. A negative 'KV write + other' means weights run faster alone than inside the engine.

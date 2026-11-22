# E05 Metal page layout

Candidate: foundation
Status: complete, park for these Metal workloads

## Question

Does changing only the order of physical KV pages affect one decode attention call enough to repay copying them into order?

## Why now

The local Metal dispatcher can consume an arbitrary table of physical pages. E05 measures the effect before changing a serving engine. The hypothesis came from the handoff. No matching candidate ID was established.

## Prediction

| Who | Prediction | Reason |
|---|---|---|
| Codex | Less than 5% median advantage for contiguous pages | Each physical page already contains 16 consecutive tokens. |
| User | No prediction recorded | |

## The one change

Each layout maps the same logical KV pages into the same number of physical slots. Contiguous pages stay ordered, clustered pages move in groups of four, and shuffled pages move individually. An additional contiguous copy estimates the noise from buffer placement and timing. The key and value tensors have shape (page, token, KV head, head dimension). A table maps each logical page to a physical page. The pool has no unused pages. This tests order in a dense pool.

Test 1k prompt, batch 1, 8k prompt, batch 1, and 1k prompt, batch 8. Use 32 query heads, eight KV heads, dimension 128 and FP16. Use two random seeds. For each layout collect 21 paired rounds per seed with random order, synchronizing each fresh attention call. Each observation is the median of nine calls. Also time gathering both tensors into fresh contiguous storage. Count a saving only if the paired median beats the contiguous-copy control variation (90th percentile absolute difference) on both seeds. Promote to a full step measurement if either scattered layout beats that threshold in the direction favoring repacking and the copying cost is repaid in at most 256 decode steps.

## How to run

From this directory run `caffeinate -i ~/dev/vllm-lab/.venv/bin/python measure.py`.

### Follow-up prediction written before the sensitivity run

The first grid showed no timing difference beyond the contiguous-copy control. A second grid checks two things the dense-pool test could miss. Use a pool with four physical slots per selected page in every layout. Scatter four-page groups or individual pages across that pool. Time 16 fresh queries together and divide by 16 to reduce synchronization cost per call. Verify all 16 outputs. Use the same two seeds, workload names and control threshold. Predict that the sparse shuffled layout is the most likely to show a measurable saving.

Run the sensitivity check with the same command, replacing `measure.py` with `sparse.py`.

### Layer working set follow-up

The first sparse run reapplied 16 queries to the same KV. For 1k prompt, batch 8, test eight distinct layers against eight queries to one layer. Eight layers touch 256 MiB of selected KV per timed sequence, versus 32 MiB for the reuse control. The experiment varies page indices inside GPU buffers. It does not measure the OS mapping of those buffers to DRAM. The three layouts together allocate about 3 GiB of KV buffers.

Before observing outcomes, predict the scattered layout costs more when each query reads a different layer. Randomize timing order, check all outputs, rotate the first layer, and record the total cost of repacking all eight layers. Promote only when the median saving for distinct layers exceeds the 90th percentile absolute contiguous-copy difference at every seed, and copying is repaid in at most 256 steps at every seed. Use seeds 37, 91 and 133, 31 rounds per seed. Run `caffeinate -i ~/dev/vllm-lab/.venv/bin/python stream.py`. This is a follow-up to the original parked result.

## Runs

| Run | Setup | Paired rounds |
|---|---|---:|
| 20261122-010814-layout | Dense pool, one synchronized query per request | 126 |
| 20261122-010922-sparse | Sparse pool, 16 queries per request issued before synchronization | 126 |

## Result

All outputs matched exactly across layout variants. The dense contiguous result also matched dense scaled dot-product attention within the declared FP16 tolerance. Each method paid a positive copy cost. None produced a saving above the prespecified control threshold. No reliable break-even estimate. Values below are medians. Full round values and ranges are in each run. Positive differences favor repacking.

| Run | Workload | Seed | Contiguous µs | Clustered minus contiguous µs | Shuffled minus contiguous µs | Control threshold µs | Shuffled copy µs |
|---|---|---:|---:|---:|---:|---:|---:|
| layout | 1k prompt, batch 1 | 37 | 479.4 | +6.2 | +2.1 | 23.2 | 448.5 |
| layout | 8k prompt, batch 1 | 37 | 506.0 | +4.6 | +1.4 | 30.1 | 793.2 |
| layout | 1k prompt, batch 8 | 37 | 690.2 | +0.4 | +6.2 | 37.3 | 753.4 |
| layout | 1k prompt, batch 1 | 91 | 476.0 | +0.6 | +2.0 | 25.0 | 445.5 |
| layout | 8k prompt, batch 1 | 91 | 470.1 | -1.5 | -1.2 | 21.4 | 758.2 |
| layout | 1k prompt, batch 8 | 91 | 673.2 | +7.8 | +11.0 | 33.0 | 770.2 |
| sparse | 1k prompt, batch 1 | 37 | 56.5 | +0.4 | -0.2 | 3.5 | 241.9 |
| sparse | 8k prompt, batch 1 | 37 | 283.0 | +2.7 | -2.9 | 15.2 | 769.2 |
| sparse | 1k prompt, batch 8 | 37 | 694.0 | +1.2 | +6.4 | 13.5 | 779.4 |
| sparse | 1k prompt, batch 1 | 91 | 58.2 | -1.4 | -1.9 | 4.0 | 245.9 |
| sparse | 8k prompt, batch 1 | 91 | 277.1 | +1.0 | -1.3 | 11.8 | 769.1 |
| sparse | 1k prompt, batch 8 | 91 | 694.1 | +0.1 | +9.0 | 15.8 | 772.5 |

## Why

Amortizing synchronization and using a larger pool reduced the control threshold from 21–37 µs to 3–16 µs. These two changes were made together, so the run cannot apportion their contributions. The closest lead was 1k prompt, batch 8 with sparse shuffled pages. The paired savings were 6.4 and 9.0 µs, below control thresholds of 13.5 and 15.8 µs. The 8k prompt, batch 1 dense route used gqa_decode, while the 1k workloads dispatched per-token kernels. The precise GPU bottleneck was not profiled. No inference about other kernels or devices follows from this screen.

## Verdict

Park physical-page repacking for these three workloads on this M5. The full step promotion condition did not trigger. The sparse batch 8 direction remains unresolved at this measurement precision.

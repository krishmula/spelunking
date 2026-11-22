# E01: why a decode step at batch 8 costs 87 ms, not 34

Candidate: foundation
Status: planned 2026-10-12; no script yet, nothing run

## Question

E00 measured a step time of 24 ms for a 1k prompt at batch 1 and about 87 ms at batch 8. Each weight matrix is read
once per decode step whatever the batch size, so batch 8 should cost about 34 ms (weights matmul 17.6 + attention
8 x 1.5 + small kernels 5). Where do the other ~53 ms go?

1. How does step time grow with batch size (1, 2, 4, 8), and how much of it is the GPU working vs idle?
2. How long does the weights matmul take alone, for all 253 weight matrices, at batch size 1, 2, 4, 8 and 16? If
   the weights matmul is the problem, this shows it directly, outside the engine.
3. At batch 8, how much of the step budget is attention and how much is small kernels?

## Why now

This was the cheapest test for the deleted R02-C8 lead. The decision rule: if the weights matmul
is under half of the batch-8 excess (under about 31 ms of the 63 ms between batch 1 and batch 8), it isn't the
bottleneck and C8 dies or pivots. Published numbers on other M5 chips say the stock 4-bit weights matmul at batch 8
runs at about 39% of bandwidth, which alone would explain most of the gap; nobody has checked on this Air inside a
real decode step.

## Prediction

Write before the first run. A number and a reason each.

| Who | Prediction | Reason |
|---|---|---|
| Krishna | | |
| Claude | | |

What to predict:

- Weights matmul alone at batch 8: how many ms (batch 1 is 17.6)?
- The step budget at 1k prompt, batch 8: weights matmul / attention / small kernels / idle.
- Of the ~53 ms gap, what share is the weights matmul?

Claude writes theirs at the same time as Krishna, so neither anchors the other.

## The one change

Batch size (1, 2, 4, 8). Model, engine settings and prompt length (1k) are the same as E00.

## How to run

Not written yet. The plan for `run_experiment.py`, same shape and safeguards as E00's (preflight, heat soak,
shuffled passes, flags when other apps use the GPU, summary on success or failure):

1. **Step time by batch size.** Serve the model as in E00. For each batch size 1, 2, 4, 8 with 1k prompts, 5 reps,
   shuffled: send the requests together, wait until every one is decoding, then time only decode steps where no
   prefill shares the forward pass. GPU working vs idle from the per-process GPU-time counter, as in E00. This
   removes the prefill stretch that made E00's TPOT at 1k prompt, batch 8 (119.8 ms) bigger than its step time
   (87.4 ms).
2. **Attention at batch 8.** The same batch-8 run at 1k, 2k and 4k context; the slope against context is
   attention, the intercept is everything else.
3. **Weights matmul alone by batch size.** In a fresh process, E00's weights matmul test at batch size 1, 2, 4, 8,
   16, plus the bandwidth test. Reports ms and effective bandwidth per batch size.
4. **The step budget at each batch size.** Weights matmul (from 3), attention (from 2), small kernels = the rest,
   idle (from 1), checked against the measured step time.

```sh
source ~/dev/vllm-lab/.venv/bin/activate
cd ~/dev/spelunking/experiments/E01-small-batch-step
python run_experiment.py --dry-run
caffeinate -i python run_experiment.py
```

Out of scope here: why the weights matmul is slow if it is (ALU, threadgroup staging, occupancy). That needs GPU
hardware counters from Xcode and is the experiment after this one, only if this one says the weights matmul is the
gap.

## Runs

| Run | What | Key numbers |
|---|---|---|

## Result

## Why

## Verdict

Surprise: . Cause known: . Next: .

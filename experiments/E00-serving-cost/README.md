# E00: what serving a model on this Air costs, and where the time goes

Candidate: foundation
Status: done 2026-10-12 (run 20261012-011313-full)

## Question

Running Qwen3-4B 4-bit through vLLM on this Air:

1. What are time to first token (mostly prefill) and TPOT (decode) for a 1k prompt, an 8k prompt,
   and eight 1k requests at once? How big a change can this setup detect?
2. Of a decode step at 1k and at 8k context, how much is weights matmul, attention, small kernels, and the GPU
   sitting idle waiting on the CPU?

## Why now

It's the first look at the machine. The weight-read floor is about 17.5 ms per decode step (2.26 GB at
about 129 GB/s), so the totals say how far above that floor the engine runs, and the split says why. Without the
split, a later result like "attention got 2x faster" has no size.

Why these three workloads: 1k prompt, batch 1 is the simplest decode step, the one place the weight-read floor can be checked
directly. 8k prompt, batch 1 is the same with a big KV cache, so the difference is attention over a long context. 1k prompt, batch 8 is the
realistic load, where each step carries 8 requests.

## Prediction

| Who | Prediction | Reason |
|---|---|---|
| Krishna | 1k prompt, batch 1 TPOT beats the 27.3 ms of the 2026-10-10 attempt by a bit | almost every other app has been quit |
| Krishna | 1k step: weights 2-3 ms, attention 12 ms, KV write 1 ms, other 6 ms | |
| Krishna | 8k step: weights 4-5 ms, attention 25 ms, KV write 2 ms, other 9 ms | |
| Krishna | The gap above the weight-read floor is the GPU not doing the math fast enough | other processes run alongside decode (Terminal, T3 Code, TickTick, Finder, macOS) |
| Claude | 1k prompt, batch 1 TPOT within 3% of 27.3 ms | same engine settings; other apps used about 3% of the GPU last time |
| Claude | 1k step: weights 20 ms, attention 1.5 ms, small kernels 5 ms, idle 0.1 ms | weights are 2.26 GB; everything else touches little memory |
| Claude | 8k step: weights 20 ms, attention 11 ms, small kernels 5 ms | only attention grows with context |
| Claude | The gap above the floor is memory not delivering full speed to the 4-bit matmuls | at one row per step the math is tiny next to the bytes read |

Note: part 3's smoke test (2026-10-12, 00:55) already showed the weights-alone matmuls at 20.6 ms, before the full
run. Claude's weights number was written before that; Krishna's was too.

## The one change

None; this measures the engine as it is.

## How to run

```sh
source ~/dev/vllm-lab/.venv/bin/activate
cd ~/dev/spelunking/experiments/E00-serving-cost
python run_experiment.py --dry-run          # checks and plan only
caffeinate -i python run_experiment.py      # the real run, about 50 minutes; read runs/<latest>/summary.md after
```

`run_experiment.py` does everything, in order, and leaves `summary.md` and `STATUS` in the run folder:

1. **Preflight.** Refuses to start if Low Power Mode is on, the engine code has uncommitted changes, or the port is
   busy. Waits up to 30 minutes for other apps to stop using the GPU, then starts anyway and records it.
2. **Server.** `vllm serve mlx-community/Qwen3-4B-4bit` with a fixed KV cache (`--gpu-memory-utilization 0.38`),
   prefix caching off, async scheduling on, seed 0. Stops if the KV cache can't hold 8 x 1,152 tokens.
3. **Heat soak.** 180 s of unrecorded requests, so every pass starts from the same hot, steady chip.
4. **Part 1, totals.** `vllm bench serve` on the three workloads, 5 passes each in a seeded shuffled order, 20 s
   cooldown between, plus a 1k prompt, batch 1 drift pass first and last. Every pass records how much GPU other apps used; above
   8% it is flagged.
5. **Part 2, context sweep.** One streaming request at a time at 128, 1k, 2k, 4k and 8k tokens of context, 256
   output tokens, 5 each, shuffled. macOS keeps a per-process total of GPU time; the script reads it at the first
   and last output token, so each step splits into GPU working and GPU idle. GPU time per step against context is a
   straight line: the slope is attention, the only work that grows with context; the intercept is the rest.
6. **Part 3, weights alone.** After the server stops, a fresh process loads the same model and times its 253 real
   4-bit matmuls at one row (the MLX kernel the engine calls), plus a memory-bandwidth test for tonight's floor.
7. **The step budget.** Weights matmul from part 3, attention from the slope, small kernels = intercept minus weights matmul, idle from
   part 2. The step budget's sum is checked against the measured step time.

Limits of this method: KV write and other small kernels come out as one number. Part 3 times the matmuls outside the
engine, so if the engine runs them slower, "small kernels" absorbs the difference (and could even come out
negative if the engine runs them faster). The GPU-time counter counts a bundle of GPU work start to end, so a pause
inside a bundle counts as busy.

## Runs

| Run | What | Key numbers |
|---|---|---|
| 20261012-010400-quick | pipeline smoke test, 1 rep each, screen busy (other apps 20-45% of GPU) | not usable; proved the script runs end to end |
| 20261012-011313-full | the run, 01:13-01:50, display asleep, other apps under 0.5% of GPU on every pass, 0 flagged | 1k prompt, batch 1 23.0 ms/token; 8k prompt, batch 1 35.9; 1k prompt, batch 8 119.8; step budget at 1k: weights matmul 17.6, attention 1.5, small kernels 5.1, idle 0.03 |

## Result

Part 1, totals (median of 5 passes, no pass flagged):

| Workload | TPOT | Time to first token | Spread across passes |
|---|---:|---:|---:|
| 1k prompt, batch 1 | 23.0 ms | 1.31 s | 4.6% |
| 8k prompt, batch 1 | 35.9 ms | 13.4 s | 8.4% |
| 1k prompt, batch 8 | 119.8 ms (inter-token median 87.4) | 6.15 s | 13.7% |

Part 2 and 3, where a decode step goes:

| | 1k step | 8k step |
|---|---:|---:|
| Weights matmul (all 253, timed alone) | 17.6 ms | 17.6 ms |
| Attention (slope 1.42 ± 0.16 ms per 1k context) | 1.5 ms | 11.6 ms |
| Small kernels (intercept 22.7 ± 0.7 minus weights matmul) | 5.1 ms | 5.1 ms |
| GPU idle | 0.03 ms | 0.03 ms |
| Sum / measured step | 24.2 / 24.2 ms | 34.4 / 33.2 ms |

- The GPU was busy 99.8-99.9% of every step and other apps used 0-0.5% of it. The CPU never makes the GPU wait.
- Memory bandwidth tonight: 133.4 GB/s best, so the weight-read floor is 17.0 ms. The real matmuls take 17.6 ms, 96% of
  the best bandwidth. The three big MLP matmuls are 12.3 ms of that.
- Smallest change this setup can detect, from the pass-to-pass spread: about 5% at 1k prompt, batch 1, about 10% at 8k prompt, batch 1, about
  15% at 1k prompt, batch 8. Single requests in part 2 vary more (8k steps ranged 30.7-38.2 ms with the GPU fully busy), so the
  GPU itself runs at a varying speed, most likely heat on a fanless chip.
- Drift: the first 1k prompt, batch 1 pass was 11% slower than the last, not faster. The first pass came straight after the
  heat soak; the cause isn't known.

## Why

The decode step is mostly weights matmul, and it runs close to the memory's limit: 17.6 of 24.2 ms at 1k, at 96% of
measured bandwidth. Attention is small at 1k (1.5 ms) and grows linearly, to 11.6 ms at
8k, which is the whole difference between the 1k and 8k steps. What remains, about 5 ms per step that doesn't grow
with context, is everything that isn't a big matmul or attention: norms, RoPE, activations, KV writes, sampling,
and any extra cost of running the matmuls inside the engine instead of alone. This method can't separate those.

The gap above the 17.0 ms floor (7.2 ms at 1k) is therefore 0.7 ms of matmul inefficiency, 1.5 ms of attention and
5.1 ms of fixed small work. It is not other apps (0% of the GPU) and not the GPU waiting on the CPU (0.03 ms).

1k prompt, batch 1 came in 16% faster than the deleted 2026-10-10 attempt (27.3 ms). That run's settings and state aren't
available to compare, so the cause isn't known; it was not same-session, so it's a hint, not a result.

## Verdict

Answered. Decode on this Air is bound by memory bandwidth for the weights matmul, plus attention at long context, plus a fixed
5 ms of small kernels. The two targets that could still move a single-request step are that 5 ms (about 21% of a 1k
step) and attention at long context. The 1k prompt, batch 8 step costs 87-120 ms per decode step when each weight matrix is read once, which
this experiment doesn't explain: that is R02-C8's question.

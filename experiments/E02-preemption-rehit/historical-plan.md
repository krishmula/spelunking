# E02: how much of a preemption the prefix cache gives back

Candidate: R03-C2
Status: full run started 2026-10-14 00:25, unattended; Krishna's prediction to be written before anyone reads summary.md

This is the kill-test E-C2'a for card C2' in [R03 candidates](../../research/R03-kv-cache-design-space/candidates.md)
(the lint reads card ids as `C<n>`, so the line says `R03-C2`).

## Question

When vLLM runs out of KV blocks it preempts a request: frees all its blocks and later recomputes it. With prefix
caching on, the freed blocks keep their hashes until something reuses them, so the resumed request may hit its own
old blocks and skip part of the recompute.

1. How often does preemption fire at E00's pool size (12,416 tokens) with thinking-length outputs, at batch 8 and 16?
2. What share of GPU time goes to re-prefill?
3. What share of a victim's computed tokens does the prefix cache give back on resume (the re-hit share)?
4. At the moment of a preemption, would a different victim have kept many more of its blocks? This decides
   whether choosing the victim is worth a thesis.

## Why now

C2' is the only R03 survivor, and it is weak. The kill conditions were fixed in
`research/R03-kv-cache-design-space/decisions.tsv` (2026-10-12T09:30) before any run:

- KILL if under 1 preemption per 100 requests, or re-prefill under 5% of GPU time.
- KILL the victim-policy direction if candidate victims differ by under 2x in surviving blocks.
- PIVOT to a short measurement note if the re-hit share is above 0.8 or below 0.2 for every victim.
- GO to a victim-policy oracle if preemptions are at least 1 per 100 requests, re-prefill is at least 10% of GPU time,
  and the 2x spread holds.

## Prediction

Write before the first run. A number and a reason each.

| Who | Prediction | Reason |
|---|---|---|
| Krishna | (written 2026-10-14 morning, before reading summary.md) | |
| Claude | Preemptions per 100 requests: b8-cap2k ~40, b16-cap2k ~150, b8-cap4k ~120, b16-cap4k ~300. Re-prefill 5-15% of GPU time at batch 8, 15-30% at batch 16. Re-hit share bimodal: about half the victims near 0, the rest 0.7-0.9; median about 0.5. Candidate spread under 2x (median about 1.1x). Verdict: KILL the victim-policy direction, PIVOT not triggered. | The pool holds 12,416 tokens; 8 thinking requests of about 2k tokens need about 17k, so the pool runs full and the free queue is empty at every preemption. Then a victim's blocks survive only until the running requests grow into them: a victim resumed quickly keeps its prefix, one that waits through two preemptions loses everything. That depends on wait time, not on which request was the victim. Written after a smoke run on a 1,536-token pool (4 preemptions: re-hit 0, 0, 0.81, 0.87; spread 1.08x), so this is not blind. |

## The one change

None against stock vLLM: this measures the stock engine. The scheduler is vLLM's own `AsyncScheduler` with logging
added through `--scheduler-cls` (`e02_scheduler.py`); it makes the same decisions. Prefix caching is on (E00 had
it off). The four configs differ in batch size (8, 16) and output cap (2,048 and 4,096 tokens).

## How to run

```sh
source ~/dev/vllm-lab/.venv/bin/activate
cd ~/dev/spelunking/experiments/E02-preemption-rehit
python run_experiment.py --dry-run                   # checks only
python run_experiment.py --smoke --out /tmp/e02-smoke # 1 minute, tiny pool, tests the pipeline
caffeinate -is python run_experiment.py              # the real run, at most about 6 hours
python run_experiment.py --analyze runs/<run>         # recompute summary.md from events.jsonl
```

Each config gets a fresh server, 64 chat requests (seeded math problems, thinking on, temperature 0.6), at most
`batch` in flight, and a deadline (70 min for 2k caps, 75 for 4k) after which no new request starts.

How the numbers are made:

- **Re-prefill time share.** Step time is the gap between consecutive `schedule()` calls. A least-squares fit of
  step time on scheduled tokens gives the cost of one more token; re-prefill time is that cost times the re-prefill
  tokens, over total busy time. The token-proportional share is printed beside it as a cross-check.
- **Re-hit share.** The prefix-cache hit at the victim's resume, over the tokens it had computed when preempted.
- **Candidate spread.** At each preemption the log holds every running request's blocks. For each candidate, the
  replay assumes it had been freed instead and that the same number of blocks were popped from the free queue as
  were popped before the real victim's resume. vLLM frees a request's last block first, unhashed blocks to the
  head and hashed ones to the tail, so the replay knows which blocks go first. The summary prints a check of the
  replay against the real victim. In the smoke run it matched the real victim's surviving prefix 4 times out of 4.
  It ignores blocks other requests free during the wait.

## Runs

| Run | What | Key numbers |
|---|---|---|
| 20261014-002308-full | all four configs, pool 12,416 tokens (same as E00) | sealed until Krishna's prediction is in |

## Result

## Why

## Verdict

Surprise: . Cause known: . Next: .

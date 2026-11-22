# How much of a preemption does the prefix cache already save?

2026-10-12 · Status: MEASURE-FIRST · Gap confidence: low

## Summary

When vLLM runs out of KV memory mid-decode, it drops one request and recomputes it later. The dropped request's
blocks are not erased. They go back into the prefix cache and stay there until other requests need the space, so
the request can get part of its work back for free when it resumes. Nobody has published how big that part is. The
one public claim says the prefix cache makes no difference to the recompute cost, which contradicts how the code
works. The first experiment is one evening of logging on the Air, with no change to any policy.

Precisely: measure the share of a preempted request's computed tokens that it re-hits from its own surviving blocks
on resume in vLLM V1 on vllm-metal, under KV pressure from reasoning-length outputs, and whether that share differs
across candidate victims at the same moment.

This dossier's survivor is weak. The search ran eight candidates, and six died in pass 1. Pass 2 killed one more and
cut the other down to this measurement. Expect a short measurement note, not a thesis mechanism, unless the
kill-test finds something the reviewers said it would not.

## Current experiment plan

[E02](../../experiments/E02-preemption-rehit/README.md) is the expanded measurement plan formulated on 2026-10-20.
It specifies 20 configurations, five runs per configuration, and 100 requests per run, with full block histories and manual decision inspection.
It changes no vLLM mechanism. The earlier scope and predictions remain in [the historical plan](../../experiments/E02-preemption-rehit/historical-plan.md).
The earlier run is preserved and excluded from the expanded sweep.

For E02, the 2x candidate-spread rule and automatic GO, PIVOT, and KILL thresholds below are historical hypotheses, not execution gates.
An unchosen victim's recovery cannot be established by replaying the stock execution as though scheduling had stayed unchanged.
E02 uses observed recomputation cost and candidate disagreements to decide whether to propose a separate policy experiment.

## Research question

Under KV pressure from reasoning-length outputs on vllm-metal, what fraction of a preempted request's computed tokens
does it re-hit on resume, and does that fraction differ enough across candidate victims that a victim policy using it
could beat least-computed-first by 10% p99 TPOT (a heuristic) at equal KV memory?

The answer is "no" if candidate victims at the same moment differ by less than 2x in blocks that survive to resume,
or if re-prefill time is under 5% of GPU time.

## Background

- Preemption in vLLM V1 frees every block of the victim, sets its computed tokens to zero, and puts it at the front
  of the waiting queue [L134]. V1 has no swap [L72].
- Freed blocks that carry a hash go to the most-recently-used end of the free queue, tail block first; new
  allocations take from the other end [L131] [L132]. Checked in the local v0.31.0 code.
- On resume, the scheduler looks the request up in the prefix cache like any new request, so it can re-hit its own
  surviving blocks [L220]. This happens on vllm-metal on Apple Silicon [L221].
- SGLang does not do this: a retracted request's KV is released without entering the radix tree [L227].

## What prior work already answers

| Work | What it solves | What it leaves open | Source |
|---|---|---|---|
| Least-computed-first PR | Victim with least sunk compute; 20-59% less wasted recompute (claimed); recompute tax 10-23% online | Says the tax is invariant to prefix caching, with no numbers | [L218] |
| Cost-aware victim PR | Victim by reclaimable blocks and progress | Moves metrics under 3% on one workload | [L219] |
| Echo | Cache-aware victim reward; keeps high-reference KV | Online/offline co-serving; no self re-hit | [L224] |
| InferMax | Evict short requests; recompute over offload | No prefix caching | [L225] |
| TOPAS | Joint prefix residency and request choice with preemption cost | SGLang, no V1 re-hit | [L226] |
| CacheOPT | Per-preemption swap or recompute, victim order | PCIe swap; no re-hit | [L84] |
| PagedAttention | Swap vs recompute on 2023 GPUs | No prefix caching | [L61] |

## The gap

I did not find public work that measures how much of a vLLM V1 preemption the victim recovers from its own surviving
blocks. The closest threat is the companion study behind [L218], which is not public and, per the PR, found no
prefix-caching effect.

## Why now

V1 merged preemption into the prefix cache by removing swap [L72]. Two victim-choice PRs are open without this
number [L218] [L219]. Small-memory Macs preempt in practice [L221].

## Hypotheses

1. With reasoning-length outputs at the E00 pool size, preemption fires often (several per 100 requests).
2. A resumed victim re-hits most of its prefix, because it waits only a few steps (prediction, not measured).
3. The re-hit share is about the same for any victim at a given moment (the reviewers' prediction, which kills the
   victim-policy direction if true).

## Evidence against

- Fable and Opus argue from code that the free queue is nearly empty at preemption, so survival depends on wait time,
  not victim identity. Not yet answered; the kill-test measures it directly.
- A user found identical waste per preemption for least-progress and LIFO victims [L223].
- Victim choice moved metrics under 3% in [L219].
- The total recompute tax is capped at 10-23% of compute under online pressure [L218].
- vLLM docs treat preemption as a tuning failure to fix with memory or concurrency flags [L71].

## Experiments already run

| Experiment | Setup | Result | What it changed |
|---|---|---|---|
| E00 (foundation) | Qwen3-4B 4-bit, vllm-metal, three workloads | 1k decode step: 17.6 ms weights matmul, 1.5 ms attention, 5.1 ms small kernels | Sets the step budget the kill-test's re-prefill time is compared against |
| [E02](../../experiments/E02-preemption-rehit/README.md) | Four configurations, recorded run from 2026-10-14 | Results sealed pending Krishna's prediction in E02's record | Preserved as the original measurement record |

E02 records the first C2' measurement attempt. Its results were not inspected during expanded-scope formulation.

## Validation plan

**E-C2'a: count preemptions and re-hits (logging only).**

- Setup: vllm-metal, Qwen3-4B 4-bit, prefix caching on, E00's pool size, 8 and 16 concurrent requests with thinking
  on and output caps of 2k and 4k tokens. Scheduler logging only: per preemption, the victim's computed tokens and
  blocks, the free-queue length, and every running request's block count (the candidate victims); per resume, re-hit
  tokens, re-prefill tokens, and re-prefill time.
- Measures: preemptions per 100 requests; re-prefill time as a share of GPU time; re-hit share per victim; the spread
  in surviving blocks between candidate victims, computed by replaying the logged allocations.
- Conditions, fixed before the run and logged in `decisions.tsv` (heuristics):
  - KILL if fewer than 1 preemption per 100 requests, or re-prefill time under 5% of GPU time.
  - KILL the victim-policy direction if candidate victims differ by under 2x in surviving blocks.
  - PIVOT to a short measurement note if the re-hit share is above 0.8 or below 0.2 for all victims.
  - GO to a victim-policy oracle if preemptions are at least 1 per 100 requests, re-prefill time is at least 10% of
    GPU time, and the 2x spread holds.
- Local: all of it. NVIDIA GPUs are needed only to check whether the result holds on datacenter memory sizes.

## Evaluation plan

Only if E-C2'a returns GO. Baselines: vLLM's last-admitted victim, least-computed-first [L218], cost-aware [L219],
and a tuned watermark or `max_num_seqs`. Metrics: p99 TPOT, TTFT, throughput, re-prefill tokens. Outputs will differ
across policies even at temperature 0 [L223], so correctness is checked per request against a no-preemption run,
not across policies.

## Plan

- Week 1: E-C2'a. Decision point: KILL, PIVOT, or GO.
- Weeks 2-3 (GO only): hindsight-oracle victim by replaying logged allocations; compare with least-computed-first.
- Weeks 4-8 (GO only): implement the policy, evaluate, write up.

## Risks and threats

- Scoop: the [L218] companion study may publish first. Its claim runs opposite to H2, so a contradicting measurement
  stays useful.
- Scale: vLLM-only [L227]; may vanish with larger memory.
- Feasibility: low; this is logging in one scheduler file.

## Candidates considered

- C1 block size on unified memory: KILL as a thesis gap; hit granularity is already decoupled from block size [L36].
  Lab tweak: sweep `--block-size` 8/16/32/64.
- C3 KV pool that yields to macOS: KILL; Splash and oMLX ship it [L216] [L217].
- C4 drop dead thinking KV: KILL; implemented in a vLLM PR and shipped in SGLang [L208] [L209]. Lab tweak: apply the
  PR on the Air.
- C5 TurboQuant vs batch size: KILL; int4 KV on Apple already beats fp16 [L211]. Lab tweak: TurboQuant on/off.
- C6 KV through agent pauses on unified memory: KILL; persistent 4-bit KV on edge and Mac SSD tiering exist [L204]
  [L207].
- C7 eviction policy for tiny pools: KILL; 14-policy study and cost-aware eviction published [L30] [L215]. Lab tweak:
  S3-FIFO in `free_blocks`.
- C8 hybrid checkpoint placement: KILL; exact DP published [L212].

## References

All rows are in `ledger.tsv` with URL, fetch date, and quote. Cited above: [L30] When Fancy Eviction Fails (arXiv
2609.28870, 2026); [L36] vLLM CacheConfig docs; [L61] PagedAttention (SOSP 2023); [L71] vLLM Optimization and Tuning
docs; [L72] vLLM V1 User Guide; [L84] CacheOPT (arXiv 2503.13773, 2025); [L131] [L132] [L134] vLLM v0.31.0 source;
[L204] Agent Memory Below the Prompt (arXiv, 2026); [L207] oMLX guide (secondary, discovery only); [L208] vLLM PR
#39806; [L209] vLLM issue #39321; [L211] When Quantization Is Free (arXiv 2605.05699, 2026); [L212] Sparse Prefix
Caching (arXiv 2605.05219, 2026); [L215] AsymCache (arXiv 2606.02964, 2026); [L216] Splash PR #246; [L217] oMLX
process_memory_enforcer.py; [L218] vLLM PR #53723; [L219] vLLM PR #56897; [L220] vllm-gaudi PR #1848; [L221]
vllm-metal issue #890; [L223] vLLM issue #55265; [L224] Echo (arXiv 2504.03651, 2025); [L225] InferMax (arXiv
2411.07447v3); [L226] TOPAS (arXiv 2608.25523, 2026); [L227] SGLang schedule_batch.py.

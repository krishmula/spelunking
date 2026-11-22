# When a prefix cache cannot recover: bistability from shared KV capacity

2026-10-11 · Status: MEASURE-FIRST · Gap confidence: medium

## Summary

LLM servers save most of their work by reusing cached prompt prefixes. The same GPU memory also holds the KV of
requests that are still running. When something knocks the cache cold (a burst, a restart, a failover), more
requests miss, prefill takes longer, more requests sit in memory at once, and less room is left for the cache. The
open question is whether this loop can keep a server stuck in a slow, low-hit state at a load where a warm server
would stay fast. Steady-state thrashing is published and so is an offline fix; the closest theory leaves prefix
reuse out. The first experiment is a fixed-point model plus a warm-start versus cold-start run on a local vLLM.

Precisely: extend KV-constrained fluid models [L255] with a hit-rate-dependent service rate and a cache capacity
equal to pool size minus in-flight KV, find the bistable band, and test it with a two-trajectory experiment.

## Research question

Does sharing one KV pool between in-flight requests and reusable prefixes create a band of offered load where a warm
and a thrashing state are both stable in a prefix-caching LLM engine, and does a transient recovery control (shed,
or prioritize cache fill, until hit rate recovers) restore the warm state at lower goodput cost than a static
concurrency cap, uncached-token admission, or extra headroom, at equal KV memory?

The answer is "no" if warm-start and cold-start runs at the same offered load converge to the same hit rate and
goodput within noise across the swept parameters, or if the band is narrower than a heuristic 5% of warm-stable
capacity, or if a static cap closes the band at no measurable steady-state cost.

## Background

- vLLM keeps one LRU queue of free blocks; cached prefix blocks are evicted only when a new allocation needs them
  [L95] [L128]. Running requests pin their blocks; finished requests leave them cached.
- A cache miss is not a slow lookup: it is extra prefill on the same GPU that is decoding [L247].
- In general distributed systems, a trigger can push a system into a bad state that persists after the trigger is
  removed [L239] [L240]. Queueing models with congestion-dependent service rates have two stable equilibria
  [L249]. Read-through caches tend to self-heal because misses refill the cache [L254].

## What prior work already answers

| Work | What it solves | What it leaves open | Source |
|---|---|---|---|
| Ao et al. 2026 | KV-constrained serving: eviction-free equilibrium unstable, admission and mixing rules | Abstracts away prefill recompute and prefix-cache reuse | [L255] |
| CONCUR | Names middle-phase thrashing in agentic batch inference; feedback admission up to 4.09x | Offline batch; no open-loop arrivals; recovery only "partial" and unmeasured | [L246] |
| XPerf | Hit-rate cliff past a request-rate threshold on real agent apps | Steady-state sweep only | [L247] |
| PrefixBench-H100 | Pressure erodes prefix reuse | Steady state | [L244] |
| Fancy eviction study | Policies beyond LRU barely help on production traces | No dynamics or queueing feedback | [L242] |
| llm-d token-aware routing | Cluster settles into stable partition or churn depending on one threshold | Router level; no recovery measurement | [L256] |
| Queueing stability with KV memory | Stability region with compute and memory constraints | No prefix caching | [L250] |
| Metronome | Metastable collapse from pinned streaming state | Different mechanism | [L248] |
| Metastability theory | Bistability with state-dependent service; read-through self-healing | No LLM terms | [L249] [L254] |

## The gap

I did not find public work that measures or models bistability caused by in-flight KV and cached prefixes sharing
one pool, at a fixed offered load, in an LLM engine. The closest threats are Ao et al. [L255], whose model omits
prefix reuse by design, and CONCUR [L246], which fixes steady-state thrashing offline but does not test whether the
warm state returns at the same load.

## Why now

- Agent workloads reuse over 90% of tokens across turns, and upstream reports that concurrent load breaks this
  [L251].
- Production length distributions shift over time and arrivals are bursty [L209]; daytime and night patterns
  differ [L201].
- Engines are adding admission and retention controls tuned for steady state [L253] [L251] [L222]. If the band is
  real, a steady-state threshold is the wrong control.

### Who cares (market angle)

- Providers price cached input tokens at a steep discount, so hit rate is margin: Anthropic bills cache reads at
  0.1x the base input price [L125]. A server stuck cold after a burst pays full prefill cost for tokens it bills at
  the cached price.
- Agent platforms and serving stacks (vLLM, SGLang, llm-d, Dynamo) are building the controls this result would set
  thresholds for [L251] [L253] [L256].
- Security is one trigger among several: a deliberate burst is one way to knock the cache cold, alongside restarts,
  failover and organic spikes. Availability attacks on the KV pool are published [L33]; this work does not build
  attacks.
- Funding fit: NSF SaTC 2.0 covers trustworthy systems [L136]; the reliability framing also fits industry
  serving-infrastructure research.

## Hypotheses

1. A bistable band exists when the miss-cost ratio (prefill cost of a miss over a hit) is high and the working set
   is close to pool size. Shows as two stable fixed points in the fluid model.
2. The band narrows with headroom and with cheaper misses (offload tiers), and widens with closed-loop agent
   clients. Shows in the parameter sweep.
3. In a real engine, warm-start and cold-start runs at one load inside the band diverge for many revisit intervals.
4. A static cap set from steady-state data either leaves part of the band or costs steady-state goodput; a transient
   control does not.

## Evidence against

- **Read-through self-healing** [L254]. Every miss refills the cache. This is still the main risk: the argument
  that refills get evicted before reuse is mine, not measured.
- **Simple baselines** [L244] [L257] [L253] [L246]. Enough headroom keeps hit rate flat; caps and admission exist.
  Stands until the band is measured with these baselines in place.
- **Backlog drain** (two reviewers). A slow recovery can be ordinary queue drain. Answered by the two-trajectory
  design at identical offered load.
- **Scale** [L256]. Cluster routing may dominate single-node effects. Stands; the cluster arm is a later phase.
- **Laptop miss cost.** M5 prefill is relatively slower than H100, overstating loop gain. Answered by sweeping the
  miss-cost ratio in the model instead of using the laptop constant.
- **Scoop** [L255] [L246]. The theory authors could add a prefix term. Stands; move fast on the model.

## Experiments already run

| Experiment | Setup | Result | What it changed |
|---|---|---|---|
| None for this topic | | | |

Related foundations in `experiments/` (E00 baseline, E01 step budget) give local prefill and decode costs for
calibration. The topic began as KV cache security; Krishna reframed it to robustness on 2026-10-11, which killed the
security-only candidates (`decisions.tsv`).

## Validation plan

Thresholds are logged in `decisions.tsv` before any run. All numeric thresholds are heuristics.

**E-C2a, fluid fixed points (local, about one week).**
- Model: arrival rate, per-request prompt and output lengths, pool size C, in-flight KV = arrivals x residence x
  footprint, cache capacity = C minus in-flight KV, hit rate from the stack-distance curve of Mooncake tool/agent
  hashes [L203], service rate dependent on hit rate. Extends [L255].
- Sweep: load, headroom, miss-cost ratio (cover H100-plausible values), open vs closed-loop arrivals.
- GO: two stable fixed points over a band at least 15% of warm-stable capacity for some H100-plausible setting.
- PIVOT: band only with closed-loop clients or only with a router term.
- KILL: no two stable fixed points anywhere in the sweep, or band below 5%.

**E-C2b, two-trajectory test (local vLLM 0.31 + vllm-metal, about two weeks).**
- Qwen3-0.6B or Qwen3-4B 4-bit; `num_gpu_blocks_override` shrinks the pool so pressure is cheap; prefix caching
  works for dense models on Metal (checked in `vllm_metal/platform.py`).
- Closed-loop agent clients with synthetic shared prefixes shaped like the Mooncake hashes; no retries.
- At loads inside the predicted band: start warm vs start cold (or warm plus a burst), same offered load after.
  Also step-up then step-down sweeps.
- GO: hit rate and goodput differ by at least 2x for at least 10 mean revisit intervals, with default
  `max_num_seqs`, over 3 seeds.
- KILL: both converge within run-to-run noise at every load in the band.
- PIVOT: divergence only with client retries (a known retry storm).
- Apply the benchmark checklist before reporting any number.

**Split by hardware.**
- Local: the model, the simulator, the two-trajectory mechanism test.
- One NVIDIA node: the final single-engine claim with realistic prefill/decode costs and a CPU offload tier.
- Multi-node: the router arm [L256], only after the single-node result.

## Evaluation plan

- Baselines: stock vLLM LRU; static `max_num_seqs` caps; uncached-token admission (Dynamo design [L253]);
  CONCUR-style feedback admission ported online [L246]; headroom sized by working-set analysis [L257]; S3-FIFO as
  an eviction baseline [L59] [L234]; CPU offload where hardware allows.
- Workloads: Mooncake tool/agent and conversation traces [L203]; closed-loop agent sessions; burst families
  (one-off long prompts, agent fan-out, restart, noisy neighbor, length shift [L209]).
- Metrics: hit rate over time, goodput under a TTFT/TPOT SLO, recovery time, band width, steady-state cost of each
  control.
- Constraint: equal KV memory across arms; rejected requests count against goodput.

## Plan

| Weeks | Milestone | Decision |
|---|---|---|
| 1 | E-C2a fluid model and sweep | GO / PIVOT / KILL per logged thresholds |
| 2-3 | E-C2b two-trajectory test on local vLLM | Does the engine show the band? |
| 4-8 | Transient control vs baselines, locally | Does any simple baseline close the band at no cost? |
| 9-16 | One NVIDIA node with offload tier | Does the band survive realistic miss costs? |
| 17-30 | Router arm (llm-d style) and write-up | Workshop or main-track paper |
| 31-52 | Upstream contribution (recovery control or metrics) | Open-source path |

## Risks and threats

- Scoop by [L255]'s authors adding prefix reuse: publish the model and the local result early (workshop).
- No band in the model: kill quickly; the fallback is a characterization of steady-state thrashing thresholds
  online, which is weaker.
- Band exists only at cluster scale: pivot to router bistability [L256], which needs multi-node access.
- No NVIDIA access: the claim stays a laptop-scale mechanism result; say so.

## Candidates considered

- C1 admission for input-dependent compressed KV: killed on relevance, since shipping compression is fixed-ratio
  [L166] [L167] [L168] and pruning is not adopted [L54].
- C3 learning-augmented prefix eviction: killed on novelty [L220] [L238] and on upside [L242].
- C4 per-tenant KV isolation: killed on novelty [L226] [L59]; noisy neighbor is now a burst family.
- C5 calibration shift in KV quantization: killed on oracle [L166] [L167] [L192].
- C6 stacked-optimization stress test: killed as a standalone paper [L158] [L171]; it is now the evaluation harness
  ("probed many ways").
- Security-first candidates from the first framing (side channels, forced occupancy): set aside when the topic was
  reframed; crowded by [L1] [L2] [L33] [L59].

## References

Full rows, quotes and URLs are in `ledger.tsv`. Cited here:
[L1] [L2] [L33] [L54] [L59] [L95] [L125] [L128] [L136] [L158] [L166] [L167] [L168] [L171] [L192] [L201]
[L203] [L209] [L220] [L222] [L226] [L234] [L238] [L239] [L240] [L242] [L244] [L246] [L247] [L248] [L249] [L250]
[L251] [L253] [L254] [L255] [L256] [L257].

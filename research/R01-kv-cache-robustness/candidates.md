# Candidates: KV cache robustness under shift

Constraints: `constraints.md`. Sources: `ledger.tsv` (frontier sweeps in `frontier/`, ids mapped in
`frontier/id-map.tsv`). Topic reframed on 2026-10-11 from security to robustness (see `decisions.tsv`).

## Frontier in one page

- **Compression gains are workload-dependent.** Realized compression ratio and quality both vary by workload, and
  ratio alone predicts serving performance poorly [L158]. Method rankings flip when the query is not visible at
  compression time, which is the reuse case [L171]. Fixed budgets tuned per domain degrade on open-domain input
  [L187]; adaptive budgets make per-request memory input-dependent [L184] [L186] [L190]. Reasoning workloads have
  large per-request KV demand variation [L157], and low budgets lengthen reasoning traces [L174].
- **Serving integration is costly and immature.** Per-head budgets in vLLM cost up to 25% of prefill time and 1.7x
  decode latency [L156]; theoretical rates are not realized in paged memory without engine changes [L153].
  Major engines have not adopted pruning [L54]. What ships is fixed-ratio: FP8 KV [L168], NVFP4 KV with about
  1 to 2 points of loss [L166] [L167], and model-native sparse attention [L165] [L162].
- **Calibration transfers fairly well.** Offline tables correlate at 0.90 across disjoint corpora [L192].
- **Prefix-cache eviction:** better-than-LRU policies exist for stationary workloads [L199] [L200] [L201] [L221]
  [L222] [L230], but on production traces fancy policies barely beat LRU [L242] [L224]. Competitive randomized
  eviction with learning exists [L220]; learning-augmented caching has worst-case bounds [L238].
- **Memory pressure:** formal KV-constrained scheduling bounds [L217] [L219]; overload-tested threshold admission
  [L218]; early rejection in production [L202]; migration [L212]; reservation under output-length
  uncertainty [L37].
- **Shift in production is real.** Length distributions "shift dynamically and independently over time" [L209];
  daytime and night patterns differ [L201]; workloads evolve over a year as agents grow [L224]. No public trace
  pairs prefix hashes with tenants over more than an hour (frontier sweep finding; Mooncake traces cover one hour
  [L203]).
- **Failure modes seen in the wild:** persistent re-prefill under memory pressure in LMCache [L233]; never-hit
  inserts displacing other tenants' prefixes in SGLang [L231]; cache pressure eroding prefix-reuse gains [L244];
  random vs precise routing at 73% KV load moves P90 TTFT from 0.54 s to 93 s [L228].
- **Multi-tenant interference:** measured and mitigated for latency [L226], prefix-cache retention [L59], and
  compute fairness [L63] [L72]. Adversarial tenants are one documented stress case [L33].
- **Metastability:** a trigger can push a system into a bad state "that persists even when the trigger is
  removed" [L239] [L240]. I found no source that tests this for KV or prefix caches.

## Pass 0: candidate cards

### C1. Admission when KV footprint depends on the input

- Problem: input-adaptive compression and reasoning make each request's KV footprint unknown at admission. A
  scheduler that admits on an expected ratio overcommits when the mix shifts and then preempts.
- Why now: adaptive budgets [L184] [L186] [L190], on-demand growth [L157], compressed paged engines [L153] [L155]
  [L156], and documented ratio variance across workloads [L158].
- Existing work: Tangram reserves static per-head budgets [L156]; GrowPage grows on demand [L157]; output-length
  reservation with misprediction handling [L213] [L37]; threshold admission under memory limits [L218].
- Exact gap: no source measures preemption and goodput when realized compression ratio shifts away from the
  ratio admission assumed.
- Research question: Can admission that tracks realized per-request compression keep goodput within the SLO under
  a workload-mix shift, compared with static-ratio admission and with no compression, at equal quality?
- Hypothesis: static-ratio admission shows preemption spikes and goodput loss after a shift toward low-ratio
  inputs; feedback admission avoids most of it.
- Cheapest decisive test: measure realized-ratio distributions per workload family locally with kvpress-style
  adaptive methods, then replay them in a scheduler simulator.
- Biggest novelty threat: S3-style reservation with misprediction recovery [L213] applied to ratio instead of
  length.
- Scale risk: at large batch the ratio averages out across requests, so variance per batch may be small.

### C2. Does prefix caching have a metastable failure mode under transient shift?

- Problem: a transient burst (one-off long prompts, an agent loop, a batch job) can evict hot prefixes. Misses
  add prefill work, which slows service, which lengthens revisit times, which causes more eviction. If this loop
  sustains itself, hit rate and goodput stay low after the burst ends.
- Why now: prefix reuse carries most of the cost advantage in production (ideal hit rates 54-62% [L201]); agents
  shrink inter-arrival times [L224]; persistent degraded states already show up in issues [L233] [L231].
- Existing work: steady-state pressure characterizations [L244] [L242]; eviction policies tuned on stationary
  traces [L201] [L221] [L222]; overload admission without prefix cache [L218]; metastability in general
  distributed systems [L239] [L240].
- Exact gap: whether the miss-to-prefill-to-eviction loop creates hysteresis in LLM serving, which regimes have
  it, and which simple control (admission, probationary insertion, shedding) removes it.
- Research question: Does a transient workload shift leave vLLM's prefix cache in a degraded state (hit rate and
  goodput below the pre-burst level) for long after the burst ends, compared with an uncapped-cache reference,
  and does probationary insertion or prefill-aware admission restore recovery time to near the reference while
  keeping steady-state hit rate?
- Hypothesis: near capacity, recovery time grows sharply with burst size (a knee); below the knee, recovery is
  fast. Probationary insertion (S3-FIFO-style [L234]) or admission removes the knee.
- Cheapest decisive test: discrete-event simulation on Mooncake hashes [L203] with a prefill-cost model and an
  injected burst; then confirm on local vLLM with a small KV pool.
- Biggest novelty threat: [L242] (prefix eviction on production traces by caching researchers) and [L244].
- Scale risk: cache-aware routing and autoscaling in a cluster may damp or amplify the loop [L228] [L243].

### C3. Learning-augmented prefix eviction with a worst-case bound

- Problem: learned or structure-aware eviction [L200] [L223] [L225] has no fallback when its predictions break
  under shift.
- Exact gap: a radix-tree, variable-miss-cost version of learning-augmented caching [L238].
- Research question: Can a learning-augmented eviction policy keep within a bounded factor of LRU's hit rate
  under mix shift while matching the learned policy's gain on stationary traces?
- Biggest novelty threat: [L220] already joins competitive randomized eviction with learning; [L242] finds
  fancy policies barely beat LRU on production traces.

### C4. Per-tenant KV memory isolation with borrowing

- Problem: one shared KV pool and one eviction order serve all tenants [L95] [L97]; a heavy tenant degrades
  others [L226] [L231].
- Exact gap: work-conserving per-tenant KV reservations on the GPU tier.
- Research question: Can per-tenant KV reservations with borrowing bound a quiet tenant's TTFT under a noisy
  neighbor, compared with shared LRU and with [L59] and [L226], at under a heuristic 5% throughput cost?
- Biggest novelty threat: FairInference isolates latency including shared-KV delay [L226]; PrefixShield
  isolates prefix retention [L59]; LMCache per-salt LRU [L108].

### C5. Calibration shift in production KV quantization

- Problem: FP8 scales default to 1.0 without calibration [L168]; calibrated formats might fail on unseen domains.
- Research question: Does KV quantization calibrated on one domain lose more than a heuristic 2 points on shifted
  domains (code, multilingual, long reasoning) than uncalibrated or in-domain calibration?
- Biggest novelty threat: NVFP4 loses about 1 to 2 points across benchmarks [L166] [L167]; calibration
  transfers [L192].

### C6. Stress-testing stacked optimizations ("probed ten ways")

- Problem: each optimization is evaluated alone on benign averages; interactions under shift are unmeasured.
- Research question: Under a fixed family of shift scenarios, which combinations of prefix caching, KV
  compression, KV quantization, and speculative decoding lose more than their sum of individual losses?
- Biggest novelty threat: serving-level compression benchmark [L158]; reuse-time compression audit [L171].

## Pass 1: adversarial review

Searches run by the lead on 2026-10-11 (beyond the six frontier sweeps): "metastable failure LLM serving prefix
cache hit rate collapse burst eviction recompute feedback loop"; "admission control KV cache compression ratio
variable per request preemption serving scheduler"; plus abstract reads of [L242] [L243] [L244] [L245].

### C1

1. Novelty: Survives. No opened source ties realized-ratio variance to admission and preemption; Tangram's
   reservation is static [L156], Minima-KV is configuration-bound [L245].
2. Adjacent work: Narrows. Reservation with misprediction recovery exists for output length [L213] [L37]. The new
   part is that ratio is known only after prefill and compression fragments pages [L156] [L153].
3. Combination: Narrows. Close to "[L213] for compression ratio"; the coupling with page fragmentation is the
   only new constraint.
4. Simplicity: Open. Admit at uncompressed size and reclaim later, or grow on demand [L157], may get most of it.
5. Scale: Open. Per-batch averaging could hide per-request variance.
6. Oracle: Open. Upside depends on realized-ratio variance under shift, which only [L158] hints at.
7. Correctness: no state issue; data movement and accounting only.
8. Feasibility: Narrows. Compressed paged engines are CUDA forks [L153] [L155]; locally only ratio measurement
   plus simulation is feasible.
9. Relevance: Kills for production today. Shipping compression is fixed-ratio [L166] [L167] [L168] [L165], and
   pruning is not adopted in major engines [L54]. Survives only for research engines.
10. Scoop risk: medium; many 2026 compressed-paging engines [L155] [L156] [L157].

Outcome: KILL as a thesis direction (relevance fails while input-adaptive compression is not deployed). Keep as
one stress family inside C2 if adaptive compression ships.

### C2

1. Novelty: Survives. Searched "metastable" plus prefix cache terms; the closest sources study steady-state
   pressure [L244] or policy hit rates in simulation [L242]; neither reports burst recovery or hysteresis.
2. Adjacent work: Narrows. Metastability is defined and catalogued for distributed systems [L239] [L240]. The LLM-
   specific part is that a miss costs prefill compute on the same GPU that serves decode, so misses feed back
   into residence time and revisit intervals. Porting the general concept is not enough; the loop gain must be
   measured.
3. Combination: Survives, conditionally. The coupling (eviction, prefill cost, queueing, and revisit time) does
   not appear in either parent alone. Fails if the loop gain is below one in every realistic regime.
4. Simplicity: Open. Early rejection [L202], threshold admission [L218], or a bigger cache may already prevent
   it. These are the baselines.
5. Scale: Open. Cluster routing changes locality [L228] [L243]; the effect could vanish or grow.
6. Oracle: Open. Need the recovery time with and without the loop; the uncapped-cache run is the reference.
7. Correctness: no state issue.
8. Feasibility: Survives. Public block-hash traces [L203], a simulator, and local vLLM with a small pool fit the
   laptop. Final claim needs one NVIDIA node.
9. Relevance: Survives on weak evidence. Persistent degraded states and displacement are reported [L233] [L231];
   pressure erodes reuse [L244]; prefix reuse is a large share of cost [L201]. No production incident that
   names this loop was found.
10. Scoop risk: medium. The [L242] group works on prefix eviction with production traces; their abstract does
    not address dynamics.

Outcome: survives to pass 2 as MEASURE-FIRST. Open on Simplicity, Scale, and Oracle.

### C3

1. Novelty: Kills. [L220] integrates provably competitive randomized eviction with learning; [L238] gives the
   worst-case framework.
9. Relevance: Narrows toward nothing: [L242] finds policies beyond LRU add little on production traces.

Outcome: KILL.

### C4

1. Novelty: Kills for latency isolation [L226] and for prefix retention [L59]. Remaining part, GPU-tier per-tenant
   reservations, is engineering on top of [L108].

Outcome: KILL as standalone. Noisy-neighbor bursts become one trigger family in C2.

### C5

6. Oracle: Kills. Shipping 4-bit KV loses about 1 to 2 points [L166] [L167]; calibration transfers across corpora
   [L192]. The maximum upside is below the heuristic margin.

Outcome: KILL.

### C6

3. Combination: Kills as standalone. A benchmark of known optimizations under known shifts has no mechanism;
   [L158] and [L171] already cover the compression parts.

Outcome: KILL as a standalone paper. Reused as C2's evaluation harness: shift scenarios (one-off long prompts,
agent loop bursts, multilingual or code mix change, noisy-neighbor tenant, length-distribution shift [L209])
probe the system from several directions, which is the advisor's "probed ten ways" idea.

## Pass 2: multi-model panel

Reviewers (provider, model, effort as confirmed by T3): claudeAgent claude-fable-5-1 (high), codex gpt-6.1-sol
(provider default), codex gpt-6-astra (provider default), claudeAgent claude-opus-5-5 (high). Prompt:
`pass2/prompt.md`. Reviews: `pass2/{fable,sol,astra,opus}.md`. The lead opened every new source cited below.

### Rejections, merged

1. **The stated loop is wrong.** (sol, astra, opus, fable: four models.) LRU in vLLM evicts on allocation, not on
   elapsed time [L128], so "slower service lengthens revisit time" does not evict anything by itself. The only
   coupling that can sustain the loop is shared capacity: in-flight request KV and idle cached prefixes live in one
   pool [L95], so slower service raises in-flight KV (Little's law) and shrinks the cache.
   Answer: accepted. C2 is rewritten around shared capacity (C2' below).
2. **A recovery knee is just backlog drain; an uncapped cache is not a causal control.** (sol, astra.) Drain time
   grows near capacity in any stable queue.
   Answer: accepted. The test becomes two trajectories at identical offered load (warm start vs cold or
   burst-perturbed start), plus step-up and step-down sweeps. Controls change one edge at a time at equal memory.
3. **A prefix cache is read-through, so it self-heals.** (opus, fable.) HotOS'21 says a read-through cache "will
   still be filled, which steadily increases the hit rate until the system is healthy again" [L254].
   Answer: partly holds. Self-healing needs the refill to survive. In C2' the refill competes with in-flight KV for
   the same pool, so a refilled block can be evicted before its next use. Whether that keeps the system stuck is
   the measurable question; I have no evidence either way.
4. **Steady-state thrashing and its fix are published.** (opus, fable, astra.) Hit rate collapses past a load
   threshold [L247]; CONCUR names "middle-phase thrashing" and fixes it with feedback admission, offline [L246];
   upstream says agentic workloads break prefix caching under load [L251]; admission designs exist [L253].
   Answer: holds for steady state and for the mitigation. None of these steps load down or compares warm and cold
   starts at the same load; CONCUR reports only partial recovery in its cooldown phase [L246].
5. **Multiple equilibria in KV-constrained serving are already modeled.** (fable; astra and opus cite the theory.)
   Ao et al. show the eviction-free equilibrium is unstable [L255]; bistability with state-dependent service is
   textbook [L249]; metastable KV collapse is named for pinned streaming state [L248].
   Answer: holds for the general idea. Ao et al. explicitly abstract away "prefill recomputation, swapping, or
   prefix-cache reuse" [L255]. The hit-rate-dependent service rate and the shrinking cache are the new terms.
6. **Simple baselines may close it.** (all four.) Headroom (pool larger than working set) [L244] [L257], a static
   concurrency cap, uncached-token admission [L253], CONCUR-style feedback [L246], offload tiers. Probationary
   insertion is already a baseline [L59] [L242].
   Answer: open. These become mandatory baselines; probationary insertion is dropped as a contribution.
7. **Scoop risk is high on mitigation.** (fable, astra, sol.) [L246] [L251] [L253] [L222] [L221]; Ao et al. share an
   author with [L249] and could extend their own model.
   Answer: holds. The defensible contribution is the measured bistability boundary and the model term, not a new
   admission policy.
8. **Scale may change everything.** (all four.) Cluster routing shows two steady states depending on one threshold
   [L256]. Laptop prefill is relatively slower than H100 prefill, which overstates miss cost.
   Answer: open. Single node first; calibrate miss cost ratio as a swept parameter, not a laptop constant.

No reviewer overturned a pass-1 kill.

### C2' (reformulated)

- Problem: a prefix-caching engine keeps running requests and reusable prefixes in one KV pool. If a transient
  (burst, cold restart, failover) lowers hit rate, prefill work and residence time rise, in-flight KV grows, and the
  cache shrinks further. It is unknown whether this leaves the engine stuck in a low-hit state at a load where it
  would otherwise stay warm.
- Why now: agentic multi-turn workloads make prefix reuse carry most of the work [L251] [L201]; thrashing is
  measured in steady state [L247] [L246]; the closest theory leaves prefix reuse out [L255].
- Existing work: steady-state collapse [L247] [L246]; offline feedback admission [L246]; KV-constrained queueing
  models without prefix reuse [L255] [L250]; general metastability [L239] [L240] [L249]; cluster bistability [L256].
- Exact gap: whether shared capacity creates a bistable band at a fixed offered load in one engine, how wide it is,
  and whether steady-state controls close it.
- Research question: Does sharing one KV pool between in-flight requests and reusable prefixes create a band of
  offered load where a warm and a thrashing state are both stable in a prefix-caching LLM engine, and does a
  transient recovery control (shed or prioritize cache fill until hit rate recovers) restore the warm state at
  lower goodput cost than a static concurrency cap, uncached-token admission, or extra headroom, at equal KV memory?
- Hypothesis: the band exists when the miss-cost ratio is high and the working set is close to pool size; its width
  shrinks with headroom and with cheaper misses; a static cap set from steady-state data either leaves part of the
  band or costs steady-state goodput.
- Cheapest decisive test: fluid fixed-point analysis extending [L255] with a hit-rate curve from Mooncake tool/agent
  hashes [L203]; then a two-trajectory test on local vLLM with a small pool.
- Biggest novelty threat: [L255] (and its authors extending it); [L246].
- Scale risk: routing and autoscaling may dominate [L256]; laptop miss cost differs from H100.

Pass-1 attacks rerun on C2': Novelty Narrows to the bistable band with prefix reuse [L255] [L246] [L247]; Adjacent
Narrows [L239] [L249]; Combination Survives conditionally (new term absent from [L255] and [L239]); Simplicity Open;
Scale Open; Oracle Open; Correctness Survives (no state issue); Feasibility Survives for the single-node claim,
Narrows for GPU claims; Relevance Survives on moderate evidence [L251] [L252] [L247]; Scoop risk Narrows to the
characterization and model term.

Status: MEASURE-FIRST. Gap confidence: medium (both passes ran on four models; the core rejection 3 is answered by
argument, not evidence).

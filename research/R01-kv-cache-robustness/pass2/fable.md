# Pass 2 review: claudeAgent claude-fable-5-1 (effort high)

Verdict: rejection sticks against C2 as written; does not stick against a narrower bistability question.

Shortest rejection: C2 is HotOS'21's look-aside cache loop with prefill as the database; steady-state collapse is
measured (CONCUR 80% to 35%, XPerf 91% to under 4%, IISWC'26 98% to 16-26%); CONCUR evaluates admission; Ao et al.
(2606.15555) formalize admission-memory-eviction feedback with multiple equilibria. Because the prefix cache is
read-through, persistence can only exist in a narrow Little's-law band.

Key new sources: Ao et al. 2606.15555 (eviction-free equilibrium unstable; abstracts away prefix-cache reuse);
llm-d "Sticky Until Saturated" (cluster settles into stable partition or eviction churn depending on tau);
RR-Evict 2609.32278; KVSET 2609.27746 (working-set sizing); py-kvcache 2609.11744 (recompute can beat loads on
fast GPUs); Dynamo #14888 (KV recovery after failover).

Points: Combination survives conditionally (miss, longer residence, more in-flight KV, smaller free pool, more
misses appears in neither Bronson nor Ao). Simplicity narrows: headroom alone prevents it (PrefixBench flat hit
rate when pool exceeds working set); CONCUR restores hit rate. Scale open; cluster loop may dominate. Feasibility:
M5 overstates miss cost relative to H100. Scoop risk high on mitigation side; none targets two-trajectory test.
PrefixShield [L59] already uses S3-FIFO in vLLM, so probationary insertion is a baseline, not a contribution.

Kill test: two-trajectory test on one vLLM node, closed-loop agent clients, warm vs cold start at fixed load.
Rejection fails if the band is wider than a heuristic 20% of warm-stable capacity and a static cap at the warm
threshold does not close it.

Sharper question: on a single vLLM node with closed-loop agent clients, is there a load band between warm-start and
cold-start stability thresholds where the prefix cache is bistable; how does the band scale with KV headroom, miss
cost ratio, and closed vs open-loop arrivals; and does a static cap set from steady state close it, or does recovery
need a transient control (shed, or prioritize cache filling)?

Pass-1 kills: all stand.

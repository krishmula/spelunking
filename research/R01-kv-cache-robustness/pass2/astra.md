# Pass 2 review: codex gpt-6-astra (reasoning effort: provider default)

Verdict: no rejection of the whole question; MEASURE-FIRST only under a narrower form.

Shared with sol (independent): "slower service causes more eviction" is not true for LRU by itself; a recovery knee
can be plain backlog drainage (drain time B/(mu - lambda)); an uncapped cache is not a causal control.

New points:
- Metronome (arXiv 2607.02640) already uses "metastable KV-cache collapse" for pinned streaming state that grows
  monotonically ("The failure is a memory cliff, not a compute drift."). Different mechanism, but the phrase is
  taken.
- SGLang #26248: decode throughput collapse (~14x) after KV retraction, reporter suggests a re-prefill spiral; no
  controlled test.
- L242 full text already evaluates S3-FIFO for prefix caches and recommends conditional quick demotion, so
  probationary insertion is a known baseline, not a contribution.
- vLLM main scheduler config has full-input reservation and a cache watermark ("Prevents over-admission and KV
  cache thrashing"); those are stock baselines.
- Mooncake trace lacks session dependencies and think times, so timestamp replay cannot test latency-dependent
  revisits. Need a closed-loop session client.
- Offload: LMCache shows a bandwidth/length crossover for load vs recompute; an LMCache MI300X blog shows offload
  helps under pressure and adds overhead otherwise. No universal "loop gain below one".
- Metastability in Queues (Queueing Systems 2022): state-dependent service, two stable equilibria, threshold
  admission. Theory precedent.

Scoop: Dynamo #9755, llm-d #2273, PRISM, PEEK occupy cache-aware admission, retention and restoration. The defensible
contribution is a validated stability boundary and causal explanation, not another policy.

Stopping rule proposed: within a month, either a reproducible engine effect with causal ablation, or a negative
result that simple controls remove it. Suggested heuristic: 10% goodput gain at equal resources and comparable
rejection.

Sharper question: at fixed finite KV capacity and unchanged post-burst client demand, can competition between
active-request KV and reusable prefixes create a persistent low-goodput regime in an otherwise functioning engine,
beyond backlog drainage and client retries; and which combinations of reuse distance, admission policy, prefill cost,
and restoration bandwidth allow or eliminate it?

Pass-1 kills: none overturned.

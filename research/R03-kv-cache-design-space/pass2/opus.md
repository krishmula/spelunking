# Pass 2 reviewer: claudeAgent claude-opus-5-5 @high, 2026-10-12

Condensed by the lead judge; reviewer's links kept.

## C2: Narrows x4, Oracle and Simplicity open; rewrite as a new card
- Code: free order (tail-first to MRU end) is already default; "order of freed blocks" half of C2 is not new.
- Shortest rejection: the victim loses only blocks the running set grows into before resume, about the same for any
  victim; victim choice has little to optimize, and InferMax (arXiv 2411.07447) published short-first with a
  CSP-optimal bound (no prefix caching).
- Splash (Mac engine) #247 keeps a suspended request's replay point as a priority; #15 picks victims by completed work.
- vllm-gaudi #1848: "a preempted request can resume with num_computed_tokens > 0". vllm-metal #890: preemption under
  KV pressure with partial prefix re-hit on M1.
- SGLang schedule_batch.py: retracted KV is not inserted into the radix tree; TODO to improve.
- New card suggested: "In vLLM V1, how much of a preemption's recompute do self-re-hits serve, and does any victim
  policy beat FCFS by more than the oracle gap?" Likely a measurement paper.
- Change of mind: on the Air, preemption fires often, hindsight victim cuts re-prefill >= 30% vs FCFS, LCF captures
  < half of that.

## C3: KILL (Novelty, Adjacent, Combination, Scoop)
- Splash #246 (merged 2026-10-03) returns free pages, then cached entries by recency, then suspends, with compaction
  to return extents to macOS; #226, #13 measured with a hog process. MTPLX has the same design (not opened).
- xnu memorystatus doc: "Consider relaxing caching policy." Metal MTLPurgeableState.volatile.
- SiliconBench full text: compression of active KV pages, wall time 2.5x on 32 GB (self-caused padding).
- Residue (not a thesis): which signal predicts danger before a panic when pressure level stays false (Splash #325,
  mlx-lm #883).

## Pass-1 kills
- None look wrong; RR-Evict strengthens C7 kill.

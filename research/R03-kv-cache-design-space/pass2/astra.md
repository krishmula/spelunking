# Pass 2 reviewer: codex gpt-6-astra (provider default effort), 2026-10-12

Condensed by the lead judge; reviewer's links kept. Sources become ledger rows only after the judge opens them.

## Ledger corrections
- L151 (LCF PR #53723) reports 20-59% less wasted recompute, with prefix-cache ablations and larger models.
- L150 (cost-aware PR #56897) reports p99 TPOT -2.6%, throughput -0.4%, p99 TTFT +1.5%, three-run averages.
- L145 (queue-informed eviction PR #54366) preserves 192 cached tokens; TTFT CI crosses zero.
- C6 simplicity kill unsupported by py-kvcache (setup-dependent); novelty kill stands.

## C2: no rejection; MEASURE-FIRST
- Novelty narrows: cache-aware scheduling/retention taken by UniBoost (Beyond Prediction, arXiv 2606.18431; disables
  prefix caching), PEEK (2607.02525), TOPAS (2608.25523). Remaining: whether future self-reuse after preemption changes
  victim rankings.
- Adjacent: cache-related preemption delay (WCET'09) estimates reload cost from useful vs evicting blocks.
- Combination: for one victim's exclusively owned blocks, tail-first free already maximizes its surviving contiguous
  prefix; rearranging a victim's blocks is not enough. Victim choice and cross-request competition remain.
- Simplicity open: beat LCF, LCF + queue retention, tuned admission limits.
- Oracle: hindsight victim on a fixed trace is invalid (counterfactual schedules differ); first bound by removing all
  measured re-prefill work.
- Correctness: victim changes can expose unwritten KV as cached (issue #59907).
- Relevance: measure natural preemptions before shrinking the pool artificially.
- Kill if removing all re-prefill has negligible benefit or tuned admission removes the opportunity.

## C3: release-order policy anticipated by oMLX; narrower question survives
- oMLX process_memory_enforcer frees unprotected hot-cache blocks by LRU under pressure and clears allocator cache.
- Metal purgeable resources exist (WWDC22).
- vllm-metal allocates one shared lazily committed backing store (storage.py); freeing a scheduler block does not
  release physical memory. Feasibility open: physical reclamation without full-pool copy.
- Elastic KV Cache (arXiv 2608.23658): lowering max_num_batched_tokens recovers more KV than an elastic controller.
- oMLX issue #2059: generation collapses from ~100 to 0.5-6.8 tok/s under pressure oscillation on 128 GB Mac.
- Surviving question: can vllm-metal release physically resident inactive KV from a shared paged allocation during
  serving, cheaply enough to beat tuned static headroom and existing pressure control?

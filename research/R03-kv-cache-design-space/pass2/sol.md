# Pass 2 reviewer: codex gpt-6.1-sol (provider default effort), 2026-10-12

Condensed by the lead judge; reviewer's links kept.

## Pass 1 correction
- C7 blanket kill too strong: [L30] excludes capacities below 24 GiB, where S3-FIFO/ARC beat LRU (0.491 vs 0.403).
  A conditional small-pool evaluation question remains; algorithms are established.

## C2: no rejection; MEASURE-FIRST, narrowed to self-prefix survival prediction
- Same PR corrections as astra: #56897 has p99 TPOT -2.6%; #54366 has 192 preserved tokens, no TTFT gain.
- Novelty narrows: Echo (arXiv 2504.03651) selects preemption victims with a cache-aware reward; TOPAS (2608.25523)
  jointly picks requests and resident prefixes. Remaining: prediction of self-prefix survival after V1 preemption.
- Adjacent: CRPD resilience analysis (LCTES'10) models which blocks survive intervening accesses.
- Simplicity: RFC #55265 reports "per-preemption waste is identical" for least-progress vs LIFO victims, and output
  changes at temperature zero when batch composition changes.
- Relevance survives: Tetris (SOSP'25 workshop PDF) reports cascading preemption in reasoning workloads; SGLang
  #26248 reports ~14x decode collapse after retraction, naming missing radix-sharing signal in victim choice.
- Scoop: RR-Evict (2609.32278) preserves partial prefixes and recomputes tails in SGLang.
- Screening threshold suggested: 10% p99 TPOT (heuristic).

## C3: KILL as stated (novelty)
- oMLX already shrinks unprotected cached KV by LRU from a macOS-derived ceiling and synchronizes GPU work before
  clearing (process_memory_enforcer.py, scheduler.py). OpenZFS ARC shrinks to avoid swap.
- oMLX #3338 (15.9 GB released in one second; release-order problem), #702 (jetsam kill; RSS exceeds tracked).
- A replacement on physical-page reclamation from vllm-metal's shared backing allocation would be a new card.

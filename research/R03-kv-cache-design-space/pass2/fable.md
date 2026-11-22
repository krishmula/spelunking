# Pass 2 reviewer: claudeAgent claude-fable-5-1 @high, 2026-10-12

Condensed by the lead judge; reviewer's links kept.

## C2: shortest rejection
vLLM V1 frees the victim tail-first into a free queue that is empty at preemption; the resumed request re-hits what
remains, so wait time decides survival, not victim identity. Victim choice by sunk compute/size is PR #53723/#56897.

- Code: kv_cache_manager.py:565-575 fails allocation only when free (incl. cached-but-free) blocks run out, so after
  preemption the victim's blocks are the only free blocks; any victim erodes at the same rate. scheduler.py:930 does
  prefix lookup on resume when num_computed_tokens == 0. kv_cache_manager.py:206 watermark default 0 is the anti-thrash
  baseline.
- Adjacent: Kills (engine already contains tail-first evict + re-hit). Simplicity: Kills pending one number.
- Oracle: tax is 10-23% online (PR #53723); open conflict: #53723 says tax invariant to prefix caching vs code re-hit.
  "One evening of instrumentation settles it."
- Sources: CacheOPT full text (victim order by SLO, remaining time, KV occupancy; ablation -27% tail TTFT); Beyond
  Prediction ICML 2026 (cache-aware preemption, hysteresis); SGLang retraction key; vllm-metal #479.
- Change of mind: (a) preemption fires at default pool, (b) candidate victims at the same moment differ > 2x in blocks
  surviving to resume, (c) hindsight-oracle victim beats LCF by > 10% p99 TPOT.

## C3: shortest rejection
SQL Server, Chromium, Apple purgeable memory all shrink caches on low-memory signals; oMLX ships a pressure-keyed
dynamic ceiling; vllm-metal fixed its pressure probe 2026-10-07 (#1033). SiliconBench 2.5x compression slowdown was
removed by headroom, not policy. Adjacent: Kills; Simplicity: Kills pending one number.
- Change of mind: static never-compress pool loses > 20% prefix hits vs default on the Air, and idle-first release
  beats uniform shrink on p99 TPOT/TTFT.

## Pass 1 kills: all stand.

# E02 verification contract

## Definition of done

The expanded E02 has runnable commands to prepare and freeze its corpus and manifest, validate technical correctness, conduct logging-cost pilots, run or resume the full sweep, and reproduce analysis from raw events.
It preserves stock mechanisms and historical results. Missing user predictions or pilot proof block measurement mode with an actionable error.
No command silently substitutes the historical four-configuration experiment.

## Required observable checks

1. Dry-run produces the exact 100-run plan with 100 requests per run, stable per-item seeds, paired repetition orders, and a pinned physical pool.
2. Corpus preparation uses the cached real Qwen tokenizer and chat template. Every frozen token sequence lies within 960 to 1,088 tokens. Repeated preparation matches the original checksum.
3. Main mode rejects missing numeric user predictions, missing corpus, incompatible engine state, missing valid pilot evidence, and insufficient disk.
4. A real scheduler scenario with fixed arrivals and completions makes identical decisions when instrumentation is enabled and disabled.
5. A request attached to a shared block releases one reference without reclaiming that block. A sole owner reclaims its nonreserved block. Null block is excluded.
6. Cache touch removes a free cached block without incrementing its content generation. Allocation of that block increments the generation.
7. Consecutive preemptions include the removed victim and all other candidates immediately before each release. All least-computed ties remain marked.
8. Replay reconstructs exact ownership, references, hashes, and ordered free queue and matches independently captured snapshots.
9. A deleted, duplicated, corrupted, or truncated event invalidates the run. Snapshot mismatches are failures, not warnings.
10. Async scheduled tokens count as completed only after a matching completion. Stale outputs cannot alter a reset progress episode.
11. Repeated partial re-prefill preserves prior completed work and the request's greatest previously completed token position.
12. Recovery partitions into original freed KV, original shared KV, and equivalent KV from elsewhere. A reused block ID cannot prove recovery.
13. The streaming client records first output including reasoning, correct final token counts, and terminal failures. Stable IDs join server history without treating chunks as tokens.
14. Reports expose every decision and candidate with links to snapshots, correct ties, undefined recovery for zero computed tokens, and unresolved episodes.
15. Interrupted or timed-out runs remain visible and cannot count as complete. Resume cannot overwrite attempts, rerun completed IDs, or apply changed settings to an existing manifest.
16. Summary keeps five scalar run values and their spread, distinguishes unobserved GPU time from wall time, and does not present alternative outcomes as measured.

## Interfaces and constraints

Use the actual `.venv/bin/python` runtime and actual scheduler where possible. Hardware timing needs the user's prediction gate first.
Keep fixtures small and literal. Test through public functions or CLI commands. Avoid assertions derived from the implementation under test.
Current free disk is about 20 GiB. Use a reserve and forecast, never delete data to make room.
Archive the current runner and observer with checksums before replacing them. Never open sealed historical summaries or event logs.
No commit to main, no PR, no deployment, no data deletion. All repository content is currently untracked.

## Throughput checkpoint

- Blocking first steps. Source grounding and design synthesis precede implementation. Fix the data contract before splitting module ownership.
- Independent workstreams. Design proposals write isolated directories. Implementation has one integrating owner, who can assign disjoint modules only after contracts are fixed.
- Shared mutable state. One owner writes each source file. The scheduler process owns the engine event log. The client owns its request log. Analysis joins them through stable IDs.
- Smallest safe decomposition. One integrating owner protects async accounting across observer, reducer, and runner. Parallel helpers are useful only for corpus/client or report modules behind a fixed contract.

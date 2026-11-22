# E02 design judgment (cross-judge, read-only)

Reviewer: Claude Fable 5.1 (`claude-fable-5-1`) as review subagent. Effort setting not exposed to the agent. Date 2026-10-20.
Inputs read in full: grounding.md, verification-contract.md, user-prediction.json, baseline.json, designs/{sol,astra,opus,fable}/design.md.
Source check, bounded to one contested detail: vllm-core `db9527a` `scheduler.py` lines 1545-1600, 2005-2035, 2962-2975. No historical run opened. No engine rescan.

## Scores (1-5)

| Dimension | sol | astra | opus | fable |
|---|---|---|---|---|
| Scientific validity | 5 | 5 | 3 | 3 |
| Passive stock-mechanism preservation | 5 | 5 | 5 | 4 |
| Replay / provenance correctness | 5 | 5 | 4 | 3 |
| Small maintainable structure, interface depth | 3 | 4 | 5 | 3 |
| Real local verification | 3 | 4 | 5 | 4 |
| Crash / reproducibility safety | 5 | 5 | 4 | 3 |
| Total | 26 | 28 | 26 | 20 |

## Blocking design errors

1. opus and fable derive completed KV as `num_computed_tokens - num_in_flight_tokens`. opus makes it the general `RequestState.completed`
   property and the source of `Preempted.completed` (section 3, section 4 pseudocode). fable uses it for episode `C` in `_preempt_request`.
   Source: `_preempt_request` sets `num_stale_output_tokens = num_in_flight_tokens` and resets `num_computed_tokens = 0`, and its comment
   says in-flight "already includes any undrained stale share" (scheduler.py 1560-1573). After resume, computed restarts from cache hits
   while in-flight still carries the prior episode's undrained stale steps. A second preemption before that drain under-counts C, possibly
   below zero. The formula is only valid when no stale share is pending, which is exactly the case E02 cannot assume (repeated preemption,
   check 7 and 11). The engine's own use at line 2971 clamps with `max(0, ...)`, which confirms it is a heuristic, not a completed series.
   Required: C and `completed` come from replay of epoch-matched work completions (sol, astra). Record the engine formula as a cross-check
   field only. Add a real-scheduler fixture: preempt with undrained in-flight work, resume, preempt again before drain; assert replay C
   equals the hand-computed value and that the formula diverges. This also satisfies "test real scheduler stale behavior".
2. fable ends history at a quiescent `SchedEnd` with no terminal seal, hash chain, or writer overflow accounting (unbounded deque, fsync
   every Nth call). A truncated tail that lands on a quiescent boundary passes `read_history`. Fails check 9 and 15. Not graftable as is.
3. Minor but must adapt: opus and fable gate predictions on README rows. The recorded prediction now lives in `user-prediction.json`
   (recovery >0.6 low pressure, 0.4-0.6 high pressure, frequency qualitative). The gate reads that file by digest and the README rows must
   match it; neither may be invented or edited by the runner. Codex-side predictions still need their own rows.

## Verdict

Base: sol. Its accounting contract is the only one with no correctness defect: work recorded with epoch, SchedulerOutput identity joined
at completion, completed intervals advanced only on matching valid completion, executed-discarded kept separate, H partitioned with
coverage evidence and "unknown" on ambiguity, sufficiency never ranked favorably, digest-chained frames with segment footers and seals.
Its weakness is structure: `expanded_e02.py` owns corpus, client, process, readiness, ledger and host sampling, and its verification plan
is a list of fixture names without a harness. Both gaps are filled by grafts below. Grafts, in priority order (three, no more):

1. opus module map, CLI and verification harness. Replace sol's four modules with opus's flat ownership table placed in an `e02/`
   package: `corpus.py`, `events.py`, `observer.py`, `history.py`, `sweep.py`, `report.py`, `tests/`. Keep opus's single-writer file
   table (engine: events, snapshots; runner: client, machine, STATUS), `O_EXCL` attempt dirs, dead-pid detection, source-hash pinning of
   every hooked method at startup, per-call `Digest` for loss localization, sticky `Fault` events, and the offline harness: real
   `AsyncScheduler` + `KVCacheManager` on a 24-block pool with scripted `ModelRunnerOutput`, twin comparison of every `SchedulerOutput`.
   Drop opus's `completed` property and `Preempted.completed`; replace with sol's work ledger.
2. astra epoch-work accounting and independent validation. Take astra's `Work` type (work ID, request epoch, SchedulerOutput identity,
   token interval, block identities, typed status with `cancelled` versus `completed_discarded` versus `unknown`), its rule that
   `DecisionSnapshot` is copied from live objects and never assembled from replay, producer counters with terminal watermarks, and its
   explicit statement that generations and completed intervals are observer concepts that snapshots cannot prove, so they are validated
   by controlled real-engine fixtures and the limitation is printed in the report.
3. fable queue-head assertion and partial-block rule. On `Alloc`, replay asserts the allocated IDs equal the replayed free-queue head in
   order; this turns free-queue order into a tested invariant at every allocation, not only at snapshots. Mark sufficiency `unknown`
   whenever a CoW partial hit is pending or a KV group is not full attention. Require the `X-Request-Id` join to be one-to-one in the
   client gate. Nothing else from fable.

Rejected whole: fable as base (10 modules, copied server code, no seal). opus as base (the completed-property defect sits in the core
data model and propagates through replay steps 3-5; fixing it means importing sol's work ledger anyway).

## Structural decisions the implementer should not reopen

- One expanded entry point: `python -m e02 <corpus|check|twin|pilot|freeze|main|status|report>`. Historical `run_experiment.py` and
  `e02_scheduler.py` stay in place byte-for-byte; their checksums are already in `baseline.json`. The README names them historical.
  No second expanded runner, no compatibility shim, no import from the historical scripts (copy what is needed, as fable did for server).
- Completion is epoch-matched work only. `num_computed_tokens - num_in_flight_tokens` appears in events as `engine_processed_hint` and is
  compared, never used.
- Generation increments on allocation pop only. `touch` and hash promotion do not change it. Recovery partition requires same
  (block ID, generation) and no intervening `HashRemove`; otherwise `equivalent` or `missing`, never `original`.
- Snapshots are direct reads of live engine objects at initial state, every decision, every 256 calls and end. Replay must reproduce them
  field by field; first mismatch raises and the run is `invalid`. Snapshots are never reducer inputs.
- History integrity: contiguous seq, per-event prior-value checks, hash chain, segment footers, terminal seal with count and SHA. Missing
  any one makes the attempt `incomplete`, never `complete`. Writer overflow or write error is a sticky fault that invalidates the run.
- Prediction gate reads `user-prediction.json` plus Codex rows, requires numeric values and reasons, and stores the digest in the manifest.
- Pilots: technical gates first (twin, fixtures, loss injection, resume), then five paired interleaved instrumented/uninstrumented runs at
  one low-pressure and one pressure cell with bounded short workloads. Pass only when median throughput and TPOT are within 5% with ranges
  recorded; fail or inconclusive blocks `freeze` and `main`. No main sweep is authorized by this judgment.
- Resume derives next action from disk only, skips validated complete IDs, never rewrites an attempt, replacement gets a new ID and reason.

## Scores rationale, one line each

sol: correct contract, god-module and abstract tests. astra: correct and well-structured, merge layer adds a global seq nobody needs.
opus: best engineering sketch and tests, one wrong formula at the center. fable: good replay checks, wrong formula, no seal, too many modules.

Next step for the integrating owner: write `events.py` and the offline twin test against the real scheduler, then the stale-across-reset
fixture from blocking error 1, before any sweep machinery.

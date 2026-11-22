# E02 implementation report

## Correction after fix round 1

This correction supersedes the readiness and pass claims in the original record below. The original text remains as an audit record.

The hardened source is implemented and offline verified with 154 CPU tests, zero skips, and passing corpus, plan, and non-main readiness checks.
All live behavior, thermal stability, logging-cost, budget, and lifecycle validation for this source remains pending explicit user instruction.
The earlier behavior pilot replays offline but predates sealed plans and current provenance. It cannot qualify the current source.
Cost-low remains interrupted, cost-pressure never ran, and `check --main` blocks. No new pilot, model server, actual warmup, main run, engine edit, commit, or PR occurred in this pickup.
The old 151-hour forecast and old pass table below are prior claims, not current validated readiness. Review-round2 N1 and N2 now have CPU regressions for bounded finalization, terminal budgets, failed cleanup, and resumed cleanup. The earlier cap-128 inevitability rationale is also superseded. An output maximum does not prove actual preemption, and the runtime gate must observe it.

The [fix-round1 report](fix-round1-report.md) records R1-R10 resolutions, exact commands, hashes, and remaining validation gaps.
The [final checksums](fix-round1-final-checksums.json) replace the old source inventory for review of the hardened source.

## Original implementation record

Date 2026-10-20. Owner: integrating implementation subagent (claude-opus-5-5). Helpers: two subagents (claude-opus-5-5, medium effort) wrote `e02/corpus.py` + `tests/test_corpus.py` and `e02/report.py` + `tests/test_report.py` against fixed interfaces. I reviewed and integrated both and applied one history fix the report helper found.

## Outcome

The expanded E02 is implemented as `python -m e02` over six modules. 76 offline tests pass on the real vllm-core scheduler and block pool.
The behavior pilot passed on real hardware. I stopped the logging-cost pilots on Krishna's instruction ("Don't run the experiment unless I tell you so"), so no cost verdict exists and `check --main` blocks the main sweep.
No main run, no freeze, no engine change, no commit.

## Stop confirmation (14:33 PDT)

- No `vllm`, EngineCore, `e02 pilot` or driver-loop process remains (`ps` shows none); port 8013 is free. The driver loop exited with 144 when stopped; its last server group (runner pid 71424) is dead.
- `runs/20261020-143026-pilot-cost-low/attempts/p2-off.a1` was running at the stop. I marked it `incomplete` with the reason "interrupted: user instruction to stop all live experiment work". Nothing was deleted. Its `sweep.lock` is stale and the lock code clears stale locks.
- The queued cost-pressure pilot never started. Nothing is queued or relaunched.

## Files (lines, sha256 prefix)

| File | Lines | sha256[:16] |
|---|---|---|
| e02/__main__.py | 123 | 826ba389431a4acc |
| e02/corpus.py | 453 | 7f91bb950b919903 |
| e02/events.py | 378 | f5b81d42a270de20 |
| e02/observer.py | 539 | 494cc3d5d6811e95 |
| e02/history.py | 956 | 61c053903f0c628e |
| e02/sweep.py | 965 | 5dbc3485c73f8cfc |
| e02/report.py | 614 | fec7bb300dc88363 |
| tests/ (8 files) | 1,325 | engine_fixture 063895ae11f44289, scenarios 0a6806e4dc79ae5f, test_twin 00e60fe9f090177e, test_replay 0a55c388cc8594f6, test_integrity 570083101b6f1b2b, test_sweep 2e0def4a72936dc2, test_report 96752b019b6bb676, test_corpus e250050b85205f5c |

Observer sha (observer.py + events.py) recorded by `check`: `4f2f98db6199e797…`. Historical `run_experiment.py` (807c716e…) and `e02_scheduler.py` (7b4b6772…) and `historical-plan.md` match `baseline.json`. README changed on purpose.
Frozen data: `data/corpus-v1.jsonl` digest `f201391309237e9c0dd41ca7f54c62c2824417c4fd71741339751f074840bfff`; `data/predictions.json` sha256 `3bcd8e66101466e2002c47c254cfcb68343bcca0602811ea8cc5f4cac5d89930`.

## Design as built

- `events.py`: closed set of 29 event kinds, positional JSON lines, contiguous seq, chained sha256 with a footer every 1,024 events and a sealing RunEnd (count, digest, faults, snapshots, unfinished). A writer error is sticky and leaves the journal unsealed.
- `observer.py`: `ObservedScheduler(AsyncScheduler)` with instance-level wrappers on the free queue, block pool, KV manager and coordinator. Every hooked stock method's source is pinned by sha256; startup refuses drift or an unsupported configuration (FCFS, no connector, no spec decode, block = hash block size, replay tokens 0, full-prompt reservation, single full-attention group, pinned block count). Unsupported paths (CoW, deferred free, unpin, partial blocks) emit a Fault. The measured history starts at the first successful prefix-cache reset with no live requests and seals at the next reset. Direct snapshots at initial, every decision (victim included), every 256 calls, terminal; a cheap state digest every entry call.
- `history.py`: pure reducer. Only the initial snapshot seeds the pool; every later snapshot and digest is compared. Allocation pops must equal the replayed queue head. Completion comes only from epoch-matched work; stale and orphaned work is discarded; computed-minus-in-flight is a cross-check count only. Recovery H is clipped to C and split into original freed, original shared, equivalent and unknown by (block, generation) and hash-removal history. Sufficiency is yes/no only for the running-path failure with one captured count and no reserve, watermark, lookahead or admission gate, else unknown. Client join is one to one by request id and corpus prompt sha.
- `sweep.py`: plan, prediction and readiness gates, owned server process group, streaming client, exclusive attempts, resume, lock, pilots, cost rule, budget, freeze, main.
- `report.py`: deterministic decisions.csv, summary.csv, index.html and per-run HTML with filter, sort, snapshot and request-timeline links.

## Commands run and results

| Command | Result |
|---|---|
| `python -m e02 corpus` then again, then `corpus --check` | created, unchanged, identical digest |
| `python -m e02 predictions` (twice) | same digest; frozen copy refuses change |
| `python -m e02 plan` | 100 runs, 10,000 requests, pool 776 (775 usable), per-round shuffled cells, one item order per round |
| `python -m e02 check` | pass: corpus, predictions, engine SHAs clean, source fingerprints, port, 20.9 GiB free |
| `python -m e02 technical` | 76 passed |
| `python -m e02 pilot behavior` (first) | runner bug marked it incomplete; evidence valid; kept and annotated |
| `python -m e02 pilot behavior` (rerun) | pass |
| `python -m e02 pilot cost-low` | stopped by user instruction after pair 1 |
| `python -m e02 check --main` | blocked: cost-low inconclusive, 1 of 5 pairs |
| `python -m e02 report runs/20261020-140632-pilot-behavior` | 166 candidate rows over 19 decisions, under 1 s |
| `python lab.py check` | 3 experiments, 0 problems |

## The 16 checks

| # | Status | Evidence |
|---|---|---|
| 1 Dry-run plan | Pass | `test_main_plan_is_the_fixed_paired_sweep`; `plan` output |
| 2 Corpus | Pass | cached tokenizer and chat template; lengths 1007-1045; rebuild identical; max common prefix 8 tokens, 0 full-block shares |
| 3 Main rejections | Pass | gate tests for predictions (missing, non-numeric), corpus, engine SHA, pilot evidence, failed cost verdict, disk, port; live `check --main` blocked |
| 4 Twin | Pass | 6 real-scheduler scripts identical outputs, deliveries, pool state; a perturbing observer is caught |
| 5 Shared release, null block | Pass | `test_shared_block_release...`, `test_null_block_is_never_counted` |
| 6 Touch vs allocation generation | Pass | hit blocks keep generation 1; final generations equal journal pop counts |
| 7 Consecutive preemptions, ties | Pass | two victims for one failed allocation with fresh snapshots; all ties marked |
| 8 Replay vs direct snapshots | Pass | every offline run; live pilot 51 snapshots, 7,731 digests |
| 9 Loss injection | Pass | 11 integrity tests, including consistently re-chained wrong values |
| 10 Async completion, stale | Pass | hand-derived C=79 with 1 discarded stale token; stale wait at two steps in flight |
| 11 Repeated re-prefill | Pass | q4: 16 + 15 = 31 re-prefill tokens, greatest 75, 25 discarded |
| 12 Recovery partition | Pass | shared + equivalent split; reused block ID 8 at a new generation is not original |
| 13 Streaming client | Pass offline and live | usage counts vs chunks; HTTP and truncated failures; live 24 of 24 joined, prompt sha matched corpus |
| 14 Report | Pass offline, rendered on live data | helper tests: rows per candidate, ties, undefined/unresolved, determinism, links |
| 15 Interruption and resume | Pass offline; live interrupted attempt preserved | resume tests; p2-off.a1 kept as incomplete |
| 16 Summary | Implemented, not exercised on main data | five-value stats with SD/median/range, Limits paragraph, no averaged p99 |

## Pilot evidence

- Behavior rerun `runs/20261020-141844-pilot-behavior`: 24 requests at target 16, cap 1,024; 671 s; 19 preemptions in history and in `vllm:num_preemptions_total`; 19 episodes resumed; stock last-running rule held at every decision; 4.2 MB journal, 5.5 MB attempt.
- Machine during pilots: load average about 6 on 10 cores from unrelated `git` and `BTLEServer` processes; swap grew from 4.0 GB to 6.1 GB. The memory profiler alone sized 391 blocks at utilization 0.38; the 776-block override sets the pool. Decode ran at about 0.2 s per step at batch 7 to 9, against E00's 87 ms at batch 8.
- Cost-low `runs/20261020-143026-pilot-cost-low`: pair 1 complete, p2-off.a1 incomplete (interrupted), pairs 2-5 not run. No verdict.

## Protocol changes recorded in decisions.tsv

- Cost-low shrank to 8 requests at target 4, cap 256 before it started.
- Cost-pressure, never started, is now 16 requests at target 16, cap 128. By arithmetic on the frozen corpus the first 12 prompts fill 772 of 775 usable blocks and need 868 at cap 128, so it must preempt. Not demonstrated live; the rule still marks any non-preempting run inconclusive. Five pairs and the 5% bound are unchanged.
- The forecast now scales from the behavior pilot.

## Open gaps

1. No cost verdict at either pressure. Both need five complete pairs, which needs Krishna's go-ahead. A new cost-low root would supersede the stopped one.
2. The forecast is a guess from one pilot. It says about 151 hours and 4.0 GiB for the 100-run sweep at this machine's current speed. The disk fits the 20 GiB free with a reserve; the time likely needs a decision on ceilings before `freeze`.
3. Memory pressure: the 776-block pool exceeds what the profiler sized under current memory use. Timing pilots under swap growth may come out inconclusive.
4. No reasoning parser is configured, so thinking text streams as `content`. First output still includes thinking; the reasoning-field counter stays 0.
5. GPU re-prefill time is unavailable by design; only completed re-prefill tokens are reported.
6. Generations and completed intervals are observer concepts validated by real-scheduler fixtures, not by snapshots.

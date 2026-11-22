# E02 implementation review, round 1

Source review only. No tests, experiment commands, model imports, servers, pilots, or GPU workloads ran. No source edits or commits. Reproduction cases below are reasoning from source for the parent's later verification, not executed results.

Reviewer: OpenAI gpt-6.1-sol, high effort, through Codex in T3. Author: Opus. No additional subagents. Kstack version `0.2.0+codex.20261012082835`.

## Actionable defects

### R1. P1: Pilot gates accept evidence for different source and settings

Location: `e02/sweep.py:708-727,821-874`; `e02/observer.py:83-89,109-110`. Applies to sweep hash `5dbc3485c73f8cfc8c425eba6638ce54a69d71bca4e84ed9ea596686960d4e8f`.

`pilot_evidence()` trusts the behavior pilot's saved verdict without replaying its evidence. Cost verdicts reread client records and `attempt.json`, but never validate the logging-on journals or compare pilot provenance with current corpus, predictions, engine, observer, client, reducer, pilot plan, or capture cadence. The observer fingerprint covers only `observer.py` and `events.py`. There is no pinned technical-verification result.

Reproduction reasoning: preserve passing pilots, change `history.py` or the client in `sweep.py`, or set inherited `E02_DIGEST_EVERY=0`, then request main readiness. Nothing compares these changes to what passed the pilots. Removing a logging-on pilot's event files after it finished also does not stop `cost_verdict()` from returning its previous numeric result. A saved behavior `{"verdict":"pass"}` alone is accepted; the existing failed-cost gate test deliberately supplies exactly that document.

Require current, validated raw evidence and fingerprints for the complete measurement implementation and effective settings. Freeze evidence digests and technical-check provenance, then reject drift. This is a gate defect, not a claim that current incomplete pilots passed. Contract 3, 8, 9, 13 and the logging-cost gate.

### R2. P1: Main resume neither freezes nor applies all settings

Location: `e02/sweep.py:312-322,387-390,879-946`; `tests/test_sweep.py:164-171`. Same sweep hash as R1.

`load_manifest()` validates only the run list. `run_main()` compares corpus, predictions, and the two-file observer hash. Server arguments, sampling, warmup, model selection and other implementation changes remain live module constants. The stored `fixed` section is never compared or used to execute. Deadlines and ceilings can be edited directly because there is no frozen manifest digest.

Reproduction reasoning: after some completed runs, change `SAMPLING["temperature"]`, `SERVER_ARGS`, or `WARMUP_TOKENS`, leaving the run list and observer files alone. Resume passes the comparisons and runs remaining IDs with changed settings. Alternatively edit `manifest.json` deadlines or ceilings; `load_manifest()` accepts them. The current test changes only a run's cap, so it does not cover this path.

Execute a fully validated frozen configuration, seal its digest, and compare all relevant source and settings on resume. Contract 1 and 15.

### R3. P1: A crash during validation leaves an unvalidated attempt marked complete

Location: `e02/sweep.py:622-644,545-558,949-963`. Same sweep hash as R1.

`run_attempt()` writes `outcome="complete"` to disk before loading history or checking logging-off client records. Validation then runs outside the protected execution block. A crash or validation exception between the two writes leaves that outcome behind. `next_action()` skips any attempt whose JSON says complete; `sweep_status()` counts it without validation.

Reproduction reasoning: interrupt immediately after line 632, or make `history.load_run()` raise at line 634. Resume skips this run even if its journal or join is invalid. This also feeds R1 because cost evidence trusts the outcome.

Persist an unfinished validation state until checks succeed. Finalize completion once, with evidence digests, and revalidate completion records on resume. Add the interruption case on both logging sides. Contract 9 and 15.

### R4. P1: Frozen disk and wall ceilings do not bound execution

Location: `e02/sweep.py:447-500,578-645,758-785,927-946`. Same sweep hash as R1.

Disk use and reserve are checked only after an entire main attempt. Pilots check disk only before their whole sequence. No active-run path watches disk, so a run can consume the reserve or fill the disk before cleanup and final metadata writes.

The total wall ceiling is a forecast check after at least 20 completed runs. There is no total deadline. Its timer resets on every resume, and startup, warmup and shutdown are outside the individual drive deadline. Before 20 completions, repeated failures with a replacement reason can consume arbitrary time without a wall check. Resume also starts another run even if the root already exceeds its disk ceiling.

Check budgets before and during every attempt, include startup and cleanup in bounded execution, and persist the total consumed budget across resumes and failed attempts. Stop visibly incomplete while reserve still permits final records. Add controlled clock and disk-limit fixtures rather than workloads. Contract 3 and 15, README pilot and run budget requirements.

### R5. P2: Server cleanup skips surviving members of an exited parent's process group

Location: `e02/sweep.py:341-353,533-542`. Same sweep hash as R1.

`Server.stop()` does nothing to the owned process group when `self.proc.poll()` already reports an exit. It also stops escalation once the group leader exits after SIGTERM, without checking remaining members. vLLM starts EngineCore in a separate process, as shown by `vllm/v1/engine/utils.py:214-231`. `reconcile()` records a dead runner but does not reconcile its recorded `server_pgid`.

Reproduction reasoning: let the API process exit while its EngineCore remains alive, then call cleanup. The `poll()` condition skips `killpg`, leaving the model process behind. A dead runner has the same orphan risk because the server was started in a new session. This can retain GPU load or memory and prevent a clean fresh-server resume.

Track owned process-group lifetime separately from its leader, verify cleanup of its members, and reconcile only positively identified owned groups. Contract 15 and fresh-server isolation.

### R6. P2: Snapshot corruption can be ignored or escape as an exception

Location: `e02/history.py:706-733`; `e02/events.py:324-328`. History hash `61c053903f0c628ecd08f6b339b591cc27edd964aa52f62f4487e36c6bccf333`.

`read_snapshots()` silently stops at malformed JSON, silently replaces duplicate indexes, and does not validate record shapes. `replay_attempt()` calls it outside its error-handling block. Event payload parsing likewise constructs dataclasses without validating field types.

Reproduction reasoning: append an invalid JSON line to a valid sealed `snapshots.jsonl`. The original snapshots are returned, the distinct count still matches the seal, and corruption is ignored. Append `null` instead and `rec["index"]` raises `TypeError`, aborting replay/report rather than returning an invalid attempt. Existing integrity tests change valid snapshot contents, so neither case is covered.

Parse every record strictly at the boundary, reject duplicate indexes and trailing corruption, validate payload types, and convert bad evidence into an explicit invalid or incomplete result. Contract 8 and 9.

### R7. P2: Final journal sync errors can still yield a complete run

Location: `e02/events.py:294-305`; `e02/observer.py:497-511`. Events hash `f5b81d42a270de20c8b13a45ae74396ebe62f2526c7545f443dc531ca0d53dd7`; observer hash `494cc3d5d6811e958591119383d9888dc4527861040691376a5a3452ea67a375`.

`Writer.seal()` writes the final seal and flushes it before `fsync()`. If final sync or close raises, it stores `self.error` but does not raise or return failure. `reset_prefix_cache()` drops the writer reference and returns the stock reset result. Replay does not receive that error.

Reproduction reasoning: make final `os.fsync()` fail after a successful flush. The readable file already contains a valid RunEnd; the runner can replay and mark it complete despite a recorded producer error. The same issue applies to a close failure after the seal.

Propagate final producer errors to the attempt outcome and persist them independently of the already-written seal. Cover final flush, sync and close failure paths. Contract 9 and required flush/error accounting.

### R8. P2: Reported time-weighted batch size clips time and omits the distribution

Location: `e02/history.py:854-858,895-899`; `e02/report.py:458-490`. History hash as R6; report hash `fec7bb300dc883632b9d73ee1a1e594aa154a67aae66ad06d76cd727d5923f6d`.

Every interval between scheduled outputs is silently capped at one second, including stalls and long prefill. The metric also combines all forward-pass kinds and reports only a mean. The README requires the time-weighted actual batch-size distribution and separate mixed-prefill and decode counts.

Literal example: batch 1 for 10 seconds followed by batch 8 for 1 second has time-weighted mean 18/11. The current formula reports 4.5. There is no distribution from which a reader can discover the clipping. The final interval is omitted too.

Define and retain the real interval boundaries, report the batch-size distribution, and separate forward-pass kinds. Label any schedule-based wall-time proxy explicitly. Contract 16 and README summary metrics.

### R9. P2: Competing GPU activity includes the experiment's EngineCore

Location: `e02/sweep.py:242-254,609-615`. Same sweep hash as R1.

The exclusion set for `other_gpu_share()` contains only the runner and API server PIDs. The process that performs model work is an EngineCore descendant with a different PID. Its AGX usage is therefore included in `other_gpu_percent` whenever the platform exposes its GPU counters.

Reproduction reasoning: the counter for an EngineCore PID rises while its API parent has no GPU usage. The comprehension treats that descendant as another application because it is absent from `ours`. Cost verdict rows then report the experiment's own activity as competing load.

Record the owned engine process identities or group membership and exclude those identities. Keep unrelated GPU users visible. Verify with literal before/after process counters. README environmental accounting and genuine logging-cost evidence.

### R10. P2: Warmup does not establish or record a steady thermal state

Location: `e02/sweep.py:601-609,648-659`. Same sweep hash as R1.

Warmup is four requests capped at 64 tokens, followed immediately by cache reset and measurement. No thermal samples, stability criterion or warmup result are recorded at that boundary. `machine_before` is captured before server startup and `machine_after` after shutdown. Neither proves the required state at measurement start.

Reproduction reasoning: a cold machine and a previously heated machine both pass the same fixed warmup, regardless of whether either reaches steady state. Their cost differences can pass as instrumentation cost with no recorded way to assess the difference.

Define a bounded warmup/stability procedure and record its samples and outcome before measured requests. Failure to establish the required state must remain visible. Do not run it without new explicit user instruction. README fixed setup, pilot and per-run procedure.

## Contract gaps and coverage limits

All 16 checks and the canonical README were inspected against every Python source file in `e02/` and `tests/`.

- Checks 1 and 15 have R2-R5 gaps despite the correct static 100-run, 100-request plan.
- Check 3 has R1, R2 and R4 gaps.
- Checks 8 and 9 have R1, R3, R6 and R7 gaps.
- Check 13's client records and joins are exercised in source, but the evidence gate has R1's provenance gap.
- Check 16 and the README summary requirements have R8 and R9 gaps. Candidate distributions and per-run recovery distributions are not produced as distributions; the report provides candidate rows and scalar recovery summaries instead. Add the required distribution outputs without treating events as independent runs.
- No additional actionable source defect identified for checks 2, 4-7, 10-12 or 14. This is not an execution pass. In particular, the real scheduler at `scheduler.py:904-910` blocks re-admission while deliverable stale work remains. No impossible immediate stale-resume trace is alleged.
- The repeated-preemption fixture has second C greater than first C. Add a literal allowed-transition case where another preemption occurs before the old greatest completed position is regained. The implementation retains `greatest`, but that specific required behavior is not independently demonstrated by the current fixtures.
- Source contains 71 test functions and one six-case parameterization, consistent with 76 cases by static counting. Zero cases ran in this review. README's pass claim belongs to the author and was not reverified here.

## No-comments findings

Read `skills/no-comments/SKILL.md`, its runtime adapter, and `agents/comment-sicko.md` before code review. Applied boundary discipline, model the domain, prove it works, test behavior not implementation, fix root causes, technical-writing and unslop. The user's no-subagent and no-source-edit instructions override the skill's spawn and fix steps.

Actual deletions: 0. Proposed deletions: 95 units, comprising 81 Python comment tokens and 14 docstrings. Examined: 82 comment tokens and 39 docstrings. Public API contracts are retained. The style-only `# noqa: E402` in `__main__.py:27` is retained because the offline environment must be set before importing the package. No legal headers or correctness suppressions were found in scope.

Proposed deletion counts by file, relative to the experiment directory:

| File | Comment tokens | Docstrings |
|---|---:|---:|
| e02/corpus.py | 4 | 2 |
| e02/events.py | 8 | 0 |
| e02/history.py | 8 | 0 |
| e02/observer.py | 9 | 0 |
| e02/report.py | 5 | 1 |
| e02/sweep.py | 12 | 0 |
| tests/engine_fixture.py | 0 | 2 |
| tests/scenarios.py | 0 | 1 |
| tests/test_integrity.py | 0 | 2 |
| tests/test_replay.py | 28 | 1 |
| tests/test_report.py | 3 | 2 |
| tests/test_sweep.py | 4 | 1 |
| tests/test_twin.py | 0 | 2 |

`MUST KILL` flags, with changes left to the owner:

- `events.Scheduled.entries`, `Completed.entries`, `Delivered.entries`: named types must express their tuple fields; the inline comments explain untyped shapes owned by this code.
- `history.Recovery.hit_blocks`, `PreemptionEpisode.originals`, `PreemptionEpisode.lookups`: encode the position and provenance fields in named types instead of requiring comments to interpret lists.
- `sweep.run_attempt`: replace the premature-completion workaround at line 631 with a real validation state. R3 describes its correctness consequence.

Skips: no edits, architect round, encodings, restoration or reruns, because this is read-only review. No separate fresh Comment Sicko agent, as explicitly prohibited. No named `IMPORTANT` or `do not remove` constraint comments required a how/why investigation. Existing source comments can be assessed without creating application-code changes. Test calculation comments are included in proposed deletions under the invoked strict policy; their literal expected values remain required.

## Audit-trail flags

- Execution authority changed during review. The current synthesis prohibits every pilot, behavior workload, cost workload, main sweep and model server until a subsequent explicit user instruction. The README now records stopped cost pilots and prohibits cost-pressure/main execution, but its runnable recipe and CLI help do not state the full boundary before the workload commands. `pilot`, `main`, and direct `run_attempt` have no suspension guard. Make that boundary explicit at those entry points and before the recipe; technical readiness or old predictions must never be treated as execution authority. This review started none of them.
- `sweep.py` changed from `02607008d710ed4403ad0ea00002fb03edc7176d3e0f6604ffad0219c4d3f8ef` to the R1 hash during review. The observed pressure-pilot cap changed from 512 to 128. An intermediate sweep hash was `6619c2815c1bb748b0b8315e384658d078526a5f8855832b2ed615221d2cd386`. The owner then changed `forecast_budget()` to use the behavior pilot instead of the pressure pilot. I re-read that function. Its per-request and cap scaling still uses current `PILOTS["behavior"]` constants rather than recorded pilot settings. Use the recorded plan values, protected by R1's evidence checks.
- README and synthesis also changed during review. The checksum sidecars record both versions. A final checksum comparison is required before using these findings against subsequent fixes.
- Source-only review did not inspect any pilot artifacts, historical `runs/20261014*`, historical summaries or results. Historical runner and observer were hashed only and still match `.audit/e02-implementation/baseline.json` byte for byte.
- Start inventory includes incidental pytest cache files. They are hashes only and were not used as test evidence. No source or test imports ran. Audit-file writes, text reads, SHA-256 calculation and Python lexical/AST inspection were the only local execution in this review.
- Start and end inventories: `review-round1-start-checksums.json` and `review-round1-end-checksums.json`. They contain full hashes for each scoped source and test file, README, synthesis and the verification contract.

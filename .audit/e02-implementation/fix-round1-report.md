# E02 fix round 1 report

## Outcome

R1-R10 and review-round2 N1-N2 are implemented and verified offline. The final full suite passed 154 CPU tests with zero skips, failures, or errors.
All live behavior, thermal stability, logging-cost, budget, and process-lifecycle validation for this source remains pending explicit user instruction.
Main remains blocked. The earlier pilots have incompatible provenance, cost-low stopped incomplete, and cost-pressure never ran.
No pilot, model inference, model server, real warmup workload, main sweep, or GPU experiment ran in this pickup.

This pickup continued the previous integrating owner's source and merged integrity-helper work after its explicit session limit.
It did not retry Claude, repeat the design arena, or start another subagent. Final owner is OpenAI gpt-6.1-sol with high effort through Codex in T3.
Kstack version is `0.2.0+codex.20261012082835`. Parent final review is pending. Review round 2 resolved eight original findings and requested N1-N2 for the remaining budget and cleanup paths.

## Finding resolutions

| Finding | Implemented behavior | Offline proof |
|---|---|---|
| R1 | Every pilot gate reloads its sealed plan, replays raw logging-on journals, joins clients, checks the planned requests and effective capture cadence, and compares complete implementation, tests, settings, corpus, predictions, and engine provenance. Final evidence digests detect changed or missing files. Saved verdicts alone cannot pass. Current full technical evidence pins source and test fingerprints plus a JUnit digest. | `test_evidence.py` covers saved-verdict rejection, deleted evidence, cadence, source drift, and raw cost verdicts. `test_sweep.py` covers the current technical record. Real `check --main` rejects the earlier unsealed pilot. |
| R2 | `ExecConfig`, `Deadlines`, `Ceilings`, and all `RunSpec` values belong to the SHA-256 sealed plan. Execution uses the parsed sealed configuration. Resume rejects source, input, engine, settings, or attempt-plan drift. Inherited E02 and VLLM environment settings cannot override capture or engine settings. | Strict configuration round trip, plan mutation, main-plan mutation, changed temperature, foreign-attempt, and effective environment tests. |
| R3 | The state remains `validating` while evidence checks run. The final atomic attempt write contains the outcome, validation result, and evidence digests. Resume reconciles interrupted validation. Status and resume revalidate complete attempts. | Crash injection on logging-on and logging-off attempts, successful interrupted validation, and post-completion evidence tampering. |
| R4 | Cumulative persisted attempt wall use includes failures and survives resume. Admission requires room for the next attempt's full bound. Startup, pending warmup, request driving, phase boundaries, cleanup, final validation, hashing, and the final complete metadata write check the disk reserve and wall budget. A signal deadline bounds synchronous finalization. The all-complete sweep path also checks its terminal budget. Cleanup escalates when the reserve check trips. Final records retain 64 MiB beyond the configured reserve. Validation time and resumed cleanup contribute to persisted consumption. Metadata writes must fit above the free-space reserve and within the root ceiling. | Controlled clocks and disk fixtures, cumulative attempts, no-room admission with no attempted launch, pending fake-HTTP warmup cancellation, CPU cleanup escalation, and validation-time accounting. |
| R5 | Owned process groups and recorded process identities survive leader exit. Cleanup applies TERM and then KILL to surviving owned members. Dead-runner reconciliation cleans recorded groups. Unidentified groups block resume without a signal. Any cleanup failure prevents completion and pilot qualification. Resume retains original process identities and stop reports, retries positively identified cleanup, and requires a replacement after it settles. | CPU subprocess leader-exit, reaped-leader, dead-runner reconciliation, and unrelated-group fixtures. N2 fixtures check survivor and unidentified reports, rejection as pilot evidence, retained identities, cleanup retries, and explicit blocking without an unrelated signal. No model server fixtures. |
| R6 | Event, snapshot, and client boundaries check exact fields and types. Snapshot trailing corruption, duplicate indexes, null records, and invalid shapes produce explicit invalid evidence. Bad content cannot escape the replay parser. | Parameterized event and snapshot corruptions, prior-value corruption, type and shape checks, and full replay tests. |
| R7 | Seal returns final writer errors. Flush, fsync, close, and prior sticky errors propagate through the observer reset and producer-error evidence. Replay rejects a producer-error marker even if RunEnd is readable. | Producer failure injection for journal flush, sync, close, snapshot sync, and earlier writes. |
| R8 | No interval clips at one second. Batch distributions retain actual batch sizes and forward-pass kinds. The final acknowledged interval ends at completion, and an unacknowledged final interval has no duration. Reports label host schedule-time proxy and unavailable device time. Candidate and recovery distributions appear per run in HTML and `distributions.csv`. | Literal batch 1 for 10 seconds then batch 8 for 1 second gives 18/11. Decode-only, mixed, prefill, final-interval, candidate, recovery, report determinism, and link tests. |
| R9 | All captured owned group and descendant identities enter the GPU exclusion set, including EngineCore. Unrelated GPU use remains visible. | Literal process tree and counters give 70% before EngineCore exclusion and 10% after it. These are fixture values, not hardware measurements. |
| R10 | Warmup has a fixed maximum of 12 rounds and 600 seconds. Each round has four requests forced to 128 output tokens. The last three throughput samples must lie within a 5% range relative to their median, with no reported thermal limit. The result and samples precede measurement. Completion recomputes the sealed rule from samples. | Controlled-rate and thermal-string fixtures, bounded fake-HTTP rounds, unstable outcomes, budget cancellation, and saved-verdict-without-samples rejection. No actual warmup ran. |

The additional repeated-preemption fixture uses real allowed scheduler transitions.
The victim first has C=50, then is preempted again at C=46 before regaining 50.
Its greatest completed position stays 50 and completed re-prefill remains cumulative.
`test_preempted_again_before_regaining_keeps_greatest` asserts the literal ledger.

## Pickup corrections and cleanup

The inherited source passed a claimed 137 tests before this pickup. I did not treat that claim as final verification.
I added budget checks while warmup requests remain pending, budget-aware cleanup escalation, raw warmup sample revalidation, finite float parsing, and persisted validation-time accounting.
Technical provenance now rejects skipped tests and checks the JUnit counts against the result record.
Eight new cases covered those pickup changes and invalid evidence at technical, client-byte, and undefined-TPOT boundaries. Nine further N1-N2 CPU regressions bring the final verified total to 154.

The named entry types in events and history preserve the old list-based wire format.
I removed 46 remaining scoped comment tokens and retained public API docstrings.
The only remaining Python comment token is the style-only `noqa: E402` in the CLI, where offline environment setup precedes imports.
No comments were restored, no public API docstrings changed, and no architecture round was repeated. After review round 2, 12 named private and test docstrings were deleted. Counts are in [private docstring cleanup](fix-round1-private-docstrings.json).
The user requested no further helpers, so the existing independent no-comments review supplied the accepted findings.
[Comment cleanup counts](fix-round1-comment-cleanup.json) record this pickup separately from earlier helper changes.

## Verification commands

All Python commands below used `/Users/krishna/dev/vllm-lab/.venv/bin/python`.
Experiment commands ran from `experiments/E02-preemption-rehit` unless noted.

| Command | Observed result |
|---|---|
| `python -m pytest -q -p no:warnings -p no:cacheprovider tests/test_evidence.py tests/test_lifecycle.py` | 42 passed in 14.05 seconds before final comment cleanup. |
| `python -m e02 technical` before N1-N2 | 141 passed in 28.47 seconds, then 145 passed in 28.71 seconds after invalid-evidence follow-ups. These intermediate source records remain preserved. |
| `python -m pytest -q -p no:warnings -p no:cacheprovider tests/test_finalization.py` | Nine N1-N2 CPU regressions passed in 6.27 seconds. |
| `python -m e02 technical` on frozen final source | 154 passed in 28.75 seconds. Record `technical/20261020-151512/result.json`. Before and after source fingerprints match. |
| `python -m e02 plan` | 100 runs, 10,000 requests, five paired repetition orders, 776 physical blocks, 775 usable. |
| `python -m e02 check` | Exit 0. Corpus, predictions, pinned engine, port, disk, and current technical evidence pass offline readiness. This grants no execution authority. |
| `python -m e02 check --main` | Exit 2. The earlier behavior plan predates sealing and cannot qualify current source. |
| `python -m e02 pilot behavior`, `pilot cost-low`, `pilot cost-pressure`, and `main runs/not-authorized-main` | Every command exits 2 at the authority gate without a workload flag. Unit fixtures make readiness and Popen fail if reached. |
| `python -m e02 pilot --help` and `main --help` | Exit 0. Both state the suspension, the user's instruction, and the `--execute` requirement. No execution flag was used. |
| `python -m e02 corpus --check` | Exit 0. Rebuilt digest equals frozen digest. |
| `python -m e02 status runs/20261020-143026-pilot-cost-low` | Exit 0. Unsealed root is unverifiable, so verified-complete count is zero. Preserved recorded outcomes stay visible. |
| `python -m e02 verdict runs/20261020-141844-pilot-behavior` | Exit 1. Legacy unsealed plan cannot supply gate evidence. |
| `python experiments/lab.py check` from the repo root | Exit 0. Three experiments, one ledger row, zero problems. The checker reads README metadata and checks run paths and env.json existence. It does not open historical run contents. |
| `python .audit/e02-implementation/fix-round1-offline-check.py` from the repo root | Exit 0. Offline replay and deterministic rendering from a new scratch copy of the permitted 20261020 behavior pilot. |

[Final CLI output and exit codes](fix-round1-final-cli-checks.json) preserve the literal commands.
Offline engine imports emitted existing duplicate Objective-C class warnings. They did not cause test or CLI failures.
The process inspection found the parent's existing static report HTTP server, not a model server. I left it unchanged.
Port 8013 had no listener at the check. No process was started except CPU tests, fake HTTP endpoints, CLI processes, and the preview renderer.

## Evidence and hashes

Technical source-and-tests fingerprint is `1113fd5cc5c2cd35575fbca020e83100fbcd1456cd24f7892e6058e97d5a9c24`.
The technical record SHA-256 is `1d8d54c875f51f32c3c6ee4ed70268cb37f0836cbb5a683b7a522fe34f58b4c2`.
Its JUnit SHA-256 is `41ea2ab29062b84c0fa2c32c239f309b44681009803223ba6b1c23533356515e`.
The full suite records 154 tests, zero failures, zero errors, and zero skips.

The preserved new behavior pilot replays 19 decisions, 51 snapshots, 7,731 digests, and 24 client joins.
Its report has 166 candidate rows and 909 distribution rows.
Two renders produce identical bytes. All candidate and index links resolve within the scratch report.
The T3 headless renderer inspected the index at width 1,280 in light appearance, with no console messages.
The preview renderer's inherited dark background made the static report's fixed dark text unreadable on the first preview. The light appearance matches the report's intended standalone display. No report styling changed in this hardening pass.
All original pilot file hashes match the pre-analysis hashes.
This proves offline replay compatibility, not current live validation.
[Scratch verification](fix-round1-offline/20261020-151537/result.json) contains complete input and report hashes.

| Preserved new pilot file | SHA-256 |
|---|---|
| `attempts/behavior.a1/attempt.json` | `3e7df8920afe738f842e08d293e74ef1ad15427c07deae01eb91525143429150` |
| `attempts/behavior.a1/client.jsonl` | `a1907f67e7cc4ebe297c986af204906c91e3b0648f36be82c6d5ab3979ea6db2` |
| `attempts/behavior.a1/events.jsonl` | `11eb14fbeee8a4388d8283d227b96361c87b2f2eb2e4db87b568f9103f796a85` |
| `attempts/behavior.a1/snapshots.jsonl` | `3ce01527ba57facb50920ad6b393daa5c4e771ad33f603dd3bef6f44fca69297` |
| `plan.json` | `73712d6483193845c3fc2e3261ed10350b2b0cd86107c3a6c15cda781da0fe22` |

| Implementation file | SHA-256 |
|---|---|
| `e02/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `e02/__main__.py` | `48c99ac57bab576a2a8c1c4ccf246fc8f5faf9ec568a6afb519c0c0850bd4e29` |
| `e02/corpus.py` | `ce5a11dac6cd4a7d5bc4fe936047c55c04717dd0156cd99c723a5fa940e93bf4` |
| `e02/events.py` | `6d63a32392d1aa2891a8b82c025fb7ecda247683ae886a4803d5fa7111ba8798` |
| `e02/history.py` | `4b4c87cb072b193210a2488945ac599b8d66acf7f7d8d88294f7dec8e7cd8add` |
| `e02/observer.py` | `1aade1b751c694aa3afd3118408ca7460e9c176483c2204406ed866fc382a620` |
| `e02/report.py` | `abb473d834036aae85bf5b8369f5eab78dadbd6bd8bfbd5f2b2758ff243d5396` |
| `e02/sweep.py` | `d17628443c5bcca311a6a3225bcd8c45ebdbd1b5e365b2ac3058bf52a69bf219` |

Frozen corpus digest remains `f201391309237e9c0dd41ca7f54c62c2824417c4fd71741339751f074840bfff`.
Predictions file SHA-256 remains `3bcd8e66101466e2002c47c254cfcb68343bcca0602811ea8cc5f4cac5d89930`.
[Final checksum inventory](fix-round1-final-checksums.json) includes all source, tests, README, synthesis, contract, and unchanged historical source checks.
`run_experiment.py`, `e02_scheduler.py`, and `historical-plan.md` match the original baseline byte for byte.
No historical `runs/20261014*` content or sealed historical result was read.
No engine file, commit, main branch, PR, or shared external state changed.

## Throughput checkpoint and principles

- Blocking first steps. Read the accepted contract, review findings, and merged source before edits. Keep all workloads stopped.
- Independent workstreams. No new fan-out. The existing helper changes were already integrated.
- Shared mutable state. One integrating owner wrote source. Original evidence was read only. Reports used a fresh scratch copy and audit corrections were appended.
- Smallest safe decomposition. Retain the six-module design and finish the remaining local checks directly.

Model the Domain kept event tuples, process ownership, configuration, and attempt lifecycle in named structures.
Boundary Discipline put corruption and settings checks at file, environment, and CLI boundaries.
Make Operations Idempotent made validation recovery and complete-attempt revalidation safe across interruption.
Prove It Works required the actual CLI, real CPU scheduler, raw pilot replay, and rendered artifact checks.
Test Behavior, Not Implementation required literal interval arithmetic, process signals, counter values, and crash outcomes.
All five leaf principle skills, runtime, Feature, how, no-comments, unslop, technical-writing, and show-me-your-work were read in this pickup.

## Review-round2 follow-up

N1 now carries the remaining sealed wall budget into synchronous validation, evidence hashing, and the complete metadata write.
The deadline uses a temporary SIGALRM handler and restores the previous handler and timer.
A deadline or reserve failure cannot leave a qualified complete outcome. If even the final failure metadata cannot fit above the reserve, finalization raises a visible budget failure and retains the earlier running or validating record and raw evidence.
The terminal `execute` path enforces cumulative wall, root size, and reserve even when `next_action` finds no remaining runs.
Controlled-clock validation and hashing overruns, an injected alarm, metadata reserve exhaustion, and the all-complete terminal path have literal CPU regressions.

N2 treats a cleanup failure, survivor list, or unidentified-member list as a failed lifecycle.
`finish_attempt`, `validate_evidence`, `verify_attempt`, and pilot verdicts cannot accept that attempt as complete.
Resume revisits finalized cleanup failures, keeps the original positive identities and prior stop reports, retries owned cleanup, and blocks if unidentified or surviving processes remain.
Settled cleanup leaves the original attempt failed and requires a replacement. Resumed cleanup time adds to persisted budget use.
The new fixtures never signal an illustrative PID. Signal calls are captured against literal synthetic process tables.

The user requested a correction to the cap-128 rationale. The append-only decision trail now supersedes its earlier inevitability claim.
An output cap is a maximum, not a forced output length. Capacity at that maximum cannot prove actual preemption.
The pressure gate still requires observed preemption on both logging sides, plus agreement between logging-on history and engine counters.
No pressure workload ran.

## Remaining validation and attention

All live behavior and cost validation must wait for a new explicit instruction for the specific workload.
No previous behavior pass or partial cost pair qualifies the final source. New compatible behavior evidence and five complete interleaved pairs at both pressures are pending.
The bounded warmup criterion is a host throughput stability proxy with thermal warnings, not a direct temperature measurement.
Its suitability, actual engine subprocess cleanup, environment accounting, and disk polling under model output growth remain unmeasured for this source.
Reserve checks poll during bounded operations. They do not impose an operating-system disk quota against an instantaneous write burst or unrelated disk consumers.
No main forecast, immutable main manifest, main run, policy improvement, or speedup is validated.
GPU re-prefill time remains unavailable. Reports retain completed re-prefill tokens and label schedule-time estimates.

Parent independent review is pending. No new reviewer was spawned, as requested.
The local decision trail remains append-only. Earlier implementation-report claims are superseded by its correction and this report.

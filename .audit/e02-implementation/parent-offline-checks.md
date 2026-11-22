# Parent offline verification

2026-10-20, after live execution was stopped at the user's instruction.
No model workload, pilot or main sweep was started by these checks.

## Initial implementation

`/Users/krishna/dev/vllm-lab/.venv/bin/python -m e02 technical` returned 76 passed in 15.81 seconds.
`python -m e02 plan` returned 100 runs and 10,000 requests with five recorded repetition seeds and a 776-block physical pool.
`python -m e02 check --main` returned exit 2 because the cost-low pilot has only one of five complete pairs.
`python experiments/lab.py check` returned 3 experiments, 1 ledger row, 0 problems.
Port 8013 had no listener, and no model server, pilot or main process remained.

## Reproduced review defects before fixes

These reproductions used temporary copies of the new behavior pilot's journal and snapshots, never the sealed historical run.

- R6. Appending malformed JSON to a valid snapshots file still returned a sealed journal and a Replay object. Corruption was ignored.
- R6. Appending `null` instead raised TypeError rather than returning an invalid evidence result.
- R3. A temporary attempt containing only `{"outcome":"complete"}` and no evidence caused `next_action` to return None. Resume trusted the marker.

The fresh implementation owner is fixing the independent review's R1-R10 findings with offline checks only.
Initial test success is not a claim that those review defects were absent.

## Hardened implementation

After the first fix owner reached a provider rate limit, the parent reran `python -m e02 technical`.
The full offline suite returned 137 passed in 26.35 seconds and wrote `technical/20261020-150619/result.json`.
The fallback owner continues integration and documentation. A fresh second reviewer checks the fixes.

Both original R6 reproductions now return explicit Invalid evidence results. Neither trailing malformed JSON nor trailing `null` is accepted or raises an uncontrolled exception.
The historical runner, observer and plan still match the baseline SHA-256 values byte for byte.
`pilot behavior` and `main /tmp/never-an-experiment`, without execution opt-in, returned blocked before any workload or server start.
`check --main` returned blocked because the earlier behavior plan predates sealed evidence. Old validation cannot silently certify changed code.
These are offline and default-denial checks, not experiment executions.

## Manual report browser check

The parent rendered a scratch copy of the already-recorded new behavior pilot at `parent-report-qa/input/report/`.
The original run files were not modified. A temporary loopback static-file server served only that saved report.
The T3 browser verified 166 candidate rows, a chosen-victim filter yielding 19 rows, a combined decision and chosen filter yielding one row, reset restoring all 166 rows, completed-token sorting, and valid snapshot and request anchors.
The browser tab and temporary static-file server were closed afterward. This check performed no model inference or measurement.

## Final verification

The fallback owner froze the implementation after addressing review-round2 N1 and N2. The parent read finalization, terminal-budget and cleanup-reconciliation paths and the nine literal CPU regressions in tests/test_finalization.py. Validation and hashing overruns cannot qualify as complete. Cleanup survivors or unidentified members prevent completion and pilot qualification; resume preserves identities and retries owned cleanup or blocks explicitly.

The parent independently reran the complete offline technical suite: **154 passed in 27.85 seconds**, with zero skips, failures, or errors. Record: technical/20261020-151703/result.json. Before and after source-and-tests fingerprints match 1113fd5cc5c2cd35575fbca020e83100fbcd1456cd24f7892e6058e97d5a9c24. Historical runner, observer and plan hashes still match the original baseline. The lab checker returned three experiments, one ledger row, zero problems.

Port 8013 had no listener. A focused process inventory found no model server, EngineCore, E02 pilot or main process. No experiment or model workload was started during final verification. Main remains blocked by the absence of compatible pilot evidence. The source-level review is closed; live behavior, warmup stability, logging cost and main feasibility remain unvalidated pending explicit user instruction.

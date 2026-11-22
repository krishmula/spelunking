# E02 independent review, round 2

R1, R2, R3, R6, R7, R8, R9 and R10 are resolved at the source level reviewed here. R4 and R5 remain partly unresolved. Two actionable lifecycle defects are below. This is not main readiness or permission to execute.

Reviewer: OpenAI gpt-6-astra, medium effort, through Codex in T3. The reviewed implementation contains Claude and Codex work. No nested subagents, architecture panel, source edits, PR or commit. Kstack version `0.2.0+codex.20261012082835`.

## Scope and evidence

Read synthesis.md, verification-contract.md, review-round1.md, fix-round1-interfaces.md, current E02 source and focused tests. Read the no-comments skill, runtime adapter and Comment Sicko instructions before code review. Applied the boundary-discipline, model-the-domain, prove-it-works, test-behavior-not-implementation, fix-root-causes, technical-writing and unslop leaves. The user's read-only and no-subagent restrictions override their edit and spawn steps.

No GPU computation, model workload, model server, live warmup, pilot or main run was started. No historical 20261014 artifact was opened. CPU scheduler fixtures use the existing cached configuration and scripted outputs. No default CUDA or uncached model fixture was introduced. No performance measurement was made.

The parent's `technical/20261020-150619/result.json` records 137 passing tests and matching before/after fingerprint `33e0353ed5620a1c0f17df49065c9c55d0ae73bb6a502bc69dddedb8b20688ba`. That is evidence for its recorded source only. During this review `check_technical()` rejected the changed source with `no passing offline technical record for the current source and tests: run python -m e02 technical`. This rejection is expected, not a new defect.

Start and end inventories are `review-round2-start-checksums.json` and `review-round2-end-checksums.json`. The fallback owner changed executable source as well as comments during review. I re-read the affected lifecycle paths. In particular, the final version adds pending-warmup budget monitoring, exact warmup-token validation, warmup-sample revalidation, finite-float parsing, budget checks around reset, cleanup budget monitoring, and ordinary validation-time accounting. Earlier concerns about missing pending-warmup monitoring and ordinary validation time being omitted are superseded by those changes.

## R1-R10 disposition

| Finding | Disposition | Evidence and limits |
|---|---|---|
| R1 stale or verdict-only pilot evidence | Resolved | `source_digests`, `technical_fingerprint`, `provenance`, `pilot_evidence` and `verify_attempt` bind current source, tests, configuration, engine and inputs. Verdicts are recomputed from replayed journals and clients, with completed-file digests checked. Environment capture cadence is stripped and set from the sealed configuration. `test_evidence.py` checks saved-verdict-only, deleted-journal, source-drift, cadence and failed-cost cases. Cleanup acceptance remains the separate R5 issue below. |
| R2 live resume settings and unsealed manifest | Resolved | `ExecConfig`, `SealedPlan`, `load_sealed` and `check_resume` validate exact typed fields, a plan sidecar digest, the fixed run plan, matching configuration/provenance and attempt plan identities. `Server`, client bodies and warmup consume the sealed config. Tests reject edited deadlines, changed settings and foreign attempts. |
| R3 complete before validation | Resolved | `finish_attempt` persists `validating` before evidence validation and writes `complete` only afterward. `verify_attempt`, `next_action` and status revalidate completed evidence. Both logging sides have interruption tests, plus successful interrupted-validation reconciliation. R4 below concerns the finalization budget, not the old premature-completion bug. |
| R4 disk and wall budgets | Unresolved in part | Admission now checks root size and cumulative attempt consumption. Startup, measured requests, pending warmup and cleanup have budget checks. Ordinary validation duration is now recorded. Final validation and evidence hashing still have no budget enforcement, and completion can be recorded after the total ceiling. See N1. |
| R5 orphan cleanup | Unresolved in part | `OwnedGroup`, recorded PID/start identities, group/descendant discovery and TERM/KILL escalation address an exited leader. `reconcile` handles dead runners still marked running. A failed stop can nevertheless become complete, pass verification and evade subsequent reconciliation. See N2. |
| R6 strict evidence parsing | Resolved | Every snapshot line is parsed with exact fields, typed nested entries and sequential unique indexes. Trailing invalid JSON, null and duplicates become Invalid. Journal headers and payloads reject wrong scalar types, bool-as-int, invalid completion kinds and wrong tuple arity. Focused integrity tests pass. |
| R7 sealing sync errors | Resolved | `Writer.seal` returns its sticky error. Observer sealing attempts snapshot flush/fsync/close and journal sealing, writes producer-error evidence and returns false on failure. The runner rejects failed sealing reset; replay rejects producer-error evidence. Flush, fsync, close and earlier-write failure tests pass. |
| R8 clipped time and missing distributions | Resolved | `batch_distribution` retains literal intervals, final acknowledgment boundaries and decode/mixed/prefill kinds. Tests cover batch 1 for ten seconds then batch 8 for one second, and an unacknowledged final interval. Reports expose batch, candidate and recovery distributions per run and label the host schedule-time proxy. No device-time claim is made. |
| R9 own EngineCore counted as competing activity | Resolved | Owned group and descendant identities are sampled before and after measurement. `other_gpu_share` excludes all recorded owned PIDs and the runner. Literal counters distinguish 70% when EngineCore is omitted from ownership from 10% when it is included. No real GPU counters were measured here. |
| R10 steady warmup | Resolved at source level | The sealed rule requires a throughput-stability window without reported thermal limits and records each sample. The latest `validate_evidence` recomputes stability, checks round order, token counts and duration. Failure blocks completion. This is a specified throughput/thermal-warning proxy, not direct temperature measurement or proof of live steadiness. No warmup workload ran. |

The allowed-transition repeated-preemption fixture now includes a second preemption below the previous greatest completed position. It does not fabricate immediate readmission while deliverable stale output remains.

## New actionable findings

### N1. P2: final validation can exceed the wall ceiling and still complete, leaving R4 open

Location: `e02/sweep.py`, `finish_attempt`, the final call from `run_attempt`, and `execute`'s `action is None` return. Reviewed sweep SHA-256: `4bfc5375e58367acdd703e7b1ba33cbb0afebb723c8155a8271c0e6a433946d3`.

The last run-attempt budget check occurs before `finish_attempt`. Validation and `evidence_digests` then run synchronously without a budget guard or enforced deadline. The updated end timestamp correctly records their duration, but recording consumption does not enforce the ceiling. The final attempt can return complete after exceeding it. On the next loop, `execute` returns when all runs are complete before calling `admit_attempt`, so no later budget check corrects that result. A large journal or slow storage can exceed the arbitrary overhead allowance during replay and hashing.

Literal controlled-clock reproduction below produced `outcome=complete`, `consumed_s=5000.0`, `ceiling_s=3600.0`. The injected elapsed time is a deterministic boundary test, not a measured replay duration. Existing `test_validation_time_counts_toward_attempt_budget` checks the end timestamp only and does not test enforcement.

Required fix: carry the remaining budget through validation and evidence hashing, bound that work, and ensure the terminal outcome cannot be complete after a budget violation. Retain enough disk reserve for final evidence and metadata. Check the terminal sweep budget as well as admission to another attempt.

### N2. P2: failed cleanup is accepted as complete and never reconciled on resume, leaving R5 open

Location: `e02/sweep.py`, `run_attempt` finally block, `validate_evidence`, `verify_attempt`, `reconcile`, and `execute` cleanup-failure branch. Same source hash as N1.

`run_attempt` records `cleanup_failed` and surviving or unidentified processes without changing its successful outcome. `finish_attempt` and `verify_attempt` accept that attempt as complete. `execute` notices the cleanup flag and halts once, but `reconcile` only handles running or validating attempts. On a later resume the complete attempt bypasses reconciliation and can be skipped. An EngineCore survivor need not hold the API port, so the port gate does not guarantee that this is blocked. Pilot verdicts also do not reject this lifecycle failure.

Literal reproduction below produced `outcome=complete`, `verified=complete`, and no cleanup attempt on reconciliation. It supplies the exact stop-report fields produced when `terminate_owned` cannot stop a member. It neither creates nor signals the illustrative PID.

Required fix: make unsuccessful cleanup a non-complete lifecycle state, reject it as pilot evidence, and preserve a resume path that retries positively identified cleanup or blocks explicitly until it is settled. Do not discard the owned identities.

### Literal CPU reproductions for N1 and N2

Run from `experiments/E02-preemption-rehit` with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /Users/krishna/dev/vllm-lab/.venv/bin/python`. This creates temporary synthetic logging-off evidence only. No workload, HTTP endpoint or process group is started.

```python
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, "tests")
import pytest
from e02 import sweep
from pilot_fixture import install, sealed_root, start_attempt

with tempfile.TemporaryDirectory() as d, pytest.MonkeyPatch.context() as m:
    install(m)
    root, ctx = sealed_root(Path(d), "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    rec.update(cleanup_failed=True,
               stop_report={"survivors": [424242], "unidentified": []},
               server_pgid=424242)
    final = sweep.finish_attempt(path, rec, "complete", None, ctx)
    checked = sweep.verify_attempt(path, ctx)
    def forbidden():
        raise AssertionError("cleanup attempted")
    sweep.reconcile(ctx, "p1-off", table=forbidden)
    print(final["outcome"], checked.status, "cleanup not retried")
    # Actual: complete complete cleanup not retried

    now = [1000.0]
    m.setattr(sweep.time, "time", lambda: now[0])
    path2, rec2 = start_attempt(ctx, "p2-off")
    validator = sweep.validate_evidence
    def slow_validation(*args):
        result = validator(*args)
        now[0] += 5000.0
        return result
    m.setattr(sweep, "validate_evidence", slow_validation)
    final2 = sweep.finish_attempt(path2, rec2, "complete", None, ctx)
    print(final2["outcome"], sweep.attempt_consumed_s(final2, 0),
          ctx.sealed.ceilings.wall_s)
    # Actual: complete 5000.0 3600.0
```

## No-comments findings

Actual reviewer deletions: 0. No source edits. The concurrent owner's cleanup removed the remaining ordinary comment tokens. The only final Python comment token is `__main__.py`'s `# noqa: E402`, retained because offline environment setup precedes package imports. No correctness or safety suppression was found.

Twelve docstrings still warrant deletion under the invoked strict policy. They describe private implementation or test contents without a public API or external-constraint exception:

- `e02/corpus.py`: module docstring and `_create_exclusive`.
- `e02/observer.py`: `_seal`.
- `tests/pilot_fixture.py`: module docstring.
- Module docstrings in `tests/test_evidence.py`, `test_history_metrics.py`, `test_integrity.py`, `test_lifecycle.py`, `test_replay.py`, `test_report.py`, `test_sweep.py`, and `test_twin.py`.

These are comment-policy flags, not correctness defects. Public schemas, callable contracts, the CLI execution boundary and the fixture's external engine constraints are retained. Prior MUST KILL flags for unnamed event/recovery tuple fields and the premature-completion workaround are resolved by named types and the validating state. No additional workaround-based MUST KILL flag, restoration, encoding offer or architecture work is requested.

## Decision-trail audit

The synthesis and CLI correctly state that `--execute` records a new explicit instruction and does not authorize an agent to run. No new permission is inferred from technical readiness.

At review time, the initial implementation-report.md still leads with 76 passing tests, a passing behavior pilot, and a broad table of contract passes. Those are earlier-version claims. They must be visibly marked superseded by round 1 and the current source provenance. Its 151-hour forecast is an old planning estimate, not a currently validated forecast. The parent-offline-checks.md update correctly distinguishes the 137-test source from ongoing fallback edits and says old pilot evidence cannot certify changed code.

The decisions.tsv cap-128 rationale says the workload "must preempt" based on capacity at the output cap. The cap is a maximum, not a forced output length. Arithmetic at that cap does not prove actual preemption. Retain the explicit unmeasured qualification and the runtime preemption gate, and weaken the inevitability claim. This review did not inspect or reinterpret historical runs.

No current cost verdict exists. Earlier unsealed pilots cannot establish current main readiness. Neither this review nor the offline tests establish live thermal stability, logging cost or main feasibility. Existing artifacts must remain preserved.

## Verification results

First targeted command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /Users/krishna/dev/vllm-lab/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_evidence.py tests/test_integrity.py tests/test_history_metrics.py tests/test_replay.py tests/test_report.py`.

Result: 82 passed, 14 dependency deprecation warnings. The two literal CPU reproductions above ran and returned the recorded results. A final targeted rerun after concurrent integration changes is recorded below.

Final targeted command used the same Python and offline variables, with `-m pytest -q -p no:cacheprovider tests/test_evidence.py tests/test_integrity.py tests/test_history_metrics.py tests/test_replay.py tests/test_report.py tests/test_lifecycle.py -k 'not warmup_runs and not warmup_without and not warmup_stops and not pending_warmup and not stop_kills and not reaped_leader and not unidentified_group and not reconcile_cleans and not reconcile_never and not cleanup_escalates'`.

Result: **94 passed, 10 deselected**, 14 dependency deprecation warnings. The exclusions avoid endpoint and process-lifecycle execution in this review; their source was inspected. This targeted result is not a replacement for a fresh full technical provenance record. No source or test hash changed between the final inventory and this test completion. The parent's report-QA documentation changed and was re-read. It reports a static saved-report browser check, not model execution.

The latest README correctly marks prior pilots unqualified under current provenance and separates offline commands from future explicitly authorized workload commands. Its updates do not close N1 or N2.

Changed since review start: README.md; e02/corpus.py, observer.py, sweep.py; tests/pilot_fixture.py, test_evidence.py, test_history_metrics.py, test_lifecycle.py, test_replay.py, test_report.py, test_sweep.py; and parent-offline-checks.md. Full start/end hashes are in the sidecars. These changes belong to concurrent work, not this reviewer.

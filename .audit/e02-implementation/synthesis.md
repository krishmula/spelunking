# E02 implementation contract

## Selection

Use sol's engine-owned event history and epoch-matched completion ledger as the base.
The independent judge agreed. Astra scored highest overall, but its cross-producer merge layer adds no needed causality.
Use opus's flat ownership modules under `experiments/E02-preemption-rehit/e02/`, its real-scheduler fixtures, source fingerprint checks, and per-call state digests.
Take astra's explicit work identities and independent-snapshot limitations. Take fable's allocation queue-head assertion and conservative allocation-sufficiency rule.
Reject the raw-minus-in-flight completion formula as an authoritative series, fable's unsealed history, and a new engine fork.
The unmeasured 1-2% digest-cost guess in opus's proposal is not evidence. Measure logging cost.

The candidates and six-criterion scores are in `designs/judgment.md`. This is the final package for implementation, not a further design exercise.

## Caller and structure

The single expanded entry point is `python -m e02` from the experiment directory.
Expose corpus preparation, check or dry-run, offline technical verification, pilot, immutable main-manifest freeze, main/resume, status, replay, and report.
Keep the old `run_experiment.py` and `e02_scheduler.py` byte-for-byte for historical use. They are not an alternate expanded entry point.

Use six ownership modules and a thin CLI:

- `corpus.py` owns meaningful frozen reasoning prompts, actual template tokenization, corpus checksums and sharing census.
- `events.py` owns schema, typed data, journal encoding, sequence/digest integrity, and producer sealing.
- `observer.py` owns the pinned experiment-local scheduler adapter, passive mutation hooks and direct snapshots.
- `history.py` owns the pure reducer, block/request/work state, episodes, provenance, decision rows and validation.
- `sweep.py` owns fixed configuration, manifests, streaming client, server lifecycle, environment, gates, budgets and resume.
- `report.py` owns deterministic CSV and self-contained HTML from validated history.

One integrating owner writes source. It may assign separate modules to helpers only after the interfaces are fixed, with one writer per file.
Do not create another arena or nested architecture panel. Implement this chosen shape.

## Scientific invariants

The named shapes are RunManifest, BlockIdentity, RequestState, Work, PreemptionEpisode, and Event.
Use typed lifecycle statuses and a pure reducer. Validate external files and network data at boundaries.
Work records scheduler output identity, request epoch, token interval, physical block generations and completion status.
Advance confirmed completion only on matching output acknowledgment. Stale executed work stays separate and cannot advance a reset epoch.
Record pending and raw scheduler progress independently. Maintain the greatest previously confirmed token position across episodes.
Generation advances on destructive allocation, never on release, touch, or hash promotion.
Snapshots are independent direct engine reads. Never initialize an intermediate replay state from the snapshot it is supposed to validate.
Every mutation event has sufficient prior-value evidence to detect missing changes. Preserve exact queue order and all hash aliases.
Validate snapshots at initial state, each preemption, periodic checkpoints and terminal state. Check allocation against the replayed queue head.
History needs contiguous sequence numbers, chained checksums, flush/error accounting and a final count/digest seal. Missing or corrupt evidence is invalid or incomplete, never pass.
Include the removed victim and all running candidates in each fresh pre-release snapshot. Mark all eligible least-computed ties.
Recovery H is clipped to confirmed C. Partition original freed, original shared, and equivalent KV. Unproven lineage is unknown, never an invented original recovery.
Keep repeated and unresolved episodes visible. GPU re-prefill time is unavailable unless independently attributable. Use completed re-prefill tokens.
Allocation sufficiency is unknown unless a pure calculation covers the actual constraints. Do not invoke the allocator hypothetically.

## Verification and operation

Implement the 16 checks in `verification-contract.md`, with the real local scheduler and block pool for technical fixtures.
Tests must respect actual scheduler behavior. In the configured no-connector path, deliverable stale output can block resumption; do not fabricate the judge's proposed immediate stale-resume trace if the engine forbids it.
Exercise both that wait and repeated preemption after real allowed transitions. Prove the epoch ledger with hand-computed expected values.
Use the cached real Qwen tokenizer and model. No external corpus download is needed.
Record the user's actual prediction from `user-prediction.json`; do not invent numerical preemption rates or detailed causal reasoning.
Move that prediction into a stable experiment-owned data artifact and freeze it by digest. Keep Codex's existing guesses distinguishable.
The user explicitly prohibited experiment execution at 14:33 PDT on 2026-10-20.
No pilot, behavior workload, cost workload, main sweep, or model server may start without a subsequent explicit user instruction.
Implementation, offline deterministic CPU tests, and read-only review may continue. Preserve already-created artifacts.

The main sweep is exactly 20 cells, five repetitions, 100 requests per run. No main measurement runs in this implementation task.
Do not silently shrink the sweep. Freeze numeric deadlines, disk ceiling and reserve, provenance and source fingerprints before main.
Pilots use short bounded workloads. Five interleaved logging-on/off pairs at low pressure and pressure are required before a cost verdict.
Confirm actual preemption for the pressure pilot and actual request work for both sides. Failed or inconclusive cost checks block main.
Tune the pilot size to test logging without turning implementation verification into the 100-run sweep. Record any different pilot pool or cap.
Current free disk is about 20 GiB. Use budget forecasts and a free-space reserve, with visible incomplete outcomes; never delete data.
Fresh server, warmup, drain, verified cache reset and fixed pool are required. Client and engine IDs must join one-to-one.
Stream thinking and answer output, final usage counts and first/last arrival. Chunks are not tokens.
Resume uses immutable attempts, exclusive ownership and explicit replacement reasons. Never overwrite attempts or silently retry requests.
Keep historical runs sealed. Log new run roots in README and provide env.json, so `experiments/lab.py check` passes.

## Delivery limits

The workspace has no commits and is entirely untracked. No commits on main, no PR, no external messages or deployment.
Use `/Users/krishna/dev/vllm-lab/.venv/bin/python` for engine commands.
Preserve engine checkouts. Instrument experiment-local objects instead of changing scheduler, allocator or cache mechanisms.
Report exact commands, test counts, pilot evidence and remaining readiness gaps. Do not declare main-ready without actual gate evidence.

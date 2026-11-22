# E02 design proposal: one engine-owned event history

## Usage (caller's view)
```python
# Proposed API, not implemented. Preparation performs no measurement.
from expanded_e02 import Experiment, FrozenPlan
plan = Experiment.prepare(corpus_spec, fixed_config, predictions, budgets)
plan.write_new(plan_path)  # immutable corpus + manifest; rejects overwrite
checks = Experiment.verify(plan)  # technical fixtures; no workload measurements
pilot = Experiment.pilot(plan, checks)  # requires frozen numeric predictions
ready = Experiment.freeze_main(plan, pilot)  # forecasts, deadlines, reset proof
result = Experiment.run(ready, resume=True)  # complete 100-run plan or explicit stop
Experiment.report(result.directory)  # offline rebuild of HTML, CSV, metrics
```
CLI mirrors these six operations; measurement commands require validated readiness artifacts. A reviewer opens `report/index.html`, filters a decision, follows its direct snapshot and request history,
and can rebuild the same report from manifest, events, snapshots, and client records without an engine.

## Problem and grounding
This is an observation-only implementation of the expanded README, not a finding or E03 policy test. The current two scripts implement the earlier scope and must remain usable under their existing names.
Source inspected: scheduler allocation failure selects `running[-1]` under FCFS, removes it, then calls `_preempt_request`; progress advances before completion. Async output handling separately drains stale work.
The pool owns references, cache hashes, and eviction order; request tables own attachments. `touch` removes zero-reference hits without allocation; release prepends unhashed blocks and appends hashed blocks.
Thus request-local counters and allocation pops cannot reconstruct original-KV survival. Sources: local `scheduler.py` (735–825, 1967–2078), `async_scheduler.py` in full, `block_pool.py` (669–808),
existing observer in full and runner workload/client setup. Grounding and canonical README read in full. Uncertain: complete hook coverage, streaming ID round trip, cache reset API, and backend completion timing
must be confirmed during implementation. No historical run, summary, measurements, or engine import was opened.

## Shape and module map
Add four experiment-local modules alongside, without replacing earlier scripts:
- `expanded_e02.py`: sole orchestration/configuration owner; corpus, streaming client, process lifetime,
  immutable readiness artifacts, execution ledger, host observations. Public API above and thin CLI.
- `e02_observer.py`: `ObservedAsyncScheduler`; installs instance-scoped passive method hooks on its own
  managers/pools/free queue. Owns event sequence, work ledger, identity generations, direct snapshot traversal.
- `e02_history.py`: domain types, strict codec, pure replay and derivations; sole accounting definition.
- `e02_report.py`: offline HTML/CSV rendering and scalar aggregation from validated history.
Observer calls each stock method exactly once, preserves arguments/results/exceptions and never chooses victims. Hooks copy immutable values synchronously; a single writer drains encoded batches. No shared client log writer:
engine history and client stream history stay separate and join by stable request identity at report time. This interface hides readiness, process, and provenance complexity; callers supply scientific inputs and budgets.
Per boundary-discipline, parse wire formats inside owners; per minimize-reader-load, no plugin framework or bus.

## Typed data model and signatures
```python
RunStatus = Literal['planned','starting','measuring','complete','incomplete','invalid','failed']
RequestStatus = Literal['waiting','running','preempted','finished','failed','aborted','unfinished']
EpisodeStatus = Literal['waiting','resumed','regained','preempted_again','finished','unresolved']
WorkStatus = Literal['scheduled','completed_valid','completed_discarded','canceled']
Sufficiency = Literal['yes','no','unknown']
@dataclass(frozen=True)
class RunManifest:
    schema: int; run_id: str; cell: tuple[int,int]; repetition_seed: int
    corpus_digest: str; request_order: tuple[int,...]; sampling_seeds: tuple[int,...]
    effective_config: FixedConfig; provenance: Provenance; limits: RunLimits
    prediction_digest: str; instrumentation_digest: str; plan_digest: str
@dataclass(frozen=True)
class BlockIdentity:
    pool: str; group: int; physical_id: int; generation: int
@dataclass
class RequestState:
    id: str; engine_id: str; status: RequestStatus; admissions: list[Admission]
    prompt_ids: tuple[int,...]; output_ids: list[int]; raw_progress: int
    completed: IntervalSet; greatest_completed_position: int
    pending: dict[WorkId, Work]; blocks: tuple[BlockIdentity,...]
@dataclass(frozen=True)
class PreemptionEpisode:
    id: str; request_id: str; failure_id: str; decision_seq: int; C: int
    originals: tuple[PositionedBlock,...]; release_classes: tuple[ReleaseClass,...]
    lookup_attempts: tuple[LookupId,...]; resume_work: WorkId | None; status: EpisodeStatus
@dataclass(frozen=True)
class Event:
    run_id: str; seq: int; monotonic_ns: int; scheduler_call: int | None
    previous_digest: str; payload: EventPayload  # closed discriminated union
# Payloads: RequestTransition, AllocationAttempt, Attach, Detach, RefChange,
# HashAssign/Evict, QueueInsert/Remove, Lookup, Schedule, Complete/Cancel,
# PreemptDecision/Release, SnapshotRef, InitialState, EndState.
def replay(history: ValidatedHistory) -> ReconstructedRun: ...
def compare_snapshot(state: ReconstructedState, direct: DirectSnapshot) -> None: ...
def decision(state: ReconstructedState, episode: PreemptionEpisode) -> DecisionTable: ...
def recover(history: ReconstructedRun, episode: PreemptionEpisode) -> Recovery: ...
```
`BlockState` owns refcount once, reserved identity, valid coverage and hash identities; owners are derived from request attachments. History indexes `(request_id, seq)`, `(BlockIdentity, seq)`, decision and work IDs
are built once by replay, supporting request traces, victim provenance and report links without extra caches. Generation advances on destructive allocation reuse, not release or `touch`; hash change and coverage writes
are separate events. Initial nonreserved blocks start at generation zero. Null blocks never count reclaimable.

## Observation and accounting invariants
Hook allocation attempts and manager attachment/detachment, pool hash/ref operations and queue primitives; record queue operation order, affected identities and anchors, including nested operations, not only net deltas.
A startup hook-coverage audit enumerates every mutation path in the pinned effective configuration; reject unsupported pool types/features. Hook arguments containing iterators must not be consumed before stock code.
Every failure captures requested slots, free capacity, watermark, full-prompt reservation, lookahead and source branch/constraint; associate repeated victim releases with the same failure ID until the retry.
Before each `_preempt_request`, directly traverse engine requests, tables, pool and queue, including removed victim and all running candidates; save running order from the enclosing allocation attempt before removal.
Take fresh snapshots per release, eligibility via stock predicate, and before/after free counts with sequence bounds. Snapshot collector reads engine objects directly, never the replay state. Also compare initial/end state.
Compute least-completed ties only among immediately eligible candidates; preserve raw progress/pending work. Allocation sufficiency uses a proven pure constraint calculation; otherwise `unknown`, never a favorable rank.
Record scheduling work with request epoch, token-position intervals and block identities; associate the exact SchedulerOutput object with completion. Do not infer completion from scheduler-call gaps or progress alone.
Completion observes stock stale classification, records executed-discarded work separately and never restores its KV after preemption. Preserve pending work on abort until delivery/cancellation is reconciled.
At actual resumed execution select its successful lookup, not the latest failed attempt; clip hits to `[0,C)`. Partition H into original-freed, original-still-shared and equivalent-elsewhere by generation/hash/position
history. Require coverage evidence; ambiguous lineage invalidates that conclusion rather than guessing. For every valid completion, count intervals below the request's prior greatest completed position as re-prefill,
then advance the maximum. Keep accumulated work across repeated preemptions; associate work once with its active episode and retain prior episode totals. Report `H/C` undefined at C=0 and unresolved episodes explicitly.
Regaining C is elapsed recovery time; attributable device time stays unavailable unless independently validated. Writer frames have schema, length, contiguous sequence, digest chain and durable segment footers. Overflow,
write errors, malformed frames, missing footer or missing final seal invalidate the run; never drop silently. Periodic producer/writer counts and direct checkpoints expose loss. A crash preserves the valid prefix as
incomplete evidence, never certifies a missing suffix. Snapshots cannot replace required mutation events.

## Fixed setup, execution and readiness
Freeze 100 distinct locally generated reasoning problems from existing templates with relevant premises/data, problem-specific opening and no system message; rendered lengths 960–1088, target 1024. Save text, tokenizer
revision, chat template, IDs, checksum, length distribution and pairwise/full-block shared-prefix distribution. Freeze model revision, thinking enabled, temperature .6, top_p .95, normal stopping, cap includes all output;
item/repetition-derived sampling seeds and identical repetition shuffle across cells. Preserve actual stop reasons. Manifest expands targets 4/8/16/24/32 × caps 512/1024/2048/4096 × seeds 11/23/37/53/71: 100 runs,
100 requests each, 10,000 total. Shuffle cells independently per round; retain immutable run IDs and attempt lineage. Pin 776 physical/1 reserved/775 usable blocks, size 16, model length 9216, token budget 2048, FCFS, async,
prefix cache enabled, fixed KV dtype, watermark/lookahead, no speculation/offload/connector. Assert hash size matches block size, replay tokens zero and full-prompt reservation true. Capture effective class/config and
SHAs, dirty diffs, versions, machine/power/thermal/swap/GPU activity. Reject pool mismatch before submission. Each run uses fresh server, recorded thermal warmup, drain then cache reset; verify empty hashes and initial
queue/ownership snapshot. Stable client IDs encode run/item; capture server/engine ID mapping and output IDs. Streaming counts first reasoning or answer token arrival, not role-only frames; chunks are not tokens. Reconcile
server token IDs/usage with client final text; timestamp first/last output, record undefined TPOT for short output. Maintain target outstanding including waiting requests; replace completion/failure immediately until 100 submitted,
then drain. No request retries. Preserve startup/steady/drain, actual batch-size time weights and mixed passes. Before measurement pilot, require both named prediction entries numeric with reasons; implementation may ask
once asynchronously but cannot invent Krishna's prediction. Technical checks remain allowed while it is pending. Pilot gates: real shared/unshared/partial/reuse/multiple-victim/repeated/unresolved fixtures; independent snapshot
replay checks; reclaimed-count/stock-choice checks; logged/unlogged identical offline arrival/completion traces; streaming ID/timing/count reconciliation; five paired interleaved runs each low/pressure, median throughput and
TPOT within 5%, ranges/errors/load/thermal/log volume recorded. Failure or inconclusive means optimize/repeat. Freeze total disk/time ceilings and run wall/no-progress deadlines after pilot forecast; reject infeasible main.
Reforecast after round one; stop incomplete if remaining work exceeds ceilings. One process lock owns sweep; atomic ledger transitions, exclusive run directories, durable seals. Resume skips only validated complete IDs;
interrupted attempts remain immutable, replacement needs new ID/reason. Never continue old engine state or shrink sweep.

## Report and rationale
HTML and CSV include every candidate/decision, chosen victim, all ties, eligibility/ranks, admission order, blocks/shared/reclaimable/reserved, continuous shared prefix, sufficiency, actual release and victim recovery.
Link immutable snapshots/events/request timelines; annotations explain alternatives worth testing, never outcomes. Report all five run values, mean/sample SD/median/range; request outcomes, caps/lengths, preemptions/rates,
candidate/recovery distributions and provenance, unresolved episodes, re-prefill, actual batch sizes, TTFT/TPOT, latencies/throughput/stalls. Define startup-through-drain throughput interval and first-to-last/output-count TPOT.
Keep incomplete/failed runs visible; no-preemption recovery undefined; show per-run tails or counted pooled tails, never mean p99 or event-level independence. Versioned reports regenerate atomically from sealed inputs.
Alternatives (whole structures, per exhaust-the-design-space): snapshot-only scheduler observer is smaller and hides replay machinery, but loses transient queue/hash/reuse changes; reject as scientifically insufficient.
Engine-wide native event API plus central collector hides mutation details better and is broadly reusable, but requires intrusive engine edits and maintenance beyond E02; reject unless passive hooks prove impossible.
Chosen history-plus-independent-snapshot structure accepts hook coverage and log-cost burden for experiment-local passivity and auditable lineage. Red-flag screen: one accounting owner, one expanded entry point, no wire API,
no temporal module pipeline; earlier scripts are explicitly historical, never an alternate expanded path. Synthesis decision: independent runner recommendation only; no claim to have synthesized other runner outputs.
Risks/questions: Can hooks capture every mutation and attributable completion within 5%? If not, readiness fails; propose the smallest explicit observation hook or mark accounting unavailable, never silently approximate.
Next implementation step: build hook-coverage and deterministic block/work fixtures before workload execution. Verification performed here: read-only source inspection and proposal line-count/scope check only; no tests,
measurements or experiments started. Model confirmed by host: gpt-6.1-sol via Codex; effort not exposed/confirmed.

# E02 proposal: one engine history, independently checked

## Usage (caller's view)
Proposed commands, not implemented or executed. Run from the experiment directory using the engine environment's `.venv/bin/python`.
```python
from e02 import freeze, execute, review
manifest = freeze(spec, corpus, predictions)  # immutable plan; no measurements
pilot = execute(manifest, mode=Pilot(technical_checks, pilot_budget))
result = execute(manifest, mode=Main(pilot.readiness, frozen_limits), resume=True)
artifacts = review(result.history)  # rebuild HTML + CSV without running the engine
```
`freeze` permits a draft with missing predictions; `execute` rejects every measurement mode until numeric predictions and reasons are frozen. Technical checks remain available without predictions.
CLI exposes these same three operations plus technical verification; no separate legacy-compatible expanded runner. Main execution is not authorized by this proposal.

## Problem and grounding
E02 must explain each stock decision and the actual victim's recovery across 100 runs. Optimistic scheduler counters and block IDs alone cannot establish completed or surviving KV.
Overview: a client owns arrivals; the scheduler owns request progress; the pool owns shared physical blocks; execution acknowledgments establish completed work. Observation must retain all four without taking over policy.
Key concepts: distinguish physical lifetime, cached hash aliases, request token positions, pending execution, and preemption episodes. The event history is evidence; report tables are derived views.
How it works: allocation failure causes FCFS to remove the last running request, then release it. Allocation retries can release more victims. Admission failure need not preempt anything.
Where things live: existing `run_experiment.py` owns templates/client/server; `e02_scheduler.py` subclasses AsyncScheduler. Core `sched/scheduler.py`, `async_scheduler.py`, `kv_cache_manager.py`, and `block_pool.py` own mechanisms; metal `v1/model_runner.py` consumes scheduler progress.
Confirmed by bounded source inspection: core HEAD `db9527a46873454610df6dbedf79a36d6bf1a7f6`, metal HEAD `572322e17584bb54afbc9e143f72009c2398bbba`; scheduler lines 745–814 choose/remove/release; lines 1967–2045 join output to SchedulerOutput and drain stale work.
Confirmed: allocation checks full-sequence fit, watermark and reserved capacity; pool `touch` removes free cached blocks without allocation; release prepends unhashed blocks and appends hashed blocks. Hash aliases and duplicate hashes exist.
Gotchas: completion acknowledgment is distinct from output-token delivery; stale executed work can complete after release. Source comments about GPU completion require backend validation, not faith. No historical artifacts were opened.

## Shape and module map
Add a flat `e02/` package alongside unchanged historical scripts: `__init__.py` exposes the three operations; `contract.py` owns domain types and pure replay; `capture.py` owns engine adaptation; `sweep.py` owns corpus, client, server and run lifecycle; `report.py` owns derived review views.
This is ownership decomposition, not load/transform/save layering. Public calls hide paths, wire formats, hooks and process management; callers supply scientific inputs and inspect typed results.
`capture.py` uses an AsyncScheduler subclass and instance-local hooks on its existing manager, pool and free queue. Every intercepted stock operation executes exactly once with unchanged arguments/results/exceptions. No replacement scheduler loop, allocator, policy or engine source edits.
Hooks instrument mutation boundaries, not only schedule boundaries: allocate/touch/free/unpin; hash insertion/removal/move; queue pop/remove/prepend/append; manager attachment/detachment; arrival/admission/preemption/finish/abort; scheduling and output completion. Audit every reachable mutation path for this fixed configuration before declaring coverage.
Use nested operation IDs so lower-level queue/hash events and outer refcount/ownership deltas describe different fields, never duplicate a mutation. Capture immutable scalar copies immediately; serialization must not retain live engine objects or consume iterators twice.
One scheduler-thread producer assigns engine sequence numbers before enqueue. Client and environment journals have separate owners. A canonical history merges them by monotonic timestamp, producer sequence and explicit request/work links, assigning global sequence numbers; tie order is deterministic, not claimed causality.
Engine causal sequence remains authoritative. Producer registration, counters and terminal watermarks make omitted producer data detectable; merged history cannot claim completeness without all journals.

### Typed data model (Python-style sketch)
```python
RunStatus = Literal['planned','running','complete','incomplete','invalid']
RequestStatus = Literal['waiting','running','preempted','finished','failed','aborted','unfinished']
EpisodeStatus = Literal['waiting','recovering','regained','repreempted','unresolved']
WorkStatus = Literal['scheduled','cancelled','completed_valid','completed_discarded','unknown']
@dataclass(frozen=True)
class BlockIdentity:
    pool: str; group: int; physical_id: int; generation: int
@dataclass(frozen=True)
class RunManifest:
    schema: int; digest: str; runs: tuple[RunSpec, ...]; corpus: FrozenCorpus
    settings: EffectiveSettings; provenance: SourceEnvironment; predictions: PredictionSet
    limits: FrozenLimits; config_order: tuple[RunId, ...]
@dataclass(frozen=True)
class Event:
    run: RunId; seq: int; mono_ns: int; producer: ProducerId; producer_seq: int
    scheduler_call: CallId | None; operation: OperationId; previous_digest: str
    payload: Arrival | Lifecycle | Ownership | RefChange | HashChange | QueueChange | Lookup | AllocationAttempt | Decision | Work | SnapshotRef
@dataclass
class RequestState:
    id: RequestId; client_id: ClientId; status: RequestStatus; token_ids: list[int]
    prompt_length: int; output_length: int; admissions: list[Admission]
    completed: IntervalSet; raw_progress: int; pending: dict[WorkId, Work]
    blocks: tuple[BlockIdentity, ...]; greatest_completed_position: int
@dataclass
class PreemptionEpisode:
    id: EpisodeId; request: RequestId; failed_allocation: AttemptId; decision: SnapshotId
    C: int; original: tuple[PositionBlockCoverage, ...]; release_classes: dict[BlockIdentity, ReleaseClass]
    attempts: list[LookupId]; resume: LookupId | None; status: EpisodeStatus
    recovery: RecoverySources | None; completed_redo: list[WorkId]
```
`BlockState` contains identity, reserved flag, refcount, valid coverage, all hash aliases and their covered lengths; owner sets and queue membership/order derive from events. Refcount is stored once per physical block, not once per request.
`Work` contains unique work ID, request epoch, SchedulerOutput identity, token interval, block identities, timestamps and typed status. `AllocationAttempt` includes arguments, needed/free/reserved/watermark/full-fit evidence and result. Sufficiency is `Yes | No(reason) | Unknown(reason)`.
`DecisionSnapshot` directly copies all running requests plus the already-removed victim, original running order, eligibility/release constraints, pool blocks, aliases, free queue, request tables and raw counters. It is never assembled from replay state.
`RunSpec` is a fixed cell/repetition slot; an `Attempt` has a separate unique ID and optional replacement reason. Completed slot selection is explicit, never “best five”.
```python
def freeze(spec: SweepSpec, corpus: FrozenCorpus, predictions: PredictionSet) -> RunManifest: ...
def execute(manifest: RunManifest, mode: Pilot | Main, resume: bool = False) -> ExecutionResult: ...
def review(history: HistoryRef) -> ReviewArtifacts: ...
def reduce(state: ReplayState, event: Event) -> ReplayState: ...  # pure
# capture schedule: increment call ID; stock schedule; record final dispatched spans
# capture output: join original SchedulerOutput; snapshot stale state; stock update;
# emit completed spans with original epoch/identities, including absent/aborted requests
# preempt: direct snapshot including removed victim; stock release; record actual delta
```

### Completion, identity and recovery invariants
Assign a new generation on allocation for new content, including unhashed reuse; a cache-hit touch retains generation. Hash promotion alone does not change generation. Content movement/replacement records source/destination explicitly. Never infer survival from a hash or allocation-pop count alone.
`completed` advances only on validated completion for the same epoch and still-valid identities, or admission of already validated cached coverage. Scheduled spans cannot become completed because raw progress increased. Pending spans removed before dispatch are cancelled; acknowledged stale spans are executed/discarded, not cancelled.
For each completed execution span, count intersection with the request's greatest previously completed position before updating that position; retain every execution occurrence. Report valid completed re-prefill and executed/discarded rework separately. Episode links may overlap, but request/run totals count each WorkId once.
At release, preserve C and old position-to-identity coverage. At successful lookup actually used for resumed execution, intersect cached prefix with [0,C). Partition H into original-freed, original-still-shared, and equivalent-elsewhere using generation/hash/coverage history. Missing provenance is unknown and invalidates the claimed partition.
Failed lookups/admissions remain recorded; they do not close an episode. Report H/C only for C>0, missing C-H, wait, intervening allocations, regain-C elapsed time, repeated preemptions and unresolved episodes. Unresolved is never zero recovery.
Capture each failed allocation and each subsequent release freshly, preserving retry links. Compute candidate total/shared/reclaimable blocks excluding reserved blocks, continuous valid shared prefix, completed count/rank/all ties, position and first/resume admission times.
Evaluate release-alone sufficiency only from a side-effect-free model of confirmed allocator constraints; otherwise Unknown. Do not call the allocator speculatively. Compare measured free-count delta with immediate release accounting, separately recording deferred frees.

### History integrity and independent checks
Frame journal records with length/checksum, contiguous producer sequence, hash chain, periodic persisted counts and a terminal seal. A bounded writer exposes attempted/enqueued/written counters, overflow, flush/fsync failures and peak buffer use. Any drop, malformed record, absent seal or ambiguous tail makes the attempt invalid/incomplete and pauses execution.
Direct snapshots at initial state, every decision and terminal state independently traverse live request tables, blocks/hash maps and linked free queue. Compare all observable fields to replay, including request presence, owners, refs, aliases and exact queue order; retain snapshots as standalone files.
Generations and completed intervals are observer concepts, so snapshots alone cannot independently prove them. Validate these against controlled actual-engine mutation/completion fixtures and backend acknowledgments; report that limitation explicitly.

## Frozen workload and operation
Freeze 100 distinct local reasoning prompts expanded from existing math templates with relevant premises/data, problem-specific opening and no system message. Enforce 960–1088 chat-formatted tokens, target 1024; store rendered text, IDs, template/revisions, checksum, lengths and pairwise full-block shared-prefix distribution. Same corpus everywhere.
Generate exactly 20 cells (targets 4/8/16/24/32, caps 512/1024/2048/4096), five rounds seeded 11/23/37/53/71, 100 requests each. Freeze all 100 slot IDs, independently shuffled cell order per round, same request permutation across cells in a round, and stable item/repetition sampling seeds.
Use Qwen3-4B-4bit thinking, temperature .6/top_p .95, normal stopping, cap including thinking; max_model_len 9216, batched-token limit 2048, async FCFS, prefix cache enabled; no speculative decoding/offload/external connector. Freeze dtype, watermark, lookahead and all effective settings.
Explicit pool override 776 physical/1 reserved/775 usable, block size 16; verify live counts, equal hash/block sizes, prefix_replay_tokens=0, scheduler_reserve_full_isl=True and loaded scheduler class. No utilization-only pool sizing.
Fresh server per attempt; recorded warmup to steady thermal state, drain, reset prefix cache through supported mechanism, then independently prove empty hashes/no owners and correct free queue before initial measured snapshot. Refuse measurement if reset cannot be verified.
Capture engine revisions and dirty diffs, instrumentation checksum, libraries/Python, machine/power/thermal/swap/GPU contention. Engine edits after freeze require a new manifest. Preserve older scripts, earlier plan and sealed runs byte-for-byte; new output root is explicit and never recursively scans historical runs.
Streaming client maintains target outstanding including engine waiting; immediately replaces terminal success/failure until 100 submissions, then drains. Record startup/submission/drain phases. Persist stable client IDs before send and explicitly join API response ID to engine request ID; source shows chatcmpl prefixing, but exact full mapping must pass verification.
Record first nonempty output arrival including reasoning, last output arrival, submission, final usage/engine token IDs, output length and stop reason. Chunks are not tokens; unreconciled token counts/IDs invalidate relevant metrics. TPOT=(last-first)/output count, undefined for fewer than two tokens; report TTFT, latency, cap hits and stalls.
Record actual per-forward-pass batch size, pure decode versus mixed prefill and measured execution intervals. If async overlap prevents unique device occupancy attribution, provide labeled host-interval time weighting plus counts; device-time weighting/share remains unavailable rather than treating scheduler gaps as GPU time.

## Verification and readiness gates
Technical verification first: use the real installed scheduler/pool with deterministic arrivals, forced allocation outcomes and controlled completions; cover sharing/unshared, partial blocks, reuse/touch, aliases, multiple victims, deferred release, repeated preemption, stale outputs, abort and unresolved recovery. Compare logged versus unlogged decisions and SchedulerOutput content for identical inputs.
Replay must match independent snapshots at every decision/end; check stock FCFS choice, reclaimed deltas, request reconciliation and every mutation path. Corrupt/drop/reorder/truncate journal records and inject writer failures; no test may certify such a run. Verify report regeneration and crash/resume at each durable lifecycle boundary.
Measurement pilots require frozen corpus plus Krishna's numeric prediction/reason and the stated Codex predictions/reasons. Preserve pending status; parent may ask once, but this runner neither invents predictions nor starts a pilot.
Pilot checks exercise actual streaming, stable ID joins, thinking-token accounting, backend completion semantics, reset and actual batch recording. Then run five paired interleaved instrumented/uninstrumented runs at each of low pressure and pressure; record errors, throughput, TPOT, thermal/load, buffering/flush/log volume.
Require median throughput and TPOT differences within 5%, reporting ranges. Failed or noise-inconclusive cost gate blocks main; reduce observation cost without dropping events and repeat. Live generated text need not match across timing-perturbed runs.
Freeze total elapsed/disk ceilings and wall-clock/no-progress deadlines from pilot forecast before main. Reject infeasible forecast; repeat forecast after first 20-cell round. Missing limits or any failed gate block execution, not technical development.
Single sweep owner uses exclusive lock and durable attempt ledger. On deadline stop submissions, abort/drain within shutdown deadline, capture final evidence and preserve unfinished requests. Reserve disk for terminal records; monitor disk/time throughout. Corrupt history/pool mismatch pauses sweep; workload failures remain outcomes.
Resume reconciles ledger with seals and live owned processes, never resubmits an ambiguous request or reuses an attempt directory. Crash tails remain incomplete. Replacement attempts require new IDs/reasons and explicit slot linkage; preserve all attempts, no silent retries or reduced repetitions/caps.

## Reconstructable manual review
Build deterministic offline HTML and CSV from manifest + journals + direct snapshots only. Decision rows link snapshots/event spans and request episode history; filter by run/decision/status and sort computed/reclaimable/shared-prefix/sufficiency, keeping Unknown distinct. Mark chosen victim and all least-computed ties.
Keep annotations as separate keyed input with reasons an alternative merits E03; rebuild retains full table, selected examples and unresolved episodes. No alternate-victim recovery predictions or oracle.
For every cell show all five run values, mean/sample SD/median/range; request counts/lengths/cap hits, preemption rates per 100 submitted and per request, ties/candidate distributions, recovery sources/distributions and redo totals, actual batch distribution, TTFT/TPOT/latency/throughputs/stalls. Denominators and startup-to-drain throughput interval are explicit.
Failed/incomplete/invalid attempts remain visible separately; no-preemption recovery is undefined. Show per-run tails and explicitly counted pooled request tails, never mean-p99 or event-level independent-run claims. Note admission-bound targets and limited sharing even if pressure is absent.

## Rationale, alternatives and synthesis decision
A: selected engine-owned event history plus independent snapshots and pure replay. Deep three-call interface hides observer mechanics and operational recovery; accepted complexity is explicit event invariants and pinned hooks in return for ordered provenance and bounded capture cost.
B: snapshot-ledger design captures complete engine state before/after every public scheduler operation; reporter diffs states. Deep replay-free interface is attractive, but intermediate touches/reuse/hash transitions can cancel out and disappear, preventing loss-sensitive provenance. Per-primitive snapshots could recover validity at much higher capture cost. Reject as primary; retain snapshots as independent checks.
C: instrumented engine fork emits native events from mutation sites and model execution, with a separate client/report pipeline. Strongest mutation visibility and fewer fragile hooks, but changes shared engine maintenance surface and grows verification scope. Reject initially; if hooks cannot expose a required boundary, stop readiness and present the precise missing hook rather than quietly using a fork.
This is one independent runner's recommendation, not arena synthesis. Applied principles: exhaust-the-design-space (A/B/C), boundary-discipline (parse once, pure reducer), encode-lessons-in-structure (typed statuses/readiness), separate-before-serializing-shared-state (owned journals), make-operations-idempotent (ledger/seals), laziness-protocol and minimize-reader-load (flat ownership modules).
Red-flag screen: no two writers of engine state; no transport objects in public API; no staged pass-through modules; derive tables from one history. Historical scripts remain only because preservation is explicit, with expanded CLI rejecting historical artifacts.
Open verification questions: Can all queue/ref mutations be intercepted without replacing engine mechanisms? Does metal output acknowledgment prove each scheduled span executed? Can stream arrival identify the first output token exactly when parsers buffer reasoning? If not, mark exact metric unavailable and block any readiness criterion that requires it; do not silently substitute counters or chunk timing.
Next implementation step: build the pinned capture adapter and real scheduler/pool fixture proving one release, reuse, resume and stale completion end-to-end before adding sweep machinery.

Proposal only; no experiments, tests of an implementation, or historical-run inspection occurred. Verified inputs/source boundaries and output length only. Confirmed host: T3 Code, Codex, gpt-6-astra, medium effort. Kstack 0.2.0+codex.20261012082835.

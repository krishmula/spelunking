# E02 implementation design (candidate: fable)

Design proposal, not a finding. [verified] = read in local source at vllm-core `db9527a`; [uncertain] = needs a pilot check.

## Problem

E02 must observe stock vLLM preemption and recovery for 20 configurations x 5 repetitions x 100 requests, and let a reader reconstruct every request and block from an event history. The hard parts: async scheduling advances `num_computed_tokens` before execution [verified, `_update_after_schedule`], so scheduled and completed progress must be two series; block IDs are reused and one block can carry several hashes (full plus partial entries in `cached_block_hashes_by_block`) [verified], so original-KV survival needs a generation counter; the victim is popped from `self.running` before `_preempt_request` [verified], so the snapshot must add it back; and event loss must invalidate a run rather than quietly bias it. The engine must not change: all instrumentation lives in an `AsyncScheduler` subclass and instance-level wrappers on the pool and manager objects it already owns, as the existing `e02_scheduler.py` does.

## Usage (caller's view)

```
cd experiments/E02-preemption-rehit; E=/Users/krishna/dev/vllm-lab/.venv/bin/python
$E -m e02 check                      # no engine: manifest, corpus checksum, predictions, readiness; exit 1 on any gap
$E -m e02 corpus freeze              # writes corpus/corpus.json + corpus/SHA256 once; refuses to overwrite
$E -m e02 smoke                      # tiny pool; exercises gate-1 shapes; allowed without predictions
$E -m e02 twin                       # gate 4 offline: observed vs stock scheduler on a fixed script; no model
$E -m e02 pilot                      # gates 2,3,5-8; refuses unless both prediction rows are numeric with a reason
$E -m e02 main --sweep 20261022-main # 100 runs, resumable; refuses unless pilot passed and ceilings are frozen
$E -m e02 replay RUN_DIR             # events -> world, decisions, episodes; prints first mismatch or OK
$E -m e02 report SWEEP_DIR           # decisions.csv, episodes.csv, requests.csv, runs.csv, report.html
# engine side, set by the runner: --scheduler-cls e02.observe.ObservedScheduler, env E02_EVENTS=<run_dir>/events.jsonl, E02_RUN_ID
```

```python
history = read_history(run_dir / "events.jsonl")   # raises HistoryCorrupt(seq, reason): gap, bad line, non-quiescent end
world = replay(history)                             # raises ReplayMismatch(snapshot_seq, path, replayed, observed)
rows = decision_rows(world); eps = world.episodes   # pure; same input, same output
```

## Module map (`experiments/E02-preemption-rehit/e02/`, new package; old scripts and `runs/` untouched)

| Module | Owns | Public surface |
|---|---|---|
| `events.py` | Event types, wire format, integrity | `Event` union, `EventWriter`, `read_history`, `HistoryCorrupt` |
| `observe.py` | Engine-side capture, startup assertions | `ObservedScheduler` only |
| `replay.py` | World reducer, episodes, decision rows, snapshot checks | `replay`, `World`, `decision_rows`, `ReplayMismatch` |
| `corpus.py` | Frozen prompt corpus, token IDs, sharing audit | `freeze`, `load`, `Corpus` |
| `manifest.py` | Sweep plan, run IDs, statuses, ceilings, readiness, resume | `RunManifest`, `RunSpec`, `Readiness`, `plan`, `resume_state` |
| `client.py` | Streaming driver with stable IDs, TTFT, replacement loop | `drive`, `ClientRecord` |
| `server.py` | One server per run, warm, reset, environment capture | `Server` (copied from `run_experiment.py`, not imported) |
| `gates.py` | Pilot gates 1-8 and cost pairing | `run_gate(n, ...) -> GateResult` |
| `report.py` | CSV and self-contained HTML | `write_report` |
| `__main__.py` | CLI; the only module that knows the directory layout | commands above |


## Data model (Python 3.12 dataclasses, `frozen=True, slots=True` unless stated)

```python
Seq, CallId, BlockId = NewType(..., int); ReqId = NewType("ReqId", str); Lifecycle = Literal["submitted","waiting","running","preempted","finished_stop","finished_length","aborted","unfinished"]
# Every event carries Hdr(seq, t_mono, call, run); seq contiguous from 0. Payloads are scalars and id lists, never engine objects.
RunBegin(cfg: EngineConfig, instr_sha256: str, t_wall: float)   # asserts block_size==hash_block_size, replay_tokens==0, reserve_full_isl, FCFS, no spec/connector
CacheReset(free: int, hashed: int); ReqAdd(req: ReqId, prompt_tokens: int, max_tokens: int, hashes: list[str]); SchedBegin(running: list[ReqId], waiting: list[ReqId], free: int)
Lookup(req: ReqId, hit_tokens: int, hit: list[tuple[BlockId,int,str]])          # (id, generation, primary hash)
AllocAttempt(attempt: int, req: ReqId, path: Literal["admit","extend"], new_tokens: int, computed: int, in_flight: int,
             owned: int, free: int, watermark: int, ok: bool)
Alloc(req: ReqId, ids: list[BlockId], evicted: list[tuple[BlockId,str]]); Touch(req: ReqId, ids: list[BlockId], removed_from_queue: list[BlockId])   # Alloc bumps generation
HashSet(items: list[tuple[BlockId,str,int]]); HashEvict(items: list[tuple[BlockId,str]]); HashMove(src: BlockId, dst: BlockId)
Free(req: ReqId|None, ordered: list[BlockId], to_head: list[BlockId], to_tail: list[BlockId], kind: Literal["release","finish","unpin","cow"])
PreemptSnapshot(decision: str, release_index: int, attempt: int, victim: ReqId, running_after_pop: list[ReqId],
                candidates: list[CandidateSnap], free_queue: list[BlockId], pool: list[BlockSnap])   # taken BEFORE release
Preempt(req: ReqId, episode: str, C: int, raw_progress: int, in_flight: int, free_before: int, free_after: int)
SchedEnd(scheduled: dict[ReqId,int], new: list[ReqId], resumed: list[ReqId], prefill_chunk: list[ReqId], total: int)
Completed(items: list[tuple[ReqId,int,bool,int]])                 # (req, tokens drained, stale, new output tokens)
ReqFinish(req: ReqId, reason: str, outputs: int, preemptions: int)
PoolSnapshot(free_queue: list[BlockId], pool: list[BlockSnap], owners: dict[ReqId, list[BlockId]]); WriterFault(reason: str)   # run end; next schedule() raises
CandidateSnap: req; eligible; computed; raw_progress; in_flight; blocks: list[BlockId]; position; admitted_at; resumed.  BlockSnap: id; generation; ref_cnt; hashes; null
```

Replay-side domain (mutable `World`, the single writer of reconstructed state):

```python
BlockIdentity: id; generation; hashes                 BlockState: identity; ref_cnt; owners: set[ReqId]; in_queue: bool; valid_tokens: int
RequestState: id; client_id: str|None; status: Lifecycle; prompt_tokens; blocks: list[BlockIdentity]; scheduled_progress; completed;
              high_water; reprefill_completed; executed_discarded; admissions: list[Admission]; episodes: list[PreemptionEpisode]; phase: Literal["warmup","measured"]
Recovery: H; H_below_C; original_freed; original_shared; equivalent_elsewhere; missing        # tokens; the three sources are disjoint
PreemptionEpisode: id; decision; request; C; owned: tuple[tuple[BlockIdentity,int],...]      # (identity, ref_cnt at release)
                   freed_now; deferred; failed_lookups; outcome: Literal["resumed","preempted_again","unresolved"]
                   recovery: Recovery|None; wait_s: float|None; intervening_allocs; reprefill_completed; regain_s: float|None
DecisionRow: decision; request; chosen; eligible; computed; raw_progress; pending; least_rank; tied; total_blocks; reserved_blocks
             shared_blocks; reclaimable; continuous_shared_prefix; sufficiency: Literal["yes","no","unknown"]; position; admitted_at
             resumed; reclaimed_actual: int|None; episode: str|None
```

## Key signatures and pseudocode

```python
# observe.py
class ObservedScheduler(AsyncScheduler):
    def __init__(...): super().__init__(...); self._w = EventWriter(env E02_EVENTS); assert_contract(self)
                       wrap_pool(self.kv_cache_manager.block_pool, self._w); wrap_manager(self.kv_cache_manager, self._w); emit RunBegin
    def schedule(self, *a, **k): self._w.raise_if_faulted(); self._call += 1; emit SchedBegin; out = super().schedule(*a, **k); emit SchedEnd(out); return out
    def _preempt_request(self, request, ts, drop_stale_output=False):
        snap = [self._candidate(r) for r in self.running] + [self._candidate(request)]   # victim already popped [verified]
        emit PreemptSnapshot(..., free_queue=pool.free_block_queue.get_all_free_blocks(), pool=self._pool_snap())
        C = request.num_computed_tokens - request.num_in_flight_tokens                   # completed KV before the reset
        free0 = pool.get_num_free_blocks(); super()._preempt_request(...); emit Preempt(C=C, free_before=free0, free_after=...)
    def update_from_output(self, so, mro): stale = {r: req.num_stale_output_tokens > 0 ...}; r = super().update_from_output(so, mro); emit Completed; return r
    add_request / finish_requests / _free_request / reset_prefix_cache: emit, then delegate. _candidate(r): eligible = self._request_blocks_can_be_freed(r)
# wrap_pool rebinds on the instance: get_new_blocks, touch, free_blocks, unpin_blocks, _insert_block_hash, _remove_cached_block_hashes, move_block_hashes.
# wrap_manager: get_computed_blocks, allocate_slots. Each wrapper calls the original, then emits from its return value and pre/post state; nothing is altered.

# events.py
class EventWriter:   # background thread, unbounded deque; fsync at every Nth SchedEnd; any write error sets a fault flag
    def emit(self, ev: Event) -> None; def raise_if_faulted(self) -> None     # RuntimeError inside schedule(): the engine dies visibly
def read_history(path) -> History   # seq contiguous; every line parses to a known kind; last event is SchedEnd with scheduled=={} and running==[]

# replay.py
def replay(h: History) -> World: w = World.from_run_begin(h[0]); for ev in h: w.apply(ev); return w   # one match over the union
apply(Alloc): per id: generation += 1; ref = 1; ids must equal the replayed queue head in order, else ReplayMismatch
apply(Free): ref -= 1; zero and non-null -> queue head or tail, mirroring FreeKVCacheBlockQueue.prepend_n/append_n [uncertain: order inside prepend_n]
apply(PreemptSnapshot|PoolSnapshot): compare ref_cnt, hashes, owners, free-queue order field by field; first difference raises
apply(Completed): a = completed; b = a + n; if stale: executed_discarded += n else: completed = b
                  reprefill_completed += max(0, min(b, high_water) - a); high_water = max(high_water, completed)
apply(Preempt): episode.owned = [(identity, ref) per owned block]; status "preempted"; outcome "unresolved".  apply(Lookup) for a preempted request: per hit block k covering [16k, min(16k+16, H)): same (id, generation, hash) as owned[k] -> original,
    ref==1 at release -> original_freed else original_shared; otherwise equivalent_elsewhere. outcome becomes "resumed" only when the next
    AllocAttempt(path="admit") for this request has ok=True; otherwise failed_lookups += 1 and the lookup is kept as an unsuccessful attempt
decision_rows(w): per PreemptSnapshot; least_rank over eligible candidates by computed; tied marks every candidate at the minimum;
    reclaimable = owned non-null blocks with ref 1 when eligible; sufficiency = "yes" iff attempt.path=="extend" and watermark==0 and
    free + reclaimable >= cdiv(computed + in_flight + new_tokens + lookahead, 16) - owned [verified for full attention on the running path];
    "unknown" when a CoW partial hit is pending or any KV group is not full attention
```

Client: `drive(spec, corpus, out) -> list[ClientRecord]`. Each request sets `X-Request-Id: <run>-<item>`; TTFT is the first SSE delta with non-empty `reasoning_content` or `content`; the last such delta is the last-token time; output tokens come from the final `usage` chunk, never from chunk counts. Replay joins `client_id` to `ReqAdd.req` by exact id or `<id>-0` [uncertain: vLLM's per-choice suffix]; gate 8 requires a one-to-one join. The loop replaces each finished request until 100 are submitted, then drains; each record carries `phase` from the manifest clock.

Manifest and resume: `RunManifest` is written once per sweep and never edited; `RunSpec` has `run_id = <sweep>-r<rep>-b<target>-c<cap>`, seeds, order index, and `Ceilings(wall_s, disk_bytes, run_deadline_s, no_progress_s)`. Each run directory has one `STATUS` file holding `planned | running | complete | incomplete:<reason> | failed:<reason> | invalid:<reason>`. Resume lists directories; `running` without a live process becomes `failed:crashed`; replacements append to `replacements.jsonl` with a new run id and reason, never a rewrite. `Readiness` is a frozen dataclass of booleans with the file each was read from; `pilot` and `main` refuse on any false field and print its name.

## Rationale

Shape. The engine process emits facts only; every derived quantity (ranks, reclaimable counts, provenance, re-prefill) comes from a pure reducer the reviewer can rerun. Snapshots are not reducer inputs; they are checkpoints the reducer must reproduce, which makes gate 2 a real test instead of a self-comparison. Generation counters live in replay, derived from `Alloc`, so `KVCacheBlock` is never mutated. Scheduled and completed progress are separate fields and separate event kinds, so no code path can confuse them. Per boundary-discipline, `events.py` parses lines into typed events once and `replay.py` trusts them. Per model-the-domain, lifecycle and outcome are literals, so an unresolved episode is a value, not a missing row. Interface depth: callers see one scheduler class, one `replay` function, and a CLI; the rebinding, queue mirroring, and provenance rules stay hidden.

Tradeoffs accepted. We accept instance-level wrappers on private pool methods in exchange for an untouched engine; the twin gate and the pinned SHA bound the risk. We accept a full pool snapshot at every release (776 entries) in exchange for one replay check covering refcounts, hashes, and free order; gate 5 prices it. We accept derived rather than tested allocation sufficiency, with `unknown` whenever a CoW partial hit or non-full-attention group is involved. We accept monotonic timestamps from two processes [uncertain: cross-process comparability on macOS], with `RunBegin.t_wall` and the client's wall clock as fallback alignment. We accept a quiescent `SchedEnd` as the history's end instead of a shutdown trailer, because no stock shutdown hook is guaranteed to run.

Alternatives considered.
1. Live observer computes rows (extend `e02_scheduler.py`): episode and provenance logic runs inside `schedule()` and writes decision rows directly. Smaller, but its arithmetic is checked only by itself, replay is impossible without the raw events it skips, and every new field adds hot-path engine-side code. Rejected on replay correctness.
2. Stock KV events plus scheduler stats (`--kv-events-config`, `BlockStored`/`BlockRemoved`, `PREEMPTED` request events): no custom hooks, but no reference counts, free-queue order, or request-to-block attachment, and hashes only [verified in `kv_events.py`]. Cannot meet the reconstruction contract; `BlockStored` could still cross-check hashes in gate 2.
3. Engine patch adding a generation field and hooks to `BlockPool`: cleanest data, but it changes the engine the sweep claims to observe and dirties the pinned repo. Rejected by the grounding rule.

## Verification

- `twin` (gate 4, no model): build stock and observed schedulers from one `VllmConfig` and KV spec, drive both with one scripted arrival/output sequence, assert equal `SchedulerOutput` sequences and equal running/waiting orders at every call.
- Unit tests on `replay.py` with hand-written histories: reused block ID with a new generation, shared then freed block, partial-block hash, two releases in one call, repeated preemption, unresolved episode, stale completion. Each asserts an exact `Recovery` or `DecisionRow`.
- `read_history` tests: deleted middle line, truncated last line, non-quiescent end, each raising `HistoryCorrupt` with the right seq. `smoke` on a 96-block pool: every event kind appears at least once, replay returns OK, `report` renders. Then pilot gates 2, 3, 5-8 as the README specifies, each writing `gates/<n>.json` with `pass | fail | inconclusive` and its evidence.

## Open questions and risks

- Does `vllm serve` honor `X-Request-Id` as the engine request id on this fork, and with which suffix? [uncertain] Gate 8 decides.
- Is `scheduler_reserve_full_isl` on by default here, and is `watermark_blocks` zero on the running path? [uncertain] `RunBegin` records both; the sufficiency rule reads them.
- The fork's partial-block caching (`cache_partial_block`, CoW on partial hits) can make `H` non-block-aligned and add a hidden allocation; the design records it and marks sufficiency `unknown`, but its frequency is unmeasured.

Next implementation step: write `events.py` with the event union, writer, and `read_history`, plus the corrupt-history tests, since every other module depends on its shape.

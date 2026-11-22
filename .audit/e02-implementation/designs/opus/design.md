# E02 expanded sweep: implementation design (opus runner)

Proposal, not a finding. `(U#)` marks an unverified assumption with its check in section 9. Source read at vllm-core `db9527a`, vllm-metal `572322e`.

## 1. Usage (caller's view, written first)
```sh
PY=~/dev/vllm-lab/.venv/bin/python        # run from experiments/E02-preemption-rehit
$PY sweep.py corpus                       # once: corpus/v1.jsonl + v1.sha256; refuses to overwrite
$PY -m pytest tests/                      # offline gates 1, 2, 4 and loss injection; allowed while predictions pend
$PY sweep.py pilot behavior|cost          # exits 2: "prediction row 'Krishna' lacks a number or reason"
$PY sweep.py freeze --wall-hours H --disk-gb G   # needs passing pilot ids; writes immutable manifest.json
caffeinate -i $PY sweep.py main runs/expanded/<manifest_id>   # resumable: same command after any crash
$PY sweep.py status runs/expanded/<manifest_id>               # per-run status, budget forecast
$PY sweep.py report runs/expanded/<manifest_id>               # index.html, per-run pages, decisions.csv
```
```python
h = history.load_run(attempt_dir)         # pure replay; never raises on bad data
match h:
    case Invalid(reason, first_bad_seq): ...     # sweep pauses; human decides
    case RunHistory(): h.decision_rows(), h.episodes, h.requests, h.metrics
```
Engine side has no source edit: `vllm serve ... --scheduler-cls observer.ObservedScheduler` with `E02_OUT=<attempt_dir>`.

## 2. Module map (new flat files; `run_experiment.py`, `e02_scheduler.py`, `runs/20261014-002308-full` untouched)
| File | Owns | Public surface |
|---|---|---|
| `corpus.py` | Template expansion, chat rendering, 960..1,088 gate, token IDs, shared-prefix census, checksum | `build()`, `load() -> Corpus` (verifies sha) |
| `events.py` | The one event schema: frozen dataclasses, strict codec; registry derived from the classes | `Event`, `encode`, `parse_line -> Event \| Malformed` |
| `observer.py` | Engine-process hooks, startup asserts, writer, digests, direct snapshots | `ObservedScheduler` |
| `history.py` | Replay, validation, async accounting, decisions, episodes, recovery, run metrics | `load_run`, `RunHistory`, `Invalid` |
| `sweep.py` | Manifest, prediction gate, server lifecycle, streaming client, machine sampler, deadlines, budget, resume | CLI only |
| `report.py` | HTML + CSV, config aggregates, annotations join; a pure function of run dirs | `render(root)` |
| `tests/` | Real `AsyncScheduler`/`BlockPool` offline with scripted model outputs | pytest |

Each file has one writer: engine writes `events.jsonl`, `snapshots.jsonl`; runner writes `client.jsonl`, `machine.jsonl`, `server.log`, `STATUS` (atomic rename). Nothing is shared-written.

## 3. Data model (`events.py`, `history.py`)
```python
class Phase(Enum): WARMUP, MEASURED                     # observer writes MEASURED only after verified empty cache
@dataclass(frozen=True) class PoolShape: physical: int; reserved_ids: tuple[int, ...]; usable: int   # 776, (0,), 775
@dataclass(frozen=True) class RunSpec: run_id: str; kind: Literal["main","pilot","cost"]; batch_target: int
    output_cap: int; rep_seed: int; round: int; order_in_round: int; item_order: tuple[int, ...]   # 100 items
    item_seeds: tuple[int, ...]; wall_deadline_s: int; no_progress_s: int
@dataclass(frozen=True) class RunManifest: manifest_id: str; corpus_sha: str; engine: EngineFingerprint
    fixed_args: tuple[str, ...]; pool: PoolShape; runs: tuple[RunSpec, ...]   # 100, rounds shuffled by rep seed
    predictions_sha: str; observer_sha: str; budget: Budget; pilot_ids: tuple[str, ...]
@dataclass(frozen=True) class Event: seq: int; t_ns: int; call: int        # + one subclass per kind below
# RunStart(pool, config) | RequestArrived(eng_id, prompt_sha, client_hint) | Lookup(eng_id, hit_ids, hit_tokens)
# AllocAttempt(eng_id, new_tokens, needed|None, watermark, reserved, free, ok, attached_ids, table_len_before)
# QueueOp(op, ids, len_before) | RefChange(id, before, after, cause) | HashAdd/HashRemove(id, gen, hash, ntok)
# Detach(eng_id, ids_in_release_order) | FailedAlloc(group, requester, ...) | Decision(id, group, victim, snap)
# Preempted(eng_id, episode, completed, raw, in_flight, free_before, free_after) | Scheduled(per_req, running_order)
# Completed(origin_call, per_req: (n, stale)) | Finished(eng_id, reason, n_out) | Digest(ref, hash, queue)
# Fault(where, repr) | RunEnd(n_events, sha256_of_lines, faults)
@dataclass(frozen=True) class BlockIdentity: block_id: int; generation: int   # gen += 1 on each allocation pop
@dataclass class BlockState: ident: BlockIdentity; ref: int; hashes: set[HashKey]; valid_tokens: int  # completion-backed
class Lifecycle(Enum): WAITING, RUNNING, PREEMPTED, FINISHED_STOP, FINISHED_LENGTH, ABORTED, UNFINISHED
LEGAL: Mapping[Lifecycle, frozenset[Lifecycle]]   # replay rejects any other transition
@dataclass class RequestState: eng_id: str; item: int; client_id: str; status: Lifecycle; table: list[BlockIdentity]
    raw_progress: int; in_flight: int; max_completed: int; reprefill: int; stale_executed: int; episodes: list[str]
    @property completed -> int: return raw_progress - in_flight      # engine's own processed basis (scheduler.py:2971)
EpisodeOutcome = Waiting | Resumed(lookup) | Regained(t_ns) | Repreempted(episode) | Finished | Unresolved
@dataclass class PreemptionEpisode: id: str; eng_id: str; ordinal: int; decision: str; C: int
    released: tuple[Released, ...]   # (ident, position, ref_after_release, hashed)
    lookups: list[LookupAttempt]; outcome: EpisodeOutcome; recovery: Recovery | None; reprefill: int
@dataclass(frozen=True) class Recovery: H: int; original_freed: int; original_shared: int; equivalent: int; missing: int
```
`Recovery` fields are disjoint token counts with `original_freed + original_shared + equivalent == H` asserted; `H/C` is `None` when `C == 0`.

## 4. Observation (`observer.py`): passive hooks at the mutation choke points
`ObservedScheduler(AsyncScheduler)` changes no return value and no argument. At `__init__` it asserts FCFS, no connector or spec decode, `block_size == hash_block_size`, `prefix_replay_tokens == 0`, `scheduler_reserve_full_isl`, `num_gpu_blocks == 776` (U1), and that `inspect.getsource` hashes of every hooked method equal the pinned values; any miss refuses startup.

| Hook (instance-level wrap) | Emits |
|---|---|
| `free_block_queue.{popleft_n, remove, append_n, prepend_n, append}` | `QueueOp` with `len_before`: insertion, removal, order |
| `block_pool.{get_new_blocks, touch, free_blocks, unpin_blocks}` | `RefChange(before, after)` for each involved block; allocation pop bumps generation |
| `block_pool._insert_block_hash`, `_remove_cached_block_hashes` | `HashAdd` / `HashRemove` (duplicates allowed; keyed by block identity) |
| `kv_cache_manager.get_computed_blocks`, `allocate_slots`, `free`, `pop_blocks_for_free`; `coordinator.get_num_blocks_to_allocate` (captures `needed`) | `Lookup`, `AllocAttempt`, `Detach` |
| `_preempt_request` (pre-super) | `Decision` + direct snapshot of victim and every running request, then `Preempted` |
| end of `schedule`, `update_from_output`, `add_request`, `finish_requests` | `Scheduled` / `Completed` / `Digest` |

```python
def _preempt_request(self, req, ts, drop_stale_output=False):
    group = self._open_failed_alloc               # set by the AllocAttempt(ok=False) just before
    cands = [req, *self.running]                  # victim was already popped by the stock loop
    snap = self._direct_snapshot(cands, eligible={r: self._request_blocks_can_be_freed(r) for r in cands})
    self._emit(Decision(id=..., group=group, victim=req.request_id, snap=snap.seq))
    done, raw, fl, free0 = req.num_computed_tokens - req.num_in_flight_tokens, req.num_computed_tokens, ...
    super()._preempt_request(req, ts, drop_stale_output)         # stock release, reset, prepend
    self._emit(Preempted(req.request_id, episode=..., completed=done, raw=raw, in_flight=fl, free_before=free0, ...))

def update_from_output(self, sched_out, mro):
    stale = {r: self.requests[r].num_stale_output_tokens > 0 for r in sched_out.num_scheduled_tokens if r in self.requests}
    out = super().update_from_output(sched_out, mro)             # stock drain
    self._emit(Completed(origin_call=self._call_of[id(sched_out)], per_req={r: (n, stale.get(r)) ...}))
```
Loss detection is structural: every change carries its prior value, every event a contiguous `seq`, every entry call ends in a `Digest` of all 776 ref counts, hash sets and free-queue order, and `RunEnd` carries the count and SHA-256 of all prior lines. Hook exceptions become sticky `Fault` events and never alter scheduling. Direct snapshots (decision, every 256 calls, final) go to `snapshots.jsonl` by direct read of live objects, never from the log. Warmup: `Phase.WARMUP` until `reset_prefix_cache` succeeds (U2) and a direct snapshot shows 775 free, no hashes, ref 0; then `RunStart` with that snapshot. `RunEnd` fires when all 100 arrived requests are finished or aborted, so it does not depend on shutdown (U7).

## 5. Replay and derivations (`history.py`, pure)
1. Parse strictly; `Malformed`, `seq` gap, missing `RunEnd`, hash mismatch, or any `Fault` returns `Invalid(first_bad_seq)`.
2. Apply raw ops to `BlockState`/`RequestState`. Each `before` must equal replayed state; each `Digest` and each snapshot must match exactly. Replay re-implements no engine policy; it applies recorded effects and checks them.
3. Async: `Scheduled` advances `raw_progress`, `in_flight`; `Completed` with `stale=False` sets `valid_tokens` on the blocks at those positions in the table recorded at schedule time, and adds `|[a,b) ∩ [0, max_completed)|` to re-prefill, attributed to the open episode; `stale=True` adds to `stale_executed` only. Hashing happens at schedule time (`allocate_slots` caches up to scheduled tokens), so a hash never implies valid KV.
4. Decisions: candidates = snapshot's running set plus victim; computed, raw, pending, least-computed rank with all ties, total/shared/reclaimable (reserved excluded), continuous shared prefix (consecutive valid shared blocks from 0), running position, first admission versus resumption. Sufficiency is `needed + watermark <= free - reserved + reclaimable(c)` from the captured failing attempt, else `Unknown` (never ranked favorable). Stock-rule check: victim == `running[-1]` of the pre-pop order.
5. Episodes: `C` from `Preempted.completed`. Each `Lookup` while waiting is a `LookupAttempt`; the successful one is the lookup whose same-call `AllocAttempt` succeeded and whose request appears in that call's `Scheduled`. `H = min(hit_tokens, C)`. Per hit position below `C`: same `BlockIdentity` as released, with `ref_after_release == 0` gives original-freed and `> 0` gives original-shared; a different identity gives equivalent. Same generation proves no reallocation, and replay asserts no `HashRemove` occurred between. Also records wait, intervening allocations, freed blocks reallocated, time to regain `C`, and unresolved status at `RunEnd`.
6. Join: `RequestArrived.prompt_sha` must equal the frozen item's rendered token-ID hash (items are distinct, so this is authoritative); the `X-Request-Id` hint (U3) must agree. Client-only or engine-only requests make the run `Invalid`.

## 6. Running (`sweep.py`)
- Prediction gate: parse README Prediction rows; both Krishna and Codex rows need a number and a reason. `pilot`, `freeze`, `main` refuse otherwise; `main` also refuses if the rows' hash differs from `manifest.predictions_sha`.
- Per run: fresh server with pinned args and `--num-gpu-blocks-override 776`; port and leftover-process check; thermal warmup with non-corpus prompts, drain, reset, verify; then a streaming client (`stream_options.include_usage`) with a semaphore of `batch_target`, replacing each finished or failed request until 100 are submitted. Phase boundaries (startup, steady, drain) are recorded. First output = first chunk with nonempty content or reasoning delta (U9). Token counts come from engine `Finished.n_out`, reconciled with usage. Chunks are never counted as tokens.
- Deadlines from the manifest: wall and no-progress (no event-file growth and no chunk). On fire, the client cancels streams, engine aborts appear as events, and status is `INCOMPLETE`.
- Status per attempt: `RUNNING(pid) -> COMPLETE | INCOMPLETE(reason) | INVALID(reason) | FAILED(reason)`. Attempt dirs are created with `O_EXCL`; a `RUNNING` with a dead pid becomes `FAILED crash`. The next action is derived only from disk: first `RunSpec` in manifest order without a terminal attempt. A replacement gets `<run_id>-a2` plus a reason. `INVALID` or a pool mismatch pauses the sweep. A sweep lock file blocks two runners.
- Budget: after each run and mandatorily after round 1, forecast = elapsed + remaining runs x per-cell median duration (pilot-derived until real data exists). If it exceeds a ceiling, write `STOPPED budget` and keep everything for resumption. Caps and repetitions never shrink.
- Machine sampler every 30 s: thermal, swap, other-process GPU time, power, and event-loop lag (client saturation). Runs over the threshold are flagged and stay visible.

## 7. Report (`report.py`)
`decisions.csv` holds every candidate row. `index.html` has per-config tables: five run values, mean, SD, median, range; pooled tails with request counts; per-run p99 never averaged. Per-run pages carry the sortable, filterable decision table (vanilla JS, inline CSS), a `<details>` snapshot per decision, and per-request histories across episodes. `annotations.tsv` (manual: decision, candidate, why worth testing) is joined into a "Selected examples" section. The footer lists the SHA-256 of every input, so rerunning `report` must reproduce identical bytes. Configs without preemptions show zero counts and "undefined" recovery.

## 8. Rationale
**Chosen: method choke-point hooks, raw-effect log, per-call digests, offline replay.** It preserves intra-call order, which matters most: a victim's freed blocks can be re-popped in the same call that preempted it. It touches no engine source.

Alternatives considered, as whole structures:
- **A. Field-level write barrier.** Patch `block_pool.KVCacheBlock` with a slots subclass whose `ref_cnt`, hash, and link writes emit. Complete for blocks by construction, but link writes are the hottest path, the object type changes inside the engine, and request progress still needs method hooks. More perturbation for no gain.
- **B. Snapshot-diff per entry call, no mutation hooks.** Smallest and loss-proof per call, but it collapses release-then-reallocate within one call, reverse release order, and generations, so it cannot decide original versus equivalent KV. Rejected; its digest idea is grafted.
- **C. Online shadow model in the engine.** It emits derived rows only. The log is small, but the logic runs in the measured process, isn't rerunnable, and a shadow bug has no independent check. Rejected for offline replay.

Tradeoffs accepted: we hook private methods (`_insert_block_hash`, `_remove_cached_block_hashes`) in exchange for complete hash history, guarded by source-hash pinning. We pay about 1-2% per call for full digests (U10) in exchange for call-level loss localization; the cost gate decides, and any reduction is recorded in the manifest. The log is larger than a summary log in exchange for rerunnable derivations.

## 9. Verification plan (real, before any measurement)
Offline, with real `AsyncScheduler` + `KVCacheManager` on a 24-block pool and scripted `ModelRunnerOutput` (U8):
- Gate 4: stock versus observed scheduler on identical scripts, comparing every `SchedulerOutput`, preempted ID, and final pool.
- Gates 1-2: scenarios with shared prompts (tests only), a partial block, reuse, two victims in one call, repeated preemption, and unresolved at stop. Replay equals direct state at every call. The stock-rule check passes.
- Loss injection: deleting, duplicating, reordering, or truncating a line, or flipping one `RefChange.after`, each yields `Invalid` at the expected `seq`.
- Re-prefill and recovery classification on hand-computed scripts. Resume logic against fake attempt dirs.

Live pilots (after predictions): tiny pool (96 blocks) for gates 1, 3, 8; one low- and one high-pressure cell at 776; cost gate as 5 paired, interleaved runs x 2 workloads against plain `AsyncScheduler`. Checks: U1 (`num_gpu_blocks` in `RunStart`), U2 (empty-cache snapshot), U3 (header in engine ID), U4 (engine arrival >= client submit for all requests, same-host monotonic clock), U5 (single-thread assert on hook entry), U6 (sufficiency reproduces the stock failure on every failed attempt), U7 (`RunEnd` present), U9 (reasoning delta field), U10 (measured overhead and log MB per run).

## 10. Requirements that cannot be met exactly
- Victim device time: no attributable per-request timing exists, so it is reported unavailable; completed re-prefill tokens are the cost measure.
- Allocation sufficiency is exact only when the failing attempt's `needed`, watermark, and reserved are all captured; otherwise `Unknown`.
- KV written by a stale (post-preemption) step into a freed block that is later hit is valid on the GPU, but it is not completion-backed. Replay tags those positions `stale_backed` inside their recovery class rather than guessing.

## 11. Open questions
- If U2 fails (no reset endpoint), use cold engines with no in-process warmup and record it, or warm in a separate prior process?
- Should a run that reaches `INCOMPLETE` from a deadline ever be replaced, or only reported?
- Who sets the wall-time and disk ceilings at `freeze`: the user, or the pilot forecast plus a margin?

Next step: write `events.py` and the offline gate-4 test against the real scheduler before any other module.

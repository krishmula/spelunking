# E02 fix round 1: fixed interfaces between owners

Integrating owner (sweep, CLI, report, budgets, gates) relies on exactly these names from the integrity owner. Wire formats of existing `events.jsonl` and `snapshots.jsonl` do not change. The two new pilot roots (`runs/20261020-14*`) must still read and replay exactly as before.

## Ownership

| Owner | Files |
|---|---|
| Integrity helper | `e02/events.py`, `e02/observer.py`, `e02/history.py`, `tests/test_integrity.py`, `tests/test_replay.py`, `tests/test_twin.py`, `tests/scenarios.py`, `tests/engine_fixture.py`, new `tests/test_history_metrics.py` |
| Integrating owner | `e02/sweep.py`, `e02/__main__.py`, `e02/report.py`, `e02/corpus.py`, `tests/test_sweep.py`, `tests/test_report.py`, `tests/test_corpus.py`, new sweep-side test files, README, audit files |

## events.py

- `ScheduledEntry(NamedTuple)`: `r: int, n: int, start: int, epoch: int`.
- `CompletedEntry(NamedTuple)`: `r: int, n: int, kind: Literal["valid", "stale", "gone"]`.
- `DeliveredEntry(NamedTuple)`: `r: int, delivered: int, processed_hint: int`.
- `Scheduled.entries`, `Completed.entries`, `Delivered.entries` hold these. Encoding stays a JSON list of lists.
- `read_journal()` parses every field strictly against its annotation: `int` rejects `bool`, `null` only where the annotation allows `None`, list elements and entry arity and element types checked. A violation returns `Journal("corrupt", ..., reason, bad_line)`. It never raises on bad content.
- `Writer.seal(...) -> str | None`: `None` only when RunEnd was written, flushed, fsynced and closed with no earlier sticky error. Otherwise the error text. A sticky error before sealing means RunEnd is never written.

## observer.py

- `RunStart.config` gains two keys: `snapshot_every` and `digest_every`, the effective integers the observer used.
- Sealing `reset_prefix_cache` calls the stock reset exactly once. Any snapshot-file flush/fsync/close error or a non-`None` `Writer.seal` result, or an earlier sticky writer error, writes `producer-error.json` (exclusive create) in the attempt directory with `{"where": str, "error": str}` and makes this reset return `False`. Exceptions from these steps never escape into the engine.

## history.py

- `PRODUCER_ERROR = "producer-error.json"`. If that file exists, `replay_attempt` returns `Invalid("producer error: <where>: <error>", None)`.
- `read_snapshots(path) -> dict[int, tuple[str, dict]]` raises `EvidenceError(reason)` on a malformed or partial line, a non-object record, wrong keys or field types, a duplicate index, or an index that differs from its line number. A missing file returns `{}`. `replay_attempt` converts `EvidenceError` into `Invalid`.
- `Replay.snapshots: dict[int, tuple[str, dict]]` carries the parsed snapshots so callers need not reparse.
- Named types: `HitBlock(NamedTuple)` `block, generation`; `OriginalBlock(NamedTuple)` `position, block, generation, ref_after_release`; `LookupAttempt` (dataclass) `call, t, hit_tokens, admitted: bool`. `Recovery.hit_blocks: list[HitBlock]`, `PreemptionEpisode.originals: list[OriginalBlock]`, `PreemptionEpisode.lookups: list[LookupAttempt]`.
- `StepKind = Literal["decode", "mixed", "prefill"]`. `SchedStep` gains `kind: StepKind` and `t_done: int | None` (time of that output's Completed event, `None` if never acknowledged).
- `BatchBin` (frozen dataclass): `kind: StepKind, batch: int, steps: int, seconds: float`.
- `BATCH_TIME_BASIS = "host schedule-time proxy: each step lasts until the next schedule call, the final step until its completion; device time unavailable"`.
- `batch_distribution(steps) -> tuple[tuple[BatchBin, ...], FinalInterval]` where `FinalInterval = Literal["completed", "unacknowledged", "none"]`. Pure. No clipping. Bins sorted by `(kind, batch)`.
- `Distributions` (frozen dataclass): `batch: tuple[BatchBin, ...]`, `batch_basis: str`, `batch_final_interval: FinalInterval`, `candidate: dict[str, tuple[int, ...]]` with keys `completed, total_blocks, shared_blocks, reclaimable_blocks, shared_prefix_tokens` (one value per candidate row, decision order), `recovery_fraction: tuple[float, ...]` (resumed episodes with `C > 0`, episode id order), `episode_reprefill_tokens: tuple[int, ...]`.
- `RunHistory.distributions: Distributions | None` (`None` when replay is invalid).
- `run_metrics` scalars: `batch_size_time_weighted` (all steps, unclipped), `batch_size_decode_time_weighted` (decode-only steps), `batch_interval_max_s`, `steps_decode_only`, `steps_mixed`, `steps_prefill_only`. Existing keys otherwise keep their names.
- `load_run(directory, item_sha=None) -> RunHistory` keeps its signature and never raises on bad evidence.

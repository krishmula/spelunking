"""The engine event journal: one closed set of event kinds, a sealed writer, and a strict reader.

Each line is a JSON array `[seq, t_ns, call, kind, *fields]` with fields in the dataclass's order.
Every SEGMENT events the writer adds a Footer carrying the count of lines before it and the
running chain digest; the last line is RunEnd with the final count and digest. A reader that finds
a gap, a duplicate, a bad digest, an unknown kind, a field that does not match its annotation, or
no RunEnd never certifies the journal.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Union, get_args, get_origin, get_type_hints

SCHEMA = 1
SEGMENT = 1024
PRODUCER_ERROR = "producer-error.json"

KINDS: dict[str, type] = {}


class ShapeError(ValueError):
    pass


def parse_value(tp, value):
    """Parse decoded JSON against a type annotation and return the typed value, or raise ShapeError.
    `int` rejects bool, `None` passes only where the annotation allows it, and a NamedTuple is read
    from a list of exactly its arity."""
    origin = get_origin(tp)
    if tp in (int, str, bool):
        if type(value) is not tp:
            raise ShapeError(f"expected {tp.__name__}, got {value!r}")
        return value
    if tp is dict:
        if type(value) is not dict:
            raise ShapeError(f"expected an object, got {value!r}")
        return value
    if origin in (Union, types.UnionType):
        options = [a for a in get_args(tp) if a is not type(None)]
        if value is None and len(options) < len(get_args(tp)):
            return None
        if len(options) != 1:
            raise TypeError(f"unsupported annotation {tp}")
        return parse_value(options[0], value)
    if origin is Literal:
        if not any(type(value) is type(a) and value == a for a in get_args(tp)):
            raise ShapeError(f"expected one of {get_args(tp)}, got {value!r}")
        return value
    if origin is list:
        if type(value) is not list:
            raise ShapeError(f"expected a list, got {value!r}")
        (item,) = get_args(tp)
        return [parse_value(item, x) for x in value]
    if origin is dict:
        if type(value) is not dict:
            raise ShapeError(f"expected an object, got {value!r}")
        k_tp, v_tp = get_args(tp)
        return {parse_value(k_tp, k): parse_value(v_tp, v) for k, v in value.items()}
    if isinstance(tp, type) and issubclass(tp, tuple) and hasattr(tp, "_fields"):
        if type(value) is not list or len(value) != len(tp._fields):
            raise ShapeError(f"expected {tp.__name__} as a list of {len(tp._fields)}, got {value!r}")
        return tp(*parse_fields(tp, value))
    raise TypeError(f"unsupported annotation {tp}")


@functools.cache
def _hints(cls) -> dict:
    return get_type_hints(cls)


def parse_fields(cls, values) -> list:
    """Parse a positional row against the fields of a dataclass or NamedTuple."""
    hints = _hints(cls)
    names = cls._fields if hasattr(cls, "_fields") else [f.name for f in dataclasses.fields(cls)]
    out = []
    for name, value in zip(names, values, strict=True):
        try:
            out.append(parse_value(hints[name], value))
        except ShapeError as e:
            raise ShapeError(f"{cls.__name__}.{name}: {e}") from None
    return out


def _kind(name):
    def register(cls):
        cls.KIND = name
        cls.FIELDS = tuple(f.name for f in dataclasses.fields(cls))
        KINDS[name] = cls
        return cls
    return register


def _ev(name):
    return lambda cls: _kind(name)(dataclass(frozen=True, slots=True)(cls))


class ScheduledEntry(NamedTuple):
    r: int
    n: int
    start: int
    epoch: int


class CompletedEntry(NamedTuple):
    r: int
    n: int
    kind: Literal["valid", "stale", "gone"]


class DeliveredEntry(NamedTuple):
    r: int
    delivered: int
    processed_hint: int


@_ev("start")
class RunStart:
    run_id: str
    schema: int
    config: dict
    num_blocks: int
    null_block: int
    block_size: int
    observer_sha: str
    sources: dict[str, str]


@_ev("snap")
class SnapshotTaken:
    index: int
    reason: str
    digest: str


@_ev("digest")
class StateDigest:
    value: str


@_ev("fault")
class Fault:
    where: str
    error: str


@_ev("foot")
class Footer:
    n: int
    digest: str


@_ev("end")
class RunEnd:
    n: int
    digest: str
    faults: int
    snapshots: int
    unfinished: int


@_ev("add")
class ReqAdd:
    r: int
    engine_id: str
    prompt_len: int
    prompt_sha: str
    max_tokens: int


@_ev("queues")
class Queues:
    running: list[int]
    waiting: list[int]


@_ev("lookup")
class Lookup:
    r: int
    hit_tokens: int
    hit_blocks: list[int]


@_ev("alloc")
class Alloc:
    """One allocate_slots call that changed a block table, failed, or admitted a request.
    `needed` holds every block count the coordinator computed during the call, in order."""
    a: int
    r: int
    ok: bool
    n_new: int
    n_hit: int
    full_fit: bool
    reserved: int
    free_before: int
    needed: list[int]
    appended: list[int]
    running: list[int] | None


@_ev("detach")
class Detach:
    r: int
    blocks: list[int]


@_ev("sched")
class Scheduled:
    out: int
    entries: list[ScheduledEntry]
    new: list[int]
    resumed: list[int]
    total: int


@_ev("done")
class Completed:
    """Emitted before the stock output handling runs, so each entry's kind is the engine's own
    state at that moment."""
    out: int
    entries: list[CompletedEntry]


@_ev("delivered")
class Delivered:
    out: int
    entries: list[DeliveredEntry]


@_ev("decide")
class Decision:
    d: int
    victim: int
    a: int
    requester: int
    running: list[int] | None
    snapshot: int


@_ev("preempt")
class Preempted:
    d: int
    r: int
    raw: int
    in_flight: int
    stale: int
    free_before: int
    free_after: int
    preemptions: int


@_ev("finish")
class Finished:
    r: int
    status: str
    n_out: int
    n_tokens: int


@_ev("new")
class PoolAlloc:
    blocks: list[int]


@_ev("touch")
class PoolTouch:
    blocks: list[int]
    refs_before: list[int]


@_ev("free")
class PoolFree:
    blocks: list[int]
    refs_before: list[int]


@_ev("hadd")
class HashAdd:
    block: int
    key: str
    num_tokens: int
    primary: bool


@_ev("hdel")
class HashDel:
    block: int
    keys: list[str]


@_ev("creset")
class CacheReset:
    pass


@_ev("qpop")
class QPop:
    blocks: list[int]


@_ev("qrm")
class QRemove:
    block: int


@_ev("qpre")
class QPrepend:
    blocks: list[int]


@_ev("qapp")
class QAppend:
    blocks: list[int]


Payload = Union[tuple(KINDS.values())]


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    t_ns: int
    call: int
    payload: object


def encode(seq: int, t_ns: int, call: int, payload) -> bytes:
    row = [seq, t_ns, call, payload.KIND, *(getattr(payload, f) for f in payload.FIELDS)]
    return json.dumps(row, separators=(",", ":")).encode() + b"\n"


class Writer:
    """Single writer owned by the scheduler thread. Never drops an event: a write error is sticky
    and the journal is left without its RunEnd, which makes the run incomplete."""

    def __init__(self, path: Path):
        self.f = open(path, "xb", buffering=1 << 20)
        self.seq = 0
        self.chain = hashlib.sha256()
        self.faults = 0
        self.error: str | None = None
        self.closed = False

    def emit(self, call: int, t_ns: int, payload) -> None:
        if self.closed or self.error:
            return
        line = encode(self.seq, t_ns, call, payload)
        try:
            self.f.write(line)
        except OSError as e:
            self.error = repr(e)
            return
        self.chain.update(line)
        self.seq += 1
        if isinstance(payload, Fault):
            self.faults += 1
        if self.seq % SEGMENT == SEGMENT - 1:
            self.emit(call, t_ns, Footer(n=self.seq, digest=self.chain.hexdigest()))
            self.flush()

    def flush(self) -> None:
        if self.error or self.closed:
            return
        try:
            self.f.flush()
        except OSError as e:
            self.error = repr(e)

    def seal(self, call: int, t_ns: int, snapshots: int, unfinished: int) -> str | None:
        """Write RunEnd and make it durable. Returns None only when RunEnd was written, flushed,
        fsynced and closed with no earlier error; otherwise the first error."""
        self.emit(call, t_ns, RunEnd(n=self.seq, digest=self.chain.hexdigest(), faults=self.faults,
                                     snapshots=snapshots, unfinished=unfinished))
        self.flush()
        if self.error is None:
            try:
                os.fsync(self.f.fileno())
            except OSError as e:
                self.error = repr(e)
        try:
            self.f.close()
        except OSError as e:
            self.error = self.error or repr(e)
        self.closed = True
        return self.error


JournalStatus = Literal["sealed", "unsealed", "corrupt"]


@dataclass(frozen=True)
class Journal:
    status: JournalStatus
    events: tuple[Event, ...]
    reason: str | None = None
    bad_line: int | None = None
    end: RunEnd | None = None


class _Header(NamedTuple):
    seq: int
    t_ns: int
    call: int
    kind: str


def _parse_line(raw: bytes) -> tuple[_Header, object]:
    row = json.loads(raw)
    if type(row) is not list or len(row) < 4:
        raise ShapeError(f"expected a list of at least 4 values, got {row!r}")
    head = _Header(*parse_fields(_Header, row[:4]))
    cls = KINDS.get(head.kind)
    if cls is None:
        raise ShapeError(f"unknown kind {head.kind!r}")
    fields = row[4:]
    if len(fields) != len(cls.FIELDS):
        raise ShapeError(f"{head.kind}: {len(fields)} fields, expected {len(cls.FIELDS)}")
    return head, cls(*parse_fields(cls, fields))


def read_journal(path: Path) -> Journal:
    """Parse and verify a journal. Never raises on bad content; returns corrupt or unsealed."""
    try:
        data = path.read_bytes()
    except OSError as e:
        return Journal("corrupt", (), f"unreadable: {e!r}", 0)
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    elif lines:
        return _truncated(lines[:-1], len(lines) - 1, "partial final line")
    return _verify(lines)


def _truncated(lines, n, why):
    j = _verify(lines)
    if j.status == "corrupt":
        return j
    return Journal("unsealed", j.events, why, n)


def _verify(lines: list[bytes]) -> Journal:
    chain = hashlib.sha256()
    events: list[Event] = []
    for i, raw in enumerate(lines):
        try:
            (seq, t_ns, call, _), payload = _parse_line(raw)
        except ValueError as e:
            return Journal("corrupt", tuple(events), f"line {i}: malformed: {e}", i)
        if seq != i:
            return Journal("corrupt", tuple(events), f"line {i}: seq {seq}, expected {i}", i)
        if isinstance(payload, (Footer, RunEnd)):
            if payload.n != i or payload.digest != chain.hexdigest():
                return Journal("corrupt", tuple(events), f"line {i}: {payload.KIND} count or digest mismatch", i)
        if isinstance(payload, RunEnd):
            if i != len(lines) - 1:
                return Journal("corrupt", tuple(events), f"line {i}: events after RunEnd", i)
            return Journal("sealed", tuple(events), None, None, payload)
        chain.update(raw + b"\n")
        events.append(Event(seq, t_ns, call, payload))
    return Journal("unsealed", tuple(events), "no RunEnd", len(lines))


def snapshot_digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

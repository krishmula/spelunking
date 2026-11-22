
import hashlib
import json
import os
import re
import shutil

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from e02 import events as ev
from e02 import history
from e02.observer import ObservedScheduler
from engine_fixture import Engine, make_scheduler
from scenarios import TWO_VICTIMS, engine_kwargs


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    d = tmp_path_factory.mktemp("sealed") / "run"
    s = make_scheduler(ObservedScheduler, out_dir=d, snapshot_every=32, **engine_kwargs(TWO_VICTIMS))
    assert s.reset_prefix_cache()
    Engine(s).run(arrivals=TWO_VICTIMS["arrivals"])
    s.reset_prefix_cache()
    journal, rep = history.replay_attempt(d)
    assert journal.status == "sealed" and not isinstance(rep, history.Invalid)
    return d


def copy(src, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(src, dst)
    return dst, (dst / "events.jsonl").read_bytes().split(b"\n")[:-1]


def write_lines(d, lines):
    (d / "events.jsonl").write_bytes(b"".join(line + b"\n" for line in lines))


def rechain(d, transform):
    """Rewrite the journal as a consistent writer would have if it had recorded transform(events)."""
    journal = ev.read_journal(d / "events.jsonl")
    kept = transform([e for e in journal.events if not isinstance(e.payload, ev.Footer)])
    (d / "events.jsonl").unlink()
    w = ev.Writer(d / "events.jsonl")
    for e in kept:
        w.emit(e.call, e.t_ns, e.payload)
    w.seal(0, 0, journal.end.snapshots, journal.end.unfinished)


def first(events, kind):
    return next(i for i, e in enumerate(events) if isinstance(e.payload, kind))


def test_deleted_line_is_corrupt(sealed, tmp_path):
    d, lines = copy(sealed, tmp_path)
    write_lines(d, lines[:40] + lines[41:])
    journal, rep = history.replay_attempt(d)
    assert (journal.status, journal.bad_line) == ("corrupt", 40)
    assert isinstance(rep, history.Invalid)


def test_duplicated_line_is_corrupt(sealed, tmp_path):
    d, lines = copy(sealed, tmp_path)
    write_lines(d, lines[:41] + lines[40:])
    journal, rep = history.replay_attempt(d)
    assert (journal.status, journal.bad_line) == ("corrupt", 41)
    assert isinstance(rep, history.Invalid)


def test_corrupted_value_breaks_the_chain(sealed, tmp_path):
    d, lines = copy(sealed, tmp_path)
    i = next(i for i, line in enumerate(lines) if b'"touch"' in line or b'"free"' in line)
    row = json.loads(lines[i])
    row[-1] = [x + 1 for x in row[-1]]
    lines[i] = json.dumps(row, separators=(",", ":")).encode()
    write_lines(d, lines)
    journal, rep = history.replay_attempt(d)
    assert journal.status == "corrupt"
    assert "digest mismatch" in journal.reason
    assert isinstance(rep, history.Invalid)


def test_truncated_journal_is_unsealed(sealed, tmp_path):
    d, lines = copy(sealed, tmp_path)
    write_lines(d, lines[:-5])
    journal, rep = history.replay_attempt(d)
    assert (journal.status, journal.reason) == ("unsealed", "no RunEnd")
    assert not isinstance(rep, history.Invalid)
    assert rep.end is None


def test_partial_final_line_is_unsealed(sealed, tmp_path):
    d, lines = copy(sealed, tmp_path)
    (d / "events.jsonl").write_bytes(b"".join(line + b"\n" for line in lines[:-1]) + lines[-1][:10])
    journal, _ = history.replay_attempt(d)
    assert (journal.status, journal.reason) == ("unsealed", "partial final line")


def test_consistently_recorded_wrong_reference_is_caught_by_replay(sealed, tmp_path):
    d, _ = copy(sealed, tmp_path)

    def bump(events):
        i = first(events, ev.PoolFree)
        p = events[i].payload
        events[i] = ev.Event(0, events[i].t_ns, events[i].call,
                             ev.PoolFree(blocks=p.blocks, refs_before=[r + 1 for r in p.refs_before]))
        return events

    rechain(d, bump)
    journal, rep = history.replay_attempt(d)
    assert journal.status == "sealed"
    assert isinstance(rep, history.Invalid)
    assert rep.reason.startswith("free of block")


def test_dropped_queue_event_is_caught_by_replay(sealed, tmp_path):
    d, _ = copy(sealed, tmp_path)
    rechain(d, lambda events: events[:first(events, ev.QAppend)] + events[first(events, ev.QAppend) + 1:])
    journal, rep = history.replay_attempt(d)
    assert journal.status == "sealed"
    assert isinstance(rep, history.Invalid)


def test_snapshot_mismatch_is_a_failure(sealed, tmp_path):
    d, _ = copy(sealed, tmp_path)
    lines = (d / "snapshots.jsonl").read_text().splitlines()
    rec = json.loads(lines[1])
    rec["blocks"][0][1] += 1
    lines[1] = json.dumps(rec, separators=(",", ":"))
    (d / "snapshots.jsonl").write_text("\n".join(lines) + "\n")
    new_digest = hashlib.sha256(lines[1].encode()).hexdigest()

    def redigest(events):
        i = next(i for i, e in enumerate(events) if isinstance(e.payload, ev.SnapshotTaken) and e.payload.index == 1)
        p = events[i].payload
        events[i] = ev.Event(0, events[i].t_ns, events[i].call, ev.SnapshotTaken(p.index, p.reason, new_digest))
        return events

    rechain(d, redigest)
    journal, rep = history.replay_attempt(d)
    assert journal.status == "sealed"
    assert isinstance(rep, history.Invalid)
    assert rep.reason.startswith("snapshot 1 blocks differ")


def test_snapshot_edit_without_new_digest_is_a_failure(sealed, tmp_path):
    d, _ = copy(sealed, tmp_path)
    text = (d / "snapshots.jsonl").read_text().replace('"free":', '"free": ', 1)
    (d / "snapshots.jsonl").write_text(text)
    _, rep = history.replay_attempt(d)
    assert isinstance(rep, history.Invalid)
    assert rep.reason == "snapshot 0 digest mismatch"


def test_fault_event_invalidates(sealed, tmp_path):
    d, _ = copy(sealed, tmp_path)
    rechain(d, lambda events: events[:5] + [ev.Event(0, 0, 0, ev.Fault("hook", "boom"))] + events[5:])
    _, rep = history.replay_attempt(d)
    assert isinstance(rep, history.Invalid)
    assert rep.reason == "observer fault at hook: boom"


def test_footers_every_segment(tmp_path):
    w = ev.Writer(tmp_path / "events.jsonl")
    for i in range(3000):
        w.emit(i, i, ev.StateDigest(value=str(i)))
    w.seal(0, 0, 0, 0)
    journal = ev.read_journal(tmp_path / "events.jsonl")
    footers = [e.seq for e in journal.events if isinstance(e.payload, ev.Footer)]
    assert footers == [1023, 2047]
    assert journal.status == "sealed"
    assert journal.end.n == 3002


def append_snapshot_line(d, text):
    with open(d / "snapshots.jsonl", "a") as f:
        f.write(text + "\n")


@pytest.mark.parametrize("line, reason", [
    ("{not json", "snapshots invalid: snapshots.jsonl line 7: Expecting property name enclosed in double quotes: "
                  "line 1 column 2 (char 1)"),
    ("null", "snapshots invalid: snapshots.jsonl line 7: expected an object with keys "
             "['blocks', 'call', 'free', 'hash_keys', 'index', 'kv_holding_waiting', 'queue', 'reason', 'requests', "
             "'running', 'seq', 'victim', 'waiting']"),
])
def test_bad_trailing_snapshot_line_is_invalid(sealed, tmp_path, line, reason):
    d, _ = copy(sealed, tmp_path)
    assert len((d / "snapshots.jsonl").read_text().splitlines()) == 7
    append_snapshot_line(d, line)
    _, rep = history.replay_attempt(d)
    assert rep == history.Invalid(reason, None)


def test_duplicate_snapshot_index_is_invalid(sealed, tmp_path):
    d, _ = copy(sealed, tmp_path)
    append_snapshot_line(d, (d / "snapshots.jsonl").read_text().splitlines()[0])
    _, rep = history.replay_attempt(d)
    assert rep == history.Invalid("snapshots invalid: snapshots.jsonl line 7: index 0 differs from its line number",
                                  None)


def test_mistyped_snapshot_field_is_invalid(sealed, tmp_path):
    d, _ = copy(sealed, tmp_path)
    lines = (d / "snapshots.jsonl").read_text().splitlines()
    lines[1] = re.sub(r'"free":\d+', '"free":true', lines[1])
    (d / "snapshots.jsonl").write_text("\n".join(lines) + "\n")
    _, rep = history.replay_attempt(d)
    assert rep == history.Invalid("snapshots invalid: snapshots.jsonl line 1: free: expected int, got True", None)


def replace_first(kind, make):
    def transform(events):
        i = first(events, kind)
        events[i] = ev.Event(0, events[i].t_ns, events[i].call, make(events[i].payload))
        return events
    return transform


@pytest.mark.parametrize("kind, make, reason", [
    (ev.PoolFree, lambda p: ev.PoolFree(p.blocks, [True] * len(p.refs_before)),
     "PoolFree.refs_before: expected int, got True"),
    (ev.Completed, lambda p: ev.Completed(p.out, [[p.entries[0].r, p.entries[0].n, "late"]]),
     "Completed.entries: CompletedEntry.kind: expected one of ('valid', 'stale', 'gone'), got 'late'"),
    (ev.Scheduled, lambda p: ev.Scheduled(p.out, [[e.r, str(e.n), e.start, e.epoch] for e in p.entries], p.new,
                                          p.resumed, p.total),
     "Scheduled.entries: ScheduledEntry.n: expected int, got '8'"),
    (ev.Scheduled, lambda p: ev.Scheduled(p.out, [[e.r, e.n, e.start] for e in p.entries], p.new, p.resumed,
                                          p.total),
     "Scheduled.entries: expected ScheduledEntry as a list of 4, got [0, 8, 0]"),
    (ev.ReqAdd, lambda p: ev.ReqAdd(p.r, None, p.prompt_len, p.prompt_sha, p.max_tokens),
     "ReqAdd.engine_id: expected str, got None"),
])
def test_mistyped_journal_field_is_corrupt(sealed, tmp_path, kind, make, reason):
    d, _ = copy(sealed, tmp_path)
    rechain(d, replace_first(kind, make))
    journal, rep = history.replay_attempt(d)
    assert journal.status == "corrupt"
    assert journal.reason == f"line {journal.bad_line}: malformed: {reason}"
    assert isinstance(rep, history.Invalid)


@pytest.mark.parametrize("kind, make, reason", [
    (ev.PoolFree, lambda p: ev.PoolFree(p.blocks, p.refs_before[1:]), "free lists differ in length"),
    (ev.Alloc, lambda p: ev.Alloc(**{**{f: getattr(p, f) for f in p.FIELDS}, "appended": [999]}),
     "unknown block 999"),
    (ev.Decision, lambda p: ev.Decision(p.d, p.victim, p.a, p.requester, p.running, 999),
     "decision 1 names missing snapshot 999"),
])
def test_consistent_journal_with_bad_reference_is_invalid(sealed, tmp_path, kind, make, reason):
    d, _ = copy(sealed, tmp_path)
    rechain(d, replace_first(kind, make))
    journal, rep = history.replay_attempt(d)
    assert journal.status == "sealed"
    assert isinstance(rep, history.Invalid)
    assert rep.reason == reason


def observed_run(d):
    s = make_scheduler(ObservedScheduler, out_dir=d, snapshot_every=32, **engine_kwargs(TWO_VICTIMS))
    assert s.reset_prefix_cache()
    Engine(s).run(arrivals=TWO_VICTIMS["arrivals"])
    return s


def count_stock_resets(monkeypatch):
    calls = []
    stock = Scheduler.reset_prefix_cache

    def counting(self, *args, **kwargs):
        calls.append(1)
        return stock(self, *args, **kwargs)
    monkeypatch.setattr(Scheduler, "reset_prefix_cache", counting)
    return calls


def fail_fsync_of(monkeypatch, path):
    stock = os.fsync

    def fsync(fd):
        if os.fstat(fd).st_ino == path.stat().st_ino:
            raise OSError(5, "injected fsync failure")
        stock(fd)
    monkeypatch.setattr(ev.os, "fsync", fsync)


class FailingFile:
    def __init__(self, f, method):
        self.f, self.method = f, method

    def __getattr__(self, name):
        attr = getattr(self.f, name)
        if name != self.method:
            return attr

        def failing(*args):
            attr(*args)
            raise OSError(5, f"injected {name} failure")
        return failing


def test_clean_seal_returns_true_and_writes_no_producer_error(tmp_path):
    s = observed_run(tmp_path / "run")
    assert s.reset_prefix_cache() is True
    assert not (tmp_path / "run" / history.PRODUCER_ERROR).exists()
    assert history.load_run(tmp_path / "run").distributions is not None


def assert_producer_error(d, where, error, journal_status):
    assert json.loads((d / history.PRODUCER_ERROR).read_text()) == {"where": where, "error": error}
    h = history.load_run(d)
    assert (h.status, h.journal.status, h.distributions) == ("invalid", journal_status, None)
    assert h.reasons[0] == f"replay: producer error: {where}: {error} (seq None)"


def test_journal_fsync_failure_at_seal_is_a_producer_error(tmp_path, monkeypatch):
    d = tmp_path / "run"
    s = observed_run(d)
    calls = count_stock_resets(monkeypatch)
    fail_fsync_of(monkeypatch, d / "events.jsonl")
    assert s.reset_prefix_cache() is False
    assert calls == [1]
    assert_producer_error(d, "seal", "OSError(5, 'injected fsync failure')", "sealed")


def test_snapshots_fsync_failure_is_a_producer_error(tmp_path, monkeypatch):
    d = tmp_path / "run"
    s = observed_run(d)
    calls = count_stock_resets(monkeypatch)
    fail_fsync_of(monkeypatch, d / "snapshots.jsonl")
    assert s.reset_prefix_cache() is False
    assert calls == [1]
    assert_producer_error(d, "snapshots fsync", "OSError(5, 'injected fsync failure')", "sealed")


@pytest.mark.parametrize("method", ["flush", "close"])
def test_journal_flush_or_close_failure_at_seal_is_a_producer_error(tmp_path, monkeypatch, method):
    d = tmp_path / "run"
    s = observed_run(d)
    calls = count_stock_resets(monkeypatch)
    s._w.f = FailingFile(s._w.f, method)
    assert s.reset_prefix_cache() is False
    assert calls == [1]
    assert_producer_error(d, "seal", f"OSError(5, 'injected {method} failure')", "sealed")


def test_earlier_write_error_leaves_no_run_end_and_is_a_producer_error(tmp_path, monkeypatch):
    d = tmp_path / "run"
    s = make_scheduler(ObservedScheduler, out_dir=d, snapshot_every=32, **engine_kwargs(TWO_VICTIMS))
    assert s.reset_prefix_cache()
    s._w.f = FailingFile(s._w.f, "write")
    Engine(s).run(arrivals=TWO_VICTIMS["arrivals"])
    calls = count_stock_resets(monkeypatch)
    assert s.reset_prefix_cache() is False
    assert calls == [1]
    assert_producer_error(d, "writer", "OSError(5, 'injected write failure')", "unsealed")


def test_effective_capture_cadence_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("E02_DIGEST_EVERY", "3")
    d = tmp_path / "run"
    s = make_scheduler(ObservedScheduler, out_dir=d, snapshot_every=5, **engine_kwargs(TWO_VICTIMS))
    assert s.reset_prefix_cache()
    s.reset_prefix_cache()
    config = ev.read_journal(d / "events.jsonl").events[0].payload.config
    assert (config["snapshot_every"], config["digest_every"]) == (5, 3)


def test_writer_seal_reports_its_error(tmp_path, monkeypatch):
    w = ev.Writer(tmp_path / "ok.jsonl")
    assert w.seal(0, 0, 0, 0) is None
    w = ev.Writer(tmp_path / "bad.jsonl")
    fail_fsync_of(monkeypatch, tmp_path / "bad.jsonl")
    assert w.seal(0, 0, 0, 0) == "OSError(5, 'injected fsync failure')"

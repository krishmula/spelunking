import signal

import pytest

from e02 import sweep
from pilot_fixture import behavior_root, install, provenance, sealed_root, start_attempt


@pytest.fixture(autouse=True)
def small_pilots(monkeypatch):
    install(monkeypatch)


@pytest.mark.parametrize("phase", ["validate_evidence", "evidence_digests"])
def test_finalization_over_wall_ceiling_cannot_complete(tmp_path, monkeypatch, phase):
    now = [1000.0]
    monkeypatch.setattr(sweep.time, "time", lambda: now[0])
    _, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    original = getattr(sweep, phase)

    def consume(*args):
        result = original(*args)
        now[0] += 5000.0
        return result

    monkeypatch.setattr(sweep, phase, consume)
    final = sweep.finish_attempt(path, rec, "complete", None, ctx)
    assert (final["outcome"], final["reason"], final["wall_ended"] - final["wall_started"]) == (
        "incomplete", "budget: wall: the attempt passed its budgeted end", 5000.0)
    assert sweep.verify_attempt(path, ctx).status == "incomplete"
    assert final["evidence"] == {}


def test_finalization_alarm_interrupts_validation(tmp_path, monkeypatch):
    _, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    old_handler = signal.getsignal(signal.SIGALRM)
    reached = []

    def interrupted(*args):
        reached.append("entered")
        signal.raise_signal(signal.SIGALRM)
        reached.append("returned")

    monkeypatch.setattr(sweep, "validate_evidence", interrupted)
    final = sweep.finish_attempt(path, rec, "complete", None, ctx)
    assert (final["outcome"], final["reason"], reached) == (
        "incomplete", "budget: wall: final validation or hashing exceeded its deadline", ["entered"])
    assert signal.getsignal(signal.SIGALRM) == old_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_final_metadata_cannot_consume_disk_reserve(tmp_path, monkeypatch):
    _, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    free = [sweep.GIB]
    monkeypatch.setattr(sweep, "free_bytes", lambda: free[0])
    original = sweep.evidence_digests

    def fill(*args):
        result = original(*args)
        free[0] = 1
        return result

    monkeypatch.setattr(sweep, "evidence_digests", fill)
    with pytest.raises(sweep.BudgetExceeded, match="final metadata would consume the free-space reserve"):
        sweep.finish_attempt(path, rec, "complete", None, ctx)
    assert sweep.read_attempt(path)["outcome"] == "validating"
    assert (path / "client.jsonl").exists()


def test_terminal_complete_sweep_still_enforces_budget(tmp_path, monkeypatch):
    root = behavior_root(tmp_path)
    current = provenance()
    monkeypatch.setattr(sweep, "readiness", lambda **k: {"provenance": current,
                                                        "provenance_digest": sweep.digest_of(current)})
    monkeypatch.setattr(sweep, "corpus_texts", lambda: {})
    monkeypatch.setattr(sweep, "root_consumed_s", lambda root: 5000.0)

    def forbidden(*args):
        raise AssertionError("an attempt started")

    monkeypatch.setattr(sweep, "run_attempt", forbidden)
    with pytest.raises(sweep.GateError, match="completed evidence exceeds the terminal sweep budget"):
        sweep.execute(root, sweep.ExecutionGrant("CPU terminal check", "now"))
    assert '"reason": "wall:' in (root / "halts.jsonl").read_text()


@pytest.mark.parametrize("field", ["survivors", "unidentified"])
def test_failed_cleanup_cannot_complete_or_qualify_a_pilot(tmp_path, field):
    root, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    rec.update(cleanup_failed=True, server_pgid=424242, owned_processes=[[424242, "owned"]],
               stop_report={"survivors": [], "unidentified": [], field: [424242]})
    final = sweep.finish_attempt(path, rec, "complete", None, ctx)
    assert (final["outcome"], final["reason"], sweep.verify_attempt(path, ctx).status) == (
        "failed", "cleanup: owned server cleanup is unresolved", "failed")
    verdict = sweep.cost_verdict(root, "cost-low")
    assert verdict["verdict"] == "inconclusive"
    assert "p1-off is failed: cleanup: owned server cleanup is unresolved" in verdict["reasons"]


def test_resume_retries_cleanup_with_preserved_positive_identities(tmp_path, monkeypatch):
    _, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    identities = [[424242, "owned"]]
    rec.update(outcome="complete", cleanup_failed=True, server_pgid=424242, owned_processes=identities,
               stop_report={"survivors": [424242], "unidentified": []})
    sweep._write_atomic(path / "attempt.json", rec)
    assert sweep.verify_attempt(path, ctx).status == "failed"
    processes = {424242: sweep.Proc(424242, 1, 424242, "owned")}
    sent = []

    def terminate(pgid, sig):
        sent.append((pgid, sig))
        processes.clear()

    monkeypatch.setattr(sweep.os, "killpg", terminate)
    sweep.reconcile(ctx, "p1-off", table=lambda: processes.copy())
    final = sweep.read_attempt(path)
    assert sent == [(424242, signal.SIGTERM)]
    assert (final["outcome"], final["cleanup_failed"], final["owned_processes"]) == ("failed", False, identities)
    assert final["cleanup_reports"] == [{"survivors": [424242], "unidentified": []}]
    assert final["stop_report"]["survivors"] == []


def test_resume_cleanup_blocks_unidentified_members_without_a_signal(tmp_path, monkeypatch):
    _, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    rec.update(outcome="complete", cleanup_failed=True, server_pgid=424242,
               owned_processes=[[424242, "old identity"]],
               stop_report={"survivors": [], "unidentified": [424242]})
    sweep._write_atomic(path / "attempt.json", rec)
    processes = {424242: sweep.Proc(424242, 1, 424242, "different identity")}

    def forbidden(*args):
        raise AssertionError("an unrelated process was signaled")

    monkeypatch.setattr(sweep.os, "killpg", forbidden)
    monkeypatch.setattr(sweep.os, "kill", forbidden)
    with pytest.raises(sweep.GateError, match="cleanup remains unresolved"):
        sweep.reconcile(ctx, "p1-off", table=lambda: processes)
    final = sweep.read_attempt(path)
    assert (final["outcome"], final["cleanup_failed"], final["owned_processes"]) == (
        "failed", True, [[424242, "old identity"]])

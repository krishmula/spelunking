
import asyncio
import dataclasses
import os
import signal
import subprocess
import sys
import time

import pytest
from aiohttp import web

from e02 import __main__ as cli
from e02 import sweep
from pilot_fixture import CFG, DEAD_PID, install, provenance, sealed_root

GIB = sweep.GIB
MIB = 2**20


def test_attempt_upper_bound_sums_every_phase_deadline():
    assert CFG.attempt_max_s(sweep.Deadlines(1200.0, 180.0)) == 900 + 600 + 1200 + 120 + 0.5 + 5 + 300


def test_admission_needs_room_for_a_whole_attempt():
    c = sweep.Ceilings(10_000.0, 10 * GIB, 2 * GIB, 64 * MIB)
    assert sweep.admit_attempt(5_000.0, 4_000.0, c, GIB, 5 * GIB) is None
    assert sweep.admit_attempt(6_500.0, 4_000.0, c, GIB, 5 * GIB) == (
        "wall: 1.81 h used of 2.78 h, and the next attempt may take up to 1.11 h")
    assert sweep.admit_attempt(0.0, 1.0, c, 10 * GIB - 32 * MIB, 5 * GIB) == (
        "disk: the run root holds 9.97 GiB of its 10.00 GiB ceiling")
    assert sweep.admit_attempt(0.0, 1.0, c, GIB, 2 * GIB) == "disk: 2.00 GiB free, at the 2.00 GiB reserve"


def write_attempt(root, name, mtime, **rec):
    d = root / "attempts" / name
    d.mkdir(parents=True)
    sweep._write_atomic(d / "attempt.json", rec)
    (d / "server.log").write_text("x")
    for f in d.iterdir():
        os.utime(f, (mtime, mtime))


def test_consumed_wall_time_persists_across_attempts_resumes_and_dead_runners(tmp_path):
    write_attempt(tmp_path, "p1-on.a1", 400, outcome="complete", wall_started=100.0, wall_ended=400.0)
    write_attempt(tmp_path, "p1-off.a1", 560, outcome="failed", wall_started=500.0, wall_ended=560.0)
    write_attempt(tmp_path, "p1-off.a2", 1100, outcome="running", wall_started=900.0, wall_checkpoint=1000.0)
    assert sweep.root_consumed_s(tmp_path) == 300 + 60 + 200


def test_budget_checks_wall_and_disk_with_a_controlled_clock(tmp_path):
    now, free, saved = [0.0], [10 * GIB], []
    c = sweep.Ceilings(1000.0, 4 * GIB, 2 * GIB, 64 * MIB)
    b = sweep.Budget(c, GIB, tmp_path, end=100.0, clock=lambda: now[0], free=lambda: free[0], checkpoint=saved.append)
    assert (b.problem(), saved) == (None, [])
    now[0] = 31.0
    assert (b.problem(), saved) == (None, [31.0])
    free[0] = 2 * GIB
    assert b.problem() == "budget: disk: 2.00 GiB free, at the 2.00 GiB reserve"
    free[0] = 10 * GIB
    (tmp_path / "events.jsonl").write_bytes(b"x" * (3 * GIB // 1024))
    b.root_bytes_before = 4 * GIB - 64 * MIB
    assert b.problem() == "budget: disk: the run root holds 3.94 GiB of its 4.00 GiB ceiling"
    b.root_bytes_before = 0
    now[0] = 101.0
    with pytest.raises(sweep.BudgetExceeded, match="budget: wall: the attempt passed its budgeted end"):
        b.check()


def test_execute_halts_before_an_attempt_that_cannot_fit(tmp_path, monkeypatch):
    install(monkeypatch)
    root, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = sweep.begin_attempt(ctx, ctx.sealed.spec("p1-on"), 1, None, None)
    sweep.finish_attempt(path, rec, "failed", "server exited", ctx, wall_ended=rec["wall_started"] + 2000.0)
    current = provenance()
    monkeypatch.setattr(sweep, "readiness", lambda **k: {"provenance": current,
                                                         "provenance_digest": sweep.digest_of(current)})
    monkeypatch.setattr(sweep, "corpus_texts", lambda: {})

    def forbidden(*a, **k):
        raise AssertionError("an attempt started")

    monkeypatch.setattr(sweep, "run_attempt", forbidden)
    with pytest.raises(sweep.GateError, match=r"budget: wall: 0.56 h used of 1.00 h, and the next attempt may take "
                                              r"up to 0.55 h; stopped with .* incomplete"):
        sweep.execute(root, sweep.ExecutionGrant("test without a server", "now"))
    assert len((root / "halts.jsonl").read_text().splitlines()) == 1


RULE = sweep.WarmupRule(prompts=2, tokens=8, min_rounds=3, max_rounds=5, window=3, max_relative_range=0.05,
                        max_s=60.0)


def samples(rates, therm=""):
    return [sweep.WarmupSample(i + 1, r, 1.0, therm) for i, r in enumerate(rates)]


def test_warmup_verdict_needs_a_steady_window_without_thermal_limits():
    assert sweep.warmup_verdict(samples([100, 80]), RULE) == (False, "2 of at least 3 rounds")
    assert sweep.warmup_verdict(samples([100, 80, 99]), RULE) == (
        False, "last 3 rounds spread 20.2%, above 5%")
    assert sweep.warmup_verdict(samples([100, 80, 99, 100, 101]), RULE) == (
        True, "last 3 rounds within 2.0% of their median 100.0 tokens/s")
    assert sweep.warmup_verdict(samples([100, 100, 100], "CPU_Speed_Limit \t= 80"), RULE) == (
        False, "thermal limit reported in the last rounds")


def test_thermal_limit_parsing():
    assert not sweep.thermal_limited("Note: No thermal warning level has been recorded\n"
                                     "Note: No performance warning level has been recorded")
    assert not sweep.thermal_limited("CPU_Speed_Limit \t= 100")
    assert sweep.thermal_limited("CPU_Speed_Limit \t= 70")
    assert sweep.thermal_limited("Thermal warning level set to 2.")


def warm_against(rule, clock_values, guard=lambda: None):
    bodies = []

    async def handler(request):
        bodies.append(await request.json())
        return web.json_response({"usage": {"completion_tokens": rule.tokens}})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        cfg = dataclasses.replace(CFG, port=site._server.sockets[0].getsockname()[1], warmup=rule)
        ticks = iter(clock_values)
        try:
            return await sweep.run_warmup(cfg, guard, clock=lambda: next(ticks), therm=lambda: "")
        finally:
            await runner.cleanup()
    return asyncio.run(go()), bodies


def test_warmup_runs_bounded_rounds_until_steady():
    clock = [0, 0, 0, 1, 1, 1, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 4.5]
    result, bodies = warm_against(RULE, clock)
    assert [(s.round, s.tokens, s.elapsed_s) for s in result.samples] == [
        (1, 16, 1), (2, 16, 0.5), (3, 16, 1), (4, 16, 1), (5, 16, 1)]
    assert (result.stable, result.reason) == (True, "last 3 rounds within 0.0% of their median 16.0 tokens/s")
    assert len(bodies) == 10
    assert {(b["max_tokens"], b["ignore_eos"], b["stream"]) for b in bodies} == {(8, True, False)}
    for b in bodies:
        assert_metal_accepts(b)


def assert_metal_accepts(body):
    """Run a request body through the server's parsing and vllm-metal's request check."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm_metal.platform import MetalPlatform
    params = ChatCompletionRequest(**body).to_sampling_params(body["max_tokens"], {})
    MetalPlatform.validate_request(None, params)


def test_warmup_without_steady_state_is_reported_unstable():
    clock = [0, 0, 0, 1, 1, 1, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0, 4.0]
    result, _ = warm_against(RULE, clock)
    assert (result.stable, len(result.samples)) == (False, 5)
    assert result.reason == "no steady state in 5 rounds: last 3 rounds spread 100.0%, above 5%"


def test_warmup_stops_when_the_budget_guard_trips():
    with pytest.raises(sweep.BudgetExceeded, match="budget: disk"):
        warm_against(RULE, [0, 0, 0, 1], guard=lambda: "budget: disk: 1.00 GiB free, at the 5.00 GiB reserve")


def test_pending_warmup_is_cancelled_when_disk_budget_trips():
    received = []

    async def go():
        async def handler(request):
            received.append(await request.json())
            await asyncio.sleep(2)
            return web.json_response({"usage": {"completion_tokens": RULE.tokens}})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        cfg = dataclasses.replace(CFG, port=site._server.sockets[0].getsockname()[1], warmup=RULE)
        try:
            with pytest.raises(sweep.BudgetExceeded, match="budget: disk: controlled reserve"):
                await sweep.run_warmup(cfg, lambda: "budget: disk: controlled reserve" if received else None,
                                       therm=lambda: "")
        finally:
            await runner.cleanup()

    asyncio.run(go())
    assert len(received) == 2


def test_cleanup_escalates_when_budget_runs_out(tmp_path):
    g, child = spawn_group(tmp_path)
    report = g.stop(60.0, 5.0, guard=lambda: "budget: disk: controlled reserve")
    assert (report["budget_problem"], report["killed"], report["survivors"]) == (
        "budget: disk: controlled reserve", [child], [])
    assert not alive(child)


LEADER = """
import subprocess, sys
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(120)"])
print(child.pid, flush=True)
"""


def alive(pid):
    return pid in sweep.process_table()


def spawn_group(tmp_path):
    """A leader that starts a TERM-ignoring child in its process group and exits at once."""
    log = open(tmp_path / "group.log", "w")
    g = sweep.OwnedGroup([sys.executable, "-c", LEADER], env=dict(os.environ), cwd=tmp_path, log=log)
    end = time.monotonic() + 20
    while time.monotonic() < end:
        lines = (tmp_path / "group.log").read_text().split()
        if "ready" in lines:
            child = int(next(x for x in lines if x.isdigit()))
            if alive(child):
                return g, child
        time.sleep(0.05)
    raise AssertionError("child did not start")


def test_stop_kills_a_surviving_member_after_the_leader_exits(tmp_path):
    g, child = spawn_group(tmp_path)
    g.sample()
    assert child in g.pids()
    report = g.stop(0.5, 5.0)
    assert (report["group_identified"], report["term"], report["killed"], report["survivors"],
            report["unidentified"]) == (True, [child], [child], [], [])
    assert not alive(child)


def test_reaped_leader_group_is_cleaned_by_recorded_identity(tmp_path):
    g, child = spawn_group(tmp_path)
    g.sample()
    g.proc.wait()
    report = g.stop(0.5, 5.0)
    assert (report["group_identified"], report["killed"], report["survivors"]) == (True, [child], [])
    assert not alive(child)


def test_unidentified_group_member_is_never_signaled(tmp_path):
    g, child = spawn_group(tmp_path)
    g.proc.wait()
    g.known.clear()
    try:
        report = g.stop(0.5, 1.0)
        assert (report["group_identified"], report["term"], report["unidentified"]) == (False, [], [child])
        assert alive(child)
    finally:
        os.kill(child, signal.SIGKILL)


def test_reconcile_cleans_a_dead_runners_server_group(tmp_path, monkeypatch):
    install(monkeypatch)
    root, ctx = sealed_root(tmp_path, "cost-low")
    g, child = spawn_group(tmp_path)
    g.sample()
    g.proc.wait()
    path, rec = sweep.begin_attempt(ctx, ctx.sealed.spec("p1-on"), 1, None, None)
    sweep._write_atomic(path / "attempt.json", {**rec, "pid": DEAD_PID, "runner_start": None,
                                                "server_pgid": g.pgid, "owned_processes": g.identities()})
    sweep.reconcile(ctx, "p1-on")
    final = sweep.read_attempt(path)
    assert (final["outcome"], final["reason"], final["stop_report"]["killed"]) == (
        "failed", "runner died while the attempt was running", [child])
    assert not alive(child)


def test_reconcile_never_signals_an_unrelated_group(tmp_path, monkeypatch):
    install(monkeypatch)
    root, ctx = sealed_root(tmp_path, "cost-low")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        path, rec = sweep.begin_attempt(ctx, ctx.sealed.spec("p1-on"), 1, None, None)
        sweep._write_atomic(path / "attempt.json", {
            **rec, "pid": DEAD_PID, "runner_start": None, "server_pgid": other.pid,
            "owned_processes": [[other.pid, "Mon Jan  1 00:00:00 2001"]]})
        with pytest.raises(sweep.GateError, match=f"server group {other.pid} that cannot be positively identified"):
            sweep.reconcile(ctx, "p1-on")
        assert other.poll() is None
        assert sweep.read_attempt(path)["outcome"] == "running"
    finally:
        other.kill()
        other.wait()


def table(*rows):
    return {pid: sweep.Proc(pid, ppid, pgid, "t") for pid, ppid, pgid in rows}


def test_owned_identities_include_the_engine_core_and_exclude_others():
    t = table((100, 1, 100), (200, 100, 200), (300, 200, 200), (301, 200, 200), (302, 300, 302), (400, 1, 400))
    assert [p.pid for p in sweep.group_members(200, 200, t)] == [200, 300, 301, 302]


def test_competing_gpu_share_excludes_every_owned_process():
    before = {(100, "python"): 0, (200, "vllm"): 0, (300, "EngineCore"): 0, (400, "WindowServer"): 1_000_000_000}
    after = {(100, "python"): 0, (200, "vllm"): 0, (300, "EngineCore"): 6_000_000_000,
             (400, "WindowServer"): 2_000_000_000}
    assert sweep.other_gpu_share(before, after, 10.0, {100, 200}) == 70.0
    assert sweep.other_gpu_share(before, after, 10.0, {100, 200, 300, 301, 302}) == 10.0


def test_workload_commands_without_execute_start_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "RUNS", tmp_path / "runs")

    def forbidden(*a, **k):
        raise AssertionError("a process or gate ran")

    monkeypatch.setattr(sweep.subprocess, "Popen", forbidden)
    monkeypatch.setattr(sweep, "readiness", forbidden)
    assert cli.main(["pilot", "behavior"]) == 2
    assert cli.main(["main", str(tmp_path / "runs" / "20261020-000000-main")]) == 2
    err = capsys.readouterr().err
    assert err.count("blocked: Workload execution is suspended. On 2026-10-20 at 14:33 PDT Krishna said "
                     "\"Don't run the experiment unless I tell you so.\"") == 2
    assert not (tmp_path / "runs").exists()


def test_server_and_attempt_refuse_without_a_grant(tmp_path, monkeypatch):
    install(monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError("a process started")

    monkeypatch.setattr(sweep.subprocess, "Popen", forbidden)
    root, ctx = sealed_root(tmp_path, "cost-low")
    spec = ctx.sealed.spec("p1-on")
    with pytest.raises(sweep.GateError, match="Workload execution is suspended"):
        sweep.Server(tmp_path, spec, CFG, None, "x")
    with pytest.raises(sweep.GateError, match="Workload execution is suspended"):
        sweep.run_attempt(ctx, spec, 1, None, None, {}, None)
    with pytest.raises(sweep.GateError, match="Workload execution is suspended"):
        sweep.run_pilot("behavior", None)
    assert list((root / "attempts").iterdir()) == []


def test_workload_help_states_the_suspended_authority(capsys):
    with pytest.raises(SystemExit):
        cli.main(["pilot", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "Don't run the experiment unless I tell you so." in out
    assert "--execute start the workload; pass only after Krishna's explicit instruction to run it" in out

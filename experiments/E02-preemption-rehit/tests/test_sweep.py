
import asyncio
import dataclasses
import hashlib
import json

import aiohttp
import pytest
from aiohttp import web

from e02 import history, sweep

CFG = sweep.current_config()


def test_main_plan_is_the_fixed_paired_sweep():
    runs = sweep.main_plan()
    assert len(runs) == 100
    assert {r.requests for r in runs} == {100}
    assert len({r.run_id for r in runs}) == 100
    for rnd, seed in enumerate((11, 23, 37, 53, 71), start=1):
        in_round = [r for r in runs if r.round == rnd]
        assert sorted((r.target, r.cap) for r in in_round) == sorted(
            (t, c) for t in (4, 8, 16, 24, 32) for c in (512, 1024, 2048, 4096))
        assert {r.rep_seed for r in in_round} == {seed}
        assert len({r.item_order for r in in_round}) == 1
        assert len({r.item_seeds for r in in_round}) == 1
        assert sorted(in_round[0].item_order) == list(range(100))
    assert runs[0].item_order[:6] == (82, 72, 51, 58, 54, 47)
    assert [r.item_order for r in runs[:20]] != [r.item_order for r in runs[20:40]]
    assert [(r.target, r.cap) for r in runs[:20]] != [(r.target, r.cap) for r in runs[20:40]]
    assert sweep.main_plan() == runs
    args = sweep.SERVER_ARGS
    assert args[args.index("--num-gpu-blocks-override") + 1] == "776"


def test_item_seed_depends_on_item_and_repetition():
    seeds = {sweep.item_seed(i, s) for i in range(100) for s in (11, 23, 37, 53, 71)}
    assert len(seeds) == 500
    assert sweep.item_seed(5, 11) == sweep.item_seed(5, 11)


def test_cost_pilot_pairs_alternate_order():
    runs = sweep.pilot_plan("cost-pressure")
    assert [r.run_id for r in runs] == ["p1-on", "p1-off", "p2-off", "p2-on", "p3-on", "p3-off", "p4-off", "p4-on",
                                        "p5-on", "p5-off"]
    assert {r.requests for r in runs} == {16}
    assert len({r.item_order for r in runs}) == 1


def test_config_round_trips_through_strict_parsing():
    data = sweep.to_json(CFG)
    assert sweep.parse_as(sweep.ExecConfig, data, "config") == CFG
    data["temperature"] = "0.6"
    with pytest.raises(sweep.GateError, match=r"config.temperature: expected float, found '0.6'"):
        sweep.parse_as(sweep.ExecConfig, data, "config")


def test_main_gate_rejects_missing_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "PREDICTIONS", tmp_path / "predictions.json")
    monkeypatch.setattr(sweep, "PREDICTIONS_SHA", tmp_path / "predictions.sha256")
    with pytest.raises(sweep.GateError, match="predictions are not frozen"):
        sweep.readiness(main=True)


def test_main_gate_rejects_prediction_without_number(monkeypatch, tmp_path):
    doc = json.loads(sweep.PREDICTIONS.read_text())
    doc["krishna"]["recovery"]["low_pressure"] = {"greater_than": "most"}
    text = json.dumps(doc)
    (tmp_path / "p.json").write_text(text)
    (tmp_path / "p.sha").write_text(hashlib.sha256(text.encode()).hexdigest())
    monkeypatch.setattr(sweep, "PREDICTIONS", tmp_path / "p.json")
    monkeypatch.setattr(sweep, "PREDICTIONS_SHA", tmp_path / "p.sha")
    with pytest.raises(sweep.GateError, match="numeric recovery value"):
        sweep.readiness(main=True)


def test_main_gate_rejects_busy_port(monkeypatch):
    monkeypatch.setattr(sweep, "port_busy", lambda port: port == 8013)
    with pytest.raises(sweep.GateError, match="port 8013 is busy"):
        sweep.readiness(main=False)


def test_main_gate_rejects_missing_corpus(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "DATA", tmp_path)
    with pytest.raises(sweep.GateError, match="corpus is not frozen"):
        sweep.readiness(main=True)


def test_main_gate_rejects_incompatible_engine(monkeypatch):
    monkeypatch.setitem(sweep.PINNED_ENGINE, "vllm-core", "0" * 40)
    with pytest.raises(sweep.GateError, match="vllm-core is at"):
        sweep.readiness(main=True)


def test_main_gate_rejects_insufficient_disk(monkeypatch):
    monkeypatch.setattr(sweep, "port_busy", lambda port: False)
    with pytest.raises(sweep.GateError, match="GiB free, need"):
        sweep.readiness(main=False, disk_need_gib=10_000, reserve_gib=5)


def test_gate_requires_a_technical_record_for_the_current_source(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "port_busy", lambda port: False)
    monkeypatch.setattr(sweep, "TECHNICAL", tmp_path)
    with pytest.raises(sweep.GateError, match="no passing offline technical record"):
        sweep.readiness(main=False)
    run = tmp_path / "20261020-000000"
    run.mkdir()
    (run / "junit.xml").write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    result = {"fingerprint": sweep.technical_fingerprint(), "fingerprint_after": sweep.technical_fingerprint(),
              "targets": ["tests"], "exit_code": 0, "tests": 3, "failures": 0, "errors": 0, "skipped": 0,
              "junit_sha256": sweep.file_sha(run / "junit.xml")}
    (run / "result.json").write_text(json.dumps(result))
    assert sweep.check_technical() == {"record": f"{tmp_path.name}/20261020-000000/result.json",
                                       "sha256": sweep.file_sha(run / "result.json"), "tests": 3}
    assert sweep.junit_counts(run / "junit.xml") == {"tests": 3, "failures": 0, "errors": 0, "skipped": 0}
    result["fingerprint"] = "0" * 64
    (run / "result.json").write_text(json.dumps(result))
    with pytest.raises(sweep.GateError, match="no passing offline technical record"):
        sweep.check_technical()


@pytest.mark.parametrize("result, junit", [("null", "<testsuites/>"), ("{}", "not xml")])
def test_corrupt_technical_evidence_is_a_gate_failure(monkeypatch, tmp_path, result, junit):
    monkeypatch.setattr(sweep, "TECHNICAL", tmp_path)
    run = tmp_path / "20261020-000000"
    run.mkdir()
    (run / "result.json").write_text(result)
    (run / "junit.xml").write_text(junit)
    with pytest.raises(sweep.GateError, match="no passing offline technical record"):
        sweep.check_technical()


def test_non_utf8_client_evidence_is_invalid(tmp_path):
    (tmp_path / "client.jsonl").write_bytes(b"\xff\n")
    records, reason = history.read_client(tmp_path / "client.jsonl")
    assert records == []
    assert reason.startswith("client.jsonl unreadable: 'utf-8' codec can't decode byte 0xff")


def test_main_gate_rejects_missing_pilot_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "port_busy", lambda port: False)
    monkeypatch.setattr(sweep, "check_technical", lambda: {"record": "x", "sha256": "y", "tests": 1})
    monkeypatch.setattr(sweep, "RUNS", tmp_path)
    with pytest.raises(sweep.GateError, match="no behavior pilot: `python -m e02 pilot behavior --execute` needs"):
        sweep.readiness(main=True)


def sse(*msgs):
    return "".join(f"data: {json.dumps(m)}\n\n" for m in msgs) + "data: [DONE]\n\n"


async def serve(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def run_client(handler, body=None):
    async def go():
        runner, url = await serve(handler)
        written = []
        async with aiohttp.ClientSession() as session:
            rec = await sweep.stream_one(session, url, "root.a1.i3", 3, 0, "startup", body or {}, [0], written.append)
        await runner.cleanup()
        return rec, written
    return asyncio.run(go())


def test_client_counts_tokens_from_usage_not_chunks():
    seen = {}

    async def handler(request):
        seen["id"] = request.headers["X-Request-Id"]
        await asyncio.sleep(0.05)
        resp = web.StreamResponse()
        await resp.prepare(request)
        rid = "chatcmpl-" + request.headers["X-Request-Id"]
        await resp.write(sse({"id": rid, "choices": [{"delta": {"role": "assistant", "content": ""}}]}).encode()[:-14])
        await asyncio.sleep(0.05)
        await resp.write(sse({"id": rid, "choices": [{"delta": {"reasoning_content": "Let me think"}}]},
                             {"id": rid, "choices": [{"delta": {"content": "<think> two tokens"}}]},
                             {"id": rid, "choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]},
                             {"id": rid, "choices": [], "usage": {"prompt_tokens": 1019, "completion_tokens": 7}}).encode())
        return resp

    rec, written = run_client(handler)
    assert seen["id"] == "root.a1.i3"
    assert (rec.status, rec.finish_reason, rec.prompt_tokens, rec.completion_tokens) == ("ok", "length", 1019, 7)
    assert (rec.chunks, rec.content_chunks, rec.reasoning_chunks) == (5, 2, 1)
    assert rec.response_id == "chatcmpl-root.a1.i3"
    assert rec.t_first_ns - rec.t_submit_ns >= 100_000_000
    assert written == [rec]


def test_client_records_http_failure():
    async def handler(request):
        return web.Response(status=400, text="prompt too long")

    rec, _ = run_client(handler)
    assert (rec.status, rec.http_status, rec.error, rec.completion_tokens) == ("http_error", 400, "prompt too long", None)


def test_client_marks_stream_without_usage_as_error():
    async def handler(request):
        resp = web.StreamResponse()
        await resp.prepare(request)
        await resp.write(sse({"id": "x", "choices": [{"delta": {"content": "a"}}]}).encode())
        return resp

    rec, _ = run_client(handler)
    assert (rec.status, rec.error) == ("error", "stream ended without finish reason and usage")


def drive_against(handler, spec, out, deadlines, guard=lambda: None):
    async def go():
        runner, url = await serve(handler)
        result = await sweep.drive(spec, {i: f"q{i}" for i in range(5)}, "root.a1", out, deadlines,
                                   lambda text, cap, seed: sweep.client_body(CFG, text, cap, seed), url, guard)
        await runner.cleanup()
        return result
    return asyncio.run(go())


def test_drive_keeps_target_outstanding_and_replaces(tmp_path):
    state = {"now": 0, "max": 0, "order": [], "bodies": []}

    async def handler(request):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        state["order"].append(request.headers["X-Request-Id"])
        state["bodies"].append(await request.json())
        await asyncio.sleep(0.02)
        resp = web.StreamResponse()
        await resp.prepare(request)
        rid = "chatcmpl-" + request.headers["X-Request-Id"]
        await resp.write(sse({"id": rid, "choices": [{"delta": {"content": "a"}, "finish_reason": "stop"}]},
                             {"id": rid, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode())
        state["now"] -= 1
        return resp

    spec = sweep.RunSpec("t", 2, 16, 11, 1, 1, (4, 1, 3, 0, 2), (9, 9, 9, 9, 9), 5, False, "test")
    records, fired, drain = drive_against(handler, spec, tmp_path, sweep.Deadlines(30.0, 30.0))
    assert fired is None and drain is not None
    assert state["max"] == 2
    assert state["order"] == ["root.a1.i4", "root.a1.i1", "root.a1.i3", "root.a1.i0", "root.a1.i2"]
    assert state["bodies"][0] == {"model": "mlx-community/Qwen3-4B-4bit", "messages": [{"role": "user", "content": "q4"}],
                                  "max_tokens": 16, "seed": 9, "stream": True, "stream_options": {"include_usage": True},
                                  "temperature": 0.6, "top_p": 0.95}
    assert sorted(r.order for r in records) == [0, 1, 2, 3, 4]
    assert [r.phase for r in sorted(records, key=lambda r: r.order)] == ["startup", "startup", "steady", "steady", "steady"]
    parsed, problem = history.read_client(tmp_path / "client.jsonl")
    assert problem is None and len(parsed) == 5


async def stalled(request):
    resp = web.StreamResponse()
    await resp.prepare(request)
    await asyncio.sleep(5)
    return resp


def test_drive_deadline_leaves_the_run_incomplete(tmp_path):
    spec = sweep.RunSpec("t", 2, 16, 11, 1, 1, (0, 1, 2), (1, 1, 1), 3, False, "test")
    records, fired, _ = drive_against(stalled, spec, tmp_path, sweep.Deadlines(60.0, 2.0))
    assert fired == "no output for 2 s"
    assert sorted(r.status for r in records) == ["cancelled", "cancelled"]


def test_drive_stops_when_the_budget_guard_trips(tmp_path):
    calls = []

    def guard():
        calls.append(1)
        return "budget: disk: 2.00 GiB free, at the 2.00 GiB reserve" if len(calls) >= 2 else None

    spec = sweep.RunSpec("t", 2, 16, 11, 1, 1, (0, 1, 2), (1, 1, 1), 3, False, "test")
    records, fired, _ = drive_against(stalled, spec, tmp_path, sweep.Deadlines(60.0, 60.0), guard)
    assert fired == "budget: disk: 2.00 GiB free, at the 2.00 GiB reserve"
    assert len(calls) == 2
    assert [r.status for r in records] == ["cancelled", "cancelled"]
    assert dataclasses.asdict(records[0])["error"] == "deadline"

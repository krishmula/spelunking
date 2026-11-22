
import json

import pytest

from e02 import sweep
from pilot_fixture import (CFG, DEAD_PID, behavior_root, cost_root, install, make_attempt, provenance, sealed_root,
                           start_attempt)


@pytest.fixture(autouse=True)
def small_pilots(monkeypatch, tmp_path):
    install(monkeypatch)
    monkeypatch.setattr(sweep, "RUNS", tmp_path)


def test_behavior_verdict_is_recomputed_from_raw_evidence(tmp_path):
    root = behavior_root(tmp_path)
    rec = sweep.read_attempt(root / "attempts" / "behavior.a1")
    assert (rec["outcome"], rec["validation"], rec["observed_preemptions"]) == (
        "complete", {"status": "complete", "reasons": []}, 1)
    assert sorted(rec["evidence"]) == ["client.jsonl", "events.jsonl", "snapshots.jsonl"]
    v = sweep.behavior_verdict(root)
    assert (v["verdict"], v["reasons"], v["preemptions"], v["requests"], v["cap"]) == ("pass", [], 1, 4, 64)


def test_pilot_gate_passes_current_evidence_and_rejects_a_deleted_journal(tmp_path):
    root = behavior_root(tmp_path)
    cost_root(tmp_path, "cost-low")
    cost_root(tmp_path, "cost-pressure")
    out = sweep.pilot_evidence(provenance())
    assert {k: v["verdict"] for k, v in out.items()} == {"behavior": "pass", "cost-low": "pass", "cost-pressure": "pass"}
    (root / "attempts" / "behavior.a1" / "events.jsonl").unlink()
    with pytest.raises(sweep.GateError, match=r"behavior pilot .* is fail: .*evidence changed after completion: "
                                              r"events.jsonl"):
        sweep.pilot_evidence(provenance())


def test_saved_verdict_without_sealed_evidence_is_rejected(tmp_path):
    b = tmp_path / "20261020-000000-pilot-behavior"
    b.mkdir()
    (b / "verdict.json").write_text(json.dumps({"verdict": "pass", "reasons": []}))
    with pytest.raises(sweep.GateError, match="has no plan.json or manifest.json"):
        sweep.pilot_evidence(provenance())
    (b / "plan.json").write_text(json.dumps({"kind": "pilot", "runs": []}))
    with pytest.raises(sweep.GateError, match="plan.json is not sealed: it predates sealed plans"):
        sweep.pilot_evidence(provenance())


def test_pilot_recorded_under_other_source_or_settings_is_rejected(tmp_path):
    behavior_root(tmp_path, prov=provenance("before-edit"))
    with pytest.raises(sweep.GateError, match=r"recorded under different source or settings than the current code "
                                              r"\(changed: \['tag'\]\)"):
        sweep.pilot_evidence(provenance("after-edit"))


def test_inherited_capture_cadence_cannot_pass_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("E02_DIGEST_EVERY", "0")
    _, ctx = sealed_root(tmp_path, "behavior")
    rec = make_attempt(ctx, "behavior")
    assert (rec["outcome"], rec["reason"]) == ("invalid", "journal digest_every 0, sealed plan 1")


def test_server_environment_comes_from_the_sealed_configuration(tmp_path, monkeypatch):
    for key in ("E02_OUT", "E02_RUN_ID", "E02_SNAPSHOT_EVERY", "E02_NUM_BLOCKS", "VLLM_LAB", "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_WORKER_MULTIPROC_METHOD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("E02_DIGEST_EVERY", "0")
    monkeypatch.setenv("VLLM_ENABLE_SOMETHING", "1")
    env, removed = sweep.server_env(CFG, tmp_path, "root/run.a1")
    assert removed == ["E02_DIGEST_EVERY", "VLLM_ENABLE_SOMETHING"]
    assert {k: env[k] for k in env if k.startswith(("E02_", "VLLM_"))} == {
        "E02_DIGEST_EVERY": "1", "E02_SNAPSHOT_EVERY": "32", "E02_NUM_BLOCKS": "24", "E02_RUN_ID": "root/run.a1",
        "E02_OUT": str(tmp_path), "VLLM_SERVER_DEV_MODE": "1", "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}


def test_cost_verdict_passes_equal_sides(tmp_path):
    v = sweep.cost_verdict(cost_root(tmp_path, "cost-pressure"), "cost-pressure")
    assert (v["verdict"], v["reasons"]) == ("pass", [])
    assert v["median_delta"] == {"throughput": pytest.approx(0, abs=1e-9), "tpot": pytest.approx(0, abs=1e-9)}
    assert [r["observed_preemptions"] for r in v["runs"] if r["logging"]] == [1, 1, 1, 1, 1]


def test_cost_verdict_fails_consistent_slowdown(tmp_path):
    v = sweep.cost_verdict(cost_root(tmp_path, "cost-pressure", on_factor=0.9), "cost-pressure")
    assert v["verdict"] == "fail"
    assert v["median_delta"]["tpot"] == pytest.approx(1 / 0.9 - 1)


def test_cost_verdict_is_inconclusive_under_noise(tmp_path):
    v = sweep.cost_verdict(cost_root(tmp_path, "cost-low", noisy=True), "cost-low")
    assert v["verdict"] == "inconclusive"


def test_cost_verdict_with_undefined_tpot_is_inconclusive(tmp_path):
    root, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    lines = [json.loads(line) for line in (path / "client.jsonl").read_text().splitlines()]
    for line in lines:
        line["completion_tokens"] = 1
    (path / "client.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert sweep.finish_attempt(path, rec, "complete", None, ctx)["outcome"] == "complete"
    verdict = sweep.cost_verdict(root, "cost-low")
    assert verdict["verdict"] == "inconclusive"
    assert "p1-off has no positive throughput or TPOT for a cost comparison" in verdict["reasons"]


def test_cost_verdict_requires_pressure_to_preempt_and_counters_to_agree(tmp_path):
    v = sweep.cost_verdict(cost_root(tmp_path, "cost-pressure", preempt=0), "cost-pressure")
    assert v["verdict"] == "inconclusive"
    assert "a pressure run did not preempt, so it does not test logging under pressure" in v["reasons"]
    assert "p1-on: engine counter 0, history 1" in v["reasons"]


def test_failed_cost_pilot_blocks_main(tmp_path):
    behavior_root(tmp_path)
    cost_root(tmp_path, "cost-low")
    cost_root(tmp_path, "cost-pressure", on_factor=0.85)
    with pytest.raises(sweep.GateError, match="cost-pressure pilot .* is fail"):
        sweep.pilot_evidence(provenance())


@pytest.mark.parametrize("run_id, outcome", [("p1-on", "invalid"), ("p1-off", "incomplete")])
def test_crash_during_validation_never_leaves_a_complete_attempt(tmp_path, monkeypatch, run_id, outcome):
    root, ctx = sealed_root(tmp_path, "cost-low")
    if run_id == "p1-off":
        make_attempt(ctx, "p1-on")
    path, rec = start_attempt(ctx, run_id)
    lines = (path / "client.jsonl").read_text().splitlines()
    (path / "client.jsonl").write_text("\n".join(lines[:3]) + "\n")
    validate = sweep.validate_evidence

    def crash(*a):
        raise RuntimeError("runner killed during validation")

    monkeypatch.setattr(sweep, "validate_evidence", crash)
    with pytest.raises(RuntimeError):
        sweep.finish_attempt(path, rec, "complete", None, ctx)
    monkeypatch.setattr(sweep, "validate_evidence", validate)
    assert sweep.read_attempt(path)["outcome"] == "validating"
    assert [r["verified"] for r in sweep.sweep_status(root)["runs"][:2]] == (
        ["validating", "planned"] if run_id == "p1-on" else ["complete", "validating"])
    sweep._write_atomic(path / "attempt.json", {**sweep.read_attempt(path), "pid": DEAD_PID, "runner_start": None})
    sweep.reconcile(ctx, run_id)
    assert sweep.read_attempt(path)["outcome"] == outcome
    spec, number, replaces, _ = sweep.next_action(ctx, {})
    assert (spec.run_id, number, replaces) == (run_id, 2, f"{run_id}.a1")


def test_interrupted_validation_of_good_evidence_finishes_complete(tmp_path):
    root, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-on")
    sweep._write_atomic(path / "attempt.json", {**rec, "outcome": "validating", "execution_outcome": "complete",
                                                "execution_reason": None, "pid": DEAD_PID, "runner_start": None})
    sweep.reconcile(ctx, "p1-on")
    assert sweep.read_attempt(path)["outcome"] == "complete"
    assert sweep.next_action(ctx, {})[:3][0].run_id == "p1-off"


def test_completed_attempt_is_revalidated_on_resume_and_status(tmp_path):
    root, ctx = sealed_root(tmp_path, "cost-low")
    make_attempt(ctx, "p1-on")
    with open(root / "attempts" / "p1-on.a1" / "client.jsonl", "a") as f:
        f.write("\n")
    spec, number, replaces, reason = sweep.next_action(ctx, {})
    assert (spec.run_id, number, replaces) == ("p1-on", 2, "p1-on.a1")
    assert reason == ("previous attempt recorded complete but fails revalidation: evidence changed after "
                      "completion: client.jsonl")
    row = sweep.sweep_status(root)["runs"][0]
    assert (row["recorded"], row["verified"]) == ("complete", "invalid")


def test_saved_warmup_verdict_without_samples_cannot_complete(tmp_path):
    _, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    rec["warmup"] = {"stable": True, "reason": "claimed", "samples": []}
    final = sweep.finish_attempt(path, rec, "complete", None, ctx)
    assert (final["outcome"], final["reason"]) == (
        "invalid", "warmup samples do not establish the sealed stability rule: 0 of at least 3 rounds")


def test_validation_time_counts_toward_attempt_budget(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(sweep.time, "time", lambda: now[0])
    _, ctx = sealed_root(tmp_path, "cost-low")
    path, rec = start_attempt(ctx, "p1-off")
    validate = sweep.validate_evidence

    def consume(*args):
        result = validate(*args)
        now[0] = 125.0
        return result

    monkeypatch.setattr(sweep, "validate_evidence", consume)
    final = sweep.finish_attempt(path, rec, "complete", None, ctx)
    assert (final["outcome"], final["wall_ended"]) == ("complete", 125.0)


def test_sealed_plan_rejects_edits(tmp_path):
    root, ctx = sealed_root(tmp_path, "cost-low")
    text = (root / "plan.json").read_text()
    (root / "plan.json").write_text(text.replace('"run_wall_s": 60.0', '"run_wall_s": 6000.0'))
    with pytest.raises(sweep.GateError, match="does not match its sealed digest"):
        sweep.load_sealed(root)


def test_resume_rejects_changed_settings_and_foreign_attempts(tmp_path):
    root, ctx = sealed_root(tmp_path, "cost-low")
    changed = {**provenance(), "config": {**sweep.to_json(CFG), "temperature": 0.7}}
    with pytest.raises(sweep.GateError, match=r"sealed under different inputs; changed since: \['config'\]"):
        sweep.check_resume(ctx.sealed, {"provenance": changed, "provenance_digest": sweep.digest_of(changed)})
    path, rec = start_attempt(ctx, "p1-on")
    sweep._write_atomic(path / "attempt.json", {**rec, "plan_digest": "0" * 64})
    current = provenance()
    with pytest.raises(sweep.GateError, match="p1-on.a1 was made under a different sealed plan"):
        sweep.check_resume(ctx.sealed, {"provenance": current, "provenance_digest": sweep.digest_of(current)})


def test_main_manifest_must_hold_the_fixed_plan(tmp_path):
    root = tmp_path / "20261020-000000-main"
    root.mkdir()
    runs = sweep.main_plan()
    runs[7] = sweep.RunSpec(**{**sweep.to_json(runs[7]), "cap": 256, "item_order": runs[7].item_order,
                               "item_seeds": runs[7].item_seeds})
    sweep.seal_plan(root, kind="main", workload="main", runs=runs, cfg=CFG, deadlines=sweep.Deadlines(600.0, 180.0),
                    ceilings=sweep.Ceilings(3600.0, sweep.GIB, 0, 0), prov=provenance(), extra={})
    with pytest.raises(sweep.GateError, match="runs differ from the fixed plan"):
        sweep.load_manifest(root)

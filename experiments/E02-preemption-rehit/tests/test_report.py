
import csv
import dataclasses
import json
import re
import shutil
from typing import NamedTuple

import pytest

from e02 import history
from e02.observer import ObservedScheduler
from e02.report import render
from engine_fixture import Engine, make_scheduler, prompt
from scenarios import SELF, TWO_VICTIMS, ZERO_C, engine_kwargs

QUIET = dict(num_blocks=24, max_num_batched_tokens=64, arrivals={0: [("a", prompt(7, 8), 4), ("b", prompt(8, 8), 4)]})


class Run(NamedTuple):
    script: dict | None
    target: int
    cap: int
    rep_seed: int
    stop_after: int | None


RUNS = {
    "r1": Run(TWO_VICTIMS, 8, 64, 1, None),
    "r2": Run(SELF, 8, 64, 2, None),
    "r3": Run(SELF, 8, 64, 3, 50),
    "r4": Run(ZERO_C, 4, 32, 1, None),
    "r5": Run(None, 4, 32, 2, None),
    "r6": Run(QUIET, 2, 16, 1, None),
}


def client_record(cid, item, n_out, max_tokens, ok):
    t_submit = 1_000_000 * item
    t_first = t_submit + 5_000_000 * (item + 1)
    return history.ClientRecord(
        client_id=cid, item=item, order=item, phase="measure", t_submit_ns=t_submit, t_first_ns=t_first,
        t_last_ns=t_first + 2_000_000 * n_out if ok else None, t_end_ns=t_first + 2_000_000 * n_out + 1,
        status="ok" if ok else "timeout", http_status=200, error=None,
        finish_reason=("length" if n_out == max_tokens else "stop") if ok else None, prompt_tokens=None,
        completion_tokens=n_out if ok else None, chunks=n_out, content_chunks=n_out, reasoning_chunks=0,
        max_gap_ns=2_000_000, response_id=f"chatcmpl-{cid}" if ok else None)


def observe(d, run_id, script, stop_after):
    """Run the real scheduler with chatcmpl-style ids and write a joining client.jsonl."""
    s = make_scheduler(ObservedScheduler, out_dir=d, **engine_kwargs(script))
    assert s.reset_prefix_cache()
    e = Engine(s)
    arrivals, wanted = {}, []
    for step, reqs in script["arrivals"].items():
        for name, p, m in reqs:
            cid = f"{run_id}.{name}"
            arrivals.setdefault(step, []).append((f"chatcmpl-{cid}-abcdefgh", p, m))
            wanted.append((cid, m))
    if stop_after is None:
        e.run(arrivals=arrivals)
    else:
        for i in range(stop_after):
            for rid, p, m in arrivals.get(i, ()):
                e.add(rid, p, m)
            e.step()
    s.reset_prefix_cache()
    lines = []
    for item, (cid, m) in enumerate(wanted):
        n = len(e.delivered.get(f"chatcmpl-{cid}-abcdefgh", []))
        lines.append(json.dumps(dataclasses.asdict(client_record(cid, item, n, m, stop_after is None))))
    (d / "client.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    specs = []
    for i, (run_id, (script, target, cap, seed, stop_after)) in enumerate(RUNS.items()):
        logging = script is not None
        specs.append(dict(run_id=run_id, target=target, cap=cap, rep_seed=seed, round=1, order_in_round=i,
                          item_order=[], item_seeds=[], requests=4, logging=logging, workload="test"))
        d = root / "attempts" / f"{run_id}.a1"
        if logging:
            observe(d, run_id, script, stop_after)
        else:
            d.mkdir(parents=True)
            (d / "client.jsonl").write_text("not read\n")
        outcome = "incomplete" if stop_after else "complete"
        n = sum(len(v) for v in script["arrivals"].values()) if script else 4
        (d / "attempt.json").write_text(json.dumps(dict(
            run_id=run_id, attempt=1, replaces=None, replacement_reason=None, outcome=outcome,
            reason="no-progress deadline" if stop_after else None, requests=n, logging=logging)))
    events = root / "attempts" / "r3.a1" / "events.jsonl"
    events.write_bytes(b"".join(line + b"\n" for line in events.read_bytes().split(b"\n")[:-2]))
    (root / "manifest.json").write_text(json.dumps(dict(kind="main", root_id="test-root", corpus_digest="c0ffee",
                                                        runs=specs)))
    (root / "annotations.tsv").write_text("decision_key\trequest\twhy\nr1.a1:1\tr1.q0\tq0 frees the same block\n")
    out = render(root)
    return root, out


def rows(out):
    with open(out / "decisions.csv", newline="") as f:
        return list(csv.DictReader(f))


def by_attempt(out, attempt):
    return [r for r in rows(out) if r["attempt"] == attempt]


def decisions(out, attempt):
    got = {}
    for r in by_attempt(out, attempt):
        got.setdefault(int(r["decision"]), []).append(r)
    return [got[d] for d in sorted(got)]


def test_one_row_per_candidate_per_decision(root):
    _, out = root
    assert [len(d) for d in decisions(out, "r1.a1")] == [4, 3, 2]
    assert [len(d) for d in decisions(out, "r2.a1")] == [4]
    assert [len(d) for d in decisions(out, "r3.a1")] == [4]
    assert {r["attempt"] for r in rows(out)} == {"r1.a1", "r2.a1", "r3.a1", "r4.a1"}
    keys = [(r["attempt"], r["decision"], r["request"]) for r in rows(out)]
    assert len(keys) == len(set(keys))
    assert {r["run_status"] for r in by_attempt(out, "r3.a1")} == {"incomplete"}


def test_chosen_and_least_tied(root):
    _, out = root
    (self_dec,) = decisions(out, "r2.a1")
    assert [(r["client_id"], r["chosen"], r["least_tied"]) for r in self_dec] == [
        ("r2.a", "no", "no"), ("r2.b", "no", "no"), ("r2.c", "no", "no"), ("r2.d", "yes", "yes")]
    victim = self_dec[3]
    assert (victim["completed"], victim["raw"], victim["C"], victim["H"], victim["recovery_fraction"]) == (
        "79", "80", "79", "32", f"{32 / 79:.4f}")
    assert (victim["reclaimed_blocks"], victim["original_freed"], victim["missing"]) == ("5", "32", "47")
    assert all(r["reclaimed_blocks"] == "" and r["episode"] == "" for r in self_dec[:3])
    d1, d2, _ = decisions(out, "r1.a1")
    assert [r["client_id"] for r in d1] == ["r1.q1", "r1.q2", "r1.q0", "r1.q3"]
    assert [r["chosen"] for r in d1] == ["no", "no", "no", "yes"]
    assert [r["chosen"] for r in d2] == ["no", "no", "yes"]
    assert [r["sufficiency"] for r in d1] == ["no", "unknown", "no", "no"]
    assert [r["sufficiency"] for r in d2] == ["yes", "unknown", "yes"]


def test_two_victims_least_tied(root):
    _, out = root
    got = [[(r["client_id"], r["chosen"], r["least_tied"], r["rank"]) for r in d] for d in decisions(out, "r1.a1")]
    assert got == [
        [("r1.q1", "no", "no", "2"), ("r1.q2", "no", "no", "4"), ("r1.q0", "no", "no", "3"),
         ("r1.q3", "yes", "yes", "1")],
        [("r1.q1", "no", "yes", "1"), ("r1.q2", "no", "no", "3"), ("r1.q0", "yes", "no", "2")],
        [("r1.q1", "no", "yes", "1"), ("r1.q2", "yes", "no", "2")],
    ]


def test_zero_c_and_unresolved_fractions(root):
    _, out = root
    victims = [r for r in by_attempt(out, "r4.a1") if r["chosen"] == "yes"]
    q3 = [(r["C"], r["H"], r["recovery_fraction"], r["episode_status"]) for r in victims if r["client_id"] == "r4.q3"]
    assert q3 == [("0", "0", "undefined", "preempted_again"), ("35", "32", "0.9143", "finished")]
    (r3,) = [r for r in by_attempt(out, "r3.a1") if r["chosen"] == "yes"]
    assert (r3["client_id"], r3["C"], r3["H"], r3["recovery_fraction"], r3["episode_status"]) == (
        "r3.d", "79", "", "unresolved", "unresolved")


def test_render_is_deterministic(root):
    path, out = root
    first = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    shutil.rmtree(out)
    render(path)
    second = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    assert sorted(map(str, first)) == ["decisions.csv", "distributions.csv", "index.html", "runs/r1.a1.html",
                                       "runs/r2.a1.html", "runs/r3.a1.html", "runs/r4.a1.html", "runs/r6.a1.html",
                                       "summary.csv"]
    assert first == second


def test_every_link_anchor_exists(root):
    _, out = root
    for r in rows(out):
        for link in (r["snapshot_link"], r["request_link"]):
            page, anchor = link.split("#")
            assert f'id="{anchor}"' in (out / page).read_text(), link
    snaps = re.findall(r'<details id="snap-(\d+)"', (out / "runs/r1.a1.html").read_text())
    assert snaps == sorted(snaps, key=int) and len(snaps) == 3


def test_status_table_and_aggregates(root):
    _, out = root
    index = (out / "index.html").read_text()
    status = index.split("<h2>Run status</h2>")[1].split("<h2>Unobserved attempts</h2>")[0]
    unobserved = index.split("<h2>Unobserved attempts</h2>")[1].split("<h2>Configurations</h2>")[0]
    for name in ("r1.a1", "r2.a1", "r3.a1", "r4.a1", "r6.a1"):
        assert f">{name}</a>" in status
    assert "r5.a1" not in status and "r5.a1" in unobserved
    assert ">incomplete</span>" in status and "no-progress deadline" in status
    assert all(name not in unobserved for name in ("r1.a1", "r3.a1"))
    with open(out / "summary.csv", newline="") as f:
        summary = {(r["target"], r["cap"], r["metric"]): r for r in csv.DictReader(f)}
    pre = summary[("8", "64", "preemptions")]
    assert (pre["values"], pre["n"], pre["mean"], pre["median"], pre["min"], pre["max"]) == (
        "3;1", "2", "2", "2", "1", "3")
    assert summary[("8", "64", "ttft_p99_s")]["mean"] == "not aggregated"
    assert summary[("8", "64", "pooled_ttft_s")]["n"] == "8"
    assert ("4", "32", "preemptions") in summary and summary[("4", "32", "preemptions")]["n"] == "1"


def test_zero_preemption_config_has_undefined_recovery(root):
    _, out = root
    with open(out / "summary.csv", newline="") as f:
        quiet = {r["metric"]: r for r in csv.DictReader(f) if (r["target"], r["cap"]) == ("2", "16")}
    assert (quiet["preemptions"]["values"], quiet["episodes"]["values"]) == ("0", "0")
    for m in ("recovery_fraction_median", "recovered_original_freed_tokens", "missing_tokens"):
        assert (quiet[m]["values"], quiet[m]["mean"], quiet[m]["n"]) == ("undefined", "undefined", "0")


def test_examples_limits_and_input_digests(root):
    path, out = root
    index = (out / "index.html").read_text()
    examples = index.split("<h2>Selected examples</h2>")[1].split("<h2>Limits</h2>")[0]
    assert "q0 frees the same block" in examples and "runs/r1.a1.html#req-" in examples
    assert "GPU re-prefill time is unavailable" in index
    listed = re.findall(r"<tr><td>([^<]+)</td><td><code>[0-9a-f]{64}</code>", index.split("<footer>")[1])
    assert listed == sorted(listed)
    assert "attempts/r5.a1/attempt.json" in listed and "attempts/r5.a1/client.jsonl" not in listed
    assert {"manifest.json", "annotations.tsv", "attempts/r1.a1/events.jsonl"} <= set(listed)


def test_per_run_distributions_are_reported_without_pooling(root):
    _, out = root
    with open(out / "distributions.csv", newline="") as f:
        dist = [r for r in csv.DictReader(f) if r["attempt"] == "r2.a1"]
    values = {}
    for r in dist:
        values.setdefault(r["distribution"], []).append(r["value"])
    assert values["candidate_completed"] == ["81", "80", "80", "79"]
    assert values["recovery_fraction"] == [f"{32 / 79:.6f}"]
    assert values["episode_reprefill_tokens"] == ["47"]
    steps = {r["key"]: int(r["value"]) for r in dist if r["distribution"] == "batch_steps"}
    assert sum(steps.values()) == int(by_metric(out, "8", "64", "steps").split(";")[1])
    page = (out / "runs/r2.a1.html").read_text()
    assert "host schedule-time proxy" in page and "device time unavailable" in page
    assert "they are not independent samples" in page
    index = (out / "index.html").read_text()
    assert index.count("Per-run distributions. Each row summarizes one run") == 3


def by_metric(out, target, cap, metric):
    with open(out / "summary.csv", newline="") as f:
        return next(r["values"] for r in csv.DictReader(f) if (r["target"], r["cap"], r["metric"]) == (target, cap, metric))


import pytest

from e02 import history
from e02.history import BatchBin, SchedStep
from e02.observer import ObservedScheduler
from engine_fixture import Engine, make_scheduler
from scenarios import SELF, engine_kwargs

S = 1_000_000_000


def step(out, t, batch, kind="decode", t_done=None):
    prefill = {"decode": 0, "mixed": 1, "prefill": batch}[kind]
    return SchedStep(out=out, call=out, t=t, batch=batch, tokens=batch, prefill_entries=prefill, kind=kind,
                     t_done=t_done)


def replay_of(steps):
    return history.Replay(run_id="r", config={}, num_blocks=0, null_block=0, block_size=16, requests={},
                          decisions=[], episodes={}, steps=steps, works_unacknowledged=0, hint_divergences=0,
                          snapshots_checked=0, digests_checked=0, alloc_failures=0, end=None, blocks={},
                          snapshots={})


def test_batch_one_for_ten_seconds_then_eight_for_one():
    steps = [step(1, 0, 1, t_done=5 * S), step(2, 10 * S, 8, t_done=11 * S)]
    assert history.batch_distribution(steps) == (
        (BatchBin("decode", 1, 1, 10.0), BatchBin("decode", 8, 1, 1.0)), "completed")
    m = history.run_metrics(replay_of(steps), [])
    assert m["batch_size_time_weighted"] == 18 / 11
    assert m["batch_size_decode_time_weighted"] == 18 / 11
    assert m["batch_interval_max_s"] == 10.0


def test_unacknowledged_final_step_has_no_interval():
    steps = [step(1, 0, 1, t_done=5 * S), step(2, 10 * S, 8)]
    assert history.batch_distribution(steps) == (
        (BatchBin("decode", 1, 1, 10.0), BatchBin("decode", 8, 1, 0.0)), "unacknowledged")
    assert history.run_metrics(replay_of(steps), [])["batch_size_time_weighted"] == 1.0


def test_decode_only_mean_leaves_out_mixed_steps():
    steps = [step(1, 0, 1), step(2, 10 * S, 4, "mixed"), step(3, 12 * S, 8, t_done=13 * S)]
    m = history.run_metrics(replay_of(steps), [])
    assert (m["batch_size_time_weighted"], m["batch_size_decode_time_weighted"]) == (2.0, 18 / 11)
    assert (m["steps_decode_only"], m["steps_mixed"], m["steps_prefill_only"]) == (2, 1, 0)


def test_no_steps():
    assert history.batch_distribution([]) == ((), "none")
    m = history.run_metrics(replay_of([]), [])
    assert (m["batch_size_time_weighted"], m["batch_interval_max_s"]) == (None, None)


def observe_self(tmp_path, stop_after=None):
    d = tmp_path / "self"
    s = make_scheduler(ObservedScheduler, out_dir=d, **engine_kwargs(SELF))
    assert s.reset_prefix_cache()
    e = Engine(s)
    if stop_after is None:
        e.run(arrivals=SELF["arrivals"])
    else:
        for i in range(stop_after):
            for rid, prompt, max_tokens in SELF["arrivals"].get(i, ()):
                e.add(rid, prompt, max_tokens)
            e.step()
    s.reset_prefix_cache()
    _, rep = history.replay_attempt(d)
    assert not isinstance(rep, history.Invalid), rep
    return rep


def test_step_kinds_on_the_real_scheduler(tmp_path):
    rep = observe_self(tmp_path)
    assert [(s.batch, s.tokens, s.kind) for s in rep.steps[:4]] == [
        (2, 64, "prefill"), (4, 64, "mixed"), (4, 36, "mixed"), (4, 4, "decode")]
    assert all(s.t_done is not None for s in rep.steps)
    assert history.batch_distribution(rep.steps)[1] == "completed"


def test_step_in_flight_at_seal_is_unacknowledged(tmp_path):
    rep = observe_self(tmp_path, stop_after=50)
    assert [s.t_done is None for s in rep.steps[-2:]] == [False, True]
    assert history.batch_distribution(rep.steps)[1] == "unacknowledged"


def test_candidate_and_recovery_distributions(tmp_path):
    dist = history.distributions(observe_self(tmp_path))
    assert dist.candidate == {"completed": (81, 80, 80, 79), "total_blocks": (6, 6, 6, 5),
                              "shared_blocks": (0, 0, 0, 0), "reclaimable_blocks": (6, 6, 6, 5),
                              "shared_prefix_tokens": (0, 0, 0, 0)}
    assert dist.recovery_fraction == pytest.approx((32 / 79,))
    assert dist.episode_reprefill_tokens == (47,)
    assert dist.batch_basis == history.BATCH_TIME_BASIS

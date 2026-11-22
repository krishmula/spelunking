
import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from e02.observer import ObservedScheduler
from engine_fixture import Engine, make_scheduler, pool_state
from scenarios import ALL, engine_kwargs


def drive(cls, script, tmp_path):
    s = make_scheduler(cls, out_dir=tmp_path / cls.__name__, **engine_kwargs(script))
    assert s.reset_prefix_cache()
    e = Engine(s)
    e.run(arrivals=script["arrivals"])
    trace = [(dict(o.num_scheduled_tokens), sorted(r.req_id for r in o.scheduled_new_reqs),
              sorted(o.scheduled_cached_reqs.resumed_req_ids), sorted(o.preempted_req_ids or ()),
              sorted(o.finished_req_ids)) for o in e.outputs]
    return trace, e.delivered, pool_state(s), sum(len(p) for p in e.preempted)


@pytest.mark.parametrize("name", sorted(ALL))
def test_observed_scheduler_matches_stock(name, tmp_path):
    stock = drive(AsyncScheduler, ALL[name], tmp_path)
    observed = drive(ObservedScheduler, ALL[name], tmp_path)
    assert observed[0] == stock[0]
    assert observed[1] == stock[1]
    assert observed[2] == stock[2]
    assert observed[3] == stock[3]


def test_twin_scripts_exercise_preemption(tmp_path):
    counts = {name: drive(AsyncScheduler, script, tmp_path / name)[3] for name, script in ALL.items()}
    assert counts == {"self": 1, "two_victims": 3, "repeated": 4, "equivalent": 2, "reused_id": 2,
                      "zero_c": 3, "before_regain": 3}


class PerturbingScheduler(ObservedScheduler):
    """Releases blocks in forward instead of reverse order, as a buggy hook might."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pool = self.kv_cache_manager.block_pool
        stock = pool.free_blocks
        pool.free_blocks = lambda blocks: stock(list(blocks)[::-1])


def test_twin_comparison_catches_a_perturbing_observer(tmp_path):
    stock = drive(AsyncScheduler, ALL["self"], tmp_path)
    perturbed = drive(PerturbingScheduler, ALL["self"], tmp_path)
    assert perturbed[2] != stock[2]

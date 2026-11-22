
from collections import Counter

import pytest

from e02 import events as ev
from e02 import history
from e02.observer import ObservedScheduler
from engine_fixture import Engine, make_scheduler
from scenarios import ALL, engine_kwargs


def observe(name, tmp_path, depth=1, stop_after=None, snapshot_every=0):
    script = ALL[name]
    d = tmp_path / name
    s = make_scheduler(ObservedScheduler, out_dir=d, snapshot_every=snapshot_every, **engine_kwargs(script))
    assert s.reset_prefix_cache()
    e = Engine(s, depth=depth)
    if stop_after is None:
        e.run(arrivals=script["arrivals"])
    else:
        for i in range(stop_after):
            for rid, prompt, max_tokens in script["arrivals"].get(i, ()):
                e.add(rid, prompt, max_tokens)
            e.step()
    s.reset_prefix_cache()
    journal, rep = history.replay_attempt(d)
    assert journal.status == "sealed"
    assert not isinstance(rep, history.Invalid), rep
    by_name = {q.engine_id: q for q in rep.requests.values()}
    return rep, by_name, d


def cand(dec, rep, name):
    r = next(q.r for q in rep.requests.values() if q.engine_id == name)
    return next(c for c in dec.candidates if c.r == r)


def test_self_preemption_ledger(tmp_path):
    rep, q, d = observe("self", tmp_path)
    assert len(rep.decisions) == 1
    dec = rep.decisions[0]
    assert (dec.needed, dec.free, dec.stock_rule_holds, dec.reclaimed_blocks) == (1, 0, True, 5)
    assert [(c.completed, c.raw, c.pending) for c in dec.candidates] == [(81, 82, 1), (80, 81, 1), (80, 81, 1),
                                                                          (79, 80, 1)]
    victim = cand(dec, rep, "d")
    assert (victim.chosen, victim.rank, victim.least_tied, victim.sufficiency) == (True, 1, True, "unknown")
    assert [c.least_tied for c in dec.candidates] == [False, False, False, True]
    ep = rep.episodes[dec.episode]
    assert (ep.C, ep.raw, ep.pending_tokens) == (79, 80, 1)
    r = ep.recovery
    assert (r.H, r.original_freed, r.original_shared, r.equivalent, r.unknown, r.missing) == (32, 32, 0, 0, 0, 47)
    assert (r.intervening_alloc_blocks, r.freed_blocks_reallocated) == (3, 3)
    assert ep.fraction == pytest.approx(32 / 79)
    assert (ep.reprefill_tokens, q["d"].reprefill_tokens, q["d"].discarded_tokens) == (47, 47, 1)
    assert [q[n].greatest for n in "abcd"] == [99, 99, 99, 99]
    assert ep.status == "finished"


def test_touch_keeps_generation_and_allocation_advances_it(tmp_path):
    rep, q, d = observe("self", tmp_path)
    ep = next(iter(rep.episodes.values()))
    assert [h.generation for h in ep.recovery.hit_blocks] == [1, 1]
    assert [h.block for h in ep.recovery.hit_blocks] == [o.block for o in ep.originals[:2]]
    journal = ev.read_journal(d / "events.jsonl")
    pops = Counter(b for e in journal.events if isinstance(e.payload, ev.PoolAlloc) for b in e.payload.blocks)
    touched_free = [b for e in journal.events if isinstance(e.payload, ev.PoolTouch)
                    for b, ref in zip(e.payload.blocks, e.payload.refs_before) if ref == 0]
    assert {h.block for h in ep.recovery.hit_blocks} <= set(touched_free)
    assert {b: s.generation for b, s in rep.blocks.items() if b} == {b: pops[b] for b in rep.blocks if b}


def test_two_victims_for_one_failed_allocation(tmp_path):
    rep, q, _ = observe("two_victims", tmp_path)
    d1, d2 = rep.decisions[:2]
    assert (d1.group, d1.call) == (d2.group, d2.call)
    assert [d1.needed, d2.needed] == [2, 2]
    assert [d1.free, d2.free] == [0, 1]
    assert d1.snapshot != d2.snapshot
    assert [rep.requests[d.victim].engine_id for d in (d1, d2)] == ["q3", "q0"]
    assert [rep.requests[c.r].engine_id for c in d1.candidates] == ["q1", "q2", "q0", "q3"]
    assert [rep.requests[c.r].engine_id for c in d2.candidates] == ["q1", "q2", "q0"]
    assert [(c.sufficiency, c.reclaimable_blocks) for c in d1.candidates] == [
        ("no", 1), ("unknown", 8), ("no", 1), ("no", 1)]
    assert [c.sufficiency for c in d2.candidates] == ["yes", "unknown", "yes"]
    assert [c.sufficiency_reason for c in d2.candidates][1] == "requester"
    assert [d1.reclaimed_blocks, d2.reclaimed_blocks] == [1, 1]


def test_repeated_preemption_preserves_greatest_position(tmp_path):
    rep, q, _ = observe("repeated", tmp_path)
    q4 = q["q4"]
    first, second = (rep.episodes[e] for e in q4.episodes)
    assert (first.C, first.raw, first.pending_tokens, first.status) == (32, 56, 24, "preempted_again")
    assert (first.recovery.H, first.recovery.original_shared, first.recovery.missing) == (16, 16, 16)
    assert (second.C, second.raw, second.status) == (63, 64, "finished")
    assert (second.recovery.H, second.recovery.original_freed, second.recovery.missing) == (48, 48, 15)
    assert (first.reprefill_tokens, second.reprefill_tokens, q4.reprefill_tokens) == (16, 15, 31)
    assert (q4.discarded_tokens, q4.greatest, q4.preemptions) == (25, 75, 2)


def test_preempted_again_before_regaining_keeps_greatest(tmp_path):
    rep, q, _ = observe("before_regain", tmp_path)
    q1 = q["q1"]
    first, second = (rep.episodes[e] for e in q1.episodes)
    assert (first.C, first.raw, first.status, first.regained_t) == (50, 51, "preempted_again", None)
    assert (first.recovery.H, first.recovery.missing) == (16, 34)
    assert (second.C, second.raw, second.pending_tokens, second.status) == (46, 52, 6, "finished")
    assert (second.recovery.H, second.recovery.missing) == (0, 46)
    assert (first.reprefill_tokens, second.reprefill_tokens, q1.reprefill_tokens) == (30, 50, 80)
    assert (q1.discarded_tokens, q1.greatest, q1.preemptions) == (7, 107, 2)


def test_all_least_computed_ties_are_marked(tmp_path):
    rep, q, _ = observe("repeated", tmp_path)
    last = rep.decisions[-1]
    assert [(rep.requests[c.r].engine_id, c.completed, c.least_tied, c.rank) for c in last.candidates] == [
        ("q0", 63, True, 1), ("q4", 63, True, 1)]


def test_shared_block_release_keeps_block_and_sole_owner_reclaims(tmp_path):
    rep, q, _ = observe("equivalent", tmp_path)
    dec = rep.decisions[0]
    victim = cand(dec, rep, "q1")
    assert (victim.total_blocks, victim.shared_blocks, victim.reclaimable_blocks) == (4, 2, 2)
    assert (victim.shared_prefix_tokens, victim.reserved_blocks) == (32, 0)
    assert dec.reclaimed_blocks == 2
    ep = rep.episodes[dec.episode]
    assert [(o.block, o.ref_after_release) for o in ep.originals] == [(5, 1), (6, 1), (8, 0), (9, 0)]


def test_null_block_is_never_counted(tmp_path):
    rep, _, d = observe("two_victims", tmp_path)
    snaps = history.read_snapshots(d / "snapshots.jsonl")
    assert rep.null_block == 0
    assert all(row[0] != 0 for _, rec in snaps.values() for row in rec["blocks"])
    assert all(0 not in rec["queue"] for _, rec in snaps.values())
    assert all(c.reserved_blocks == 0 for dec in rep.decisions for c in dec.candidates)


def test_recovery_partitions_shared_and_equivalent(tmp_path):
    rep, q, _ = observe("equivalent", tmp_path)
    ep = rep.episodes[q["q1"].episodes[0]]
    r = ep.recovery
    assert [h.block for h in r.hit_blocks] == [5, 7]
    assert (r.H, r.original_freed, r.original_shared, r.equivalent, r.missing) == (32, 0, 16, 16, 0)
    assert ep.fraction == 1.0


def test_reused_block_id_is_not_original_recovery(tmp_path):
    rep, q, _ = observe("reused_id", tmp_path)
    ep = rep.episodes[q["q1"].episodes[0]]
    assert (ep.originals[1].block, ep.originals[1].generation) == (8, 1)
    assert ep.recovery.hit_blocks == [history.HitBlock(7, 1), history.HitBlock(8, 3)]
    assert (ep.recovery.original_shared, ep.recovery.equivalent) == (16, 16)


def test_zero_computed_episode_has_undefined_fraction(tmp_path):
    rep, q, _ = observe("zero_c", tmp_path)
    ep = rep.episodes[q["q3"].episodes[0]]
    assert (ep.C, ep.raw, ep.recovery.H) == (0, 32, 0)
    assert ep.fraction is None


def test_unresolved_episode_when_run_stops_waiting(tmp_path):
    rep, q, _ = observe("self", tmp_path, stop_after=50)
    ep = rep.episodes[q["d"].episodes[0]]
    assert (ep.status, ep.recovery, ep.fraction) == ("unresolved", None, None)
    assert rep.end.unfinished == 4


def test_stale_wait_at_depth_two(tmp_path):
    rep, q, _ = observe("self", tmp_path, depth=2)
    ep = rep.episodes[q["d"].episodes[0]]
    assert (ep.C, ep.raw, ep.pending_tokens) == (78, 80, 2)
    assert ep.lookups[0].call == ep.call + 4
    assert q["d"].discarded_tokens == 3


def test_snapshots_and_digests_are_checked(tmp_path):
    rep, _, _ = observe("repeated", tmp_path, snapshot_every=16)
    assert rep.snapshots_checked == rep.end.snapshots
    assert rep.end.snapshots > 2 + len(rep.decisions)
    assert rep.digests_checked > 100

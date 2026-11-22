"""Stock AsyncScheduler plus passive observation. Load with
`--scheduler-cls e02.observer.ObservedScheduler` and E02_OUT=<attempt dir>.

Every hook calls the stock method exactly once with the same arguments and returns its result.
Observation code runs in try blocks; a failure becomes a sticky Fault event that invalidates the
run and never changes a scheduling decision.

The measured phase starts at the first successful reset_prefix_cache with no live requests, after
an empty-cache check, and ends at the next reset_prefix_cache, which seals the journal. That
sealing reset is the one exception to returning the stock result: if the evidence could not be
made durable it writes producer-error.json in the attempt directory and returns False.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from array import array
from pathlib import Path

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SingleTypeKVCacheManager

from e02 import events as ev

PINNED_SOURCES = {
    "Scheduler.schedule": "73b6a4084e76ac6c",
    "Scheduler._preempt_request": "4113f5de8a84ec25",
    "Scheduler.update_from_output": "85b29db96051135c",
    "Scheduler.add_request": "ab9efd2a1069dbe9",
    "Scheduler.finish_requests": "b7a4ef7b8aa588bf",
    "Scheduler._free_request": "21f109d2b2c2d8ef",
    "Scheduler._free_blocks": "b24b618a274d6f5c",
    "Scheduler._free_request_blocks": "f967ef09115484c6",
    "Scheduler.reset_prefix_cache": "36b4355be090877e",
    "Scheduler._update_after_schedule": "2b527eb8070d3cce",
    "Scheduler._request_blocks_can_be_freed": "d3dd6fc13e67f3f1",
    "AsyncScheduler._update_after_schedule": "138d47daea23cdfd",
    "AsyncScheduler._update_request_with_output": "c4c4ccac872af040",
    "BlockPool.get_new_blocks": "ab54592edae48a7c",
    "BlockPool.touch": "08fcc52c2ec9e624",
    "BlockPool.free_blocks": "f12f94bc787cc15e",
    "BlockPool._insert_block_hash": "14a8749486bcd9f6",
    "BlockPool._remove_cached_block_hashes": "6fa2ab7286e56f01",
    "BlockPool.reset_prefix_cache": "160053665bf4d269",
    "BlockPool.cache_full_blocks": "5fbbcc0cf6eb1c7e",
    "BlockPool._maybe_evict_cached_block": "9d7fec709e2b25d9",
    "KVCacheManager.get_computed_blocks": "77f434d88d573bfe",
    "KVCacheManager.allocate_slots": "ffa596d58dfbbc62",
    "KVCacheManager.free": "6ee26cbc5b6fc234",
    "FreeKVCacheBlockQueue.popleft": "11b3c2d342a026fb",
    "FreeKVCacheBlockQueue.popleft_n": "2ef81e822bb4aef9",
    "FreeKVCacheBlockQueue.remove": "3e5a74b88b2db25f",
    "FreeKVCacheBlockQueue.append": "68fbb7af73d462a9",
    "FreeKVCacheBlockQueue.append_n": "e6670d2f826c89c0",
    "FreeKVCacheBlockQueue.prepend_n": "1dc241a97fade1fd",
    "SingleTypeKVCacheManager.free": "8293f02e5c2b8e0e",
    "SingleTypeKVCacheManager.allocate_new_blocks": "b8edfbf2e1667228",
    "SingleTypeKVCacheManager.add_local_computed_blocks": "8c415d4822aa3d0e",
    "SingleTypeKVCacheManager.pop_blocks_for_free": "bc0da22a739b2d16",
    "SingleTypeKVCacheManager.get_num_blocks_to_allocate": "f50d7c5b545195c3",
    "FullAttentionManager.cache_blocks": "981125e4caafbec8",
}

_CLASSES = {c.__name__: c for c in (Scheduler, AsyncScheduler, BlockPool, KVCacheManager, FreeKVCacheBlockQueue,
                                     SingleTypeKVCacheManager, FullAttentionManager)}


def source_fingerprints() -> dict[str, str]:
    out = {}
    for name in PINNED_SOURCES:
        cls, meth = name.split(".")
        src = inspect.getsource(_CLASSES[cls].__dict__[meth])
        out[name] = hashlib.sha256(src.encode()).hexdigest()[:16]
    return out


def observer_sha() -> str:
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("observer.py", "events.py"):
        h.update((pkg / name).read_bytes())
    return h.hexdigest()


def token_sha256(token_ids) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()


def keys_digest(hex_keys) -> str:
    """Short digest of a block's hash aliases; replay computes the same from its own key set."""
    return hashlib.blake2b("|".join(sorted(hex_keys)).encode(), digest_size=8).hexdigest()


class ConfigError(RuntimeError):
    pass


class ObservedScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._e02_out = Path(os.environ["E02_OUT"])
        self._e02_run = os.environ.get("E02_RUN_ID", self._e02_out.name)
        self._snapshot_every = int(os.environ.get("E02_SNAPSHOT_EVERY", "256"))
        self._digest_every = int(os.environ.get("E02_DIGEST_EVERY", "1"))
        self._pool = self.kv_cache_manager.block_pool
        self._mgr = self.kv_cache_manager.coordinator.single_type_managers[0]
        self._config = self._check_config()
        self._sources = source_fingerprints()
        drift = {k: v for k, v in self._sources.items() if PINNED_SOURCES[k] != v}
        if drift:
            raise ConfigError(f"stock source differs from the pinned engine: {sorted(drift)}")
        self._w: ev.Writer | None = None
        self._sealed = False
        self._call = 0
        self._rid: dict[str, int] = {}
        self._outs: dict[int, tuple[int, object]] = {}
        self._out_seq = 0
        self._alloc_seq = 0
        self._decision_seq = 0
        self._snapshots = 0
        self._snap_file = None
        self._last_queues = None
        self._last_failed_alloc: tuple[int, int, list] | None = None
        self._needed: list[int] | None = None
        self._install_hooks()

    def _check_config(self) -> dict:
        pool, km = self._pool, self.kv_cache_manager
        facts = {
            "scheduler_class": type(self).__name__,
            "policy": self.policy.name,
            "async": True,
            "connector": self.connector is not None,
            "ec_connector": self.ec_connector is not None,
            "num_spec_tokens": self.num_spec_tokens,
            "num_lookahead_tokens": self.num_lookahead_tokens,
            "block_size": self.block_size,
            "hash_block_size": pool.hash_block_size,
            "prefix_replay_tokens": self.prefix_replay_tokens,
            "scheduler_reserve_full_isl": self.scheduler_reserve_full_isl,
            "enable_caching": pool.enable_caching,
            "defer_block_free": self.defer_block_free,
            "requires_kv_delivery": self.requires_kv_delivery,
            "watermark_blocks": km.watermark_blocks,
            "num_gpu_blocks": pool.num_gpu_blocks,
            "kv_groups": [type(m).__name__ for m in km.coordinator.single_type_managers],
            "max_num_running_reqs": self.max_num_running_reqs,
            "max_num_scheduled_tokens": self.max_num_scheduled_tokens,
            "max_model_len": self.max_model_len,
            "lora": self.lora_config is not None,
        }
        required = {
            "policy": SchedulingPolicy.FCFS.name, "connector": False, "ec_connector": False, "num_spec_tokens": 0,
            "num_lookahead_tokens": 0, "hash_block_size": self.block_size, "prefix_replay_tokens": 0,
            "scheduler_reserve_full_isl": True, "enable_caching": True, "defer_block_free": False,
            "requires_kv_delivery": False, "kv_groups": ["FullAttentionManager"], "lora": False,
        }
        expected_blocks = os.environ.get("E02_NUM_BLOCKS")
        if expected_blocks is not None:
            required["num_gpu_blocks"] = int(expected_blocks)
        bad = {k: (facts[k], v) for k, v in required.items() if facts[k] != v}
        if bad:
            raise ConfigError(f"unsupported engine configuration (actual, required): {bad}")
        return facts

    def _emit(self, payload) -> None:
        self._w.emit(self._call, time.monotonic_ns(), payload)

    def _fault(self, where: str, e: BaseException) -> None:
        if self._w is not None:
            self._emit(ev.Fault(where=where, error=repr(e)[:500]))

    def _r(self, request_id: str) -> int:
        return self._rid.get(request_id, -1)

    def _install_hooks(self) -> None:
        pool, q, km, mgr = self._pool, self._pool.free_block_queue, self.kv_cache_manager, self._mgr
        coord = km.coordinator
        obs = self

        def wrap(owner, name, before=None, after=None):
            stock = getattr(owner, name)

            def hooked(*args, **kwargs):
                if obs._w is None:
                    return stock(*args, **kwargs)
                state = None
                if before is not None:
                    try:
                        state = before(*args, **kwargs)
                    except Exception as e:
                        obs._fault(f"{name}:before", e)
                if isinstance(state, tuple) and state and state[0] == "args":
                    args = state[1]
                result = stock(*args, **kwargs)
                if after is not None:
                    try:
                        after(result, state, *args, **kwargs)
                    except Exception as e:
                        obs._fault(f"{name}:after", e)
                return result

            setattr(owner, name, hooked)

        def ids(blocks):
            return [b.block_id for b in blocks]

        wrap(q, "popleft", after=lambda res, st: obs._emit(ev.QPop(blocks=[res.block_id])))
        wrap(q, "popleft_n", after=lambda res, st, n: res and obs._emit(ev.QPop(blocks=ids(res))))
        wrap(q, "remove", after=lambda res, st, b: obs._emit(ev.QRemove(block=b.block_id)))
        wrap(q, "append", after=lambda res, st, b: obs._emit(ev.QAppend(blocks=[b.block_id])))
        wrap(q, "append_n", after=lambda res, st, bs: bs and obs._emit(ev.QAppend(blocks=ids(bs))))
        wrap(q, "prepend_n", after=lambda res, st, bs: bs and obs._emit(ev.QPrepend(blocks=ids(bs))))

        wrap(pool, "get_new_blocks", after=lambda res, st, n: res and obs._emit(ev.PoolAlloc(blocks=ids(res))))

        def touch_before(blocks):
            blocks = list(blocks)
            if blocks:
                obs._emit(ev.PoolTouch(blocks=ids(blocks), refs_before=[b.ref_cnt for b in blocks]))
            return ("args", (blocks,))
        wrap(pool, "touch", before=touch_before)

        def free_before(ordered_blocks):
            blocks = list(ordered_blocks)
            if blocks:
                obs._emit(ev.PoolFree(blocks=ids(blocks), refs_before=[b.ref_cnt for b in blocks]))
            return ("args", (blocks,))
        wrap(pool, "free_blocks", before=free_before)

        def insert_before(key, block, num_tokens):
            effective = not (block.block_hash == key or pool.cached_block_hash_to_block.contain(key, block.block_id))
            return effective, block.block_hash is None

        def insert_after(res, st, key, block, num_tokens):
            if st and st[0]:
                obs._emit(ev.HashAdd(block=block.block_id, key=key.hex(), num_tokens=num_tokens, primary=st[1]))
        wrap(pool, "_insert_block_hash", before=insert_before, after=insert_after)

        def remove_before(block):
            keys = [] if block.block_hash is None else [block.block_hash]
            keys.extend(pool.cached_block_hashes_by_block.get(block.block_id, ()))
            return [k.hex() for k in keys]

        def remove_after(res, keys, block):
            if keys:
                obs._emit(ev.HashDel(block=block.block_id, keys=keys))
        wrap(pool, "_remove_cached_block_hashes", before=remove_before, after=remove_after)
        wrap(pool, "reset_prefix_cache", after=lambda res, st: res and obs._emit(ev.CacheReset()))

        for name in ("unpin_blocks", "move_block_hashes", "evict_blocks", "cache_partial_block"):
            wrap(pool, name, before=lambda *a, _n=name, **k: obs._emit(ev.Fault(where=f"unsupported:{_n}", error="")))
        wrap(km, "pop_blocks_for_free", before=lambda *a, **k: obs._emit(ev.Fault(where="unsupported:deferred_free", error="")))
        wrap(mgr, "_apply_cow", before=lambda *a, **k: obs._emit(ev.Fault(where="unsupported:cow", error="")))

        def lookup_after(res, st, request):
            blocks, hit_tokens, _ = res
            obs._emit(ev.Lookup(r=obs._r(request.request_id), hit_tokens=hit_tokens, hit_blocks=ids(blocks.blocks[0])))
        wrap(km, "get_computed_blocks", after=lookup_after)

        def needed_after(res, st, *a, **k):
            if obs._needed is not None:
                obs._needed.append(res)
        wrap(coord, "get_num_blocks_to_allocate", after=needed_after)

        def alloc_before(request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                         num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
                         num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0,
                         has_scheduled_reqs=True):
            obs._needed = []
            table = mgr.req_to_blocks.get(request.request_id)
            return (len(table) if table else 0, pool.get_num_free_blocks(), list(obs.running), request.status)

        def alloc_after(res, st, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None,
                        num_lookahead_tokens=0, num_external_computed_tokens=0, delay_cache_blocks=False,
                        num_encoder_tokens=0, full_sequence_must_fit=False, reserved_blocks=0,
                        has_scheduled_reqs=True):
            len_before, free_before, running, status = st
            needed, obs._needed = obs._needed, None
            table = mgr.req_to_blocks.get(request.request_id) or []
            appended = ids(table[len_before:])
            ok = res is not None
            admitting = status.name in ("WAITING", "PREEMPTED")
            if ok and not appended and not admitting:
                return
            obs._alloc_seq += 1
            r = obs._r(request.request_id)
            running_r = None if ok else [obs._r(x.request_id) for x in running]
            obs._emit(ev.Alloc(a=obs._alloc_seq, r=r, ok=ok, n_new=num_new_tokens, n_hit=num_new_computed_tokens,
                               full_fit=full_sequence_must_fit, reserved=reserved_blocks, free_before=free_before,
                               needed=needed, appended=appended, running=running_r))
            if not ok:
                obs._last_failed_alloc = (obs._alloc_seq, r, running_r, needed, free_before)
        wrap(km, "allocate_slots", before=alloc_before, after=alloc_after)

        def kv_free_before(request):
            table = mgr.req_to_blocks.get(request.request_id) or []
            obs._emit(ev.Detach(r=obs._r(request.request_id), blocks=ids(table)))
        wrap(km, "free", before=kv_free_before)

    def _queue_ids(self) -> list[int]:
        q = self._pool.free_block_queue
        out, b = [], q.fake_free_list_head.next_free_block
        while b is not q.fake_free_list_tail:
            out.append(b.block_id)
            b = b.next_free_block
        return out

    def _block_keys(self, b) -> list[str]:
        keys = [] if b.block_hash is None else [b.block_hash.hex()]
        keys.extend(k.hex() for k in self._pool.cached_block_hashes_by_block.get(b.block_id, ()))
        return keys

    def _state_digest(self) -> str:
        pool = self._pool
        refs = array("i", (b.ref_cnt for b in pool.blocks if not b.is_null))
        q = pool.free_block_queue
        head = q.fake_free_list_head.next_free_block.block_id
        tail = q.fake_free_list_tail.prev_free_block.block_id
        tail_info = array("i", [q.num_free_blocks, head, tail, len(pool.cached_block_hash_to_block)])
        return hashlib.blake2b(refs.tobytes() + tail_info.tobytes(), digest_size=8).hexdigest()

    def _snapshot(self, reason: str, victim=None) -> int:
        pool, mgr = self._pool, self._mgr
        blocks = []
        for b in pool.blocks:
            if b.is_null:
                continue
            keys = self._block_keys(b)
            if b.ref_cnt or keys:
                blocks.append([b.block_id, b.ref_cnt, len(keys), keys_digest(keys) if keys else ""])
        reqs = list(self.requests.values())
        if victim is not None and victim.request_id not in self.requests:
            reqs.append(victim)
        requests = []
        for req in reqs:
            table = mgr.req_to_blocks.get(req.request_id) or []
            requests.append([self._r(req.request_id), req.status.name, [b.block_id for b in table],
                             req.num_computed_tokens, req.num_in_flight_tokens, req.num_stale_output_tokens,
                             req.num_tokens, req.num_output_tokens, req.num_preemptions,
                             bool(self._request_blocks_can_be_freed(req))])
        requests.sort()
        index = self._snapshots
        record = {
            "index": index, "reason": reason, "call": self._call, "seq": self._w.seq,
            "blocks": blocks, "queue": self._queue_ids(), "free": pool.get_num_free_blocks(),
            "hash_keys": len(pool.cached_block_hash_to_block),
            "requests": requests,
            "running": [self._r(x.request_id) for x in self.running],
            "waiting": [self._r(x.request_id) for x in self.waiting],
            "kv_holding_waiting": len(self.kv_holding_waiting),
            "victim": None if victim is None else self._r(victim.request_id),
        }
        line = json.dumps(record, separators=(",", ":"))
        self._snap_file.write(line + "\n")
        self._snapshots += 1
        self._emit(ev.SnapshotTaken(index=index, reason=reason, digest=hashlib.sha256(line.encode()).hexdigest()))
        return index

    def _end_of_call(self) -> None:
        try:
            queues = ([self._r(x.request_id) for x in self.running], [self._r(x.request_id) for x in self.waiting])
            if queues != self._last_queues:
                self._last_queues = queues
                self._emit(ev.Queues(running=queues[0], waiting=queues[1]))
            if self._digest_every and self._call % self._digest_every == 0:
                self._emit(ev.StateDigest(value=self._state_digest()))
            if self._snapshot_every and self._call % self._snapshot_every == 0:
                self._snapshot("periodic")
        except Exception as e:
            self._fault("end_of_call", e)

    def add_request(self, request) -> None:
        if self._w is None:
            return super().add_request(request)
        self._call += 1
        super().add_request(request)
        try:
            r = len(self._rid)
            self._rid[request.request_id] = r
            self._emit(ev.ReqAdd(r=r, engine_id=request.request_id, prompt_len=request.num_prompt_tokens,
                                 prompt_sha=token_sha256(request.prompt_token_ids),
                                 max_tokens=request.max_tokens))
        except Exception as e:
            self._fault("add_request", e)
        self._end_of_call()

    def schedule(self, *args, **kwargs):
        if self._w is None:
            return super().schedule(*args, **kwargs)
        self._call += 1
        out = super().schedule(*args, **kwargs)
        try:
            self._out_seq += 1
            self._outs[id(out)] = (self._out_seq, out)
            entries = []
            for rid, n in out.num_scheduled_tokens.items():
                req = self.requests[rid]
                entries.append(ev.ScheduledEntry(self._r(rid), n, req.num_computed_tokens - n, req.num_preemptions))
            self._emit(ev.Scheduled(out=self._out_seq, entries=entries,
                                    new=[self._r(x.req_id) for x in out.scheduled_new_reqs],
                                    resumed=sorted(self._r(x) for x in out.scheduled_cached_reqs.resumed_req_ids),
                                    total=out.total_num_scheduled_tokens))
        except Exception as e:
            self._fault("schedule", e)
        self._end_of_call()
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        if self._w is None:
            return super().update_from_output(scheduler_output, model_runner_output)
        self._call += 1
        pending = []
        try:
            k, _ = self._outs.pop(id(scheduler_output), (-1, None))
            entries = []
            for rid, n in scheduler_output.num_scheduled_tokens.items():
                req = self.requests.get(rid)
                if req is None or req.is_finished():
                    kind = "gone"
                elif req.num_stale_output_tokens > 0:
                    kind = "stale"
                else:
                    kind = "valid"
                entries.append(ev.CompletedEntry(self._r(rid), n, kind))
                if req is not None:
                    pending.append((rid, req, req.num_output_tokens))
            self._emit(ev.Completed(out=k, entries=entries))
        except Exception as e:
            self._fault("update_from_output:before", e)
        result = super().update_from_output(scheduler_output, model_runner_output)
        try:
            self._emit(ev.Delivered(out=k, entries=[
                ev.DeliveredEntry(self._r(rid), req.num_output_tokens - before,
                                  req.num_computed_tokens - req.num_in_flight_tokens)
                for rid, req, before in pending]))
        except Exception as e:
            self._fault("update_from_output:after", e)
        self._end_of_call()
        return result

    def finish_requests(self, request_ids, finished_status):
        if self._w is None:
            return super().finish_requests(request_ids, finished_status)
        self._call += 1
        result = super().finish_requests(request_ids, finished_status)
        self._end_of_call()
        return result

    def _free_request(self, request, delay_free_blocks: bool = False):
        if self._w is not None:
            try:
                self._emit(ev.Finished(r=self._r(request.request_id), status=request.status.name,
                                       n_out=request.num_output_tokens, n_tokens=request.num_tokens))
            except Exception as e:
                self._fault("_free_request", e)
        return super()._free_request(request, delay_free_blocks)

    def _preempt_request(self, request, timestamp, drop_stale_output: bool = False):
        if self._w is None:
            return super()._preempt_request(request, timestamp, drop_stale_output)
        d = -1
        raw = in_flight = free_before = -1
        try:
            self._decision_seq += 1
            d = self._decision_seq
            a, requester, running, _, _ = self._last_failed_alloc or (-1, -1, None, None, None)
            snap = self._snapshot("decision", victim=request)
            self._emit(ev.Decision(d=d, victim=self._r(request.request_id), a=a, requester=requester,
                                   running=running, snapshot=snap))
            raw, in_flight = request.num_computed_tokens, request.num_in_flight_tokens
            free_before = self._pool.get_num_free_blocks()
        except Exception as e:
            self._fault("_preempt_request:before", e)
        super()._preempt_request(request, timestamp, drop_stale_output)
        try:
            self._emit(ev.Preempted(d=d, r=self._r(request.request_id), raw=raw, in_flight=in_flight,
                                    stale=request.num_stale_output_tokens, free_before=free_before,
                                    free_after=self._pool.get_num_free_blocks(),
                                    preemptions=request.num_preemptions))
        except Exception as e:
            self._fault("_preempt_request:after", e)

    def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        if self._w is not None:
            self._call += 1
            error = self._seal()
            self._w = None
            self._sealed = True
            ok = super().reset_prefix_cache(reset_running_requests, reset_connector)
            if error is None:
                return ok
            self._write_producer_error(*error)
            return False
        ok = super().reset_prefix_cache(reset_running_requests, reset_connector)
        if ok and not self._sealed and not self.requests:
            self._begin_measurement()
        return ok

    def _seal(self) -> tuple[str, str] | None:
        errors = []
        if self._w.error is not None:
            errors.append(("writer", self._w.error))
        for where, step in (("terminal snapshot", lambda: self._snapshot("terminal")),
                            ("snapshots flush", self._snap_file.flush),
                            ("snapshots fsync", lambda: os.fsync(self._snap_file.fileno())),
                            ("snapshots close", self._snap_file.close)):
            try:
                step()
            except Exception as e:
                errors.append((where, repr(e)[:500]))
        try:
            unfinished = sum(1 for r in self.requests if r in self._rid)
            sealed = self._w.seal(self._call, time.monotonic_ns(), self._snapshots, unfinished)
            if sealed is not None:
                errors.append(("seal", sealed))
        except Exception as e:
            errors.append(("seal", repr(e)[:500]))
        return errors[0] if errors else None

    def _write_producer_error(self, where: str, error: str) -> None:
        try:
            with open(self._e02_out / ev.PRODUCER_ERROR, "x") as f:
                json.dump({"where": where, "error": error}, f)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def _begin_measurement(self) -> None:
        self._e02_out.mkdir(parents=True, exist_ok=True)
        self._w = ev.Writer(self._e02_out / "events.jsonl")
        self._snap_file = open(self._e02_out / "snapshots.jsonl", "x", buffering=1 << 20)
        self._call = 0
        pool = self._pool
        config = {**self._config, "snapshot_every": self._snapshot_every, "digest_every": self._digest_every}
        self._emit(ev.RunStart(run_id=self._e02_run, schema=ev.SCHEMA, config=config,
                               num_blocks=pool.num_gpu_blocks, null_block=pool.null_block.block_id,
                               block_size=self.block_size, observer_sha=observer_sha(), sources=self._sources))
        nonempty = [b.block_id for b in pool.blocks if not b.is_null and (b.ref_cnt or self._block_keys(b))]
        if nonempty or pool.get_num_free_blocks() != pool.num_gpu_blocks - 1 or len(pool.cached_block_hash_to_block):
            self._emit(ev.Fault(where="initial_state_not_empty", error=str(nonempty[:20])))
        self._snapshot("initial")
        self._last_queues = ([], [])
        self._emit(ev.StateDigest(value=self._state_digest()))
        self._snap_file.flush()
        self._w.flush()

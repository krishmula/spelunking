"""vLLM's own scheduler plus logging, loaded with `--scheduler-cls e02_scheduler.LoggingScheduler`.

Changes nothing the scheduler decides. Writes one JSON line per event to $E02_LOG:

  step     every schedule() call: tokens scheduled, how many of them re-prefill a preempted request, free blocks
  preempt  the victim and every other running request (the candidate victims): computed tokens, block ids,
           which blocks are hashed, ref counts; the free-queue length before the victim's blocks are freed
  resume   the victim's prefix-cache hit when it is scheduled again, and which of its old blocks were reused
           (popped from the free queue) while it waited
  finish   a request leaving, with its preemption count

`pops` counts every block popped from the free queue since start, so two events bound how many pops happened
between them.
"""

import json
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class LoggingScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log = open(os.environ["E02_LOG"], "a", buffering=1)
        self._step = 0
        self._pops = 0
        self._last_pop = {}  # block id -> value of _pops when it was last popped
        self._last_hit = {}  # request id -> (hit tokens, hit block ids) from the latest lookup
        self._waiting_resume = {}  # request id -> state at its latest preemption
        self._reprefill_to = {}  # request id -> computed tokens it had when preempted (re-prefill ends there)

        pool = self.kv_cache_manager.block_pool
        self._pool = pool
        get_new_blocks = pool.get_new_blocks

        def logged_get_new_blocks(num_blocks):
            blocks = get_new_blocks(num_blocks)
            for b in blocks:
                self._pops += 1
                self._last_pop[b.block_id] = self._pops
            return blocks

        pool.get_new_blocks = logged_get_new_blocks

        get_computed_blocks = self.kv_cache_manager.get_computed_blocks

        def logged_get_computed_blocks(request):
            result = get_computed_blocks(request)  # (blocks, hit tokens, shared prefix boundary)
            if request.request_id in self._waiting_resume:
                self._last_hit[request.request_id] = (result[1], [b.block_id for b in result[0].blocks[0]], self._pops)
            return result

        self.kv_cache_manager.get_computed_blocks = logged_get_computed_blocks
        self._emit("start", num_blocks=pool.num_gpu_blocks, block_size=self.block_size)

    def _emit(self, ev, **fields):
        self._log.write(json.dumps({"ev": ev, "t": time.monotonic(), "step": self._step, "pops": self._pops,
                                    **fields}) + "\n")

    def _describe(self, request):
        blocks = self.kv_cache_manager.get_blocks(request.request_id).blocks[0]
        return {"id": request.request_id, "computed": request.num_computed_tokens, "tokens": request.num_tokens,
                "outputs": request.num_output_tokens, "blocks": [b.block_id for b in blocks],
                "hashed": [b.block_hash is not None for b in blocks], "refs": [b.ref_cnt for b in blocks]}

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        victim = self._describe(request)
        others = [self._describe(r) for r in self.running if r is not request]
        free_before = self._pool.get_num_free_blocks()
        super()._preempt_request(request, timestamp, drop_stale_output)
        self._emit("preempt", victim=victim, candidates=others, free_before=free_before,
                   free_after=self._pool.get_num_free_blocks(), waiting=len(self.waiting))
        self._waiting_resume[request.request_id] = {"pops": self._pops, "step": self._step, "victim": victim}

    def schedule(self, *args, **kwargs):
        out = super().schedule(*args, **kwargs)
        self._step += 1
        reprefill = 0
        for rid, n in out.num_scheduled_tokens.items():
            req = self.requests.get(rid)
            if req is None:
                continue
            if rid in self._waiting_resume:
                at = self._waiting_resume.pop(rid)
                # The window ends at the victim's lookup, before it allocates its own new blocks.
                hit, hit_ids, lookup_pops = self._last_hit.pop(rid, (0, [], self._pops))
                old = at["victim"]["blocks"]
                reused = [b for b in old if at["pops"] < self._last_pop.get(b, 0) <= lookup_pops]
                self._emit("resume", id=rid, hit_tokens=hit, hit_blocks=hit_ids, tokens=req.num_tokens,
                           computed_at_preempt=at["victim"]["computed"], waited_steps=self._step - at["step"],
                           pops_since_preempt=lookup_pops - at["pops"], old_blocks_reused=reused)
                self._reprefill_to[rid] = at["victim"]["computed"]
            end = self._reprefill_to.get(rid)
            if end is not None:
                start = req.num_computed_tokens - n  # schedule() already advanced num_computed_tokens
                reprefill += max(0, min(start + n, end) - start)
                if start + n >= end:
                    del self._reprefill_to[rid]
        self._emit("step", tokens=out.total_num_scheduled_tokens, reqs=len(out.num_scheduled_tokens),
                   reprefill=reprefill, free=self._pool.get_num_free_blocks(), running=len(self.running),
                   waiting=len(self.waiting))
        return out

    def _free_request(self, request, *args, **kwargs):
        self._emit("finish", id=request.request_id, preemptions=request.num_preemptions,
                   outputs=request.num_output_tokens, prompt=request.num_prompt_tokens)
        return super()._free_request(request, *args, **kwargs)

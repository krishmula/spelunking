"""The real vllm-core AsyncScheduler and block pool on a tiny pool, driven by a scripted model runner.

Adapted from vllm-core tests/v1/core/utils.py and test_async_scheduler.py, without their default
uncached OPT model: the model config comes from the cached Qwen3 checkpoint, and tokenization is
skipped. One step stays in flight while the next is scheduled, as in the async engine loop.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import torch

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

MODEL = "mlx-community/Qwen3-4B-4bit"
BLOCK = 16
EOS = 151645
_model_config = None
_hasher = None


def model_config(max_model_len: int) -> ModelConfig:
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig(model=MODEL, dtype="float16", seed=0, max_model_len=max_model_len,
                                    skip_tokenizer_init=True)
    return _model_config


def make_scheduler(cls=AsyncScheduler, *, out_dir: Path | None = None, num_blocks: int = 24, max_num_seqs: int = 4,
                   max_num_batched_tokens: int = 64, max_model_len: int = 256, snapshot_every: int = 0,
                   long_prefill_token_threshold: int = 0, run_id: str | None = None):
    if out_dir is not None:
        os.environ["E02_OUT"] = str(out_dir)
        os.environ["E02_RUN_ID"] = run_id or out_dir.name
        os.environ["E02_SNAPSHOT_EVERY"] = str(snapshot_every)
        os.environ["E02_NUM_BLOCKS"] = str(num_blocks)
    mc = model_config(max_model_len)
    sc = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                         max_model_len=max_model_len, async_scheduling=True, is_encoder_decoder=False,
                         watermark=0.0, enable_chunked_prefill=True,
                         long_prefill_token_threshold=long_prefill_token_threshold)
    cc = CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=True)
    vc = VllmConfig(scheduler_config=sc, model_config=mc, cache_config=cc, parallel_config=ParallelConfig())
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
    kc = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)])
    cc.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vc)
    s = cls(vllm_config=vc, kv_cache_config=kc, block_size=BLOCK, log_stats=True,
            structured_output_manager=StructuredOutputManager(vc))
    s.use_v2_model_runner = False
    return s


def make_request(rid: str, prompt: list[int], max_tokens: int) -> Request:
    global _hasher
    if _hasher is None:
        init_none_hash(sha256)
        _hasher = get_request_block_hasher(BLOCK, sha256)
    params = SamplingParams(ignore_eos=False, max_tokens=max_tokens)
    params.update_from_generation_config({}, EOS)
    return Request(request_id=rid, prompt_token_ids=list(prompt), sampling_params=params, pooling_params=None,
                   block_hasher=_hasher)


class Engine:
    """Scheduler plus a scripted runner. `depth` steps are in flight while the next is scheduled.
    A step samples one token for a request whose scheduled tokens reach the end of its known
    sequence at schedule time; tokens are globally unique so delivery is checkable."""

    def __init__(self, scheduler, depth: int = 1):
        self.s = scheduler
        self.depth = depth
        self.queue: deque = deque()
        self.outputs: list = []
        self.preempted: list[list[str]] = []
        self.delivered: dict[str, list[int]] = {}
        self.next_token = 1000

    def add(self, rid: str, prompt: list[int], max_tokens: int) -> None:
        self.s.add_request(make_request(rid, prompt, max_tokens))

    def schedule(self):
        out = self.s.schedule()
        samples = {rid: not self.s.requests[rid].is_prefill_chunk for rid in out.num_scheduled_tokens}
        self.queue.append((out, samples))
        self.outputs.append(out)
        self.preempted.append(sorted(out.preempted_req_ids or ()))
        return out

    def complete_oldest(self) -> None:
        out, samples = self.queue.popleft()
        req_ids = list(out.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            if samples[rid]:
                sampled.append([self.next_token])
                self.next_token += 1
            else:
                sampled.append([])
        mro = ModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={}, pooler_output=[])
        result = self.s.update_from_output(out, mro)
        for outs in result.values():
            for o in outs.outputs:
                self.delivered.setdefault(o.request_id, []).extend(o.new_token_ids)

    def step(self) -> None:
        self.schedule()
        if len(self.queue) > self.depth:
            self.complete_oldest()

    def drain(self) -> None:
        while self.queue:
            self.complete_oldest()

    def run(self, max_steps: int = 500, arrivals: dict | None = None) -> None:
        arrivals = arrivals or {}
        for i in range(max_steps):
            for rid, prompt, max_tokens in arrivals.get(i, ()):
                self.add(rid, prompt, max_tokens)
            if not self.s.has_requests() and i > max(arrivals, default=-1):
                self.drain()
                if not self.s.has_requests():
                    return
            self.step()
        raise AssertionError("engine did not drain")


def prompt(seed: int, n: int) -> list[int]:
    return [seed * 1000 + i for i in range(n)]


def pool_state(s) -> tuple:
    pool = s.kv_cache_manager.block_pool
    q = pool.free_block_queue
    order, b = [], q.fake_free_list_head.next_free_block
    while b is not q.fake_free_list_tail:
        order.append(b.block_id)
        b = b.next_free_block
    refs = [blk.ref_cnt for blk in pool.blocks]
    hashed = sorted(blk.block_id for blk in pool.blocks if blk.block_hash is not None)
    return refs, order, hashed

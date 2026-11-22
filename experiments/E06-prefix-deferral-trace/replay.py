"""Replay timestamped, length-controlled requests in the released Zipage CPU scheduler."""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import types
from collections import Counter
from datetime import datetime
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xxhash

REV = "3cf00ad547e5fe084fa640f431a31e02b45a2869"
TRACE_BLOB = "5a371794b81d7fc7487ec8a02a2c631bcc7fbdf8"
CAPACITIES = (1.05, 1.25, 1.5, 2, 3)
BLOCK = 256


def load_scheduler(repo):
    assert subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip() == REV
    subprocess.run(["git", "-C", repo, "diff", "--exit-code", "HEAD", "--", "zipage"], check=True)
    # Skip package __init__, which eagerly imports the full CUDA model runner.
    package = types.ModuleType("zipage")
    package.__path__ = [str(repo / "zipage")]
    sys.modules["zipage"] = package
    config = types.ModuleType("zipage.config")
    config.Config = SimpleNamespace
    sys.modules[config.__name__] = config
    from zipage.engine.scheduler import Scheduler
    from zipage.engine.sequence import Sequence
    from zipage.sampling_params import SamplingParams

    source = (repo / "zipage/engine/scheduler.py").read_text()
    anchor = "                while not self.block_manager.can_append_or_compress(seq):\n"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + """                    if self.pause_if_blocked(seq):
                        running_seqs.append(seq)
                        break
""")
    scope = {}
    exec(compile(source, "scheduler-with-deferral.py", "exec"), scope)
    return Scheduler, scope["Scheduler"], Sequence, SamplingParams


def prompt(record, index):
    length = record["input_length"]
    assert len(record["hash_ids"]) == math.ceil(length / 512)
    result = []
    for pos, value in enumerate(record["hash_ids"]):
        n = min(length - 512 * pos, 512)
        value = value + 1 if n == 512 else -index - 1
        result.extend([value] * n)
    return result


def percentile(values):
    return float(np.percentile(values, 95)) if values else None


def simulate(source, requests, prompts, capacity, bound, step_ms, prefill_rate, control=False):
    original, modified, sequence_type, params_type = source
    seqs = [sequence_type(token_ids, params_type(max_tokens=r["output_length"], ignore_eos=True))
            for token_ids, r in zip(prompts, requests)]
    for i, seq in enumerate(seqs):
        seq.request_id = i
    cls = original if control else modified
    config = SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=max(32768, max(map(len, seqs)) * 2),
                             eos=-1, num_kvcache_blocks=capacity, kvcache_block_size=BLOCK,
                             max_cache_blocks_per_seq=8, enable_prefix_cache=True, max_concurrency=16,
                             enable_hybrid_engine=True, query_cache_len=16)
    sched = cls(config)
    allocator = sched.block_manager
    events = []
    first, done, deadlines = {}, {}, {}
    counts = Counter()
    idle_ms = 0
    now = tick = submitted = empty = 0
    lower = requests[0]["timestamp"]
    arrivals = [r["timestamp"] - lower for r in requests]
    assert arrivals == sorted(arrivals)
    peak = 0
    original_preempt = sched.preempt

    def preempt(seq):
        events.append([tick, now, "preempt", seq.request_id])
        original_preempt(seq)
        deadlines.pop(seq.request_id, None)

    def pause_if_blocked(seq):
        if len(seq.block_table) < allocator.max_blocks_per_seq:
            return False
        shared = 0
        for bid in seq.block_table:
            if allocator.blocks[bid].ref_count == 1:
                break
            shared += 1
        assert min(shared, allocator.max_blocks_per_seq - 1) > len(allocator.free_block_ids)
        deadline = deadlines.setdefault(seq.request_id, tick + bound)
        eligible = any(not (s.compressed or s.require_compress) for s in sched.running)
        pause = tick < deadline and eligible
        events.append([tick, now, "pause" if pause else "blocked", seq.request_id, shared,
                       len(allocator.free_block_ids), eligible])
        return pause

    sched.pause_if_blocked = pause_if_blocked
    sched.preempt = preempt
    while submitted < len(seqs) or not sched.is_finished():
        if sched.is_finished() and submitted < len(seqs):
            idle_ms += max(0, arrivals[submitted] - now)
            now = max(now, arrivals[submitted])
        while submitted < len(seqs) and arrivals[submitted] <= now:
            sched.add(seqs[submitted])
            submitted += 1
        active, prefill = sched.schedule()
        peak = max(peak, len(allocator.used_block_ids))
        if not active:
            counts["empty_ticks"] += 1
            # A bounded pause may temporarily skip all candidates.
            empty += 1
            if empty > bound + 2 and submitted == len(seqs):
                break
            now += step_ms
            idle_ms += step_ms
        else:
            empty = 0
            if prefill:
                uncached = sum(len(s) - s.num_cached_tokens for s in active)
                counts["prefill_tokens"] += uncached
                for s in active:
                    s.num_cached_tokens = len(s)
                now += max(1, 1000 * uncached / prefill_rate)
            else:
                compressing = [s for s in active if s.require_compress]
                counts["compressions"] += len(compressing)
                for s in compressing:
                    deadlines.pop(s.request_id, None)
                sched.postprocess(compressing)
                now += step_ms
            # The negative token range is disjoint from all prompt tokens.
            tokens = [-1000000 - s.request_id for s in active]
            sched.postprocess(active, tokens)
            for s in active:
                first.setdefault(s.request_id, now)
                if s.is_finished:
                    assert s.num_completion_tokens == requests[s.request_id]["output_length"]
                    done[s.request_id] = now
        tick += 1
        if tick > 1500000:
            break
    assert peak <= capacity
    if len(done) == len(seqs):
        assert not allocator.used_block_ids
        assert len(allocator.free_block_ids) == capacity
    per_req = [dict(id=i, arrival_ms=t, first_ms=first.get(i), complete_ms=done.get(i),
                    output_tokens=seqs[i].num_completion_tokens) for i, t in enumerate(arrivals)]
    return dict(capacity=capacity, bound=bound, step_ms=step_ms, prefill_rate=prefill_rate,
                ticks=tick, peak=peak, arrival_span_ms=arrivals[-1], last_tick_ms=now,
                outstanding_at_last_arrival=len(seqs)-sum(t <= arrivals[-1] for t in done.values()),
                service_fraction=(now-idle_ms)/now,
                preemptions_after_last_arrival=sum(e[2]=="preempt" and e[1]>arrivals[-1] for e in events),
                counts=dict(counts), events=events, per_request=per_req,
                completed=len(done), preemptions=sum(e[2] == "preempt" for e in events),
                pauses=sum(e[2] == "pause" for e in events),
                ttft95_ms=percentile([v-arrivals[i] for i,v in first.items()]),
                completion95_ms=percentile([v-arrivals[i] for i,v in done.items()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zipage", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--start", type=int, default=1, help="first request, indexed from one")
    parser.add_argument("--lower-load", action="store_true", help="100,000 prefill tokens/s and 1 ms decode")
    args = parser.parse_args()
    if args.start < 1:
        parser.error("--start must be positive")
    data = args.trace.read_bytes()
    git_hash = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    assert git_hash == TRACE_BLOB
    requests = [json.loads(row) for row in islice(data.splitlines(), args.start - 1, args.start - 1 + 256)]
    assert len(requests) == 256
    assert all(r["input_length"] > 0 and r["output_length"] > 0 for r in requests)
    prompts = [prompt(r, i) for i, r in enumerate(requests)]
    biggest = math.ceil(max(map(len, prompts)) / BLOCK)
    caps = [math.ceil(f * biggest) + 1 for f in CAPACITIES]
    caps.append(sum(math.ceil((r["input_length"]+r["output_length"])/BLOCK) for r in requests))
    sources = load_scheduler(args.zipage.resolve())
    run = Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-trace")
    run.mkdir(parents=True)
    env = dict(first_request=args.start, request_count=len(requests), zipage=REV, trace_blob=TRACE_BLOB, python=sys.version, platform=platform.platform(),
               numpy=np.__version__, xxhash=xxhash.VERSION,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               timing="simulated", logical_cpus=os.cpu_count())
    (run / "env.json").write_text(json.dumps(env, indent=2))
    summaries = []
    prefill_rate = 100000 if args.lower_load else 2000
    for step_ms in ((1,) if args.lower_load else (10, 5)):
        for cap in caps:
            for bound in (0, 2):
                result = simulate(sources, requests, prompts, cap, bound, step_ms, prefill_rate)
                if step_ms in (1, 10) and cap == caps[0] and bound == 0:
                    orig = simulate(sources, requests, prompts, cap, 0, step_ms, prefill_rate, control=True)
                    assert result["per_request"] == orig["per_request"]
                    assert result["preemptions"] == orig["preemptions"]
                (run / f"{step_ms}-{cap}-{bound}.json").write_text(json.dumps(result))
                sums = {k:v for k,v in result.items() if k not in ("events", "per_request")}
                sums["blocked_attempts"] = sum(e[2] in ("blocked", "pause") for e in result["events"])
                summaries.append(sums)
                print(json.dumps(sums), flush=True)
    (run / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

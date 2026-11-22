"""Matched heterogeneous-depth target verification on the Metal GPU."""
import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model
from vllm_metal.attention.context import OffsetCache, clear_context, prepare_unified
from vllm_metal.attention.runtime.sdpa import SDPAPagedAttentionRuntime
from vllm_metal.metal import get_ops

MODEL = "mlx-community/Qwen3-4B-4bit"
REVISION = "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
CONTEXTS = [1024] * 4 + [3072] * 4
LIMITS = [0, 1, 3, 7] * 2
DEPTHS = (0, 1, 2, 3, 4, 6, 8)
BLOCK_SIZE = 16
ROUNDS = 12
REPEATS = 4
ROOT = Path(__file__).resolve().parent


def command(*args):
    p = subprocess.run(args, text=True, capture_output=True)
    return (p.stdout + p.stderr).strip()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


class Target:
    def __init__(self, model_path):
        self.model, config = load_model(Path(model_path))
        self.n_layers = config["num_hidden_layers"]
        self.block_tables = []
        block_count = 0
        for context in CONTEXTS:
            count = math.ceil((context + 1 + max(DEPTHS)) / BLOCK_SIZE)
            self.block_tables.append(list(range(block_count, block_count + count)))
            block_count += count
        self.runtime = SDPAPagedAttentionRuntime(
            num_layers=self.n_layers, num_kv_heads=config["num_key_value_heads"],
            head_dim=config["head_dim"], block_size=BLOCK_SIZE,
            dtype=self.model.layers[0].input_layernorm.weight.dtype,
        )
        self.runtime.initialize(block_count)
        assert self.runtime.patch_model(self.model) == self.n_layers
        self.cache = [OffsetCache(max(CONTEXTS)) for _ in range(self.n_layers)]
        self.prompt = [[1033 + (i % 4) * 11 + j % 4 for j in range(n + 1)]
                       for i, n in enumerate(CONTEXTS)]

    def fill(self):
        for table, n, prompt in zip(self.block_tables, CONTEXTS, self.prompt):
            for start in range(0, n, 256):
                end = min(start + 256, n)
                prepare_unified([], [(table, end - start, start)], BLOCK_SIZE)
                try:
                    hidden = self.model.model(mx.array([prompt[start:end]]), cache=self.cache)
                    mx.eval(hidden)
                finally:
                    clear_context()

    def greedy(self, tapes, offsets):
        ids = mx.array([list(itertools.chain.from_iterable(tapes))], dtype=mx.int32)
        specs = [(blocks, offset, len(tape)) for blocks, offset, tape in
                 zip(self.block_tables, offsets, tapes)]
        prepare_unified(specs, [], BLOCK_SIZE)
        try:
            start = time.perf_counter_ns()
            result = mx.argmax(self.model(ids, cache=self.cache)[0], axis=-1)
            choices = result.tolist()
            elapsed = (time.perf_counter_ns() - start) / 1e6
        finally:
            clear_context()
        assert len(choices) == ids.shape[1]
        return choices, elapsed

    def draft_tapes(self):
        tapes = [[prompt[-1]] for prompt in self.prompt]
        for position in range(max(DEPTHS)):
            choices, _ = self.greedy([[t[-1]] for t in tapes], [n + position for n in CONTEXTS])
            for tape, prediction in zip(tapes, choices):
                tape.append(prediction)
        for tape, limit in zip(tapes, LIMITS):
            if limit < max(DEPTHS):
                tape[limit + 1] = (tape[limit + 1] + 1) % self.model.args.vocab_size
        return tapes


def depths_for(k, assignment):
    if k == 0:
        return [0] * 8
    deep_context = 1024 if assignment == "short" else 3072
    return [k if n == deep_context else 1 for n in CONTEXTS]


def sample(target, tapes, depths):
    start = time.perf_counter_ns()
    inputs = [tape[:n + 1] for tape, n in zip(tapes, depths)]
    predicted, verify_ms = target.greedy(inputs, CONTEXTS)
    cursor = 0
    emitted = []
    for tokens in inputs:
        count = 0
        for proposed in tokens[1:]:
            if proposed != predicted[cursor + count]:
                break
            count += 1
        emitted.append(tokens[1:count + 1] + [predicted[cursor + count]])
        cursor += len(tokens)
    whole_ms = (time.perf_counter_ns() - start) / 1e6
    counts = [len(x) for x in emitted]
    expected = [1 + min(n, limit) for n, limit in zip(depths, LIMITS)]
    assert counts == expected, (depths, counts, expected)
    return verify_ms, whole_ms, sum(counts)


def audit(target, tapes, depths):
    ops = get_ops()
    original = ops.paged_attention_primitive
    seen = []
    expected_lengths = [c + i + 1 for c, k in zip(CONTEXTS, depths) for i in range(k + 1)]

    def observed(*args, **kwargs):
        q, lens, cu = args[0], args[7], args[8]
        assert q.shape[0] == len(expected_lengths)
        assert lens.tolist() == expected_lengths
        assert cu.tolist() == list(range(len(expected_lengths) + 1))
        seen.append(list(q.shape))
        return original(*args, **kwargs)

    previous = ops._set_paged_dispatch_diagnostics(True)
    try:
        with patch.object(ops, "paged_attention_primitive", observed):
            _, _, committed = sample(target, tapes, depths)
        route = ops.last_paged_dispatch()
        assert route
    finally:
        ops._set_paged_dispatch_diagnostics(previous)
    assert len(seen) == target.n_layers and len({tuple(shape) for shape in seen}) == 1
    return dict(query_shape=seen[0], layers=len(seen), last_layer_route=route, context_lengths=expected_lengths,
                committed=committed, work_pairs=sum(expected_lengths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-first", action="store_true")
    parser.add_argument("--high-acceptance", action="store_true")
    args = parser.parse_args()
    if args.high_acceptance:
        LIMITS[:] = [0, 2, 4, 8] * 2
    if args.long_first:
        CONTEXTS.reverse()
    label = "long-first" if args.long_first else "depth"
    if args.high_acceptance:
        label += "-high"
    run = ROOT / "runs" / datetime.now().strftime(f"%Y%m%d-%H%M%S-{label}")
    run.mkdir(parents=True)
    env = dict(model=MODEL, revision=REVISION, python=sys.version, platform=platform.platform(),
               versions={p: version(p) for p in ("mlx", "mlx-lm", "vllm", "vllm-metal")},
               source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               metal_head=command("git", "-C", str(Path.home() / "dev/vllm-lab/vllm-metal"), "rev-parse", "HEAD"),
               load=command("uptime"), cpu=command("sysctl", "-n", "hw.logicalcpu"),
               thermal=command("pmset", "-g", "therm"), swap=command("sysctl", "vm.swapusage"))
    save(run / "env.json", env)
    model_path = snapshot_download(MODEL, revision=REVISION, allow_patterns=["model.safetensors", "config.json"])
    print(f"Loading {model_path} for {run}", flush=True)
    target = Target(model_path)
    target.fill()
    tapes = target.draft_tapes()
    save(run / "fixture.json", dict(contexts=CONTEXTS, acceptance=LIMITS, tapes=tapes))
    arms = list(itertools.product(DEPTHS, ("short", "long")))
    dispatch = {f"{k}-{a}": audit(target, tapes, depths_for(k, a)) for k, a in arms}
    save(run / "dispatch.json", dispatch)
    rng = random.Random(7091)
    started = time.monotonic()
    while time.monotonic() - started < 60:
        for k, a in arms:
            sample(target, tapes, depths_for(k, a))
    print("Warmup complete", flush=True)
    with (run / "samples.jsonl").open("x") as f:
        for r in range(ROUNDS):
            ks = list(DEPTHS)
            rng.shuffle(ks)
            for k in ks:
                order = ["short", "long"]
                if (r + k) % 2:
                    order.reverse()
                for a in order:
                    depths = depths_for(k, a)
                    values = [sample(target, tapes, depths) for _ in range(REPEATS)]
                    f.write(json.dumps(dict(round=r,depth=k,assignment=a,values=values)) + "\n")
                    f.flush()
            print(f"Round {r + 1}/{ROUNDS}", flush=True)
    env.update(end_thermal=command("pmset", "-g", "therm"), end_swap=command("sysctl", "vm.swapusage"))
    save(run / "env.json", env)
    print(run, flush=True)


if __name__ == "__main__":
    main()

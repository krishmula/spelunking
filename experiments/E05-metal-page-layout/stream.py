"""Compare page order when successive attention calls use distinct KV buffers."""
import datetime
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import median

import mlx.core as mx
import numpy as np
from vllm_metal.metal import get_ops

from measure import sample, system

LAYERS = 8
COUNT = 512
SLOTS = 4 * COUNT
NAMES = ("contiguous", "control", "shuffled")


def run(seed):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    pages = np.arange(SLOTS, dtype=np.int32)
    queries = [mx.random.normal((8, 32, 128), dtype=mx.float16) for _ in range(LAYERS)]
    lengths = mx.array([1024] * 8, dtype=mx.int32)
    offsets = mx.arange(9, dtype=mx.int32)
    mx.eval(queries, lengths, offsets)
    banks = {name: [] for name in NAMES}
    ops = get_ops()
    for layer in range(LAYERS):
        selected = rng.permutation(pages)[:COUNT]
        inverse = np.empty_like(pages)
        inverse[selected] = pages[:COUNT]
        inverse[np.setdiff1d(pages, selected)] = pages[COUNT:]
        baseline = [mx.random.normal((SLOTS, 16, 8, 128), dtype=mx.float16) for _ in range(2)]
        mx.eval(baseline)
        for name in rng.permutation(NAMES):
            source = mx.array(inverse if name == 'shuffled' else pages)
            keys, values = [a[source] for a in baseline]
            table = mx.array(selected if name == 'shuffled' else pages[:COUNT]).reshape(8, -1)
            mx.eval(keys, values, table)
            banks[name].append((keys, values, table))
            if name == 'shuffled':
                for tensor, original in zip((keys, values), baseline):
                    assert mx.array_equal(tensor[table.reshape(-1)], original[:COUNT]).item()
        del baseline, original
    mx.clear_cache()

    def attend(name, layers):
        results = []
        for query, layer in zip(queries, layers):
            keys, values, table = banks[name][layer]
            out = mx.array(0)
            ops.paged_attention_primitive(query, keys, values, 8, 128**-0.5, 0., table,
                lengths, offsets, 16, 1024, -1, out, window_seqlen_q=1, num_decode_requests=8,
                num_decode_tokens=8, max_decode_context_len=1024)
            results.append(out)
        return results

    def repack():
        return [tensor[table.reshape(-1)] for keys, values, table in banks['shuffled']
                for tensor in (keys, values)]

    for layers in (range(LAYERS), *([i] * LAYERS for i in range(LAYERS))):
        ref = [np.asarray(x) for x in attend('contiguous', layers)]
        for name in ('control', 'shuffled'):
            for out, expected in zip(attend(name, layers), ref, strict=True):
                np.testing.assert_array_equal(np.asarray(out), expected)
    mx.clear_cache()
    warm = time.monotonic()
    while time.monotonic() - warm < 20:
        for name in NAMES:
            sample(lambda: attend(name, range(LAYERS)), 1)

    rounds = []
    for r in range(31):
        record = {}
        shift = r % LAYERS
        for mode in rng.permutation(['rotate', 'reuse']):
            indices = [(shift + i) % LAYERS for i in range(LAYERS)] if mode == 'rotate' else [shift] * LAYERS
            record[mode] = {name: sample(lambda: attend(name, indices), 3)
                            for name in rng.permutation(NAMES)}
        record['repack_us'] = sample(repack, 1)
        rounds.append(record)
    summary = {}
    for mode in ('rotate', 'reuse'):
        def delta(name):
            return [r[mode][name] - r[mode]['contiguous'] for r in rounds]
        gain = median(delta('shuffled'))
        control = np.percentile(np.abs(delta('control')), 90)
        cost = median(r['repack_us'] for r in rounds)
        summary[mode] = dict(gain_us=gain, control_us=control, cost_us=cost,
                             steps=math.ceil(cost/gain) if gain > control else None)
    return dict(seed=seed, layers=LAYERS, rounds=rounds, summary=summary)


if __name__ == '__main__':
    root = Path('runs') / datetime.datetime.now().strftime('%Y%m%d-%H%M%S-stream')
    root.mkdir(parents=True)
    env = json.loads(next(Path('runs').glob('*-layout/env.json')).read_text())
    env.update(load=system(['uptime']), heat=system(['pmset','-g','therm']),
               swap=system(['sysctl','vm.swapusage']), device=mx.device_info(),
               sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'env.json').write_text(json.dumps(env, indent=2))
    for seed in (37, 91, 133):
        report = run(seed)
        report.update(swap=system(['sysctl','vm.swapusage']),heat=system(['pmset','-g','therm']))
        (root/f'{seed}.json').write_text(json.dumps(report, indent=2))
        print(seed, report['summary'], flush=True)
        mx.clear_cache()

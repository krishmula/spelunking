import datetime
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from statistics import median

import mlx.core as mx
import numpy as np
from vllm_metal.metal import get_ops


def system(command):
    return subprocess.run(command, text=True, capture_output=True).stdout.strip()


def sample(operation, count):
    elapsed = []
    for _ in range(count):
        start = time.perf_counter_ns()
        result = operation()
        mx.eval(result)
        elapsed.append((time.perf_counter_ns() - start) / 1000)
    return median(elapsed)


def measure(batch, length, seed):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    n = batch * length // 16
    pages = np.arange(n, dtype=np.int32)
    layouts = dict(contiguous=pages, control=pages.copy(),
                   clustered=rng.permutation(pages.reshape(-1, 4)).ravel(),
                   shuffled=rng.permutation(pages))
    kv_shape = (n, 16, 8, 128)
    k, v = [mx.random.normal(kv_shape, dtype=mx.float16) for _ in range(2)]
    q = mx.random.normal((batch, 32, 128), dtype=mx.float16)
    lengths = mx.array([length] * batch, dtype=mx.int32)
    offsets = mx.arange(batch + 1, dtype=mx.int32)
    ops = get_ops()
    ops._set_paged_dispatch_diagnostics(True)
    calls, copies, routes = {}, {}, {}
    reference = mx.fast.scaled_dot_product_attention(
        q[:, :, None, :], k.reshape(batch, length, 8, 128).transpose(0, 2, 1, 3),
        v.reshape(batch, length, 8, 128).transpose(0, 2, 1, 3), scale=128**-0.5)
    for name, order in layouts.items():
        table = mx.array(order.reshape(batch, -1))
        reverse = mx.array(np.argsort(order).astype(np.int32))
        keys, values = k[reverse], v[reverse]
        flat = mx.array(order)
        mx.eval(q, keys, values, table, flat, lengths, offsets)
        def attention(keys=keys, values=values, table=table):
            out = mx.array(0)
            ops.paged_attention_primitive(q, keys, values, 8, 128**-0.5, 0.,
                table, lengths, offsets, 16, length, -1, out, window_seqlen_q=1,
                num_decode_requests=batch, num_decode_tokens=batch,
                max_decode_context_len=length)
            return out
        def repack(keys=keys, values=values, flat=flat):
            return [keys[flat], values[flat]]
        out = np.asarray(attention())
        if name == 'contiguous':
            expected = out
            np.testing.assert_allclose(out, np.asarray(reference).reshape(out.shape),
                                       atol=0.001, rtol=0.01)
        np.testing.assert_array_equal(out, expected)
        assert all(mx.array_equal(a, b).item() for a, b in zip(repack(), (k, v)))
        routes[name] = ops.last_paged_dispatch()
        calls[name] = attention
        copies[name] = repack
    assert len(set(routes.values())) == 1 and all(routes.values())
    ops._set_paged_dispatch_diagnostics(False)
    warm = time.monotonic()
    while time.monotonic() - warm < 4:
        for op in calls.values():
            sample(op, 9)
    rounds = []
    for _ in range(21):
        times = {name: sample(calls[name], 9) for name in rng.permutation(list(calls))}
        costs = {name: sample(copies[name], 3) for name in rng.permutation(['clustered','shuffled'])}
        rounds.append(dict(attention_us=times, repack_us=costs))
    def saving(name):
        return [r['attention_us'][name] - r['attention_us']['contiguous'] for r in rounds]
    noise = np.percentile(np.abs(saving('control')), 90)
    summary = {}
    for name in copies:
        times = [r['attention_us'][name] for r in rounds]
        diff = median(saving(name))
        cost = median(r['repack_us'][name] for r in rounds) if name in costs else None
        summary[name] = dict(us=median(times), limits=[min(times),max(times)],
                             saving_us=diff, repack_us=cost,
                             steps=math.ceil(cost / diff) if cost and diff > noise else None)
    return dict(batch=batch,length=length,seed=seed,route=routes['contiguous'],
                control_noise_us=noise,summary=summary,rounds=rounds)


if __name__ == '__main__':
    root = Path('runs') / datetime.datetime.now().strftime('%Y%m%d-%H%M%S-layout')
    root.mkdir(parents=True)
    repo = Path.home() / 'dev/vllm-lab/vllm-metal'
    env = dict(system=platform.platform(), python=platform.python_version(),
               mlx=importlib.metadata.version('mlx'), device=mx.device_info(),
               metal_head=system(['git','-C',str(repo),'rev-parse','HEAD']),
               metal_status=system(['git','-C',str(repo),'status','--short']),
               load=system(['uptime']), heat=system(['pmset','-g','therm']),
               swap=system(['sysctl','vm.swapusage']),
               sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'env.json').write_text(json.dumps(env, indent=2))
    for seed in (37, 91):
        for batch, length in [(1,1024),(1,8192),(8,1024)]:
            result = measure(batch, length, seed)
            (root/f'{seed}-{batch}-{length}.json').write_text(json.dumps(result,indent=2))
            print(seed, batch, length, result['summary'], flush=True)

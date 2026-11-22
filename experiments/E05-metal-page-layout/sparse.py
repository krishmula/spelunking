"""Sparse-pool and synchronization sensitivity check for E05."""
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


def run(batch, length, seed):
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    count = batch * length // 16
    slots = np.arange(4 * count, dtype=np.int32)
    layouts = dict(contiguous=slots[:count], control=slots[:count],
                   clustered=rng.permutation(slots.reshape(-1, 4))[:count//4].ravel(),
                   shuffled=rng.permutation(slots)[:count])
    base = [mx.random.normal((len(slots), 16, 8, 128), dtype=mx.float16) for _ in range(2)]
    queries = [mx.random.normal((batch, 32, 128), dtype=mx.float16) for _ in range(16)]
    lengths = mx.array([length] * batch, dtype=mx.int32)
    offsets = mx.arange(batch+1, dtype=mx.int32)
    mx.eval(base, queries, lengths, offsets)
    calls, copies = {}, {}
    ops = get_ops()
    for name, selected in layouts.items():
        inverse = np.empty_like(slots)
        inverse[selected] = slots[:count]
        inverse[np.setdiff1d(slots, selected)] = slots[count:]
        keys, values = [a[mx.array(inverse)] for a in base]
        flat = mx.array(selected)
        table = flat.reshape(batch, -1)
        mx.eval(keys, values, flat, table)
        def issue(keys=keys, values=values, table=table):
            outputs = []
            for query in queries:
                out = mx.array(0)
                ops.paged_attention_primitive(query,keys,values,8,128**-0.5,0.,
                    table,lengths,offsets,16,length,-1,out,window_seqlen_q=1,
                    num_decode_requests=batch,num_decode_tokens=batch,max_decode_context_len=length)
                outputs.append(out)
            return outputs
        def copy(keys=keys,values=values,flat=flat):
            return [keys[flat],values[flat]]
        output = issue()
        mx.eval(output)
        if name == 'contiguous':
            expected = [np.array(out) for out in output]
        for out, ref in zip(output, expected, strict=True):
            np.testing.assert_array_equal(np.array(out), ref)
        for packed, original in zip(copy(),base):
            assert mx.array_equal(packed, original[:count]).item()
        calls[name], copies[name] = issue, copy
    warm = time.monotonic()
    while time.monotonic()-warm < 4:
        for op in calls.values():
            sample(op, 1)
    rounds = []
    for _ in range(21):
        att = {n: sample(calls[n],3)/16 for n in rng.permutation(list(calls))}
        rep = {n: sample(copies[n],3) for n in rng.permutation(['clustered','shuffled'])}
        rounds.append(dict(attention_us=att,repack_us=rep))
    def deltas(name):
        return [r['attention_us'][name]-r['attention_us']['contiguous'] for r in rounds]
    noise = np.percentile(np.abs(deltas('control')),90)
    summary = {}
    for name in layouts:
        times = [r['attention_us'][name] for r in rounds]
        cost = median(r['repack_us'][name] for r in rounds) if name in rep else None
        gain = median(deltas(name))
        summary[name] = dict(us=median(times),limits=[min(times), max(times)],saving_us=gain,
                             repack_us=cost,steps=math.ceil(cost/gain) if cost and gain>noise else None)
    return dict(seed=seed,batch=batch,length=length,rounds=rounds,summary=summary,control_noise_us=noise)


if __name__ == '__main__':
    root = Path('runs') / datetime.datetime.now().strftime('%Y%m%d-%H%M%S-sparse')
    root.mkdir(parents=True)
    env = json.loads(next(Path('runs').glob('*-layout/env.json')).read_text())
    env.update(load=system(['uptime']),heat=system(['pmset','-g','therm']),
               swap=system(['sysctl','vm.swapusage']),
               sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'env.json').write_text(json.dumps(env,indent=2))
    for seed in (37,91):
        for batch,length in ((1,1024),(1,8192),(8,1024)):
            data = run(batch,length,seed)
            (root/f'{seed}-{batch}-{length}.json').write_text(json.dumps(data,indent=2))
            print(seed,batch,length,data['summary'],flush=True)

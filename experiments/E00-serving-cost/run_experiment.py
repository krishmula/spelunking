"""E00: what serving Qwen3-4B 4-bit costs on this Air, and where a decode step's time goes.

Runs unattended, start to finish, and leaves a morning summary. Three parts:

  1. Totals. `vllm bench serve` on three workloads (1k x1, 8k x1, 1k x8), REPS passes in a seeded shuffled order,
     a drift pass first and last. Gives time to first token and time per output token.
  2. Context sweep. One streaming request at a time at several context lengths. For each, the GPU driver's
     per-process GPU-time counter is read at the first and last output token, so every decode step splits into
     GPU working time and GPU idle time. GPU time per step against context length is a straight line: the slope is
     attention (the only work that grows with context), the intercept is everything else.
  3. Weights alone. In a fresh process, the model's real 4-bit matmuls at one row (the same MLX kernel the engine
     calls), plus a raw memory-bandwidth test for tonight's floor. Intercept minus weights is KV write plus other.

Usage:
  python run_experiment.py --dry-run      # checks and plan only, touches nothing
  python run_experiment.py --quick        # 5-minute smoke: tiny reps, to test the pipeline
  caffeinate -i python run_experiment.py  # the real run, about 50 minutes

Writes runs/<timestamp>-<label>/: env.json, plan.json, server.log, part1/*.json, part2.jsonl, part3.json,
summary.md (read this first), STATUS (OK or FAILED <reason>).
"""

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = Path(os.environ.get("VLLM_LAB", "~/dev/vllm-lab")).expanduser()
MODEL = "mlx-community/Qwen3-4B-4bit"
PORT = 8011
URL = f"http://127.0.0.1:{PORT}"
SEED = 0

SERVER_ARGS = ["--max-model-len", "9216", "--max-num-seqs", "8", "--max-num-batched-tokens", "2048",
               "--gpu-memory-utilization", "0.38", "--no-enable-prefix-caching", "--seed", str(SEED)]
KV_TOKENS_NEEDED = 8 * (1024 + 128)  # the 1k x8 workload holds 8 sequences at once; 8k+256 is smaller

# Part 1: (name, input_len, output_len, concurrency, num_prompts)
WORKLOADS = [("in1k_c1", 1024, 128, 1, 8), ("in8k_c1", 8192, 128, 1, 4), ("in1k_c8", 1024, 128, 8, 32)]
# Part 2: context lengths and decode tokens per request
SWEEP_LENS = [128, 1024, 2048, 4096, 8192]
SWEEP_OUT = 256

OTHER_GPU_LIMIT = 8.0  # % of wall time other processes may use the GPU before a pass is flagged
SOAK_S = 180
COOLDOWN_S = 20
FULL = {"reps1": 5, "reps2": 5, "soak": SOAK_S, "cooldown": COOLDOWN_S}
QUICK = {"reps1": 1, "reps2": 1, "soak": 10, "cooldown": 2}


# ---------- shell and machine state ----------

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def gpu_time_by_pid():
    """Cumulative GPU nanoseconds per (pid, name), from the GPU driver's client records. No sudo needed."""
    totals, who = {}, None
    for line in sh("ioreg -r -c AGXDeviceUserClient -l -w0").splitlines():
        m = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]+)"', line)
        if m:
            who = (int(m[1]), m[2])
        if '"AppUsage"' in line and who:
            totals[who] = totals.get(who, 0) + sum(int(x) for x in re.findall(r'"accumulatedGPUTime"=(\d+)', line))
    return totals


def descendants(pid):
    kids = [int(p) for p in sh(f"pgrep -P {pid}").split()]
    return {pid, *[d for k in kids for d in descendants(k)]}


def gpu_ns(totals, pids):
    return sum(v for (pid, _), v in totals.items() if pid in pids)


def other_gpu(before, after, seconds, ours):
    """Total % of wall time the GPU spent on processes outside ours, and the top users."""
    use = {f"{n}({p})": (after[(p, n)] - before.get((p, n), 0)) / 1e9 / seconds * 100
           for (p, n) in after if p not in ours}
    use = {k: round(v, 2) for k, v in use.items() if v >= 0.1}
    return round(sum(use.values()), 2), dict(sorted(use.items(), key=lambda kv: -kv[1])[:5])


def snapshot():
    """Machine state worth recording before and after every pass."""
    return {"time": datetime.now().isoformat(timespec="seconds"), "swap": sh("sysctl -n vm.swapusage"),
            "power": sh("pmset -g batt | head -1"), "therm": sh("pmset -g therm | grep -v '^Note' | head -3")}


def environment():
    import mlx.core as mx
    import vllm
    import vllm_metal
    return {
        "time": datetime.now().isoformat(), "machine": sh("sysctl -n machdep.cpu.brand_string"),
        "gpu_cores": sh("ioreg -l | grep -m1 gpu-core-count | grep -oE '[0-9]+$'"),
        "ram_bytes": int(sh("sysctl -n hw.memsize")), "macos": sh("sw_vers -productVersion"),
        "lowpowermode": sh("pmset -g | grep -i lowpowermode"), **snapshot(),
        "versions": {"vllm": vllm.__version__, "vllm_metal": getattr(vllm_metal, "__version__", "?"),
                     "mlx": mx.__version__, "python": sys.version.split()[0]},
        "git": {r: {"sha": sh(f"git -C {ENGINE / r} rev-parse HEAD"),
                    "dirty": sh(f"git -C {ENGINE / r} status --porcelain") != ""} for r in ("vllm-core", "vllm-metal")},
        "model": MODEL, "server_args": SERVER_ARGS,
    }


# ---------- preflight ----------

def preflight(wait_quiet_s=0):
    """Problems that make the run not worth starting. Returns (problems, warnings)."""
    problems, warnings = [], []
    if "1" in sh("pmset -g | grep -i lowpowermode"):
        problems.append("Low Power Mode is on (it slows the GPU)")
    if "AC Power" not in sh("pmset -g batt"):
        warnings.append("on battery; plug in for an overnight run")
    for r in ("vllm-core", "vllm-metal"):
        if sh(f"git -C {ENGINE / r} status --porcelain"):
            problems.append(f"{r} has uncommitted changes; the run would not be reproducible")
    if sh(f"lsof -ti tcp:{PORT}"):
        problems.append(f"port {PORT} is busy (a server left over from an earlier run?)")
    for attempt in range(wait_quiet_s // 30 + 1):  # an unattended start waits for the screen and apps to go quiet
        a = gpu_time_by_pid(); time.sleep(5); b = gpu_time_by_pid()
        total, top = other_gpu(a, b, 5, ours={os.getpid()})
        if total <= OTHER_GPU_LIMIT or attempt == wait_quiet_s // 30:
            break
        print(f"other apps use {total:.1f}% of the GPU ({top}); waiting for quiet", flush=True)
        time.sleep(25)
    if total > OTHER_GPU_LIMIT:  # not a refusal: the passes record their own interference and get flagged
        warnings.append(f"other apps are using {total:.1f}% of the GPU at start: {top}")
    elif total > 0:
        warnings.append(f"other apps use {total:.1f}% of the GPU at idle (screen drawing alone is 3-6%): {top}")
    m = re.search(r"used = ([\d.]+)M", sh("sysctl -n vm.swapusage"))
    if m and float(m[1]) > 4000:
        warnings.append(f"swap already at {float(m[1]) / 1024:.1f} GB; memory pressure may slow the run")
    return problems, warnings


# ---------- server ----------

class Server:
    def __init__(self, out):
        self.log = open(out / "server.log", "w")
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        self.proc = subprocess.Popen(["vllm", "serve", MODEL, "--port", str(PORT), *SERVER_ARGS],
                                     stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.out = out

    def wait_healthy(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode} during startup; see server.log")
            try:
                urllib.request.urlopen(f"{URL}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError(f"server not healthy after {timeout} s")
        text = (self.out / "server.log").read_text()
        m = re.findall(r"KV cache size: ([\d,]+) tokens", text)
        kv = int(m[-1].replace(",", "")) if m else None
        if kv is not None and kv < KV_TOKENS_NEEDED:
            raise RuntimeError(f"KV cache holds {kv} tokens, need {KV_TOKENS_NEEDED}: memory too tight tonight")
        return kv

    def pids(self):
        return descendants(self.proc.pid)

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, 15)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
        self.log.close()


def random_prompt(rng, n):
    return [rng.randrange(1000, 150000) for _ in range(n)]


def stream(prompt_ids, max_tokens, on_first=None, timeout=180):
    """One greedy streaming completion. Returns arrival times of chunks, completion token count, and on_first's value."""
    body = json.dumps({"model": MODEL, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"{URL}/v1/completions", body, {"Content-Type": "application/json"})
    t_send, times, tokens, first = time.perf_counter(), [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
                times.append(time.perf_counter())
                if len(times) == 1 and on_first:
                    first = on_first()
    return t_send, times, tokens, first


# ---------- part 1 ----------

def bench_pass(out, name, inp, outp, conc, n, seed):
    f = f"{name}.json"
    cmd = ["vllm", "bench", "serve", "--backend", "vllm", "--base-url", URL, "--model", MODEL,
           "--dataset-name", "random", "--random-input-len", str(inp), "--random-output-len", str(outp),
           "--num-prompts", str(n), "--max-concurrency", str(conc), "--ignore-eos", "--temperature", "0",
           "--seed", str(seed), "--num-warmups", "1", "--percentile-metrics", "ttft,tpot,itl,e2el",
           "--save-result", "--result-dir", str(out), "--result-filename", f, "--disable-tqdm"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    if r.returncode != 0:
        raise RuntimeError(f"vllm bench serve failed on {name}: {r.stderr[-800:]}")
    return json.loads((out / f).read_text())


def part1(out, server, cfg, log):
    d = out / "part1"
    d.mkdir()
    order = [w for w in WORKLOADS for _ in range(cfg["reps1"])]
    random.Random(SEED).shuffle(order)
    order = [WORKLOADS[0]] + order + [WORKLOADS[0]]  # drift passes: same 1k x1 workload first and last
    rows = []
    for i, (name, inp, outp, conc, n) in enumerate(order):
        tag = "drift_start" if i == 0 else "drift_end" if i == len(order) - 1 else "pass"
        ours, before, t0, snap = server.pids(), gpu_time_by_pid(), time.time(), snapshot()
        res = bench_pass(d, f"{i:02d}_{name}_{tag}", inp, outp, conc, n, seed=SEED + i)
        secs = time.time() - t0
        other, top = other_gpu(before, gpu_time_by_pid(), secs, ours | {os.getpid()})
        row = {"i": i, "workload": name, "tag": tag, "secs": round(secs, 1), "other_gpu_pct": other, "other_top": top,
               "flagged": other > OTHER_GPU_LIMIT, "state_before": snap,
               **{k: res.get(k) for k in ("output_throughput", "median_ttft_ms", "median_tpot_ms", "median_itl_ms",
                                          "p99_itl_ms", "median_e2el_ms")}}
        rows.append(row)
        (out / "part1.jsonl").open("a").write(json.dumps(row) + "\n")
        log(f"part 1 [{i + 1}/{len(order)}] {name} {tag}: tpot {row['median_tpot_ms']:.1f} ms, "
            f"ttft {row['median_ttft_ms']:.0f} ms, other GPU {other}%")
        time.sleep(cfg["cooldown"])
    return rows


# ---------- part 2 ----------

def part2(out, server, cfg, log):
    rng = random.Random(SEED + 1000)
    order = [L for L in SWEEP_LENS for _ in range(cfg["reps2"])]
    rng.shuffle(order)
    rows = []
    for i, L in enumerate(order):
        ours = server.pids()

        def read_counter():
            t_a = time.perf_counter()
            totals = gpu_time_by_pid()
            return (t_a + time.perf_counter()) / 2, gpu_ns(totals, ours), totals

        t_send, times, tokens, first = stream(random_prompt(rng, L), SWEEP_OUT, on_first=read_counter)
        t_mid, g_first, totals_first = first
        totals_last = gpu_time_by_pid()
        g_last = gpu_ns(totals_last, ours)
        t_last = times[-1]
        steps = (tokens or len(times)) - 1
        wall_step = (t_last - times[0]) / steps * 1000
        busy = (g_last - g_first) / 1e9 / (t_last - t_mid)
        other, _ = other_gpu(totals_first, totals_last, t_last - t_mid, ours | {os.getpid()})
        row = {"i": i, "context": L, "tokens": tokens, "ttft_ms": round((times[0] - t_send) * 1000, 1),
               "wall_ms_per_step": round(wall_step, 3), "gpu_busy": round(busy, 4),
               "gpu_ms_per_step": round(wall_step * busy, 3), "idle_ms_per_step": round(wall_step * (1 - busy), 3),
               "other_gpu_pct": other, "flagged": other > OTHER_GPU_LIMIT}
        rows.append(row)
        (out / "part2.jsonl").open("a").write(json.dumps(row) + "\n")
        log(f"part 2 [{i + 1}/{len(order)}] context {L}: {wall_step:.2f} ms/step, GPU busy {busy * 100:.1f}%")
        time.sleep(cfg["cooldown"] / 4)
    return rows


# ---------- part 3 (runs in its own process, after the server is gone) ----------

def part3(path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    model, _ = load(MODEL)
    mats = []  # (kind, w, scales, biases, group_size, bits)
    for name, m in model.named_modules():
        kind = name.split(".")[-1]
        if isinstance(m, nn.QuantizedLinear):
            mats.append((kind, m.weight, m.scales, m.get("biases"), m.group_size, m.bits))
        elif isinstance(m, nn.QuantizedEmbedding):  # tied embeddings: the output projection reuses this matrix
            mats.append(("lm_head", m.weight, m.scales, m.get("biases"), m.group_size, m.bits))
    dtype = mats[0][2].dtype

    def nbytes(group):
        return sum(w.nbytes + s.nbytes + (b.nbytes if b is not None else 0) for _, w, s, b, _, _ in group)

    def time_group(group, iters=40):
        xs = [mx.random.normal((1, w.shape[1] * 32 // bits)).astype(dtype) for _, w, _, _, _, bits in group]
        mx.eval(xs)

        def once():
            mx.eval([mx.quantized_matmul(x, w, s, b, transpose=True, group_size=g, bits=bits)
                     for x, (_, w, s, b, g, bits) in zip(xs, group)])
        for _ in range(5):
            once()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); once(); ts.append((time.perf_counter() - t0) * 1000)
        return statistics.median(ts), min(ts)

    a = mx.random.uniform(shape=(512 * 1024 * 1024,)).astype(mx.float16)  # 1 GiB
    mx.eval(a)
    bw = []
    for _ in range(25):
        t0 = time.perf_counter(); mx.eval(a.sum()); bw.append(a.nbytes / (time.perf_counter() - t0) / 1e9)
    del a
    result = {"bandwidth_gbs_median": round(statistics.median(bw[5:]), 1), "bandwidth_gbs_max": round(max(bw), 1),
              "kinds": {}}
    for kind in sorted({k for k, *_ in mats}):
        group = [m for m in mats if m[0] == kind]
        med, best = time_group(group)
        result["kinds"][kind] = {"count": len(group), "mb": round(nbytes(group) / 1e6, 1),
                                 "ms_median": round(med, 3), "ms_min": round(best, 3)}
    med, best = time_group(mats)
    result["all"] = {"count": len(mats), "mb": round(nbytes(mats) / 1e6, 1), "ms_median": round(med, 3),
                     "ms_min": round(best, 3)}
    result["floor_ms_at_best_bandwidth"] = round(nbytes(mats) / 1e9 / result["bandwidth_gbs_max"] * 1000, 2)
    Path(path).write_text(json.dumps(result, indent=1))


# ---------- summary ----------

def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def clean(rows):
    """Rows where other apps stayed under the GPU limit; all rows if fewer than two are clean (the summary says so)."""
    ok = [r for r in rows if not r["flagged"]]
    return ok if len(ok) >= 2 or len(ok) == len(rows) else rows


def fit(rows):
    """Least-squares GPU ms per step = intercept + slope * context."""
    xs, ys = [r["context"] for r in rows], [r["gpu_ms_per_step"] for r in rows]
    slope, intercept = statistics.linear_regression(xs, ys)
    return intercept, slope


def summary(out, p1, p2, p3, status, kv):
    L = [f"# E00 run {out.name}", "", f"STATUS: {status}", "", f"KV cache: {kv} tokens", ""]
    if p1:
        L += ["## Part 1: totals", "", "Medians over the recorded passes (drift passes excluded; flagged passes, where other apps used more than "
              f"{OTHER_GPU_LIMIT}% of the GPU, are left out unless fewer than two are clean). Spread is "
              "(max - min) / median of output tokens per second.", "",
              "| Workload | Passes used | Time per token (ms) | Inter-token median (ms) | Time to first token (ms) | "
              "Output tok/s | Spread | Flagged passes |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, *_ in WORKLOADS:
            rs = clean([r for r in p1 if r["workload"] == name and r["tag"] == "pass"])
            if not rs:
                continue
            tput = [r["output_throughput"] for r in rs]
            spread = (max(tput) - min(tput)) / statistics.median(tput) * 100
            nflag = sum(r["flagged"] for r in p1 if r["workload"] == name and r["tag"] == "pass")
            L.append(f"| {name} | {len(rs)} | {med(r['median_tpot_ms'] for r in rs):.2f} | "
                     f"{med(r['median_itl_ms'] for r in rs):.2f} | {med(r['median_ttft_ms'] for r in rs):.0f} | "
                     f"{statistics.median(tput):.2f} | {spread:.1f}% | {nflag} |")
        drift = [r for r in p1 if r["tag"].startswith("drift")]
        if len(drift) == 2:
            a, b = drift[0]["output_throughput"], drift[1]["output_throughput"]
            L += ["", f"Drift (1k x1, first vs last pass): {a:.2f} -> {b:.2f} tok/s ({(b - a) / a * 100:+.1f}%). "
                  "A big drop means the chip got hotter over the run."]
        L += ["", f"Swap at start / end: {p1[0]['state_before']['swap']} / {p1[-1]['state_before']['swap']}", ""]
    if p2:
        L += ["## Part 2: GPU working vs idle, by context length", "",
              "Idle here is time our process had no GPU work running. Other apps' GPU work also shows up as our idle, "
              "so read it next to the last column.", "",
              "| Context | Reps used | Wall ms/step | GPU ms/step | Idle ms/step | GPU busy | Other apps' GPU |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        good = [r for c in SWEEP_LENS for r in clean([r for r in p2 if r["context"] == c])]
        for c in SWEEP_LENS:
            rs = [r for r in good if r["context"] == c]
            if rs:
                L.append(f"| {c} | {len(rs)} | {med(r['wall_ms_per_step'] for r in rs):.2f} | "
                         f"{med(r['gpu_ms_per_step'] for r in rs):.2f} | {med(r['idle_ms_per_step'] for r in rs):.2f} | "
                         f"{med(r['gpu_busy'] for r in rs) * 100:.1f}% | {med(r['other_gpu_pct'] for r in rs):.1f}% |")
        if len({r["context"] for r in good}) >= 2:
            a, b = fit(good)
            L += ["", f"Fit: GPU ms per step = {a:.2f} + {b * 1000:.3f} x (context / 1000). The slope is attention; "
                  "the intercept is everything that doesn't grow with context.", ""]
    if p3:
        L += ["## Part 3: weights alone", "",
              f"Memory bandwidth tonight: {p3['bandwidth_gbs_median']} GB/s median, {p3['bandwidth_gbs_max']} best. "
              f"Weight bytes: {p3['all']['mb']} MB. Floor at the best bandwidth: {p3['floor_ms_at_best_bandwidth']} ms.",
              "", "| Matmul | Count | MB | ms (median) |", "|---|---:|---:|---:|"]
        for k, v in p3["kinds"].items():
            L.append(f"| {k} | {v['count']} | {v['mb']} | {v['ms_median']:.2f} |")
        L.append(f"| all, one pass | {p3['all']['count']} | {p3['all']['mb']} | {p3['all']['ms_median']:.2f} |")
        L.append("")
    good = [r for c in SWEEP_LENS for r in clean([r for r in p2 if r["context"] == c])] if p2 else []
    if p3 and len({r["context"] for r in good}) >= 2:
        a, b = fit(good)
        w = p3["all"]["ms_median"]
        L += ["## The bill", "", "Weights from part 3; attention = slope x context; KV write + other = intercept - "
              "weights; idle from part 2.", "", "| | 1k step | 8k step |", "|---|---:|---:|",
              f"| Weights | {w:.2f} ms | {w:.2f} ms |",
              f"| Attention | {b * 1024:.2f} ms | {b * 8192:.2f} ms |",
              f"| KV write + other | {a - w:.2f} ms | {a - w:.2f} ms |"]
        idle = {c: med(r["idle_ms_per_step"] for r in good if r["context"] == c) for c in (1024, 8192)}
        L.append(f"| GPU idle | {idle[1024] or 0:.2f} ms | {idle[8192] or 0:.2f} ms |")
        L.append(f"| Sum | {w + b * 1024 + (a - w) + (idle[1024] or 0):.2f} ms | "
                 f"{w + b * 8192 + (a - w) + (idle[8192] or 0):.2f} ms |")
        L += ["", "Check: the Sum should match part 2's wall ms/step at 1k and 8k. A negative 'KV write + other' "
              "means weights run faster alone than inside the engine."]
    (out / "summary.md").write_text("\n".join(L) + "\n")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="run the checks and print the plan; touch nothing")
    p.add_argument("--quick", action="store_true", help="tiny reps, to test the pipeline (results not usable)")
    p.add_argument("--force", action="store_true", help="start even if a preflight check fails")
    p.add_argument("--part3-only", metavar="OUT_JSON", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.part3_only:
        return part3(args.part3_only)

    cfg = QUICK if args.quick else FULL
    n1 = (len(WORKLOADS) * cfg["reps1"] + 2)
    n2 = len(SWEEP_LENS) * cfg["reps2"]
    plan = {"model": MODEL, "server_args": SERVER_ARGS, "seed": SEED, "config": cfg,
            "part1": f"{n1} bench passes: {[w[0] for w in WORKLOADS]} x {cfg['reps1']} shuffled, plus drift first and last",
            "part2": f"{n2} streaming requests: contexts {SWEEP_LENS} x {cfg['reps2']} shuffled, {SWEEP_OUT} tokens each",
            "part3": "weights-only matmuls and a bandwidth test in a fresh process",
            "estimate_min": 50 if not args.quick else 8}
    problems, warnings = preflight(wait_quiet_s=0 if args.dry_run or args.force else 1800)
    print(json.dumps(plan, indent=1))
    for w in warnings:
        print("warning:", w)
    for pr in problems:
        print("PROBLEM:", pr)
    if args.dry_run:
        return 1 if problems else 0
    if problems and not args.force:
        sys.exit("refusing to start; fix the problems above or pass --force")

    out = HERE / "runs" / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{'quick' if args.quick else 'full'}"
    out.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps({**plan, "warnings": warnings, "problems_forced": problems}, indent=1))
    (out / "env.json").write_text(json.dumps(environment(), indent=1))
    logf = (out / "progress.log").open("a")

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    p1 = p2 = p3 = None
    kv, status, server = None, "OK", None
    try:
        server = Server(out)
        log("starting server")
        kv = server.wait_healthy()
        log(f"server healthy, KV cache {kv} tokens; heat soak {cfg['soak']} s")
        rng, t0 = random.Random(SEED + 2000), time.time()
        while time.time() - t0 < cfg["soak"]:
            stream(random_prompt(rng, 1024), 128)
        p1 = part1(out, server, cfg, log)
        p2 = part2(out, server, cfg, log)
        server.stop(); server = None
        log("server stopped; part 3 in a fresh process")
        time.sleep(cfg["cooldown"])
        subprocess.run([sys.executable, __file__, "--part3-only", str(out / "part3.json")], check=True, timeout=1200,
                       env={**os.environ, "HF_HUB_OFFLINE": "1"})
        p3 = json.loads((out / "part3.json").read_text())
        log("part 3 done")
    except BaseException as e:
        status = f"FAILED {type(e).__name__}: {e}"
        (out / "error.txt").write_text(traceback.format_exc())
        log(status)
    finally:
        if server:
            server.stop()
        try:
            summary(out, p1, p2, p3, status, kv)
        except Exception:
            (out / "summary_error.txt").write_text(traceback.format_exc())
        (out / "STATUS").write_text(status + "\n")
        log(f"done: {status}. Read {out / 'summary.md'}")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())

"""E02: how often vLLM preempts on this Air, and how much of a preempted request the prefix cache gives back.

Runs unattended and leaves a summary. Four configs, each on a fresh server (empty prefix cache): batch 8 and 16,
output caps 2k and 4k, thinking on, prefix caching on, E00's pool size. The scheduler is vLLM's own with logging
added (e02_scheduler.py); it decides exactly what stock vLLM decides.

Usage:
  python run_experiment.py --dry-run           # checks and plan only, touches nothing
  python run_experiment.py --smoke --out DIR   # a few minutes with a tiny pool, to test the pipeline
  caffeinate -i python run_experiment.py       # the real run, at most about 5 hours
  python run_experiment.py --analyze RUN_DIR   # recompute summary.md from the logged events

Writes runs/<timestamp>-<label>/: env.json, plan.json, progress.log, <config>/{server.log, events.jsonl,
requests.jsonl}, summary.json, summary.md (read this first), STATUS (OK or FAILED <reason>).
"""

import argparse
import asyncio
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
PORT = 8012
URL = f"http://127.0.0.1:{PORT}"
SEED = 0
E00_POOL_TOKENS = 12416  # E00's pool at --gpu-memory-utilization 0.38

BASE_ARGS = ["--max-model-len", "9216", "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.38",
             "--enable-prefix-caching", "--async-scheduling", "--seed", str(SEED),
             "--scheduler-cls", "e02_scheduler.LoggingScheduler"]

# (name, batch size, output cap, requests, deadline in minutes)
CONFIGS = [("b8-cap2k", 8, 2048, 64, 70), ("b16-cap2k", 16, 2048, 64, 70),
           ("b8-cap4k", 8, 4096, 64, 75), ("b16-cap4k", 16, 4096, 64, 75)]
SMOKE = [("smoke-b4-cap512", 4, 512, 8, 6)]
SMOKE_ARGS = ["--num-gpu-blocks-override", "96", "--max-model-len", "1024"]  # 1,536 tokens: forces preemption in a few minutes

OTHER_GPU_LIMIT = 8.0
COOLDOWN_S = 60


# ---------- shell and machine state (copied from E00) ----------

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def gpu_time_by_pid():
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


def other_gpu(before, after, seconds, ours):
    use = {f"{n}({p})": (after[(p, n)] - before.get((p, n), 0)) / 1e9 / seconds * 100
           for (p, n) in after if p not in ours}
    use = {k: round(v, 2) for k, v in use.items() if v >= 0.1}
    return round(sum(use.values()), 2), dict(sorted(use.items(), key=lambda kv: -kv[1])[:5])


def snapshot():
    return {"time": datetime.now().isoformat(timespec="seconds"), "swap": sh("sysctl -n vm.swapusage"),
            "power": sh("pmset -g batt | head -1"), "therm": sh("pmset -g therm | grep -v '^Note' | head -3")}


def environment(configs, extra_args):
    import mlx.core as mx
    import vllm
    import vllm_metal
    return {
        "time": datetime.now().isoformat(), "machine": sh("sysctl -n machdep.cpu.brand_string"),
        "ram_bytes": int(sh("sysctl -n hw.memsize")), "macos": sh("sw_vers -productVersion"),
        "lowpowermode": sh("pmset -g | grep -i lowpowermode"), **snapshot(),
        "versions": {"vllm": vllm.__version__, "vllm_metal": getattr(vllm_metal, "__version__", "?"),
                     "mlx": mx.__version__, "python": sys.version.split()[0]},
        "git": {r: {"sha": sh(f"git -C {ENGINE / r} rev-parse HEAD"),
                    "dirty": sh(f"git -C {ENGINE / r} status --porcelain") != ""} for r in ("vllm-core", "vllm-metal")},
        "model": MODEL, "server_args": BASE_ARGS + extra_args, "configs": configs,
    }


def preflight():
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
    a = gpu_time_by_pid(); time.sleep(5); b = gpu_time_by_pid()
    total, top = other_gpu(a, b, 5, ours={os.getpid()})
    if total > OTHER_GPU_LIMIT:
        warnings.append(f"other apps are using {total:.1f}% of the GPU at start: {top}")
    m = re.search(r"used = ([\d.]+)M", sh("sysctl -n vm.swapusage"))
    if m and float(m[1]) > 4000:
        warnings.append(f"swap already at {float(m[1]) / 1024:.1f} GB; memory pressure may slow the run")
    return problems, warnings


# ---------- server ----------

class Server:
    def __init__(self, out, args):
        self.out = out
        self.log = open(out / "server.log", "w")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "E02_LOG": str(out / "events.jsonl"),
               "PYTHONPATH": os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))}
        self.proc = subprocess.Popen(["vllm", "serve", MODEL, "--port", str(PORT), *args],
                                     stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

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
        if "LoggingScheduler" not in text and not (self.out / "events.jsonl").exists():
            raise RuntimeError("the logging scheduler did not load; see server.log")
        m = re.findall(r"KV cache size: ([\d,]+) tokens", text)
        return int(m[-1].replace(",", "")) if m else None

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


# ---------- workload ----------

TEMPLATES = [
    "How many positive integers less than {n} are divisible by {a} or {b} but not by {c}?",
    "Find the remainder when {a}^{e} is divided by {m}.",
    "A bag has {r} red, {g} green and {b} blue marbles. You draw {k} marbles without replacement. "
    "What is the probability that exactly {j} of them are red? Give the answer as a reduced fraction.",
    "Find all integer pairs (x, y) with x^2 + {k}*y^2 = {n}.",
    "In how many ways can {n} people sit around a round table if {a} particular people must not sit next to "
    "each other? Rotations count as the same seating.",
    "What is the sum of the digits of all the integers from 1 to {n}?",
    "A sequence starts a_1 = {a} and continues a_(n+1) = (a_n^2 + {b}*a_n + {c}) mod {m}. Find a_{k}.",
    "Two trains start {d} km apart and drive toward each other at {v1} km/h and {v2} km/h. A bird flies back "
    "and forth between them at {v3} km/h until they meet. How far does the bird fly, and how many full trips "
    "does it complete if each trip takes it from one train to the other?",
]


def prompts(n):
    rng = random.Random(SEED)
    out = []
    for i in range(n):
        t = TEMPLATES[i % len(TEMPLATES)]
        out.append(t.format(n=rng.randrange(500, 5000), a=rng.randrange(3, 12), b=rng.randrange(13, 30),
                            c=rng.randrange(31, 60), e=rng.randrange(100, 2000), m=rng.randrange(13, 200),
                            r=rng.randrange(3, 9), g=rng.randrange(3, 9), k=rng.randrange(3, 7),
                            j=rng.randrange(1, 3), d=rng.randrange(100, 400), v1=rng.randrange(40, 90),
                            v2=rng.randrange(40, 90), v3=rng.randrange(100, 200)))
    return out


async def drive(batch, cap, texts, deadline_s, out):
    """Send every prompt with at most `batch` in flight. Stops starting new ones at the deadline."""
    import aiohttp
    sem = asyncio.Semaphore(batch)
    t_end = time.monotonic() + deadline_s
    records = []

    async def one(session, i, text):
        async with sem:
            if time.monotonic() > t_end:
                return
            body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": cap,
                    "temperature": 0.6, "top_p": 0.95, "seed": SEED + i}
            t0 = time.monotonic()
            try:
                async with session.post(f"{URL}/v1/chat/completions", json=body) as r:
                    data = await r.json()
                choice = data["choices"][0]
                rec = {"i": i, "ok": True, "seconds": time.monotonic() - t0, "finish": choice["finish_reason"],
                       "prompt_tokens": data["usage"]["prompt_tokens"],
                       "output_tokens": data["usage"]["completion_tokens"]}
            except Exception as e:
                rec = {"i": i, "ok": False, "seconds": time.monotonic() - t0, "error": repr(e)[:300]}
            records.append(rec)
            with open(out / "requests.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")

    timeout = aiohttp.ClientTimeout(total=None, sock_read=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(one(session, i, t)) for i, t in enumerate(texts)]
        _, pending = await asyncio.wait(tasks, timeout=deadline_s + 900)
        for t in pending:
            t.cancel()
    return records


# ---------- analysis ----------

def surviving_prefix(c, lost_count):
    """Leading blocks of candidate c that a resume could hit, if its last `lost_count` freed hashed blocks are gone.

    vLLM frees a request's blocks last block first: unhashed ones go to the head of the free queue, hashed ones to
    the tail. So the tail end of a request's hashed blocks is reused first, and the prefix survives longest.
    Blocks shared with another request (ref count above 1) are not freed and always survive."""
    freed_hashed = [i for i, (h, r) in enumerate(zip(c["hashed"], c["refs"])) if h and r == 1]
    lost = set(freed_hashed[len(freed_hashed) - lost_count:]) if lost_count else set()
    n = 0
    for i, h in enumerate(c["hashed"]):
        if not h or i in lost:
            break
        n += 1
    return n


def predicted_lost(c, free_before, window):
    """Hashed blocks of c that `window` pops would reuse: pops take c's unhashed blocks, then the old free queue,
    then c's hashed blocks. Ignores blocks freed by others during the window (checked against the real victim)."""
    unhashed = sum(1 for h, r in zip(c["hashed"], c["refs"]) if not h and r == 1)
    hashed = sum(1 for h, r in zip(c["hashed"], c["refs"]) if h and r == 1)
    return max(0, min(hashed, window - unhashed - free_before))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def analyze_config(d):
    ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    reqs = [json.loads(l) for l in (d / "requests.jsonl").read_text().splitlines()] if (d / "requests.jsonl").exists() else []
    start = next(e for e in ev if e["ev"] == "start")
    bs = start["block_size"]
    steps = [e for e in ev if e["ev"] == "step"]
    pre = [e for e in ev if e["ev"] == "preempt"]
    res = [e for e in ev if e["ev"] == "resume"]
    fin = [e for e in ev if e["ev"] == "finish"]

    # Step time: the gap between consecutive schedule() calls while busy.
    rows = []
    for a, b in zip(steps, steps[1:]):
        dt = b["t"] - a["t"]
        if a["tokens"] > 0 and dt < 5.0:
            rows.append((dt, a["tokens"], a["reprefill"]))
    busy = sum(r[0] for r in rows)
    prop = sum(dt * rp / tok for dt, tok, rp in rows)
    slope = None
    if len(rows) > 10:
        mx = statistics.fmean(r[1] for r in rows); my = statistics.fmean(r[0] for r in rows)
        sxx = sum((r[1] - mx) ** 2 for r in rows)
        slope = sum((r[1] - mx) * (r[0] - my) for r in rows) / sxx if sxx else None
    marginal = slope * sum(r[2] for r in rows) if slope else None

    # Re-hits: what the prefix cache gave back on resume.
    rehit = [r["hit_tokens"] / r["computed_at_preempt"] for r in res if r["computed_at_preempt"] > 0]

    # Model check and candidate spread, using the real victim's wait as the window for every candidate.
    by_id = {}
    for p in pre:
        by_id.setdefault(p["victim"]["id"], []).append(p)
    checks, spreads, lost_spreads = [], [], []
    for r in res:
        p = by_id.get(r["id"], []).pop(0) if by_id.get(r["id"]) else None
        if p is None:
            continue
        v, w = p["victim"], r["pops_since_preempt"]
        real_lost = sum(1 for b, h, rc in zip(v["blocks"], v["hashed"], v["refs"])
                        if h and rc == 1 and b in set(r["old_blocks_reused"]))
        checks.append((predicted_lost(v, p["free_before"], w), real_lost,
                       surviving_prefix(v, predicted_lost(v, p["free_before"], w)), r["hit_tokens"] // bs))
        cands = [v] + p["candidates"]
        surv = [surviving_prefix(c, predicted_lost(c, p["free_before"], w)) for c in cands]
        lost = [predicted_lost(c, p["free_before"], w) for c in cands]
        if len(cands) >= 2:
            hi, lo = max(surv), min(surv)
            spreads.append(1.0 if hi == lo else (float("inf") if lo == 0 else hi / lo))
            hi, lo = max(lost), min(lost)
            lost_spreads.append(1.0 if hi == lo else (float("inf") if lo == 0 else hi / lo))

    done = [q for q in reqs if q.get("ok")]
    n_req = max(len(fin), len(done))
    return {
        "requests_finished": n_req, "requests_failed": sum(1 for q in reqs if not q.get("ok")),
        "output_tokens_median": statistics.median(q["output_tokens"] for q in done) if done else None,
        "hit_cap_share": (sum(1 for q in done if q["finish"] == "length") / len(done)) if done else None,
        "pool_blocks": start["num_blocks"], "block_size": bs,
        "preemptions": len(pre), "resumes": len(res),
        "preemptions_per_100_requests": 100 * len(pre) / n_req if n_req else None,
        "requests_preempted_share": (sum(1 for e in fin if e["preemptions"] > 0) / len(fin)) if fin else None,
        "busy_seconds": busy, "steps": len(rows),
        "reprefill_tokens": sum(r[2] for r in rows), "all_tokens": sum(r[1] for r in rows),
        "reprefill_time_share_proportional": prop / busy if busy else None,
        "seconds_per_token_slope": slope,
        "reprefill_time_share_marginal": marginal / busy if marginal is not None and busy else None,
        "rehit_share": {"n": len(rehit), "min": min(rehit, default=None), "p10": pct(rehit, 0.1),
                        "median": pct(rehit, 0.5), "p90": pct(rehit, 0.9), "max": max(rehit, default=None),
                        "above_0.8": sum(x > 0.8 for x in rehit) / len(rehit) if rehit else None,
                        "below_0.2": sum(x < 0.2 for x in rehit) / len(rehit) if rehit else None},
        "free_before_median": statistics.median(p["free_before"] for p in pre) if pre else None,
        "candidates_median": statistics.median(len(p["candidates"]) + 1 for p in pre) if pre else None,
        "replay_check": {"n": len(checks),
                         "lost_exact_share": sum(a == b for a, b, _, _ in checks) / len(checks) if checks else None,
                         "lost_mean_abs_error": statistics.fmean(abs(a - b) for a, b, _, _ in checks) if checks else None,
                         "surviving_exact_share": sum(c == d for _, _, c, d in checks) / len(checks) if checks else None},
        "spread_surviving": {"n": len(spreads), "median": pct(spreads, 0.5),
                             "share_at_least_2x": sum(s >= 2 for s in spreads) / len(spreads) if spreads else None},
        "spread_lost": {"n": len(lost_spreads), "median": pct(lost_spreads, 0.5),
                        "share_at_least_2x": sum(s >= 2 for s in lost_spreads) / len(lost_spreads) if lost_spreads else None},
    }


def verdict(s):
    """The conditions fixed in research/R03-kv-cache-design-space/decisions.tsv before the run."""
    rate, share = s["preemptions_per_100_requests"], s["reprefill_time_share_marginal"]
    rh, sp = s["rehit_share"], s["spread_surviving"]["median"]
    if not s["requests_finished"]:
        return ["no data"]
    out = []
    if rate is None or rate < 1:
        out.append("KILL: under 1 preemption per 100 requests")
    if share is not None and share < 0.05:
        out.append("KILL: re-prefill under 5% of GPU time")
    if sp is not None and sp < 2:
        out.append("KILL victim-policy direction: candidates differ by under 2x in surviving blocks (median)")
    if rh["n"] and (rh["min"] > 0.8 or rh["max"] < 0.2):
        out.append("PIVOT: re-hit share above 0.8 or below 0.2 for every victim")
    if rate is not None and rate >= 1 and share is not None and share >= 0.10 and sp is not None and sp >= 2:
        out.append("GO: victim-policy oracle")
    return out or ["no rule fired (between thresholds)"]


def fmt(x, p=2):
    return "n/a" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def write_summary(run_dir, results, meta):
    (run_dir / "summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    lines = [f"# E02 summary: {run_dir.name}", "",
             "Do not read past this line until the prediction table in README.md is filled in.", "",
             f"Pool: {meta.get('pool_tokens')} tokens. Thresholds from R03 decisions.tsv (2026-10-12T09:30).", ""]
    for name, s in results.items():
        if "error" in s:
            lines += [f"## {name}", "", f"FAILED: {s['error']}", ""]
            continue
        rh = s["rehit_share"]
        lines += [f"## {name}", "",
                  f"- Verdict: {'; '.join(verdict(s))}",
                  f"- Requests finished {s['requests_finished']} (failed {s['requests_failed']}), median output "
                  f"{fmt(s['output_tokens_median'])} tokens, {fmt(s['hit_cap_share'])} hit the cap",
                  f"- Preemptions {s['preemptions']} ({fmt(s['preemptions_per_100_requests'], 1)} per 100 requests); "
                  f"{fmt(s['requests_preempted_share'])} of requests preempted at least once",
                  f"- Re-prefill: {s['reprefill_tokens']} of {s['all_tokens']} scheduled tokens; time share "
                  f"{fmt(s['reprefill_time_share_marginal'], 3)} (marginal cost) / "
                  f"{fmt(s['reprefill_time_share_proportional'], 3)} (token-proportional) of {fmt(s['busy_seconds'], 0)} s busy",
                  f"- Re-hit share over {rh['n']} resumes: min {fmt(rh['min'])}, p10 {fmt(rh['p10'])}, median "
                  f"{fmt(rh['median'])}, p90 {fmt(rh['p90'])}, max {fmt(rh['max'])}",
                  f"- Free blocks at preemption (median) {fmt(s['free_before_median'])}; candidates per preemption "
                  f"(median) {fmt(s['candidates_median'])}",
                  f"- Replay check on the real victim ({s['replay_check']['n']}): lost blocks exact "
                  f"{fmt(s['replay_check']['lost_exact_share'])}, mean abs error {fmt(s['replay_check']['lost_mean_abs_error'])}; "
                  f"surviving prefix exact {fmt(s['replay_check']['surviving_exact_share'])}",
                  f"- Candidate spread in surviving blocks: median {fmt(s['spread_surviving']['median'])}x, "
                  f"{fmt(s['spread_surviving']['share_at_least_2x'])} of preemptions at least 2x "
                  f"(lost blocks: median {fmt(s['spread_lost']['median'])}x)", ""]
    (run_dir / "summary.md").write_text("\n".join(lines))


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--analyze", type=Path)
    a = ap.parse_args()

    if a.analyze:
        meta = json.loads((a.analyze / "plan.json").read_text())
        results = {}
        for name, *_ in meta["configs"]:
            try:
                results[name] = analyze_config(a.analyze / name)
            except Exception as e:
                results[name] = {"error": repr(e)}
        write_summary(a.analyze, results, meta)
        print((a.analyze / "summary.md").read_text())
        return

    configs = SMOKE if a.smoke else CONFIGS
    extra = SMOKE_ARGS if a.smoke else []
    problems, warnings = preflight()
    for w in warnings:
        print("warning:", w)
    if problems:
        sys.exit("cannot start: " + "; ".join(problems))
    if a.dry_run:
        print(json.dumps({"configs": configs, "args": BASE_ARGS + extra, "first_prompt": prompts(1)[0]}, indent=1))
        return

    label = "smoke" if a.smoke else "full"
    run_dir = a.out or HERE / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}-{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(environment(configs, extra), indent=1))
    meta = {"configs": configs, "warnings": warnings, "pool_tokens": None}
    (run_dir / "plan.json").write_text(json.dumps(meta, indent=1))
    log = open(run_dir / "progress.log", "a", buffering=1)

    def say(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True); log.write(line + "\n")

    results, failed = {}, None
    for name, batch, cap, n, deadline_min in configs:
        d = run_dir / name
        d.mkdir(exist_ok=True)
        server = Server(d, BASE_ARGS + extra + ["--max-num-seqs", str(batch)])
        try:
            pool = server.wait_healthy()
            meta["pool_tokens"] = pool
            (run_dir / "plan.json").write_text(json.dumps(meta, indent=1))
            say(f"{name}: server up, pool {pool} tokens; {snapshot()}")
            if not a.smoke and pool != E00_POOL_TOKENS:
                say(f"{name}: WARNING pool is {pool} tokens, E00 had {E00_POOL_TOKENS}")
            g0, t0 = gpu_time_by_pid(), time.time()
            recs = asyncio.run(drive(batch, cap, prompts(n), deadline_min * 60, d))
            other, top = other_gpu(g0, gpu_time_by_pid(), time.time() - t0, server.pids() | {os.getpid()})
            if not any(r["ok"] for r in recs):
                raise RuntimeError("no request succeeded; see server.log and requests.jsonl")
            say(f"{name}: {sum(r['ok'] for r in recs)}/{n} requests done in {time.time() - t0:.0f} s; "
                f"other apps {other}% of GPU {top}")
        except Exception as e:
            say(f"{name}: FAILED {e!r}\n{traceback.format_exc()}")
            failed = failed or f"{name}: {e!r}"
        finally:
            server.stop()
        try:
            results[name] = analyze_config(d)
        except Exception as e:
            results[name] = {"error": repr(e)}
        if not a.smoke:
            time.sleep(COOLDOWN_S)

    write_summary(run_dir, results, meta)
    (run_dir / "STATUS").write_text("OK\n" if not failed else f"FAILED {failed}\n")
    say(f"done: {(run_dir / 'STATUS').read_text().strip()}")


if __name__ == "__main__":
    main()

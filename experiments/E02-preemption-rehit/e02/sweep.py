"""Fixed configuration, sealed plans and manifests, readiness gates, owned server process groups,
streaming client, warmup, budgets, attempts, pilots and resume for the expanded E02.

Run roots live under runs/<YYYYMMDD-HHMMSS>-<label>/ with env.json, a plan.json or manifest.json
sealed by a .sha256 sidecar, and attempts/<run_id>.a<k>/. An attempt moves from running to
validating to one final outcome, written once together with digests of its evidence files. A
replacement is a new attempt naming the one it replaces and why.

Workload execution is suspended; see AUTHORITY.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import typing
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

from e02 import corpus as corpus_mod
from e02 import history

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
RUNS = HERE / "runs"
TECHNICAL = HERE / "technical"
README = HERE / "README.md"
ENGINE = Path(os.environ.get("VLLM_LAB", "~/dev/vllm-lab")).expanduser()
PYTHON = ENGINE / ".venv" / "bin" / "python"
VLLM = ENGINE / ".venv" / "bin" / "vllm"
MODEL = corpus_mod.MODEL
PORT = 8013

AUTHORITY = (
    "Workload execution is suspended. On 2026-10-20 at 14:33 PDT Krishna said \"Don't run the experiment "
    "unless I tell you so.\" `pilot` and `main` start a model server and GPU work only with --execute. Pass "
    "--execute only after Krishna's new, explicit instruction to run that specific workload. The flag records "
    "that instruction; it does not grant it.")

TARGETS = (4, 8, 16, 24, 32)
CAPS = (512, 1024, 2048, 4096)
REP_SEEDS = (11, 23, 37, 53, 71)
REQUESTS = 100
NUM_BLOCKS = 776
BLOCK_SIZE = 16
GIB = 2**30
PINNED_ENGINE = {"vllm-core": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
                 "vllm-metal": "572322e17584bb54afbc9e143f72009c2398bbba"}
SERVER_ARGS = ["--max-model-len", "9216", "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.38",
               "--num-gpu-blocks-override", str(NUM_BLOCKS), "--block-size", str(BLOCK_SIZE),
               "--enable-prefix-caching", "--async-scheduling", "--scheduling-policy", "fcfs", "--seed", "0"]
SERVER_ENV = (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("VLLM_SERVER_DEV_MODE", "1"),
              ("VLLM_USE_V2_MODEL_RUNNER", "0"), ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"))
STRIPPED_ENV_PREFIXES = ("VLLM_", "E02_")
OBSERVER_CLS = "e02.observer.ObservedScheduler"
PREDICTIONS = DATA / "predictions.json"
PREDICTIONS_SHA = DATA / "predictions.sha256"
CHECKPOINT_S = 30.0
FINAL_RECORDS_BYTES = 64 * 2**20
COST_PAIRS = 5
COST_BOUND = 0.05
COST_NOISE_BOUND = 0.10

Outcome = Literal["running", "validating", "complete", "incomplete", "invalid", "failed"]
FINAL_OUTCOMES = ("complete", "incomplete", "invalid", "failed")


class GateError(Exception):
    """A readiness gate failed. The message says what to do."""


class BudgetExceeded(Exception):
    pass


class Incomplete(Exception):
    pass


@dataclass(frozen=True)
class ExecutionGrant:
    command: str
    granted_at: str


def execution_grant(execute: bool, command: str) -> ExecutionGrant:
    if not execute:
        raise GateError(AUTHORITY)
    return ExecutionGrant(command, datetime.now().astimezone().isoformat(timespec="seconds"))


def require_grant(grant) -> ExecutionGrant:
    if not isinstance(grant, ExecutionGrant):
        raise GateError(AUTHORITY)
    return grant


@dataclass(frozen=True)
class WarmupRule:
    prompts: int
    tokens: int
    min_rounds: int
    max_rounds: int
    window: int
    max_relative_range: float
    max_s: float


@dataclass(frozen=True)
class Deadlines:
    run_wall_s: float
    no_progress_s: float


@dataclass(frozen=True)
class Ceilings:
    wall_s: float
    disk_bytes: int
    reserve_bytes: int
    final_records_bytes: int


@dataclass(frozen=True)
class ExecConfig:
    model: str
    vllm: str
    port: int
    server_args: tuple[str, ...]
    server_env: tuple[tuple[str, str], ...]
    observer_cls: str
    temperature: float
    top_p: float
    num_blocks: int
    snapshot_every: int
    digest_every: int
    startup_timeout_s: float
    seal_timeout_s: float
    shutdown_grace_s: float
    kill_grace_s: float
    overhead_allowance_s: float
    warmup: WarmupRule

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def attempt_max_s(self, d: Deadlines) -> float:
        """Upper bound on one attempt's wall time: every phase has its own deadline."""
        return (self.startup_timeout_s + self.warmup.max_s + d.run_wall_s + self.seal_timeout_s
                + self.shutdown_grace_s + self.kill_grace_s + self.overhead_allowance_s)


WARMUP = WarmupRule(prompts=4, tokens=128, min_rounds=3, max_rounds=12, window=3, max_relative_range=0.05,
                    max_s=600.0)


def current_config() -> ExecConfig:
    return ExecConfig(model=MODEL, vllm=str(VLLM), port=PORT, server_args=tuple(SERVER_ARGS), server_env=SERVER_ENV,
                      observer_cls=OBSERVER_CLS, temperature=0.6, top_p=0.95, num_blocks=NUM_BLOCKS,
                      snapshot_every=256, digest_every=1, startup_timeout_s=900.0, seal_timeout_s=120.0,
                      shutdown_grace_s=60.0, kill_grace_s=30.0, overhead_allowance_s=300.0, warmup=WARMUP)


@dataclass(frozen=True)
class PilotWorkload:
    """A bounded pilot. Each uses the first `requests` items of repetition 11's order."""
    target: int
    cap: int
    requests: int
    deadlines: Deadlines
    ceilings: Ceilings


PILOTS = {
    "behavior": PilotWorkload(16, 1024, 24, Deadlines(2400.0, 180.0),
                              Ceilings(3 * 3600.0, 4 * GIB, 5 * GIB, FINAL_RECORDS_BYTES)),
    "cost-low": PilotWorkload(4, 256, 8, Deadlines(1200.0, 180.0),
                              Ceilings(4 * 3600.0, 4 * GIB, 5 * GIB, FINAL_RECORDS_BYTES)),
    "cost-pressure": PilotWorkload(16, 128, 16, Deadlines(1200.0, 180.0),
                                   Ceilings(6 * 3600.0, 4 * GIB, 5 * GIB, FINAL_RECORDS_BYTES)),
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    target: int
    cap: int
    rep_seed: int
    round: int
    order_in_round: int
    item_order: tuple[int, ...]
    item_seeds: tuple[int, ...]
    requests: int
    logging: bool
    workload: str


def item_order(rep_seed: int, n: int = corpus_mod.SIZE) -> list[int]:
    order = list(range(n))
    random.Random(rep_seed).shuffle(order)
    return order


def item_seed(item: int, rep_seed: int) -> int:
    return int(hashlib.sha256(f"e02:{item}:{rep_seed}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def cell_order(rep_seed: int) -> list[tuple[int, int]]:
    cells = [(t, c) for t in TARGETS for c in CAPS]
    random.Random(f"cells:{rep_seed}").shuffle(cells)
    return cells


def main_plan() -> list[RunSpec]:
    runs = []
    for rnd, seed in enumerate(REP_SEEDS, start=1):
        order = item_order(seed)
        seeds = tuple(item_seed(i, seed) for i in order)
        for pos, (t, c) in enumerate(cell_order(seed), start=1):
            runs.append(RunSpec(f"r{rnd}-{pos:02d}-t{t}-c{c}", t, c, seed, rnd, pos, tuple(order), seeds,
                                REQUESTS, True, "main"))
    return runs


def pilot_plan(kind: str) -> list[RunSpec]:
    seed = REP_SEEDS[0]
    w = PILOTS[kind]
    items = item_order(seed)[:w.requests]
    seeds = tuple(item_seed(i, seed) for i in items)
    if kind == "behavior":
        return [RunSpec("behavior", w.target, w.cap, seed, 1, 1, tuple(items), seeds, w.requests, True, "behavior")]
    runs = []
    for pair in range(COST_PAIRS):
        sides = (True, False) if pair % 2 == 0 else (False, True)
        for pos, logging in enumerate(sides, start=1):
            runs.append(RunSpec(f"p{pair + 1}-{'on' if logging else 'off'}", w.target, w.cap, seed, pair + 1,
                                pos, tuple(items), seeds, w.requests, logging, kind))
    return runs


def to_json(obj) -> object:
    return json.loads(json.dumps(dataclasses.asdict(obj)))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest_of(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_as(tp, v, where: str):
    """Strictly parse JSON `v` into `tp`: frozen dataclasses with exact keys, tuples, and scalars."""
    if dataclasses.is_dataclass(tp):
        names = [f.name for f in dataclasses.fields(tp)]
        if not isinstance(v, dict) or sorted(v) != sorted(names):
            raise GateError(f"{where}: expected an object with fields {names}")
        hints = typing.get_type_hints(tp)
        return tp(**{n: parse_as(hints[n], v[n], f"{where}.{n}") for n in names})
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is tuple:
        if not isinstance(v, list):
            raise GateError(f"{where}: expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(parse_as(args[0], x, f"{where}[{i}]") for i, x in enumerate(v))
        if len(v) != len(args):
            raise GateError(f"{where}: expected {len(args)} items")
        return tuple(parse_as(a, x, f"{where}[{i}]") for i, (a, x) in enumerate(zip(args, v)))
    ok = {bool: lambda x: isinstance(x, bool),
          int: lambda x: isinstance(x, int) and not isinstance(x, bool),
          float: lambda x: isinstance(x, float) and math.isfinite(x),
          str: lambda x: isinstance(x, str)}[tp](v)
    if not ok:
        raise GateError(f"{where}: expected {tp.__name__}, found {v!r}")
    return v


def freeze_predictions(source: Path) -> str:
    """Copy the user's recorded prediction into the experiment's data directory with the earlier
    Codex guesses kept distinct, and freeze the copy by digest. Refuses to change a frozen copy."""
    user = json.loads(source.read_text())
    doc = {
        "frozen_before_measurement": True,
        "krishna": user,
        "codex": [
            {"claim": "At batch-size target 4 with cap 512, fewer than 1 preemption per 100 requests. "
                      "At target 32 with cap 4,096, more than 10 per 100.",
             "numbers": {"t4_c512_preemptions_per_100_lt": 1, "t32_c4096_preemptions_per_100_gt": 10},
             "reason": "Guess. The nominal pool holds 12,416 token slots; 32 prompts of about 1k tokens exceed it."},
            {"claim": "The chosen victim is tied for least computed in fewer than 80% of decisions.",
             "numbers": {"victim_least_tied_frac_lt": 0.8},
             "reason": "Guess. The default rule uses running-list order rather than computed-token count."},
            {"claim": "At least one pressure configuration has resumed victims with recovery below 0.2 and above 0.8.",
             "numbers": {"recovery_below": 0.2, "recovery_above": 0.8},
             "reason": "Guess. Waiting and intervening allocations can differ between victims."},
        ],
    }
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    if PREDICTIONS.exists():
        if PREDICTIONS.read_text() != text:
            raise GateError(f"{PREDICTIONS} is frozen and differs from {source}; predictions cannot change")
        return digest
    DATA.mkdir(exist_ok=True)
    _write_new(PREDICTIONS, text)
    _write_new(PREDICTIONS_SHA, digest + "\n")
    return digest


def check_predictions() -> str:
    if not PREDICTIONS.exists() or not PREDICTIONS_SHA.exists():
        raise GateError("predictions are not frozen: run `python -m e02 predictions`")
    text = PREDICTIONS.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != PREDICTIONS_SHA.read_text().strip():
        raise GateError("data/predictions.json does not match its frozen digest")
    doc = json.loads(text)
    k = doc["krishna"]
    rec = k.get("recovery", {})
    low, high = rec.get("low_pressure", {}), rec.get("high_pressure", {})
    numeric = [low.get("greater_than"), high.get("min"), high.get("max")]
    if not all(isinstance(x, (int, float)) for x in numeric) or not k.get("reason"):
        raise GateError("Krishna's prediction lacks a numeric recovery value or a reason")
    for row in doc["codex"]:
        if not row.get("numbers") or not row.get("reason"):
            raise GateError("a Codex prediction row lacks a number or a reason")
    readme = README.read_text()
    section = readme.split("## Prediction", 1)[1].split("\n## ", 1)[0]
    if "| Krishna |" not in section or "above 60%" not in section or "40-60%" not in section:
        raise GateError("README Prediction table does not state Krishna's frozen numbers")
    return digest


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def engine_state() -> dict:
    out = {}
    for repo in PINNED_ENGINE:
        path = ENGINE / repo
        out[repo] = {"sha": sh(f"git -C {path} rev-parse HEAD"),
                     "dirty_diff_sha256": hashlib.sha256(sh(f"git -C {path} diff HEAD").encode()).hexdigest(),
                     "dirty": sh(f"git -C {path} status --porcelain") != ""}
    return out


def check_engine() -> dict:
    state = engine_state()
    for repo, sha in PINNED_ENGINE.items():
        if state[repo]["sha"] != sha:
            raise GateError(f"{repo} is at {state[repo]['sha']}, the experiment pins {sha}")
        if state[repo]["dirty"]:
            raise GateError(f"{repo} has uncommitted changes")
    from e02 import observer
    stock = observer.source_fingerprints()
    drift = {k for k, v in stock.items() if observer.PINNED_SOURCES[k] != v}
    if drift:
        raise GateError(f"stock scheduler sources differ from the pinned fingerprints: {sorted(drift)}")
    state["stock_sources"] = stock
    return state


def source_digests() -> dict[str, str]:
    return {p.relative_to(HERE).as_posix(): file_sha(p) for p in sorted((HERE / "e02").glob("*.py"))}


def test_digests() -> dict[str, str]:
    return {p.relative_to(HERE).as_posix(): file_sha(p) for p in sorted((HERE / "tests").glob("*.py"))}


def technical_fingerprint() -> str:
    return digest_of({"source": source_digests(), "tests": test_digests()})


def provenance(cfg: ExecConfig, corpus_digest: str, predictions_digest: str, engine: dict) -> dict:
    """Everything the measurement depends on. Its digest must match between a sealed plan, every
    attempt run under it, and the code that later resumes it or uses it as evidence."""
    return {"source": source_digests(), "config": to_json(cfg), "corpus_digest": corpus_digest,
            "predictions_digest": predictions_digest, "engine": engine,
            "technical_fingerprint": technical_fingerprint()}


def provenance_drift(old: dict, new: dict) -> list[str]:
    keys = sorted(set(old) | set(new))
    out = [k for k in keys if k != "source" and old.get(k) != new.get(k)]
    a, b = old.get("source", {}), new.get("source", {})
    out += [f"source {f}" for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
    return out


def run_technical(target: list[str]) -> tuple[int, Path]:
    """Run the offline pytest suite and keep a provenance record of its result."""
    TECHNICAL.mkdir(exist_ok=True)
    out = TECHNICAL / f"{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir()
    before = technical_fingerprint()
    junit = out / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-p", "no:cacheprovider",
           f"--junitxml={junit}", *target]
    code = subprocess.call(cmd, cwd=HERE)
    counts = junit_counts(junit) if junit.exists() else {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    result = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "fingerprint": before,
              "fingerprint_after": technical_fingerprint(), "targets": target, "exit_code": code,
              "python": sys.executable, **counts, "junit_sha256": file_sha(junit) if junit.exists() else None}
    _write_new(out / "result.json", json.dumps(result, indent=1, sort_keys=True) + "\n")
    return code, out


def junit_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}


def check_technical() -> dict:
    fp = technical_fingerprint()
    for result in sorted(TECHNICAL.glob("*/result.json"), reverse=True):
        try:
            r = json.loads(result.read_text())
            if not isinstance(r, dict):
                continue
            junit = result.parent / "junit.xml"
            counts = junit_counts(junit)
        except (OSError, ValueError, ET.ParseError):
            continue
        if (r.get("fingerprint") == fp and r.get("fingerprint_after") == fp and r.get("targets") == ["tests"]
                and r.get("exit_code") == 0 and r.get("failures") == 0 and r.get("errors") == 0
                and r.get("skipped") == 0 and (r.get("tests") or 0) > 0 and junit.exists()
                and r.get("junit_sha256") == file_sha(junit)
                and counts == {k: r.get(k) for k in ("tests", "failures", "errors", "skipped")}):
            return {"record": result.relative_to(TECHNICAL.parent).as_posix(), "sha256": file_sha(result), "tests": r["tests"]}
    raise GateError("no passing offline technical record for the current source and tests: "
                    "run `python -m e02 technical`")


def machine() -> dict:
    return {"time": datetime.now().isoformat(timespec="seconds"), "swap": sh("sysctl -n vm.swapusage"),
            "power": sh("pmset -g batt | head -1"), "therm": sh("pmset -g therm"),
            "load": sh("sysctl -n vm.loadavg"), "lowpowermode": sh("pmset -g | grep -i lowpowermode")}


def thermal_limited(therm: str) -> bool:
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    if m and int(m[1]) < 100:
        return True
    lines = [x for x in therm.splitlines() if re.search(r"thermal warning level", x, re.I)]
    return any("No thermal warning level has been recorded" not in x for x in lines)


def gpu_time_by_pid() -> dict:
    totals, who = {}, None
    for line in sh("ioreg -r -c AGXDeviceUserClient -l -w0").splitlines():
        m = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]+)"', line)
        if m:
            who = (int(m[1]), m[2])
        if '"AppUsage"' in line and who:
            totals[who] = totals.get(who, 0) + sum(int(x) for x in re.findall(r'"accumulatedGPUTime"=(\d+)', line))
    return totals


def other_gpu_share(before: dict, after: dict, seconds: float, ours: set) -> float:
    """GPU percent used by processes outside `ours` (every owned engine process and the runner)."""
    use = sum((after[k] - before.get(k, 0)) / 1e9 / seconds * 100 for k in after if k[0] not in ours)
    return round(use, 2)


def environment(extra: dict) -> dict:
    import mlx.core as mx
    import vllm
    import vllm_metal
    return {"time": datetime.now().isoformat(), "machine": sh("sysctl -n machdep.cpu.brand_string"),
            "ram_bytes": int(sh("sysctl -n hw.memsize") or 0), "macos": sh("sw_vers -productVersion"),
            **machine(), "python": sys.version.split()[0],
            "versions": {"vllm": vllm.__version__, "vllm_metal": getattr(vllm_metal, "__version__", "?"),
                         "mlx": mx.__version__},
            "engine": engine_state(), "model": MODEL, "server_args": SERVER_ARGS, **extra}


def free_bytes() -> int:
    return shutil.disk_usage(HERE).free


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / GIB


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _write_new(path: Path, text: str) -> None:
    with open(path, "x") as f:
        f.write(text)


def _write_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def new_root(label: str) -> Path:
    RUNS.mkdir(exist_ok=True)
    root = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}-{label}"
    root.mkdir()
    (root / "attempts").mkdir()
    return root


def register_root(root: Path, what: str) -> None:
    """List a run root under README Runs so the lab checker can find it."""
    text = README.read_text()
    if root.name in text:
        return
    head, rest = text.split("## Runs\n", 1)
    section, tail = rest.split("\n## ", 1)
    lines = section.split("\n")
    last = max(i for i, line in enumerate(lines) if line.startswith("|"))
    lines.insert(last + 1, f"| {root.name} | {what} | see report; status in attempts/ |")
    README.write_text(head + "## Runs\n" + "\n".join(lines) + "\n## " + tail)


@dataclass(frozen=True)
class SealedPlan:
    root: Path
    kind: Literal["pilot", "main"]
    workload: str
    digest: str
    doc: dict
    config: ExecConfig
    deadlines: Deadlines
    ceilings: Ceilings
    runs: tuple[RunSpec, ...]
    provenance_digest: str

    def spec(self, run_id: str) -> RunSpec | None:
        return next((r for r in self.runs if r.run_id == run_id), None)


def plan_path(root: Path) -> Path:
    return root / "manifest.json" if (root / "manifest.json").exists() else root / "plan.json"


def seal_plan(root: Path, *, kind: str, workload: str, runs: list[RunSpec], cfg: ExecConfig, deadlines: Deadlines,
              ceilings: Ceilings, prov: dict, extra: dict) -> str:
    """Write plan.json (pilot) or manifest.json (main) once, with its digest in a sidecar."""
    if prov.get("config") != to_json(cfg):
        raise GateError("provenance does not describe the configuration being sealed")
    doc = {**extra, "kind": kind, "workload": workload, "root_id": root.name,
           "runs": [to_json(r) for r in runs], "config": to_json(cfg), "deadlines": to_json(deadlines),
           "ceilings": to_json(ceilings), "provenance": prov, "provenance_digest": digest_of(prov)}
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    path = root / ("manifest.json" if kind == "main" else "plan.json")
    digest = hashlib.sha256(text.encode()).hexdigest()
    _write_new(path, text)
    _write_new(path.with_suffix(".sha256"), digest + "\n")
    return digest


def load_sealed(root: Path) -> SealedPlan:
    path = plan_path(root)
    if not path.exists():
        raise GateError(f"{root.name} has no plan.json or manifest.json, so it has no evidence to check")
    side = path.with_suffix(".sha256")
    if not side.exists():
        raise GateError(f"{root.name}/{path.name} is not sealed: it predates sealed plans (fix round 1), so it "
                        "cannot be resumed or used as gate evidence")
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != side.read_text().strip():
        raise GateError(f"{root.name}/{path.name} does not match its sealed digest; a sealed plan cannot change")
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise GateError(f"{root.name}/{path.name} is not JSON: {e}")
    kind, workload = doc.get("kind"), doc.get("workload")
    if kind == "main" and workload == "main":
        expected = main_plan()
    elif kind == "pilot" and workload in PILOTS:
        expected = pilot_plan(workload)
    else:
        raise GateError(f"{root.name}: unknown plan kind {kind!r} workload {workload!r}")
    if not isinstance(doc.get("runs"), list):
        raise GateError(f"{root.name}: runs must be a list")
    runs = tuple(parse_as(RunSpec, r, f"runs[{i}]") for i, r in enumerate(doc["runs"]))
    if list(runs) != expected:
        raise GateError(f"{root.name}: runs differ from the fixed plan; a changed plan needs a new sealed root")
    prov = doc.get("provenance")
    if not isinstance(prov, dict) or digest_of(prov) != doc.get("provenance_digest"):
        raise GateError(f"{root.name}: provenance digest does not match its provenance")
    if prov.get("config") != doc.get("config"):
        raise GateError(f"{root.name}: executed configuration differs from its provenance")
    return SealedPlan(root, kind, workload, digest, doc, parse_as(ExecConfig, doc.get("config"), "config"),
                      parse_as(Deadlines, doc.get("deadlines"), "deadlines"),
                      parse_as(Ceilings, doc.get("ceilings"), "ceilings"), runs, doc["provenance_digest"])


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    pgid: int
    start: str


def process_table() -> dict[int, Proc]:
    """Live (non-zombie) processes. A process identity is its pid plus its start time."""
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,pgid=,stat=,lstart="], capture_output=True, text=True,
                         check=True).stdout
    table = {}
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5 or parts[3].startswith("Z"):
            continue
        pid, ppid, pgid = map(int, parts[:3])
        table[pid] = Proc(pid, ppid, pgid, " ".join(parts[4].split()))
    return table


def group_members(pgid: int, leader: int, table: dict[int, Proc]) -> list[Proc]:
    """Members of the process group plus every descendant of the leader or a member."""
    children = defaultdict(list)
    for p in table.values():
        children[p.ppid].append(p)
    found = {p.pid: p for p in table.values() if p.pgid == pgid}
    stack = [leader, *found]
    while stack:
        for c in children.get(stack.pop(), ()):
            if c.pid not in found:
                found[c.pid] = c
                stack.append(c.pid)
    return sorted(found.values(), key=lambda p: p.pid)


def terminate_owned(pgid: int, known: dict[int, str], leader_holds_group: bool, grace_s: float, kill_grace_s: float,
                    table: Callable[[], dict[int, Proc]] = process_table, clock=time.monotonic,
                    sleep=time.sleep, guard: Callable[[], str | None] = lambda: None) -> dict:
    """TERM, then KILL, every process positively identified as owned, whether or not the group
    leader still runs. The group is owned when its unreaped leader holds the group id or a live
    member matches a recorded identity; known identities outside the group are owned too.
    Unidentified processes are reported, never signaled."""
    t = table()
    members = [p for p in t.values() if p.pgid == pgid]
    identified = leader_holds_group or any(known.get(p.pid) == p.start for p in members)
    report = {"pgid": pgid, "group_identified": identified, "term": [], "killed": [], "survivors": [],
              "unidentified": [] if identified else [p.pid for p in members]}

    def owned(t):
        procs = {p.pid: p for p in t.values() if identified and p.pgid == pgid}
        procs.update({p.pid: p for p in t.values() if known.get(p.pid) == p.start})
        return list(procs.values())

    def send(procs, sig):
        if any(p.pgid == pgid for p in procs):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        for p in procs:
            if p.pgid != pgid:
                try:
                    os.kill(p.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass

    def wait_gone(limit, stop_on_budget=False):
        end = clock() + limit
        while True:
            left = owned(table())
            problem = guard()
            if problem:
                report["budget_problem"] = problem
            if not left or clock() >= end or (problem and stop_on_budget):
                return left
            sleep(0.1)

    left = owned(t)
    if not left:
        return report
    report["term"] = sorted(p.pid for p in left)
    send(left, signal.SIGTERM)
    left = wait_gone(grace_s, stop_on_budget=True)
    if left:
        report["killed"] = sorted(p.pid for p in left)
        send(left, signal.SIGKILL)
        left = wait_gone(kill_grace_s)
    report["survivors"] = sorted(p.pid for p in left)
    return report


class OwnedGroup:
    """A process started in its own session. Its members are recorded while the unreaped leader
    still reserves the group id, so cleanup can identify them after the leader exits."""

    def __init__(self, argv: list[str], *, env: dict, cwd: Path, log):
        self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(cwd),
                                     start_new_session=True)
        self.pgid = self.proc.pid
        self.known: dict[int, str] = {}
        self.sample()

    def sample(self) -> None:
        if self.proc.returncode is not None:
            return
        for p in group_members(self.pgid, self.proc.pid, process_table()):
            self.known.setdefault(p.pid, p.start)

    def identities(self) -> list[list]:
        return [[pid, start] for pid, start in sorted(self.known.items())]

    def pids(self) -> set[int]:
        return set(self.known)

    def stop(self, grace_s: float, kill_grace_s: float, guard: Callable[[], str | None] = lambda: None) -> dict:
        self.sample()
        report = terminate_owned(self.pgid, self.known, self.proc.returncode is None, grace_s, kill_grace_s,
                                 guard=guard)
        try:
            self.proc.wait(timeout=kill_grace_s)
        except subprocess.TimeoutExpired:
            report["survivors"] = sorted(set(report["survivors"]) | {self.pgid})
        return report


def server_env(cfg: ExecConfig, attempt: Path, run_label: str) -> tuple[dict, list[str]]:
    """The server's environment: inherited VLLM_ and E02_ variables are removed so that only the
    sealed configuration sets engine and capture behavior."""
    removed = sorted(k for k in os.environ if k.startswith(STRIPPED_ENV_PREFIXES))
    env = {k: v for k, v in os.environ.items() if not k.startswith(STRIPPED_ENV_PREFIXES)}
    env.update(dict(cfg.server_env))
    env.update({"PYTHONPATH": os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")])),
                "E02_OUT": str(attempt), "E02_RUN_ID": run_label, "E02_NUM_BLOCKS": str(cfg.num_blocks),
                "E02_SNAPSHOT_EVERY": str(cfg.snapshot_every), "E02_DIGEST_EVERY": str(cfg.digest_every)})
    return env, removed


class Server:
    """One owned vllm serve process group. Starting it needs an ExecutionGrant."""

    def __init__(self, attempt: Path, spec: RunSpec, cfg: ExecConfig, grant: ExecutionGrant, run_label: str):
        require_grant(grant)
        self.cfg = cfg
        env, self.env_removed = server_env(cfg, attempt, run_label)
        args = [cfg.vllm, "serve", cfg.model, "--port", str(cfg.port), *cfg.server_args,
                "--max-num-seqs", str(spec.target)]
        if spec.logging:
            args += ["--scheduler-cls", cfg.observer_cls]
        self.args = args
        self.log = open(attempt / "server.log", "x")
        self.group = OwnedGroup(args, env=env, cwd=attempt, log=self.log)

    def wait_healthy(self, budget: Budget, on_sample: Callable[[], None]) -> None:
        end = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < end:
            self.group.sample()
            on_sample()
            if self.group.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.group.proc.returncode} during startup")
            budget.check()
            try:
                urllib.request.urlopen(f"{self.cfg.url}/health", timeout=2)
                return
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"server not healthy after {self.cfg.startup_timeout_s:.0f} s")

    def stop(self, guard: Callable[[], str | None] = lambda: None) -> dict:
        try:
            return self.group.stop(self.cfg.shutdown_grace_s, self.cfg.kill_grace_s, guard)
        finally:
            self.log.close()


def post(url: str, path: str, timeout: float = 30) -> dict:
    req = urllib.request.Request(f"{url}{path}", method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def metrics_text(url: str) -> str:
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as r:
        return r.read().decode()


def metric_value(text: str, name: str) -> float | None:
    total, seen = 0.0, False
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith(name + "_created"):
            rest = line[len(name):]
            if rest[:1] in ("{", " "):
                total += float(line.rsplit(" ", 1)[1])
                seen = True
    return total if seen else None


def pool_blocks(text: str) -> int | None:
    m = re.search(r'vllm:cache_config_info\{[^}]*num_gpu_blocks="(\d+)"', text)
    return int(m[1]) if m else None


def pool_blocks_from_log(path: Path) -> int | None:
    m = re.findall(r"cache_config_info with initialization after num_gpu_blocks is: (\d+)", path.read_text())
    return int(m[-1]) if m else None


def port_busy(port: int) -> bool:
    return bool(sh(f"lsof -ti tcp:{port}"))


@dataclass
class Budget:
    """Bounds one attempt: its wall end and the root's disk ceiling and free-space reserve, with
    room kept for the attempt's final records. `checkpoint` persists elapsed time as it goes."""
    ceilings: Ceilings
    root_bytes_before: int
    attempt: Path
    end: float
    clock: Callable[[], float] = time.time
    free: Callable[[], int] = free_bytes
    checkpoint: Callable[[float], None] | None = None
    last_checkpoint: float = 0.0

    def problem(self) -> str | None:
        now = self.clock()
        if self.checkpoint is not None and now - self.last_checkpoint >= CHECKPOINT_S:
            self.last_checkpoint = now
            self.checkpoint(now)
        c = self.ceilings
        if now > self.end:
            return "budget: wall: the attempt passed its budgeted end"
        used = self.root_bytes_before + dir_bytes(self.attempt)
        if used > c.disk_bytes - c.final_records_bytes:
            return f"budget: disk: the run root holds {used / GIB:.2f} GiB of its {c.disk_bytes / GIB:.2f} GiB ceiling"
        free = self.free()
        if free < c.reserve_bytes + c.final_records_bytes:
            return f"budget: disk: {free / GIB:.2f} GiB free, at the {c.reserve_bytes / GIB:.2f} GiB reserve"
        return None

    def check(self) -> None:
        p = self.problem()
        if p:
            raise BudgetExceeded(p)


def admit_attempt(consumed_s: float, attempt_max_s: float, c: Ceilings, used_bytes: int, free: int) -> str | None:
    """Whether a new attempt may start: it must fit the remaining wall budget at its own upper
    bound, and the disk must leave the reserve and room for final records."""
    if consumed_s + attempt_max_s > c.wall_s:
        return (f"wall: {consumed_s / 3600:.2f} h used of {c.wall_s / 3600:.2f} h, and the next attempt may "
                f"take up to {attempt_max_s / 3600:.2f} h")
    if used_bytes > c.disk_bytes - c.final_records_bytes:
        return f"disk: the run root holds {used_bytes / GIB:.2f} GiB of its {c.disk_bytes / GIB:.2f} GiB ceiling"
    if free < c.reserve_bytes + c.final_records_bytes:
        return f"disk: {free / GIB:.2f} GiB free, at the {c.reserve_bytes / GIB:.2f} GiB reserve"
    return None


def _timestamp(iso: str | None) -> float | None:
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else None
    except ValueError:
        return None


def attempt_consumed_s(rec: dict, latest_mtime: float) -> float:
    """Wall time an attempt used. An attempt without an end counts until its last checkpoint or
    its newest file, whichever is later."""
    start = rec.get("wall_started") or _timestamp(rec.get("started"))
    if start is None:
        return 0.0
    end = rec.get("wall_ended") or _timestamp(rec.get("ended"))
    if end is None:
        end = max(rec.get("wall_checkpoint") or start, latest_mtime)
    return max(0.0, end - start) + rec.get("cleanup_wall_s", 0.0)


def root_consumed_s(root: Path, exclude: Path | None = None) -> float:
    total = 0.0
    for path in sorted((root / "attempts").glob("*.a*")):
        if path == exclude:
            continue
        mtimes = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
        total += attempt_consumed_s(read_attempt(path), max(mtimes, default=0.0))
    return total


def halt(root: Path, reason: str) -> None:
    with open(root / "halts.jsonl", "a") as f:
        f.write(json.dumps({"time": datetime.now().astimezone().isoformat(timespec="seconds"), "reason": reason})
                + "\n")


@contextmanager
def finalization_deadline(budget: Budget):
    budget.check()
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise BudgetExceeded("budget: wall: final validation or hashing exceeded its deadline")

    signal.signal(signal.SIGALRM, expired)
    started = time.monotonic()
    old_timer = signal.setitimer(signal.ITIMER_REAL, max(0.000001, budget.end - budget.clock()))
    try:
        yield
        budget.check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, old_timer[0] - (time.monotonic() - started)),
                             old_timer[1])


@dataclass(frozen=True)
class WarmupSample:
    round: int
    tokens: int
    elapsed_s: float
    therm: str

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


@dataclass(frozen=True)
class WarmupResult:
    stable: bool
    reason: str
    samples: tuple[WarmupSample, ...]


def warmup_verdict(samples: list[WarmupSample], rule: WarmupRule) -> tuple[bool, str]:
    """Steady when the last `window` rounds' decode throughput lies within `max_relative_range`
    of their median and none of them reports a thermal limit."""
    if len(samples) < max(rule.min_rounds, rule.window):
        return False, f"{len(samples)} of at least {max(rule.min_rounds, rule.window)} rounds"
    window = samples[-rule.window:]
    if any(thermal_limited(s.therm) for s in window):
        return False, "thermal limit reported in the last rounds"
    rates = [s.tokens_per_s for s in window]
    mid = statistics.median(rates)
    spread = (max(rates) - min(rates)) / mid if mid > 0 else float("inf")
    if spread <= rule.max_relative_range:
        return True, f"last {rule.window} rounds within {spread:.1%} of their median {mid:.1f} tokens/s"
    return False, f"last {rule.window} rounds spread {spread:.1%}, above {rule.max_relative_range:.0%}"


async def run_warmup(cfg: ExecConfig, guard: Callable[[], str | None], clock=time.monotonic,
                     therm: Callable[[], str] = lambda: sh("pmset -g therm")) -> WarmupResult:
    """Repeat fixed rounds of `prompts` concurrent requests of exactly `tokens` output tokens until
    the throughput is steady, within `max_rounds` and `max_s`. Every round finishes before the next
    starts, so warmup work is drained when this returns."""
    import aiohttp
    rule = cfg.warmup
    samples: list[WarmupSample] = []
    why = "no rounds"
    t_begin = clock()
    timeout = aiohttp.ClientTimeout(total=rule.max_s)

    async def one(session, rnd, i):
        body = {"model": cfg.model, "max_tokens": rule.tokens, "ignore_eos": True,
                "seed": i, "temperature": cfg.temperature, "top_p": cfg.top_p, "stream": False,
                "messages": [{"role": "user", "content": f"Warmup round {rnd}, request {i}: list the prime "
                                                         f"numbers between {100 * i} and {100 * i + 60}."}]}
        async with session.post(f"{cfg.url}/v1/chat/completions", json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"warmup request failed with HTTP {r.status}")
            tokens = (await r.json())["usage"]["completion_tokens"]
            if type(tokens) is not int or tokens != rule.tokens:
                raise Incomplete(f"warmup returned {tokens!r} tokens; expected {rule.tokens}")
            return tokens

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for rnd in range(1, rule.max_rounds + 1):
            problem = guard()
            if problem:
                raise BudgetExceeded(problem)
            if clock() - t_begin > rule.max_s:
                return WarmupResult(False, f"no steady state within {rule.max_s:.0f} s: {why}", tuple(samples))
            t0 = clock()
            work = asyncio.gather(*(one(session, rnd, i) for i in range(rule.prompts)))
            end = asyncio.get_running_loop().time() + max(0.0, rule.max_s - (t0 - t_begin))
            try:
                while not work.done():
                    problem = guard()
                    if problem:
                        raise BudgetExceeded(problem)
                    remaining = end - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise Incomplete(f"warmup exceeded {rule.max_s:.0f} s")
                    await asyncio.wait({work}, timeout=min(1.0, remaining))
                tokens = sum(work.result())
            finally:
                if not work.done():
                    work.cancel()
                await asyncio.gather(work, return_exceptions=True)
            samples.append(WarmupSample(rnd, tokens, clock() - t0, therm()))
            stable, why = warmup_verdict(samples, rule)
            if stable:
                return WarmupResult(True, why, tuple(samples))
    return WarmupResult(False, f"no steady state in {rule.max_rounds} rounds: {why}", tuple(samples))


def client_body(cfg: ExecConfig, text: str, cap: int, seed: int) -> dict:
    return {"model": cfg.model, "messages": [{"role": "user", "content": text}], "max_tokens": cap, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}, "temperature": cfg.temperature,
            "top_p": cfg.top_p}


async def stream_one(session, url: str, client_id: str, item: int, order: int, phase: str, body: dict,
                     progress: list, write) -> history.ClientRecord:
    t_submit = time.monotonic_ns()
    rec = dict(client_id=client_id, item=item, order=order, phase=phase, t_submit_ns=t_submit, t_first_ns=None,
               t_last_ns=None, t_end_ns=t_submit, status="error", http_status=None, error=None, finish_reason=None,
               prompt_tokens=None, completion_tokens=None, chunks=0, content_chunks=0, reasoning_chunks=0,
               max_gap_ns=None, response_id=None)
    last = None
    try:
        async with session.post(f"{url}/v1/chat/completions", json=body, headers={"X-Request-Id": client_id}) as r:
            rec["http_status"] = r.status
            if r.status != 200:
                rec["status"], rec["error"] = "http_error", (await r.text())[:300]
            else:
                async for raw in r.content:
                    line = raw.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    msg = json.loads(line[6:])
                    rec["chunks"] += 1
                    rec["response_id"] = msg.get("id", rec["response_id"])
                    for choice in msg.get("choices") or ():
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if content or reasoning:
                            now = time.monotonic_ns()
                            progress[0] = now
                            if rec["t_first_ns"] is None:
                                rec["t_first_ns"] = now
                            if last is not None:
                                rec["max_gap_ns"] = max(rec["max_gap_ns"] or 0, now - last)
                            last = rec["t_last_ns"] = now
                            rec["content_chunks"] += bool(content)
                            rec["reasoning_chunks"] += bool(reasoning)
                        if choice.get("finish_reason"):
                            rec["finish_reason"] = choice["finish_reason"]
                    if msg.get("usage"):
                        rec["prompt_tokens"] = msg["usage"]["prompt_tokens"]
                        rec["completion_tokens"] = msg["usage"]["completion_tokens"]
                ok = rec["finish_reason"] is not None and rec["completion_tokens"] is not None
                rec["status"] = "ok" if ok else "error"
                if not ok:
                    rec["error"] = "stream ended without finish reason and usage"
    except asyncio.CancelledError:
        rec["status"], rec["error"] = "cancelled", "deadline"
    except Exception as e:
        rec["status"], rec["error"] = "error", repr(e)[:300]
    rec["t_end_ns"] = time.monotonic_ns()
    out = history.ClientRecord(**rec)
    write(out)
    return out


async def drive(spec: RunSpec, texts: dict[int, str], client_prefix: str, out: Path, deadlines: Deadlines,
                body: Callable[[str, int, int], dict], url: str,
                guard: Callable[[], str | None] = lambda: None) -> tuple[list, str | None, int | None]:
    """Keep `target` requests outstanding, replacing each finished or failed one until all are
    submitted, then drain. Returns the records, the reason a deadline or the budget stopped the
    run if one did, and the time the last request was submitted, which starts the drain phase."""
    import aiohttp
    f = open(out / "client.jsonl", "x", buffering=1)

    def write(rec):
        f.write(json.dumps(dataclasses.asdict(rec)) + "\n")

    items = list(zip(spec.item_order[:spec.requests], spec.item_seeds[:spec.requests]))
    progress = [time.monotonic_ns()]
    records, pending, fired, drain_start = [], set(), None, [None]
    t_end = time.monotonic() + deadlines.run_wall_s
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        nxt = 0

        def submit():
            nonlocal nxt
            item, seed = items[nxt]
            phase = "startup" if nxt < spec.target else "steady"
            task = asyncio.create_task(stream_one(session, url, f"{client_prefix}.i{item}", item, nxt, phase,
                                                  body(texts[item], spec.cap, seed), progress, write))
            pending.add(task)
            nxt += 1
            if nxt == len(items):
                drain_start[0] = time.monotonic_ns()

        while nxt < len(items) and len(pending) < spec.target:
            submit()
        while pending:
            done, _ = await asyncio.wait(pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pending.discard(task)
                records.append(task.result())
                if nxt < len(items):
                    submit()
            now = time.monotonic()
            idle = (time.monotonic_ns() - progress[0]) / 1e9
            problem = guard()
            if problem:
                fired = problem
            elif now > t_end:
                fired = "wall deadline"
            elif idle > deadlines.no_progress_s:
                fired = f"no output for {deadlines.no_progress_s:.0f} s"
            if fired and pending:
                for task in pending:
                    task.cancel()
                for task in pending:
                    try:
                        records.append(await task)
                    except asyncio.CancelledError:
                        pass
                pending.clear()
    f.close()
    return records, fired, drain_start[0]


def attempts_of(root: Path, run_id: str) -> list[Path]:
    found = []
    for p in (root / "attempts").glob(f"{run_id}.a*"):
        m = re.fullmatch(re.escape(run_id) + r"\.a(\d+)", p.name)
        if m:
            found.append((int(m[1]), p))
    return [p for _, p in sorted(found)]


def read_attempt(path: Path) -> dict:
    try:
        rec = json.loads((path / "attempt.json").read_text())
    except (OSError, ValueError):
        return {"outcome": "failed", "reason": "attempt.json missing or unreadable"}
    return rec if isinstance(rec, dict) else {"outcome": "failed", "reason": "attempt.json is not an object"}


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def own_start() -> str | None:
    p = process_table().get(os.getpid())
    return p.start if p else None


def owner_alive(rec: dict) -> bool:
    pid = rec.get("pid")
    if not isinstance(pid, int) or not pid_alive(pid):
        return False
    start = rec.get("runner_start")
    if start is None:
        return True
    p = process_table().get(pid)
    return p is not None and p.start == start


EVIDENCE_EXCLUDED = ("attempt.json", "attempt.json.tmp")


def evidence_digests(path: Path) -> dict[str, str]:
    return {p.relative_to(path).as_posix(): file_sha(p) for p in sorted(path.rglob("*"))
            if p.is_file() and p.name not in EVIDENCE_EXCLUDED}


def cleanup_problem(rec: dict) -> str | None:
    report = rec.get("stop_report") or {}
    if rec.get("cleanup_failed") or report.get("survivors") or report.get("unidentified"):
        return "cleanup: owned server cleanup is unresolved"
    return None


def finalization_budget(path: Path, rec: dict, ctx: RunContext) -> Budget:
    cfg, c = ctx.sealed.config, ctx.sealed.ceilings
    start = rec.get("wall_started") or _timestamp(rec.get("started")) or time.time()
    before = root_consumed_s(ctx.root, exclude=path)
    end = start + min(cfg.attempt_max_s(ctx.sealed.deadlines), c.wall_s - before)
    return Budget(c, dir_bytes(ctx.root) - dir_bytes(path), path, end, clock=lambda: time.time(),
                  free=lambda: free_bytes())


def write_final_record(path: Path, rec: dict, budget: Budget) -> None:
    size = len((json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    c = budget.ceilings
    if budget.free() < c.reserve_bytes + size:
        raise BudgetExceeded("budget: disk: final metadata would consume the free-space reserve")
    if budget.root_bytes_before + dir_bytes(path) + size > c.disk_bytes:
        raise BudgetExceeded("budget: disk: final metadata would exceed the root ceiling")
    _write_atomic(path / "attempt.json", rec)


@dataclass(frozen=True)
class RunContext:
    root: Path
    sealed: SealedPlan
    item_sha: dict[int, str]


@dataclass(frozen=True)
class Verification:
    status: str
    reasons: tuple[str, ...]
    record: dict
    history: history.RunHistory | None
    clients: tuple[history.ClientRecord, ...]


def validate_evidence(path: Path, spec: RunSpec, ctx: RunContext) -> Verification:
    """Validate one attempt's raw evidence against its sealed plan: the logging-on journal replays
    and joins with the run's own capture settings, and the client recorded exactly the planned
    requests, each finished."""
    rec = read_attempt(path)
    problem = cleanup_problem(rec)
    if problem:
        return Verification("failed", (problem,), rec, None, ())
    try:
        warm = parse_as(WarmupResult, rec.get("warmup"), "warmup")
        rule = ctx.sealed.config.warmup
        steady, reason = warmup_verdict(list(warm.samples), rule)
        expected_rounds = list(range(1, len(warm.samples) + 1))
        if (not warm.stable or not steady or len(warm.samples) > rule.max_rounds
                or [s.round for s in warm.samples] != expected_rounds
                or any(s.tokens != rule.prompts * rule.tokens or s.elapsed_s <= 0 for s in warm.samples)
                or sum(s.elapsed_s for s in warm.samples) > rule.max_s):
            raise GateError(f"warmup samples do not establish the sealed stability rule: {reason}")
    except GateError as e:
        return Verification("invalid", (str(e),), rec, None, ())
    clients, problem = history.read_client(path / "client.jsonl")
    clients = tuple(clients)
    if problem:
        return Verification("invalid", (problem,), rec, None, clients)
    hist = None
    if spec.logging:
        hist = history.load_run(path, item_sha=ctx.item_sha)
        if hist.status != "complete":
            return Verification(hist.status, tuple(hist.reasons[:20]), rec, hist, clients)
        cfg = ctx.sealed.config
        run_cfg = hist.replay.config
        wrong = [f"journal {key} {run_cfg.get(key)!r}, sealed plan {want!r}" for key, want in (
            ("num_gpu_blocks", cfg.num_blocks), ("snapshot_every", cfg.snapshot_every),
            ("digest_every", cfg.digest_every)) if run_cfg.get(key) != want]
        if hist.replay.run_id != f"{ctx.root.name}/{path.name}":
            wrong.append(f"journal run id {hist.replay.run_id!r} does not name this attempt")
        if wrong:
            return Verification("invalid", tuple(wrong), rec, hist, clients)
    elif (path / "events.jsonl").exists():
        return Verification("invalid", ("a logging-off attempt has an engine journal",), rec, None, clients)
    expected = {f"{ctx.root.name}.{path.name}.i{item}" for item in spec.item_order[:spec.requests]}
    ids = [c.client_id for c in clients]
    if len(ids) != len(set(ids)) or not set(ids) <= expected:
        return Verification("invalid", ("client records name requests outside the plan or repeat one",), rec,
                            hist, clients)
    reasons = []
    if set(ids) != expected:
        reasons.append(f"{len(expected - set(ids))} planned requests have no client record")
    bad = [c for c in clients if c.status != "ok" or c.finish_reason not in ("stop", "length")
           or c.completion_tokens is None or c.completion_tokens > spec.cap]
    if bad:
        reasons.append(f"{len(bad)} of {len(clients)} requests did not finish cleanly")
    return Verification("incomplete" if reasons else "complete", tuple(reasons), rec, hist, clients)


def begin_attempt(ctx: RunContext, spec: RunSpec, number: int, replaces: str | None, reason: str | None,
                  authority: dict | None = None) -> tuple[Path, dict]:
    attempt = ctx.root / "attempts" / f"{spec.run_id}.a{number}"
    attempt.mkdir()
    now = time.time()
    rec = {"run_id": spec.run_id, "attempt": number, "replaces": replaces, "replacement_reason": reason,
           "spec": to_json(spec), "requests": spec.requests, "logging": spec.logging,
           "plan_digest": ctx.sealed.digest, "provenance_digest": ctx.sealed.provenance_digest,
           "authority": authority, "outcome": "running", "reason": None, "pid": os.getpid(),
           "runner_start": own_start(), "started": datetime.now().isoformat(), "wall_started": now,
           "wall_checkpoint": now}
    _write_atomic(attempt / "attempt.json", rec)
    return attempt, rec


def finish_attempt(path: Path, rec: dict, outcome: str, reason: str | None, ctx: RunContext | None,
                   wall_ended: float | None = None) -> dict:
    """Move an attempt to its final outcome. A complete execution is first recorded as validating,
    so a crash during validation leaves a state that resume validates again rather than trusts."""
    rec = {**rec, "execution_outcome": outcome, "execution_reason": reason,
           "wall_ended": wall_ended if wall_ended is not None else time.time(),
           "ended": datetime.now().isoformat()}
    budget = finalization_budget(path, rec, ctx) if ctx is not None else None
    if cleanup_problem(rec):
        outcome, reason = "failed", cleanup_problem(rec)
    if outcome == "complete":
        rec.update(outcome="validating", reason=None)
    try:
        if budget is None:
            raise BudgetExceeded("budget: no sealed budget for final validation")
        with finalization_deadline(budget):
            write_final_record(path, rec, budget)
            if outcome == "complete":
                spec = ctx.sealed.spec(rec["run_id"])
                v = validate_evidence(path, spec, ctx)
                outcome, reason = v.status, "; ".join(v.reasons[:3]) or None
                rec["validation"] = {"status": v.status, "reasons": list(v.reasons)}
                if v.history is not None:
                    rec["observed_preemptions"] = v.history.metrics.get("preemptions")
            evidence = evidence_digests(path)
            size = dir_bytes(path)
            budget.check()
    except BudgetExceeded as e:
        outcome, reason, evidence, size = "incomplete", str(e), {}, rec.get("bytes")
    rec.update(outcome=outcome, reason=reason, evidence=evidence, bytes=size)
    if wall_ended is None:
        rec.update(wall_ended=time.time(), ended=datetime.now().isoformat())
    if budget is not None:
        problem = budget.problem()
        if problem:
            rec.update(outcome="incomplete", reason=problem)
        if rec["outcome"] == "complete":
            try:
                with finalization_deadline(budget):
                    write_final_record(path, rec, budget)
            except BudgetExceeded as e:
                rec.update(outcome="incomplete", reason=str(e), wall_ended=time.time())
                write_final_record(path, rec, budget)
        else:
            write_final_record(path, rec, budget)
    else:
        _write_atomic(path / "attempt.json", rec)
    return rec


def verify_attempt(path: Path, ctx: RunContext) -> Verification:
    """Revalidate a recorded attempt: its record must name the sealed plan and provenance, its
    evidence files must be byte-identical to those digested at completion, and the evidence must
    validate again from scratch."""
    rec = read_attempt(path)
    outcome = rec.get("outcome")
    if cleanup_problem(rec):
        return Verification("failed", (cleanup_problem(rec),), rec, None, ())
    if outcome != "complete":
        return Verification(str(outcome), (str(rec.get("reason")),), rec, None, ())
    spec = ctx.sealed.spec(rec.get("run_id"))
    problems = []
    if spec is None or rec.get("spec") != to_json(spec):
        problems.append("record names a run outside the sealed plan")
    if rec.get("plan_digest") != ctx.sealed.digest:
        problems.append("record was made under a different sealed plan")
    if rec.get("provenance_digest") != ctx.sealed.provenance_digest:
        problems.append("record was made under different source or settings")
    evidence = rec.get("evidence")
    if not isinstance(evidence, dict):
        problems.append("record has no evidence digests")
    else:
        now = evidence_digests(path)
        changed = sorted(f for f in set(evidence) | set(now) if evidence.get(f) != now.get(f))
        if changed:
            problems.append(f"evidence changed after completion: {', '.join(changed)}")
    if problems:
        return Verification("invalid", tuple(problems), rec, None, ())
    return validate_evidence(path, spec, ctx)


def reconcile(ctx: RunContext, run_id: str, table: Callable[[], dict[int, Proc]] = process_table) -> None:
    """Settle the latest attempt of a run whose runner died: clean up its owned processes and mark
    it failed, or finish an interrupted validation. Unidentified processes are never signaled."""
    paths = attempts_of(ctx.root, run_id)
    if not paths:
        return
    pending = [p for p in paths if cleanup_problem(read_attempt(p))]
    if pending:
        for path in pending:
            rec = read_attempt(path)
            known = {int(pid): start for pid, start in rec.get("owned_processes") or ()}
            pgid = rec.get("server_pgid")
            if not isinstance(pgid, int):
                raise GateError(f"{path} has unresolved cleanup without a recorded server group; resume blocked")
            cfg = ctx.sealed.config
            cleanup_started = time.time()
            report = terminate_owned(pgid, known, False, cfg.shutdown_grace_s, cfg.kill_grace_s, table=table)
            rec.setdefault("cleanup_reports", []).append(rec.get("stop_report"))
            rec.update(stop_report=report, cleanup_failed=bool(report["survivors"] or report["unidentified"]),
                       cleanup_wall_s=rec.get("cleanup_wall_s", 0.0) + max(0.0, time.time() - cleanup_started),
                       outcome="failed", reason="cleanup: a replacement is required after cleanup failure")
            _write_atomic(path / "attempt.json", rec)
            if rec["cleanup_failed"]:
                raise GateError(f"{path} cleanup remains unresolved ({report}); resume blocked")
            rec["cleanup_resolved_at"] = datetime.now().isoformat()
            _write_atomic(path / "attempt.json", rec)
    path = paths[-1]
    rec = read_attempt(path)
    state = rec.get("outcome")
    if state not in ("running", "validating") or owner_alive(rec):
        return
    if state == "validating":
        finish_attempt(path, rec, rec.get("execution_outcome", "failed"), rec.get("execution_reason"), ctx,
                       wall_ended=rec.get("wall_ended"))
        return
    known = {int(pid): start for pid, start in rec.get("owned_processes") or ()}
    pgid = rec.get("server_pgid")
    report = None
    if isinstance(pgid, int):
        cfg = ctx.sealed.config
        report = terminate_owned(pgid, known, False, cfg.shutdown_grace_s, cfg.kill_grace_s, table=table)
        if report["unidentified"] or report["survivors"]:
            raise GateError(f"{path} has processes in its server group {pgid} that cannot be positively identified "
                            f"or stopped ({report}); inspect them before resuming")
    mtimes = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
    end = max([time.time(), rec.get("wall_checkpoint") or 0.0, *mtimes])
    rec["stop_report"] = report
    finish_attempt(path, rec, "failed", "runner died while the attempt was running", ctx, wall_ended=end)


def latest_verification(ctx: RunContext, spec: RunSpec) -> Verification:
    paths = attempts_of(ctx.root, spec.run_id)
    if not paths:
        return Verification("missing", ("no attempt",), {}, None, ())
    return verify_attempt(paths[-1], ctx)


def next_action(ctx: RunContext, verified: dict) -> tuple[RunSpec, int, str | None, str | None] | None:
    """First run in plan order without an attempt that verifies complete. `verified` caches
    verifications by attempt name and record digest within one runner."""
    for spec in ctx.sealed.runs:
        paths = attempts_of(ctx.root, spec.run_id)
        results = []
        for p in paths:
            key = (p.name, file_sha(p / "attempt.json") if (p / "attempt.json").exists() else None)
            if key not in verified:
                verified[key] = verify_attempt(p, ctx)
            results.append(verified[key])
        if any(v.status == "complete" for v in results):
            if any(cleanup_problem(v.record) for v in results):
                raise GateError(f"{spec.run_id} has unresolved cleanup; reconcile it before continuing")
            continue
        if any(cleanup_problem(v.record) for v in results):
            raise GateError(f"{spec.run_id} has unresolved cleanup; reconcile it before replacing the attempt")
        if results and results[-1].status in ("running", "validating"):
            raise GateError(f"{paths[-1]} is still {results[-1].status} under pid {results[-1].record.get('pid')}")
        if not results:
            return spec, 1, None, None
        last = results[-1]
        recorded = last.record.get("outcome")
        why = f"previous attempt {recorded}: {last.record.get('reason')}" if recorded != "complete" else \
            f"previous attempt recorded complete but fails revalidation: {'; '.join(last.reasons)}"
        return spec, len(paths) + 1, paths[-1].name, why
    return None


class Lock:
    def __init__(self, root: Path):
        self.path = root / "sweep.lock"

    def __enter__(self):
        if self.path.exists():
            pid = int(self.path.read_text().strip() or 0)
            if pid_alive(pid):
                raise GateError(f"another runner (pid {pid}) owns {self.path.parent}")
            self.path.unlink()
        _write_new(self.path, str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def run_attempt(ctx: RunContext, spec: RunSpec, number: int, replaces: str | None, reason: str | None,
                texts: dict[int, str], grant: ExecutionGrant) -> dict:
    grant = require_grant(grant)
    sealed, cfg = ctx.sealed, ctx.sealed.config
    root_before = dir_bytes(ctx.root)
    attempt, rec = begin_attempt(ctx, spec, number, replaces, reason, dataclasses.asdict(grant))
    rec["machine_before"] = machine()

    def persist():
        _write_atomic(attempt / "attempt.json", rec)

    def checkpoint(now):
        rec["wall_checkpoint"] = now
        persist()

    budget = Budget(sealed.ceilings, root_before, attempt, rec["wall_started"] + cfg.attempt_max_s(sealed.deadlines),
                    checkpoint=checkpoint)
    server = None
    outcome, why = "failed", None
    try:
        budget.check()
        if port_busy(cfg.port):
            raise RuntimeError(f"port {cfg.port} is busy before server start")
        run_label = f"{ctx.root.name}/{attempt.name}"
        server = Server(attempt, spec, cfg, grant, run_label)
        rec.update(server_args=server.args, server_env_removed=server.env_removed, server_pgid=server.group.pgid,
                   owned_processes=server.group.identities())
        persist()

        def sampled():
            rec["owned_processes"] = server.group.identities()
            persist()
        server.wait_healthy(budget, sampled)
        budget.check()
        m0 = metrics_text(cfg.url)
        rec["pool_blocks"] = pool_blocks(m0) or pool_blocks_from_log(attempt / "server.log")
        if rec["pool_blocks"] != cfg.num_blocks:
            raise RuntimeError(f"pool has {rec['pool_blocks']} blocks, expected {cfg.num_blocks}")
        warm = asyncio.run(run_warmup(cfg, budget.problem))
        rec["warmup"] = to_json(warm)
        persist()
        if not warm.stable:
            raise Incomplete(f"warmup: {warm.reason}")
        budget.check()
        if not post(cfg.url, "/reset_prefix_cache").get("success"):
            raise RuntimeError("prefix cache reset after warmup failed")
        if spec.logging:
            events = attempt / "events.jsonl"
            first = events.read_bytes().split(b"\n", 1)[0] if events.exists() else b""
            if not first.startswith(b'[0,') or b'"start"' not in first:
                raise RuntimeError("observer did not start its measured history after the reset")
        m1 = metrics_text(cfg.url)
        budget.check()
        server.group.sample()
        rec["owned_processes"] = server.group.identities()
        rec["machine_at_start"] = machine()
        persist()
        g0, t0 = gpu_time_by_pid(), time.monotonic()
        records, fired, rec["t_drain_start_ns"] = asyncio.run(drive(
            spec, texts, f"{ctx.root.name}.{attempt.name}", attempt, sealed.deadlines,
            lambda text, cap, seed: client_body(cfg, text, cap, seed), cfg.url, budget.problem))
        elapsed = time.monotonic() - t0
        server.group.sample()
        rec["owned_processes"] = server.group.identities()
        rec["other_gpu_percent"] = other_gpu_share(g0, gpu_time_by_pid(), elapsed, server.group.pids() | {os.getpid()})
        m2 = metrics_text(cfg.url)
        rec["engine_preemptions"] = (metric_value(m2, "vllm:num_preemptions_total") or 0) - \
            (metric_value(m1, "vllm:num_preemptions_total") or 0)
        rec["elapsed_s"] = elapsed
        budget.check()
        rec["seal_reset"] = post(cfg.url, "/reset_prefix_cache", timeout=cfg.seal_timeout_s)
        if not rec["seal_reset"].get("success"):
            raise RuntimeError("the sealing prefix-cache reset reported failure")
        budget.check()
        outcome, why = ("incomplete", fired) if fired else ("complete", None)
    except BudgetExceeded as e:
        outcome, why = "incomplete", str(e)
    except Incomplete as e:
        outcome, why = "incomplete", str(e)
    except Exception as e:
        outcome, why = "failed", repr(e)[:500]
    finally:
        if server is not None:
            report = server.stop(budget.problem)
            rec["stop_report"] = report
            rec["cleanup_failed"] = bool(report["survivors"] or report["unidentified"])
            if rec["cleanup_failed"]:
                outcome, why = "failed", cleanup_problem(rec)
        rec["machine_after"] = machine()
        problem = budget.problem()
        if problem:
            outcome, why = "incomplete", problem
    return finish_attempt(attempt, rec, outcome, why, ctx)


_SHAS = None


def item_shas() -> dict[int, str]:
    global _SHAS
    if _SHAS is None:
        c = corpus_mod.load(DATA)
        _SHAS = {it.item: it.sha256 for it in c.items}
    return _SHAS


def corpus_texts() -> dict[int, str]:
    c = corpus_mod.load(DATA)
    return {it.item: it.text for it in c.items}


def readiness(main: bool, disk_need_gib: float = 0.0, reserve_gib: float = 0.0) -> dict:
    """Every gate measurement depends on. Raises GateError naming the first failure."""
    facts = {}
    try:
        c = corpus_mod.load(DATA)
    except (corpus_mod.CorpusError, OSError) as e:
        raise GateError(f"corpus is not frozen or does not verify ({e}): run `python -m e02 corpus`")
    facts["corpus_digest"] = c.digest
    facts["predictions_digest"] = check_predictions()
    facts["engine"] = check_engine()
    cfg = current_config()
    if port_busy(cfg.port):
        raise GateError(f"port {cfg.port} is busy; stop the process holding it")
    if "1" in sh("pmset -g | grep -i lowpowermode"):
        raise GateError("Low Power Mode is on")
    free = free_gib(HERE)
    facts["free_gib"] = round(free, 2)
    if free < disk_need_gib + reserve_gib:
        raise GateError(f"{free:.1f} GiB free, need {disk_need_gib:.1f} GiB plus a {reserve_gib:.1f} GiB reserve")
    facts["technical_record"] = check_technical()
    prov = provenance(cfg, c.digest, facts["predictions_digest"], facts["engine"])
    facts["provenance"], facts["provenance_digest"] = prov, digest_of(prov)
    if main:
        facts["pilots"] = pilot_evidence(prov)
    return facts


def latest_root(label: str) -> Path | None:
    roots = sorted(p for p in RUNS.glob(f"*-{label}") if p.is_dir())
    return roots[-1] if roots else None


def pilot_evidence(current: dict) -> dict:
    """Main is allowed only when the latest root of every pilot was sealed under the current
    provenance and its verdict, recomputed from revalidated raw evidence, passes. Saved verdict
    files are never trusted."""
    out = {}
    for kind in PILOTS:
        root = latest_root(f"pilot-{kind}")
        if root is None:
            raise GateError(f"no {kind} pilot: `python -m e02 pilot {kind} --execute` needs Krishna's explicit "
                            "instruction to run it")
        sealed = load_sealed(root)
        if sealed.provenance_digest != digest_of(current):
            drift = provenance_drift(sealed.doc["provenance"], current)
            raise GateError(f"{kind} pilot {root.name} was recorded under different source or settings than the "
                            f"current code (changed: {drift}); rerun it under the current code")
        verdict = behavior_verdict(root) if kind == "behavior" else cost_verdict(root, kind)
        if verdict["verdict"] != "pass":
            raise GateError(f"{kind} pilot {root.name} is {verdict['verdict']}: {verdict['reasons']}")
        out[kind] = {"root": root.name, "plan_digest": sealed.digest, **verdict}
    return out


def run_pilot(kind: str, grant: ExecutionGrant) -> Path:
    grant = require_grant(grant)
    if kind not in PILOTS:
        raise GateError(f"unknown pilot {kind}; choose one of {sorted(PILOTS)}")
    w = PILOTS[kind]
    facts = readiness(main=False, disk_need_gib=w.ceilings.disk_bytes / GIB, reserve_gib=w.ceilings.reserve_bytes / GIB)
    runs = pilot_plan(kind)
    root = new_root(f"pilot-{kind}")
    seal_plan(root, kind="pilot", workload=kind, runs=runs, cfg=current_config(), deadlines=w.deadlines,
              ceilings=w.ceilings, prov=facts["provenance"],
              extra={"technical_record": facts["technical_record"], "item_sha": {str(k): v for k, v in item_shas().items()},
                     "corpus_digest": facts["corpus_digest"], "predictions_digest": facts["predictions_digest"],
                     "cost_rule": cost_rule_text()})
    _write_new(root / "env.json", json.dumps(environment({"root": root.name, "kind": kind}), indent=1) + "\n")
    register_root(root, f"Expanded E02 {kind} pilot, {len(runs)} run(s), not part of the main sweep")
    execute(root, grant)
    verdict = behavior_verdict(root) if kind == "behavior" else cost_verdict(root, kind)
    _write_new(root / "verdict.json", json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    return root


def check_resume(sealed: SealedPlan, facts: dict) -> None:
    """Refuse to continue a sealed root when source, settings, inputs or engine changed, or when
    an attempt was made under another sealed plan."""
    if facts["provenance_digest"] != sealed.provenance_digest:
        drift = provenance_drift(sealed.doc["provenance"], facts["provenance"])
        raise GateError(f"{sealed.root.name} was sealed under different inputs; changed since: {drift}. "
                        "Seal a new root instead of resuming this one")
    for path in sorted((sealed.root / "attempts").glob("*.a*")):
        rec = read_attempt(path)
        if rec.get("plan_digest") != sealed.digest or rec.get("provenance_digest") != sealed.provenance_digest:
            raise GateError(f"{path.name} was made under a different sealed plan or provenance")


def execute(root: Path, grant: ExecutionGrant, after_run: Callable[[RunContext, dict], None] | None = None) -> None:
    grant = require_grant(grant)
    sealed = load_sealed(root)
    c = sealed.ceilings
    facts = readiness(main=sealed.kind == "main", reserve_gib=c.reserve_bytes / GIB)
    check_resume(sealed, facts)
    ctx = RunContext(root, sealed, item_shas())
    texts = corpus_texts()
    attempt_max = sealed.config.attempt_max_s(sealed.deadlines)
    with Lock(root):
        verified: dict = {}
        while True:
            for spec in sealed.runs:
                reconcile(ctx, spec.run_id)
            action = next_action(ctx, verified)
            if action is None:
                problem = admit_attempt(root_consumed_s(root), 0.0, c, dir_bytes(root), free_bytes())
                if problem:
                    halt(root, problem)
                    raise GateError(f"budget: {problem}; completed evidence exceeds the terminal sweep budget")
                return
            spec, number, replaces, reason = action
            if replaces is not None and sealed.kind == "main" and not os.environ.get("E02_REPLACE_REASON"):
                raise GateError(f"{spec.run_id} needs a replacement attempt ({reason}); rerun with "
                                f"E02_REPLACE_REASON='<why>' to record an explicit reason")
            if replaces is not None and os.environ.get("E02_REPLACE_REASON"):
                reason = f"{reason}; operator: {os.environ['E02_REPLACE_REASON']}"
            problem = admit_attempt(root_consumed_s(root), attempt_max, c, dir_bytes(root), free_bytes())
            if problem:
                halt(root, problem)
                raise GateError(f"budget: {problem}; stopped with {root.name} incomplete")
            print(f"{datetime.now():%H:%M:%S} {spec.run_id} attempt {number} starting", flush=True)
            rec = run_attempt(ctx, spec, number, replaces, reason, texts, grant)
            print(f"{datetime.now():%H:%M:%S} {spec.run_id} attempt {number}: {rec['outcome']} {rec['reason'] or ''}",
                  flush=True)
            if rec.get("cleanup_failed"):
                halt(root, f"{spec.run_id} cleanup left processes: {rec['stop_report']}")
                raise GateError(f"{spec.run_id} left server processes behind ({rec['stop_report']}); "
                                "the sweep pauses for review")
            if (rec.get("reason") or "").startswith("budget:"):
                halt(root, rec["reason"])
                raise GateError(f"{rec['reason']}; stopped with {root.name} incomplete")
            if rec["outcome"] == "invalid" or (rec["outcome"] == "failed" and "pool has" in (rec["reason"] or "")):
                raise GateError(f"{spec.run_id} is {rec['outcome']}: {rec['reason']}; the sweep pauses for review")
            if rec["outcome"] != "complete" and sealed.kind == "pilot":
                return
            if after_run is not None:
                after_run(ctx, rec)


def behavior_verdict(root: Path) -> dict:
    try:
        sealed = load_sealed(root)
    except GateError as e:
        return {"verdict": "fail", "reasons": [str(e)]}
    ctx = RunContext(root, sealed, item_shas())
    spec = sealed.runs[0]
    v = latest_verification(ctx, spec)
    if v.status != "complete":
        return {"verdict": "fail", "reasons": [f"attempt {v.status}: {'; '.join(v.reasons[:3])}"]}
    rec, m = v.record, v.history.metrics
    reasons = []
    if not m.get("preemptions"):
        reasons.append("no preemption occurred, so decision and recovery capture are untested")
    if m.get("stock_rule_violations"):
        reasons.append(f"{m['stock_rule_violations']} decisions differ from the last-running rule")
    if rec.get("engine_preemptions") != m.get("preemptions"):
        reasons.append(f"engine counter {rec.get('engine_preemptions')} preemptions, history {m.get('preemptions')}")
    path = attempts_of(root, spec.run_id)[-1]
    return {"verdict": "fail" if reasons else "pass", "reasons": reasons, "requests": spec.requests, "cap": spec.cap,
            "preemptions": m.get("preemptions"), "episodes_resumed": m.get("episodes_resumed"),
            "snapshots_checked": m.get("snapshots_checked"), "digests_checked": m.get("digests_checked"),
            "clients_with_reasoning_field": sum(1 for c in v.clients if c.reasoning_chunks),
            "events_bytes": (path / "events.jsonl").stat().st_size, "attempt_bytes": rec.get("bytes"),
            "elapsed_s": rec.get("elapsed_s"), "wall_s": attempt_consumed_s(rec, 0.0)}


def cost_rule_text() -> str:
    return (f"Pass when the median paired change in output-token throughput and in median TPOT, logging on "
            f"versus off, is within {COST_BOUND:.0%}, and the unobserved runs' range is within {COST_NOISE_BOUND:.0%} "
            f"of their median for both metrics. Fail when the median paired change exceeds {COST_BOUND:.0%} in the "
            f"costly direction and at least 4 of {COST_PAIRS} pairs agree. Otherwise inconclusive. Every run must "
            f"revalidate complete from its raw evidence, with all requests finished; the pressure workload must "
            f"preempt on both sides, and each observed run's engine counter must equal its history.")


def cost_verdict(root: Path, kind: str) -> dict:
    try:
        sealed = load_sealed(root)
    except GateError as e:
        return {"verdict": "inconclusive", "reasons": [str(e)], "rule": cost_rule_text()}
    if sealed.workload != kind:
        return {"verdict": "inconclusive", "reasons": [f"{root.name} is a {sealed.workload} root"],
                "rule": cost_rule_text()}
    ctx = RunContext(root, sealed, item_shas())
    reasons, rows = [], []
    for spec in sealed.runs:
        v = latest_verification(ctx, spec)
        if v.status != "complete":
            reasons.append(f"{spec.run_id} is {v.status}: {'; '.join(v.reasons[:3])}")
            continue
        rec, clients = v.record, v.clients
        timing = [history.request_timing(c) for c in clients]
        span = (max(c.t_end_ns for c in clients) - min(c.t_submit_ns for c in clients)) / 1e9
        if span <= 0:
            reasons.append(f"{spec.run_id} has no positive client elapsed interval")
            continue
        at_start = rec.get("machine_at_start") or {}
        rows.append({"run_id": spec.run_id, "pair": spec.round, "logging": spec.logging,
                     "throughput": sum(c.completion_tokens for c in clients) / span,
                     "tpot": history.median([t["tpot_s"] for t in timing]),
                     "preemptions": rec.get("engine_preemptions"),
                     "observed_preemptions": v.history.metrics.get("preemptions") if v.history else None,
                     "therm": at_start.get("therm"), "load": at_start.get("load"),
                     "warmup_stable": (rec.get("warmup") or {}).get("stable"),
                     "other_gpu": rec.get("other_gpu_percent"), "bytes": rec.get("bytes"),
                     "elapsed_s": rec.get("elapsed_s")})
    if kind == "cost-pressure" and any(not r["preemptions"] for r in rows):
        reasons.append("a pressure run did not preempt, so it does not test logging under pressure")
    for r in rows:
        if r["logging"] and r["observed_preemptions"] != r["preemptions"]:
            reasons.append(f"{r['run_id']}: engine counter {r['preemptions']}, history {r['observed_preemptions']}")
    pairs = {}
    for r in rows:
        if any(r[m] is None or not math.isfinite(r[m]) or r[m] <= 0 for m in ("throughput", "tpot")):
            reasons.append(f"{r['run_id']} has no positive throughput or TPOT for a cost comparison")
            continue
        pairs.setdefault(r["pair"], {})[r["logging"]] = r
    deltas = {"throughput": [], "tpot": []}
    for p in sorted(pairs):
        if set(pairs[p]) != {True, False}:
            continue
        on, off = pairs[p][True], pairs[p][False]
        deltas["throughput"].append(on["throughput"] / off["throughput"] - 1)
        deltas["tpot"].append(on["tpot"] / off["tpot"] - 1)
    summary = {"rule": cost_rule_text(), "runs": rows, "paired_deltas": deltas}
    if reasons or len(deltas["throughput"]) < COST_PAIRS:
        reasons.append(f"{len(deltas['throughput'])} of {COST_PAIRS} complete pairs")
        return {"verdict": "inconclusive", "reasons": reasons, **summary}
    offs = [r for r in rows if not r["logging"]]
    noise = {m: (max(r[m] for r in offs) - min(r[m] for r in offs)) / statistics.median(r[m] for r in offs)
             for m in ("throughput", "tpot")}
    med = {m: statistics.median(v) for m, v in deltas.items()}
    costly = {"throughput": lambda d: d < -COST_BOUND, "tpot": lambda d: d > COST_BOUND}
    summary.update({"median_delta": med, "off_relative_range": noise})
    for m in ("throughput", "tpot"):
        if costly[m](med[m]) and sum(costly[m](d) for d in deltas[m]) >= 4:
            return {"verdict": "fail", "reasons": [f"{m} median paired change {med[m]:+.1%}"], **summary}
    if all(abs(med[m]) <= COST_BOUND and noise[m] <= COST_NOISE_BOUND for m in med):
        return {"verdict": "pass", "reasons": [], **summary}
    return {"verdict": "inconclusive",
            "reasons": [f"median changes {med['throughput']:+.1%} throughput, {med['tpot']:+.1%} TPOT; off-run "
                        f"ranges {noise['throughput']:.1%} and {noise['tpot']:.1%}"], **summary}


def forecast_budget(pilots: dict) -> dict:
    """Scale the verified behavior pilot's per-request time and bytes, at its recorded cap and
    request count, to every main run by cap, and add its measured per-attempt overhead (startup,
    warmup, shutdown and validation). A planning estimate, not a measurement: per-request time
    differs by target, and caps change output length."""
    b = pilots["behavior"]
    per_request_s = b["elapsed_s"] / b["requests"]
    per_request_bytes = b["attempt_bytes"] / b["requests"]
    overhead_s = max(0.0, b["wall_s"] - b["elapsed_s"])
    runs = main_plan()
    seconds = sum(per_request_s * r.requests * r.cap / b["cap"] + overhead_s for r in runs)
    gib = sum(per_request_bytes * r.requests * r.cap / b["cap"] for r in runs) / GIB
    return {"hours": seconds / 3600, "gib": gib,
            "basis": f"behavior pilot {b['root']}: per-request elapsed time and bytes at cap {b['cap']}, scaled by "
                     f"each run's cap, plus {overhead_s:.0f} s measured overhead per attempt"}


def freeze_main(wall_hours: float, disk_gib: float, reserve_gib: float, run_wall_min: float,
                no_progress_s: float) -> Path:
    facts = readiness(main=True, disk_need_gib=0, reserve_gib=reserve_gib)
    forecast = forecast_budget(facts["pilots"])
    if forecast["hours"] > wall_hours or forecast["gib"] > disk_gib:
        raise GateError(f"pilot forecast {forecast['hours']:.1f} h and {forecast['gib']:.1f} GiB exceeds the "
                        f"ceilings {wall_hours} h and {disk_gib} GiB: execution is infeasible under that budget")
    if free_gib(HERE) < disk_gib + reserve_gib:
        raise GateError(f"{free_gib(HERE):.1f} GiB free is below the {disk_gib} GiB ceiling plus {reserve_gib} GiB reserve")
    cfg = current_config()
    deadlines = Deadlines(float(run_wall_min) * 60, float(no_progress_s))
    ceilings = Ceilings(float(wall_hours) * 3600, int(disk_gib * GIB), int(reserve_gib * GIB), FINAL_RECORDS_BYTES)
    if cfg.attempt_max_s(deadlines) > ceilings.wall_s:
        raise GateError("one attempt's upper bound exceeds the whole wall ceiling")
    root = new_root("main")
    seal_plan(root, kind="main", workload="main", runs=main_plan(), cfg=cfg, deadlines=deadlines, ceilings=ceilings,
              prov=facts["provenance"],
              extra={"forecast": forecast, "pilots": facts["pilots"], "technical_record": facts["technical_record"],
                     "corpus_digest": facts["corpus_digest"], "predictions_digest": facts["predictions_digest"],
                     "item_sha": {str(k): v for k, v in item_shas().items()}})
    _write_new(root / "env.json", json.dumps(environment({"root": root.name, "kind": "main"}), indent=1) + "\n")
    register_root(root, "Expanded E02 main sweep: 20 configurations x 5 repetitions x 100 requests")
    return root


def load_manifest(root: Path) -> SealedPlan:
    sealed = load_sealed(root)
    if sealed.kind != "main":
        raise GateError(f"{root.name} is not a main-sweep root")
    return sealed


def run_main(root: Path, grant: ExecutionGrant) -> None:
    grant = require_grant(grant)
    sealed = load_manifest(root)
    first_round = len(TARGETS) * len(CAPS)

    def after_run(ctx, rec):
        recs = [read_attempt(p) for p in sorted((root / "attempts").glob("*.a*"))]
        done = [r for r in recs if r.get("outcome") == "complete"]
        if len(done) < first_round:
            return
        remaining = len(sealed.runs) - len({r["run_id"] for r in done})
        per_run = statistics.median(attempt_consumed_s(r, 0.0) for r in done)
        consumed = root_consumed_s(root)
        if consumed + per_run * remaining > sealed.ceilings.wall_s:
            reason = (f"forecast {per_run * remaining / 3600:.1f} h remaining after {consumed / 3600:.1f} h used "
                      f"exceeds the {sealed.ceilings.wall_s / 3600:.1f} h ceiling")
            halt(root, reason)
            raise GateError(f"{reason}; stopped incomplete")

    execute(root, grant, after_run)


def sweep_status(root: Path) -> dict:
    """Per-run status from revalidated evidence. Read-only: never rewrites an attempt."""
    halts = (root / "halts.jsonl").read_text().splitlines() if (root / "halts.jsonl").exists() else []
    base = {"root": root.name, "bytes": dir_bytes(root), "free_gib": round(free_gib(HERE), 2),
            "consumed_hours": round(root_consumed_s(root) / 3600, 3), "halts": [json.loads(h) for h in halts]}
    try:
        sealed = load_sealed(root)
    except GateError as e:
        plan = json.loads(plan_path(root).read_text()) if plan_path(root).exists() else {"runs": []}
        rows = []
        for r in plan.get("runs", []):
            recs = [read_attempt(p) for p in attempts_of(root, r["run_id"])]
            rows.append({"run_id": r["run_id"], "attempts": len(recs),
                         "recorded": recs[-1].get("outcome") if recs else "planned", "verified": None})
        return {**base, "sealed": False, "unverifiable": str(e), "runs": rows, "complete": 0,
                "remaining": len(rows), "forecast_remaining_hours": None}
    ctx = RunContext(root, sealed, item_shas())
    rows, durations = [], []
    for spec in sealed.runs:
        paths = attempts_of(root, spec.run_id)
        checks = [verify_attempt(p, ctx) for p in paths]
        good = [v for v in checks if v.status == "complete"]
        if good:
            durations.append(attempt_consumed_s(good[-1].record, 0.0))
        latest = checks[-1] if checks else None
        state = latest.status if latest else "planned"
        if latest and state in ("running", "validating") and not owner_alive(latest.record):
            state = f"{state} (runner dead; resume reconciles it)"
        rows.append({"run_id": spec.run_id, "attempts": len(paths),
                     "recorded": latest.record.get("outcome") if latest else "planned",
                     "verified": "complete" if good else state,
                     "reasons": list(latest.reasons) if latest and not good else []})
    complete = sum(1 for x in rows if x["verified"] == "complete")
    remaining = len(rows) - complete
    forecast = statistics.median(durations) * remaining / 3600 if durations else None
    return {**base, "sealed": True, "plan_digest": sealed.digest, "ceilings": to_json(sealed.ceilings),
            "runs": rows, "complete": complete, "remaining": remaining, "forecast_remaining_hours": forecast}

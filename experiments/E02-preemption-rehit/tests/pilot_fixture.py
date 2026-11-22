
import dataclasses
import json

from e02 import history, sweep
from e02.observer import ObservedScheduler, token_sha256
from engine_fixture import Engine, make_scheduler
from scenarios import SELF, engine_kwargs

GIB = sweep.GIB
DEAD_PID = 2**22 + 12345
SMALL = {k: sweep.PilotWorkload(4, 64, 4, sweep.Deadlines(60.0, 30.0), sweep.Ceilings(3600.0, GIB, 0, 0))
         for k in ("behavior", "cost-low", "cost-pressure")}
CFG = dataclasses.replace(sweep.current_config(), num_blocks=SELF["num_blocks"], snapshot_every=32, shutdown_grace_s=0.5,
                          kill_grace_s=5.0)
ITEMS = sweep.item_order(11)[:4]
REQUESTS = [r for step in sorted(SELF["arrivals"]) for r in SELF["arrivals"][step]]
ITEM_OF = {name: item for (name, _, _), item in zip(REQUESTS, ITEMS)}
ITEM_SHA = {ITEM_OF[name]: token_sha256(prompt) for name, prompt, _ in REQUESTS}


def install(monkeypatch):
    monkeypatch.setattr(sweep, "PILOTS", SMALL)
    monkeypatch.setattr(sweep, "item_shas", lambda: ITEM_SHA)
    monkeypatch.delenv("E02_DIGEST_EVERY", raising=False)


def provenance(tag="current"):
    return {"config": sweep.to_json(CFG), "tag": tag}


def sealed_root(runs_dir, kind, prov=None, stamp="20261020-000000"):
    root = runs_dir / f"{stamp}-pilot-{kind}"
    (root / "attempts").mkdir(parents=True)
    w = sweep.PILOTS[kind]
    sweep.seal_plan(root, kind="pilot", workload=kind, runs=sweep.pilot_plan(kind), cfg=CFG, deadlines=w.deadlines,
                    ceilings=w.ceilings, prov=prov or provenance(), extra={})
    return root, sweep.RunContext(root, sweep.load_sealed(root), ITEM_SHA)


def observe_self(path, label, prefix):
    s = make_scheduler(ObservedScheduler, out_dir=path, run_id=label, snapshot_every=CFG.snapshot_every,
                       **engine_kwargs(SELF))
    assert s.reset_prefix_cache()
    arrivals = {step: [(f"chatcmpl-{prefix}.i{ITEM_OF[name]}-abcdefgh", prompt, m) for name, prompt, m in reqs]
                for step, reqs in SELF["arrivals"].items()}
    Engine(s).run(arrivals=arrivals)
    assert s.reset_prefix_cache()


def write_clients(path, prefix, rate=1.0, tokens=60, items=ITEMS):
    with open(path / "client.jsonl", "w") as f:
        for i, item in enumerate(items):
            start = i * 1_000_000
            decode = int(tokens / (100 * rate) * 1e9)
            cid = f"{prefix}.i{item}"
            rec = history.ClientRecord(cid, item, i, "startup", start, start + 10_000_000, start + 10_000_000 + decode,
                                       start + 10_000_000 + decode, "ok", 200, None, "length", 40, tokens, tokens,
                                       tokens, 0, 1_000_000, f"chatcmpl-{cid}")
            f.write(json.dumps(dataclasses.asdict(rec)) + "\n")


def start_attempt(ctx, run_id, number=1, rate=1.0, preemptions=1):
    """An attempt with its evidence written, before the runner finishes it."""
    spec = ctx.sealed.spec(run_id)
    path, rec = sweep.begin_attempt(ctx, spec, number, None, None)
    prefix = f"{ctx.root.name}.{path.name}"
    if spec.logging:
        observe_self(path, f"{ctx.root.name}/{path.name}", prefix)
    write_clients(path, prefix, rate)
    rule = ctx.sealed.config.warmup
    warm = sweep.WarmupResult(True, "controlled CPU samples", tuple(
        sweep.WarmupSample(i + 1, rule.prompts * rule.tokens, 1.0, "")
        for i in range(max(rule.min_rounds, rule.window))))
    rec.update(engine_preemptions=preemptions, elapsed_s=10.0, machine_at_start={"therm": "", "load": ""},
               warmup=sweep.to_json(warm))
    return path, rec


def make_attempt(ctx, run_id, rate=1.0, preemptions=1):
    path, rec = start_attempt(ctx, run_id, rate=rate, preemptions=preemptions)
    return sweep.finish_attempt(path, rec, "complete", None, ctx)


def behavior_root(runs_dir, prov=None):
    root, ctx = sealed_root(runs_dir, "behavior", prov, stamp="20261020-000000")
    make_attempt(ctx, "behavior")
    return root


def cost_root(runs_dir, kind, on_factor=1.0, noisy=False, preempt=1, prov=None, stamp="20261020-000001"):
    root, ctx = sealed_root(runs_dir, kind, prov, stamp=stamp)
    for k, spec in enumerate(ctx.sealed.runs):
        rate = (on_factor if spec.logging else 1.0) * (1.0 + (0.3 * (k % 3) if noisy else 0.0))
        make_attempt(ctx, spec.run_id, rate=rate, preemptions=preempt)
    return root

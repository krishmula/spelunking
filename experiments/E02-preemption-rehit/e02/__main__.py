"""Expanded E02 command line. Run from the experiment directory with the engine's Python.

Execution authority: workload commands are suspended. On 2026-10-20 at 14:33 PDT Krishna said
"Don't run the experiment unless I tell you so." `pilot` and `main` start a model server and GPU
work only with --execute. Pass --execute only after Krishna's new, explicit instruction to run that
workload. The flag records that instruction; it does not grant it. Every other command is offline.

  python -m e02 corpus [--check]       build and freeze the 100-prompt corpus, or verify it rebuilds identically
  python -m e02 predictions            freeze the recorded predictions by digest
  python -m e02 plan [--json]          dry run: the exact 100-run main plan, touches nothing
  python -m e02 check [--main]         readiness gates; --main adds revalidated pilot evidence
  python -m e02 technical [--twin]     offline real-scheduler tests (pytest); records technical/<ts>/result.json
  python -m e02 pilot KIND --execute   behavior, cost-low, or cost-pressure; bounded workloads (suspended)
  python -m e02 verdict ROOT           recompute a pilot verdict from revalidated raw evidence
  python -m e02 freeze --wall-hours H --disk-gib G --reserve-gib R --run-wall-min M --no-progress-s S
  python -m e02 main ROOT --execute    run or resume the sealed main sweep (suspended)
  python -m e02 status ROOT            per-run revalidated status and budget use; read-only
  python -m e02 replay ATTEMPT_DIR     validate one attempt's journal from its raw files
  python -m e02 report ROOT            HTML and CSV review from validated history
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from e02 import corpus, history, sweep  # noqa: E402

AUDIT_PREDICTION = sweep.HERE.parents[1] / ".audit" / "e02-implementation" / "user-prediction.json"
EXECUTE_HELP = "start the workload; pass only after Krishna's explicit instruction to run it"


def cmd_corpus(a):
    if a.check:
        frozen = corpus.load(sweep.DATA)
        rebuilt = corpus.build()
        same = rebuilt.digest == frozen.digest
        print(json.dumps({"frozen": frozen.digest, "rebuilt": rebuilt.digest, "identical": same}))
        return 0 if same else 1
    c = corpus.build()
    print(json.dumps({"result": corpus.write(c, sweep.DATA), "digest": c.digest, "census": c.census}, indent=1))
    return 0


def cmd_plan(a):
    runs = sweep.main_plan()
    if a.json:
        print(json.dumps([r.__dict__ for r in runs], default=list, indent=1))
        return 0
    rounds = {}
    for r in runs:
        rounds.setdefault(r.round, []).append(r)
    print(f"{len(runs)} runs, {sum(r.requests for r in runs)} requests, pool {sweep.NUM_BLOCKS} blocks of "
          f"{sweep.BLOCK_SIZE} ({sweep.NUM_BLOCKS - 1} usable), server args {' '.join(sweep.SERVER_ARGS)}")
    for rnd, rs in rounds.items():
        print(f"round {rnd} rep seed {rs[0].rep_seed}: first items {list(rs[0].item_order[:6])}, "
              f"first item seeds {list(rs[0].item_seeds[:3])}")
        print("  order: " + ", ".join(f"t{r.target}/c{r.cap}" for r in rs))
    return 0


def cmd_technical(a):
    code, out = sweep.run_technical(["tests/test_twin.py"] if a.twin else ["tests"])
    print(out / "result.json")
    return code


def cmd_pilot(a):
    grant = sweep.execution_grant(a.execute, f"pilot {a.kind}")
    print(sweep.run_pilot(a.kind, grant))
    return 0


def cmd_main(a):
    grant = sweep.execution_grant(a.execute, f"main {a.root}")
    sweep.run_main(Path(a.root), grant)
    return 0


def cmd_verdict(a):
    root = Path(a.root)
    kind = root.name.split("pilot-", 1)[-1]
    v = sweep.behavior_verdict(root) if kind == "behavior" else sweep.cost_verdict(root, kind)
    print(json.dumps(v, indent=1))
    return 0 if v["verdict"] == "pass" else 1


def cmd_freeze(a):
    root = sweep.freeze_main(a.wall_hours, a.disk_gib, a.reserve_gib, a.run_wall_min, a.no_progress_s)
    print(root)
    return 0


def cmd_replay(a):
    h = history.load_run(Path(a.attempt), item_sha=sweep.item_shas())
    print(json.dumps({"status": h.status, "reasons": h.reasons, "journal": h.journal.status,
                      "metrics": h.metrics}, indent=1, default=str))
    return 0 if h.status == "complete" else 1


def cmd_report(a):
    from e02 import report
    print(report.render(Path(a.root)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m e02", description=__doc__.split("\n\n")[0],
                                 epilog=sweep.AUTHORITY)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("corpus"); p.add_argument("--check", action="store_true"); p.set_defaults(f=cmd_corpus)
    p = sub.add_parser("predictions"); p.add_argument("--source", default=str(AUDIT_PREDICTION))
    p.set_defaults(f=lambda a: print(sweep.freeze_predictions(Path(a.source))) or 0)
    p = sub.add_parser("plan"); p.add_argument("--json", action="store_true"); p.set_defaults(f=cmd_plan)
    p = sub.add_parser("check"); p.add_argument("--main", action="store_true")
    p.set_defaults(f=lambda a: print(json.dumps(sweep.readiness(main=a.main), indent=1, default=str)) or 0)
    p = sub.add_parser("technical"); p.add_argument("--twin", action="store_true"); p.set_defaults(f=cmd_technical)
    p = sub.add_parser("pilot", description=sweep.AUTHORITY); p.add_argument("kind", choices=sorted(sweep.PILOTS))
    p.add_argument("--execute", action="store_true", help=EXECUTE_HELP); p.set_defaults(f=cmd_pilot)
    p = sub.add_parser("verdict"); p.add_argument("root"); p.set_defaults(f=cmd_verdict)
    p = sub.add_parser("freeze")
    for flag in ("--wall-hours", "--disk-gib", "--reserve-gib", "--run-wall-min", "--no-progress-s"):
        p.add_argument(flag, type=float, required=True)
    p.set_defaults(f=cmd_freeze)
    p = sub.add_parser("main", description=sweep.AUTHORITY); p.add_argument("root")
    p.add_argument("--execute", action="store_true", help=EXECUTE_HELP); p.set_defaults(f=cmd_main)
    p = sub.add_parser("status"); p.add_argument("root")
    p.set_defaults(f=lambda a: print(json.dumps(sweep.sweep_status(Path(a.root)), indent=1, default=str)) or 0)
    p = sub.add_parser("replay"); p.add_argument("attempt"); p.set_defaults(f=cmd_replay)
    p = sub.add_parser("report"); p.add_argument("root"); p.set_defaults(f=cmd_report)
    a = ap.parse_args(argv)
    try:
        return a.f(a)
    except (sweep.GateError, sweep.BudgetExceeded) as e:
        print(f"blocked: {e}", file=sys.stderr)
        return 2
    except corpus.CorpusError as e:
        print(f"corpus: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

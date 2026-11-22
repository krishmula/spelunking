"""Replay only the permitted new behavior pilot and render a fresh scratch copy.

Run with the engine Python. Each invocation preserves its own output directory.
This script cannot start a model server or select historical evidence.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "experiments" / "E02-preemption-rehit"
PILOT = "20261020-141844-pilot-behavior"
sys.path.insert(0, str(EXPERIMENT))

from e02 import corpus, history, report


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "fix-round1-offline" /
                    datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = EXPERIMENT / "runs" / PILOT
    before = hashes(source)
    scratch = args.output / PILOT
    shutil.copytree(source, scratch)
    c = corpus.load(EXPERIMENT / "data")
    attempt = scratch / "attempts" / "behavior.a1"
    h = history.load_run(attempt, {i.item: i.sha256 for i in c.items})
    assert (h.status, len(h.replay.decisions), h.replay.snapshots_checked, h.replay.digests_checked,
            len(h.joined)) == ("complete", 19, 51, 7731, 24)
    out = report.render(scratch)
    first = hashes(out)
    report.render(scratch)
    assert hashes(out) == first
    with (out / "decisions.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 166
    for row in rows:
        for key in ("snapshot_link", "request_link"):
            page, anchor = row[key].split("#")
            assert f'id="{anchor}"' in (out / page).read_text()
    page = (out / "runs" / "behavior.a1.html").read_text()
    assert "host schedule-time proxy" in page
    assert "device time unavailable" in page
    with (out / "distributions.csv").open() as f:
        distributions = list(csv.DictReader(f))
    assert distributions
    links = re.findall(r'href="([^\"]+)"', (out / "index.html").read_text())
    for link in links:
        if "://" not in link and not link.startswith("#"):
            assert (out / link.split("#")[0]).exists(), link
    assert hashes(source) == before
    result = {"source": str(source), "scratch": str(scratch), "status": h.status,
              "decisions": 19, "snapshots": 51, "digests": 7731, "joined": 24, "candidate_rows": len(rows),
              "distribution_rows": len(distributions), "deterministic_report": True,
              "original_evidence_unchanged": True, "source_hashes": before, "report_hashes": first,
              "limits": "Earlier unsealed pilot; replay compatibility is not current pilot qualification."}
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output / "result.json")


if __name__ == "__main__":
    main()

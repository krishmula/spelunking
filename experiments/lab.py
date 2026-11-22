"""Filing system for experiments. `python lab.py new <slug>` scaffolds a folder; `python lab.py check` lints everything.

An experiment is one folder E<nn>-<slug>/ holding README.md (the review page), its scripts, and
runs/<timestamp>-<label>/ (raw results, one folder per run). LEDGER.tsv indexes them, one row per
finding. The check enforces the rules that keep reviews cheap: predictions written before the
first run, every run carrying its env.json, every README and ledger row naming a real
dossier candidate as R<nn>-C<n> (topic folder number, then card id).
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "research"
SECTIONS = ["Question", "Why now", "Prediction", "The one change", "How to run", "Runs", "Result", "Why", "Verdict"]
LEDGER_COLUMNS = ["id", "date", "exp", "candidate", "change", "predicted", "measured", "noise", "surprise", "cause_known", "next"]
TEMPLATE = """# {id}: {title}

Candidate: foundation
Status: planned

## Question

## Why now

## Prediction

Write before the first run. A number and a reason each.

| Who | Prediction | Reason |
|---|---|---|
| Krishna | | |
| Claude | | |

## The one change

What differs from the baseline. Exactly one thing.

## How to run

```sh
source ~/dev/vllm-lab/.venv/bin/activate
cd ~/dev/spelunking/experiments/{id}-{slug}
```

## Runs

| Run | What | Key numbers |
|---|---|---|

## Result

## Why

## Verdict

Surprise: . Cause known: . Next: .
"""


def experiments():
    return sorted(p for p in HERE.iterdir() if re.match(r"E\d\d-", p.name) and p.is_dir())


def section(text, name):
    m = re.search(rf"^## {re.escape(name)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m[1] if m else None


def new(slug):
    n = max((int(p.name[1:3]) for p in experiments()), default=-1) + 1
    exp_id = f"E{n:02d}"
    folder = HERE / f"{exp_id}-{slug}"
    folder.mkdir()
    (folder / "runs").mkdir()
    (folder / "README.md").write_text(TEMPLATE.format(id=exp_id, slug=slug, title=slug.replace("-", " ")))
    print(folder)


def predicted(prediction):
    """True when the Prediction table has a data row with something written after the first cell."""
    rows = [l for l in prediction.splitlines() if l.startswith("|") and not re.fullmatch(r"[|\s:-]+", l)]
    return any(any(c.strip() for c in r.strip("|").split("|")[1:]) for r in rows[1:])


def candidates():
    """'foundation' plus R<nn>-C<n> for every card in research/R<nn>-<slug>/candidates.md."""
    ids = {"foundation"}
    for card in RESEARCH.glob("R[0-9][0-9]-*/candidates.md"):
        topic = card.parent.name.split("-")[0]
        ids |= {f"{topic}-{c}" for c in re.findall(r"^#+\s*(C\d+)\b", card.read_text(), re.M)}
    return ids


def check():
    problems = []
    folders = {p.name for p in experiments()}
    known = candidates()
    for topic in RESEARCH.iterdir():
        if topic.is_dir() and not re.match(r"R\d{2}-", topic.name):
            problems.append(f"research/{topic.name}: topic folder is not named R<nn>-<slug>")
    for folder in experiments():
        readme = folder / "README.md"
        if not readme.exists():
            problems.append(f"{folder.name}: no README.md")
            continue
        text = readme.read_text()
        named = re.search(r"^Candidate:\s*(\S+)", text, re.M)
        if not named or named.group(1) not in known:
            problems.append(f"{folder.name}/README.md: Candidate line must be 'foundation' or R<nn>-C<n> naming a real card")
        for name in SECTIONS:
            if section(text, name) is None:
                problems.append(f"{folder.name}/README.md: missing section '## {name}'")
        runs = sorted(p for p in (folder / "runs").glob("*") if p.is_dir()) if (folder / "runs").exists() else []
        if runs and not predicted(section(text, "Prediction") or ""):
            problems.append(f"{folder.name}/README.md: runs exist but the Prediction table is empty")
        for run in runs:
            if not re.match(r"\d{8}-\d{6}-[\w.-]+$", run.name):
                problems.append(f"{folder.name}/runs/{run.name}: name is not <timestamp>-<label>")
            if not (run / "env.json").exists():
                problems.append(f"{folder.name}/runs/{run.name}: no env.json (was this run recorded by a script?)")
        for run in runs:
            if run.name not in text:
                problems.append(f"{folder.name}/README.md: run {run.name} is not listed under Runs")
    ledger = (HERE / "LEDGER.tsv").read_text().rstrip("\n").split("\n")
    if ledger[0].split("\t") != LEDGER_COLUMNS:
        problems.append(f"LEDGER.tsv: header must be {' '.join(LEDGER_COLUMNS)}")
    seen = set()
    for i, line in enumerate(ledger[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(LEDGER_COLUMNS):
            problems.append(f"LEDGER.tsv:{i}: {len(cells)} cells, expected {len(LEDGER_COLUMNS)}")
            continue
        row = dict(zip(LEDGER_COLUMNS, cells))
        if row["id"] in seen:
            problems.append(f"LEDGER.tsv:{i}: duplicate id {row['id']}")
        seen.add(row["id"])
        if row["exp"] not in folders:
            problems.append(f"LEDGER.tsv:{i}: exp '{row['exp']}' is not an experiment folder")
        if row["candidate"] not in known:
            problems.append(f"LEDGER.tsv:{i}: candidate '{row['candidate']}' is not 'foundation' or R<nn>-C<n> naming a card in research/R<nn>-<slug>/candidates.md")
        for col in ("surprise", "cause_known"):
            if not re.match(r"(yes|no|partly|n/a)\b", row[col]):
                problems.append(f"LEDGER.tsv:{i}: {col} must start with yes, no, partly, or n/a")
    for p in problems:
        print(p)
    print(f"{len(folders)} experiments, {len(ledger) - 1} ledger rows, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "new":
        new(sys.argv[2])
    elif sys.argv[1:] == ["check"]:
        sys.exit(check())
    else:
        sys.exit("usage: lab.py new <slug> | lab.py check")

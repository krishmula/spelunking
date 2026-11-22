# experiments

One folder per experiment. One row per finding in `LEDGER.tsv`. One short entry per experiment in `JOURNAL.md`.

## Layout

```
experiments/
  README.md           this file: how experiments are filed and reviewed
  JOURNAL.md          the story: prediction scoreboard, open questions, one entry per experiment (read this first)
  LEDGER.tsv          the index: one row per finding, predicted vs measured
  lab.py              `python lab.py new <slug>` scaffolds a folder; `python lab.py check` lints everything
```

## An experiment folder

```
E<nn>-<slug>/
  README.md                       the review page (sections below, in this order)
  <script>.py                     the rerunnable measurement; every script the experiment needs lives here
  runs/<timestamp>-<label>/       raw results of one run: env.json, server.log, summary.*, per-pass JSON
```

README sections: Question, Why now, Prediction, The one change, How to run, Runs, Result, Why, Verdict.
The top two lines give the dossier candidate it tests as `R<nn>-C<n>` (`foundation` for measurements that precede any candidate)
and the status.

## Rules the check enforces

- The Prediction table is filled before the first run folder exists. A wrong prediction is the valuable kind;
  an absent one makes the result unexplainable.
- Every run folder has an `env.json`, which means a script produced it and recorded versions, SHAs, heat, swap.
- Every run folder is listed in the README's Runs table, so nothing is quietly dropped.
- Every ledger row points at an existing folder and a candidate written as `R<nn>-C<n>` (for example `R02-C8`) that exists in that topic's `candidates.md`.

## Words

The `## Glossary` in [../AGENTS.md](../AGENTS.md) gives one name for each thing. Every document here uses those
words and no synonyms.

## The journal

`JOURNAL.md` opens with the prediction scoreboard (every prediction, both of ours, against the measurement) and the
open questions list. Then one entry per experiment, newest last:

- A `**Surprise:**` line first: what broke our model of the system.
- Two or three paragraphs in plain words: what we asked, what happened, what it changed.
- `**Decision (who, date):** what. Why: one line.` for every choice made along the way.
- A link to the experiment's README, which holds the full reasoning and numbers.

Update the scoreboard when predictions are written and again when results land.

## Measurement rules

Each experiment has its own script and its own "before" measurement; there is no shared baseline. What carries over is
the method, which the check cannot enforce:

- One change per experiment.
- Warm the chip up before recording. The Air has no fan and slows by about 30% once hot, so measure from a steady hot
  state, not a cold start.
- Run "before" and "after" in the same session, interleaved or shuffled, so heat drift hits both equally. Never
  compare numbers from different sessions.
- Record the environment with every run (versions, commits, heat, swap) in `env.json`.
- Repeat enough to know this measurement's noise floor before calling a change real.
- Camera and mic apps closed; the Mac left alone during a run.

To reuse code from another experiment, copy it into the new folder. Pull it into a shared place only once two
experiments actually need it.

## Ledger columns

| Column | Meaning |
|---|---|
| id | `E<nn>` plus a letter when a folder yields several findings |
| exp | the folder |
| candidate | `foundation` or a card id from the research dossier |
| change | the one thing that differed, with the run folders |
| predicted, measured | the delta each of us wrote down, and the delta the run showed |
| noise | the smallest change the setup could have detected |
| surprise | did the result disagree with the prediction |
| cause_known | do we know why, after the run |
| next | what the result makes worth doing |

A row with surprise `yes` or cause_known `no` is worth following up. A row with neither is a reproduction and stops
there.

## How to review one experiment

1. Read its row in `LEDGER.tsv`: predicted against measured, and whether the cause is known.
2. Open the folder's README: Question, Prediction, Result, Why, Verdict. Five short sections.
3. If a number looks off, the run folder holds everything: `summary.txt` first, then `env.json` for heat, swap, and
   KV cache size, then `server.log`.
4. Its JOURNAL entry tells the story in two or three paragraphs, with the decisions made along the way.

## Starting one

```sh
source ~/dev/vllm-lab/.venv/bin/activate
cd ~/dev/spelunking/experiments
python lab.py new <slug>          # fills in the README skeleton
python lab.py check               # before and after every run
```

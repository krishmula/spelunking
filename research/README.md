# research

Home of the `research-gap-dossier` skill's artifacts. One folder per topic, numbered in start order as
`R<nn>-<slug>` (`R01-kv-cache-robustness`, `R02-vllm-metal-serving`). Name the next one `R03-<slug>` when you
start the skill; it otherwise writes to an un-numbered `research/<slug>/`, which `lab.py check` flags.

| File | What |
|---|---|
| `ledger.tsv` | every source opened, with a verbatim quote and what it proves |
| `candidates.md` | candidate cards `C1`, `C2`, ... and both adversarial review passes |
| `dossier.md` | the final write-up for the strongest survivor |
| `decisions.tsv` | kill, reformulate, GO / PIVOT / KILL moves, each with its evidence |

Card ids restart at C1 in every topic, so experiments in `../experiments/` cite a card as `R<nn>-C<n>` (`R02-C8`) in
their `Candidate:` line and in `LEDGER.tsv`. An experiment's
verdict is logged back here in `decisions.tsv`. `foundation` marks the measurements that come before any
candidate (the baseline and the step budget).

Check a topic with `bun ~/plugins/kstack/skills/kstack/scripts/research/ledger.ts research/R<nn>-<slug>`.

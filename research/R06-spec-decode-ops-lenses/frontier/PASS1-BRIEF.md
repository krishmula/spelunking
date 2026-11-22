# Pass 1 attack brief (R06)

You are a hostile pass-1 attacker. Read:
- /Users/krishna/dev/spelunking/research/R06-spec-decode-ops-lenses/constraints.md
- /Users/krishna/dev/spelunking/research/R06-spec-decode-ops-lenses/candidates.md (your assigned cards only)
- /Users/krishna/plugins/kstack/skills/research-gap-dossier/references/adversarial-checklist.md (the 10 attacks)
- The existing ledger /Users/krishna/dev/spelunking/research/R06-spec-decode-ops-lenses/ledger.tsv (grep it; do not
  re-add rows it already has) and frontier/g-code.md for engine facts.

For each assigned card run all 10 attacks. Search hard for the work that would kill it: the exact mechanism and
synonyms, the simple baseline, the last 6-12 months of arXiv, and vllm/sglang/TensorRT-LLM issues and PRs.

Write new sources to the TSV your task names, same format and rules as frontier/BRIEF.md (header
`id	title	year	venue	type	url	fetched	quote	proves	does_not_prove	bearing`, only pages you opened count,
verbatim quote, fetched 2026-10-20, single-line cells). Use your class letter for ids.

Do not write anywhere else. Reply (at most 500 words) with, per card: one line per attack as
`<attack>: Survives | Kills | Narrows | Open - reason [ledger id or your new row id]`, the queries you ran, and a
suggested verdict (KILL, NARROW to <smaller question>, or SURVIVE to pass 2).

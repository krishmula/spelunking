# Frontier brief (shared by all R05 frontier subagents)

Topic: speculative decoding in LLM serving (draft models, EAGLE/EAGLE-3, Medusa, MTP heads, n-gram/prompt lookup,
suffix decoding, tree verification, self-speculation, DFlash, P-EAGLE/DSpark and other recent proposers).
The research lenses for this run are systems lenses: resource disaggregation, leases and reservations, speculative
execution (in the systems sense: doing work before knowing it is needed, and rolling back), work stealing, load
balancing and elasticity, isolation and multi-tenancy. We look for systems questions about WHERE drafting and
verification run, WHO pays for them, and HOW speculation interacts with shared resources (GPUs, KV memory, token
budgets, data-parallel ranks, tenants, LoRA adapters, idle cycles) that current work leaves unanswered.

The user is a masters student; hardware is a MacBook Air M5 16 GB (vLLM 0.31.0 + vllm-metal) plus single Colab GPUs,
possibly real GPUs later. Today is 2026-10-20.

A previous run (R04) used other lenses (checkpoint/restart, live migration, preemption, scheduling/admission,
caching) on the same topic and found every candidate answered. Its ledger is at
/Users/krishna/dev/spelunking/research/R04-spec-decode-systems-lenses/ledger.tsv. You may use it for leads, but a
row only counts in YOUR file if you reopen the page yourself today. Do not spend effort re-covering adaptive
speculation length, draft-state loss on preemption, or suffix-cache eviction; R04 covered those.

## Your output

Write a TSV file at the path your task names. Header line exactly:

id	title	year	venue	type	url	fetched	quote	proves	does_not_prove	bearing

- One row per source. Tab-separated, every cell single-line, no tabs or newlines inside a cell.
- id: your class letter plus a number (A1, A2, ...).
- type: one of paper, preprint, engineering, repo, trace, secondary.
- ONLY a source you opened yourself counts. fetched = 2026-10-20 and quote = a short VERBATIM passage from the page
  you opened that supports `proves`. If you only saw a search snippet or a citation, set fetched = unverified and
  leave quote empty. Never write a title, venue, or year you did not see on the source. If a fetch fails, say so in
  does_not_prove and mark unverified.
- proves: the concrete technical fact the source establishes (numbers where given). does_not_prove: the limit of
  that fact, especially what it leaves open.
- bearing: which kind of candidate idea this threatens or supports, in a few words.
- Aim for 15-30 fetched rows. Prefer primary sources: peer-reviewed papers, arXiv, official docs, repos, issues,
  RFCs, PRs, traces. Include negative results and workload studies that could make an opportunity vanish.
  Prefer work from the last 18 months, and check the last 6 months explicitly.

Do not write anywhere except your TSV file. Then reply with at most 250 words: the 3-6 most important facts for
finding a gap (with row ids), and any area that looked only shallowly studied (with the searches that showed it).

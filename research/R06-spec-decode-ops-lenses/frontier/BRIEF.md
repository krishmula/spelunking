# Frontier brief (shared by all R06 frontier subagents)

Topic: speculative decoding in LLM serving (draft models, EAGLE/EAGLE-3, Medusa, MTP heads, n-gram/prompt lookup,
suffix decoding, tree verification, self-speculation, DFlash, P-EAGLE/DSpark and other recent proposers).
The research lenses for this run are operational systems lenses: consistency and ownership transfer, backpressure
and queueing, fault tolerance, topology changes, hardware-aware placement, power/capacity constraints. We look for
systems questions about what happens to speculative decoding when state changes hands (drafter/target versions,
weight updates, KV handoff, rollback ownership), when load and queues build up, when components fail, when the
parallel layout changes (TP/PP/EP resharding, scale up/down, elastic EP), where drafting and verification should
run on heterogeneous hardware, and how speculation behaves under power caps, thermal limits, and capacity limits.

The user is a masters student; hardware is a MacBook Air M5 16 GB fanless (vLLM 0.31.0 + vllm-metal) plus single
Colab GPUs, possibly real GPUs later. Today is 2026-10-20.

Two previous runs on the same topic found little:
- R04 (checkpoint/restart, live migration, preemption, scheduling/admission, caching): no survivor. Killed:
  adaptive speculation length, draft-state loss on preemption, live migration of draft KV, suffix-cache eviction,
  hybrid recurrent-state rollback, prefix-cache block drop under EAGLE, unified token budget.
- R05 (disaggregation, reservations, speculative execution, work stealing, load balancing, isolation): one weak
  survivor (cross-tenant leak through a shared suffix draft cache). Killed: shrink-K before preempt, PD draft-state
  placement, shared drafter pools, all-accepted async scheduling cost, multi-LoRA drafter, acceptance-aware DP
  routing, cross-replica draft caches, batch-level controllers, unified-memory overlap on Apple silicon.
Ledgers: /Users/krishna/dev/spelunking/research/R04-spec-decode-systems-lenses/ledger.tsv and
/Users/krishna/dev/spelunking/research/R05-spec-decode-resource-lenses/ledger.tsv. Use them for leads only; a row
counts in YOUR file only if you reopen the page yourself today. Do not spend effort re-covering the killed ideas.

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

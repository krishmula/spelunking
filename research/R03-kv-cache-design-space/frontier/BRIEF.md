# Frontier brief (shared by all frontier subagents)

Topic: the KV cache side of LLM serving, broadly: any mechanism that decides how KV is stored, laid out, sized,
allocated, shared, evicted, compressed, moved, or scheduled. The user (a masters student) wants to change one such
mechanism at a time and measure the effect, looking for changes that are new or only shallowly studied. Hardware:
MacBook Air M5 16 GB unified memory, vLLM 0.31.0 + vllm-metal plugin, MLX, Qwen3 0.6B-4B 4-bit. Already read by the
user: vLLM/PagedAttention, Mooncake, Llumnix, RetrievalAttention, FlexiCache, Strata. Today is 2026-10-12.

## Your output

Write a TSV file at the path your task names. Header line exactly:

id	title	year	venue	type	url	fetched	quote	proves	does_not_prove	bearing

- One row per source. Tab-separated, every cell single-line, no tabs or newlines inside a cell.
- id: your class letter plus a number (A1, A2, ...).
- type: one of paper, preprint, engineering, repo, trace, secondary.
- ONLY a source you opened yourself counts. fetched = 2026-10-12 and quote = a short VERBATIM passage from the page
  you opened that supports `proves`. If you only saw a search snippet or a citation, set fetched = unverified and
  leave quote empty. Never write a title, venue, or year you did not see on the source. If a fetch fails, say so in
  does_not_prove and mark unverified.
- proves: the concrete technical fact the source establishes (numbers where given). does_not_prove: the limit of
  that fact, especially what it leaves open.
- bearing: which kind of candidate idea this threatens or supports, in a few words (e.g. "threatens: cost-aware
  prefix eviction; supports: block-size tradeoff gap").
- Aim for 15-30 fetched rows. Prefer primary sources: peer-reviewed papers, arXiv, official docs, repos, issues,
  RFCs, PRs, traces. Include negative results and workload studies that could make an opportunity vanish.

Do not write anywhere except your TSV file. Then reply with at most 200 words: the 3-6 most important facts for
finding a gap (with row ids), and any area that looked only shallowly studied (with the searches that showed it).

# Pass 2: codex gpt-6-astra (condensed; full reply in the T3 delegated task r05-pass2-astra)

Tools: web search and page open, shell file read, code search. No files written, no experiments run. No pass-1 kill
reversed.

## C5': MEASURE-FIRST
- Shortest rejection: none for the complete question. Conditional rejection: "reproduces the REST datastore attack on
  another structure and ports a known namespace boundary". Must show something consequential about online updates,
  target verification, or realistic tenant workloads.
- Corrections to the card:
  - Repetition is conditional, not absolute. `_speculate_path` starts at probability 1 and multiplies branch-count
    ratios; a unique context can propose a span seen once, and 3 occurrences can lose to a common competing context
    (code reading).
  - The attacker can supply the answer to its own cache: prompts fill the local tree, and the local draft wins ties.
    Later probes can measure the attacker's own earlier output.
  - Membership does not identify a tenant without controlled background histories.
  - Stream chunks are not automatically accepted-token counts (`output_processor.py:315`, chat `serving.py:648`).
  - The 80% retention metric can hide loss of nearly all sharing benefit. Report incremental retention
    (A_tenant - A_local) / (A_global - A_local), plus absolute acceptance, TPOT, goodput, at equal total capacity.
- Remaining core: causal detection of another tenant's contribution to online draft state, with realistic
  observations and an uncontaminated probe protocol. Salting itself is not novel.
- Scale: Arctic #185 is about 20 outstanding requests, not a batch cutoff; vLLM GPU-suffix RFC #51788 reports
  benefit at higher load. Record actual scheduled batch size.
- Change mind: paired histories differing only in a victim contribution; first-probe vs adaptive probing; held-out
  spans; precision at realistic prevalence, not only balanced AUC. If local-only drafting is nearly free, the result
  is a security finding but a weak systems thesis.
- New sources: RFC 9842 Compression Dictionary Transport; Debreach (1909.05977, ASE 2019); dedup side channel
  (1703.05126); vLLM `security.md`.
- Compute: CPU screen 1-4 h; live pilot 4-12 GPU-hours on L4; replicated 20-60 GPU-hours.

## C2': KILL as a standalone contribution
- CacheGen (2310.07240, SIGCOMM 2024, §5.3) already adapts between sending KV and sending text for recompute as
  bandwidth changes ("fall back to sending a chunk in text format"). An independent drafter's state is built from
  prompt tokens, so no new constraint. SPECTRE (2605.08151) covers independent-drafter serving.

## C9': weak MEASURE-FIRST
- DSpark in SGLang blog (2026-07-06): "The scheduler is sizing each request, not applying one batch average."
  SGLang RFC #30263 already raises shared-EMA contamination. Remaining core: causal TPOT harm from shared controller
  state beyond extra load. Intervention: frozen vs adaptive controller on the same replay. Lower K lowers both
  acceptance and step time, so measure TPOT directly. Heuristic screen: at least 10% p95 TPOT harm.
- New sources: DSpark SGLang blog; Pulsar (OSDI 2014).

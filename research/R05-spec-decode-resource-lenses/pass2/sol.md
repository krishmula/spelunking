# Pass 2: codex gpt-6.1-sol (condensed; full reply in the T3 delegated task r05-pass2-sol)

Tools: web search and page open, read-only shell. No files written, no experiments run. No pass-1 kill reversed.

## C5': MEASURE-FIRST, novelty narrower
- Shortest rejection: none for the exact live-cache question. But "Speculation Spills Secrets" §5 (2411.01076v4)
  already describes user-interaction-derived datastores, malicious probes, accepted-token observation, iterative
  extraction. "Speculation leaks a datastore" and "chunk sizes expose acceptance" are not contributions.
- Combination (Open): the new constraint is that a victim-derived global proposal must beat the attacker's local
  prompt tree (Arctic `cache.py` picks the higher score, local wins ties) and then agree with the target.
- Oracle (Open): "at least 3 repetitions" is not a valid threshold; selection depends on conditional counts,
  competing continuations, context, cumulative probability. First measure whether victim insertion changes the
  selected draft at all.
- Correctness (Narrow): a hit shows worker-cache membership, not membership in a specific victim's output. Public
  boilerplate and attacker-generated copies confound.
- Relevance (Narrow): vLLM `security.md` disclaims isolation between tenants sharing one server process. Must show
  a deployment that shares suffix state across trust boundaries, ideally one that already scopes prefix-cache salts.
- Scale (Open): Arctic #185 reports degradation at high load. Test batch 1, 8, 16 with background arrivals and churn.
- Required controls: identical target continuations across member and non-member; log global proposal change, winner,
  accepted length separately; keep guessed continuation out of attacker prompt tree; reset or clone cache between
  probes; report first-probe accuracy, FPR, query budget; confirm through client-visible streaming
  (`output_processor.py:315-337` aggregates when `stream_interval > 1`).
- New sources: RFC 9841 shared Brotli ("data compressed with the dictionary can reveal contents of the dictionary");
  memory-compression timing attacks (2111.08404); cross-user dedup side channels (IBM, 2010); vLLM `security.md`;
  llama.cpp #27852 (per-slot ngram cache kept across requests); TensorRT-LLM ngram README; acceptance-collapse
  (2607.21804).
- Kill early: if victim insertion rarely changes the winning proposal, or only for public boilerplate.
- Compute: CPU screen on the Air, hours to a day. Live pilot 5-20 GPU-hours on L4/4B (low confidence).

## C2': open but weak
- No source measures the crossover for a separate drafter. Strongest objection: an ordinary compute-vs-transfer
  benchmark; missing baseline is eagerly starting drafter prefill on D from prompt tokens, overlapped with target
  prefill. Adjacent: PPD (2603.13358, ICML 2026), Prefill Deflection (2607.02043). Endpoint: if P emits the first
  token, the D-side gain affects the next gap, not TTFT.
- Compute: blocked; distributed validation 20-80 GPU-hours on 2 GPUs.

## C9': characterization of a known limitation
- AdaSpec (2503.05096v2) and SGLang #23705 (adaptive roadmap: "the main challenge for request-level policy is CUDA
  graph") already recognise batch-average limits; #30263 asks whether opt-outs drag the shared EMA. Current SGLang
  `adaptive_spec_params.py` has a `[0]` tier at batch 64+, so no acceptance-driven choice there. Oracle: mean is
  (1-f)a_h + f a_l; check whether a minority fraction crosses the hysteresis threshold at all.
- Compute: CPU controller replay now; 10-30 GPU-hours live.

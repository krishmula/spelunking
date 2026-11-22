# When a draft vocabulary misses the token a tool call needs

November 17, 2026 · R07-C2 · MEASURE-FIRST · Low confidence in a public research gap

## What to investigate

A tool schema might allow either “Paris” or “Lyon”. A small drafter could represent the first choice but lack the token needed to start the second. Its grammar mask still has a legal token, so an empty-mask counter reports no problem. If the target chooses the missing token, the proposed sequence breaks there. Even then, correcting that guess might save no time. It matters where the first rejection falls and whether fixing it eliminates an entire target verification pass. This example illustrates the mechanism. Actual token boundaries depend on the tokenizer. [L18] [L20] [L21]

**First question:** At reached positions where the schema permits a choice, could repairing only first rejections caused by missing vocabulary save 5% of decode time on ordinary tool calls, after forced-token drafting and prompt copying? **First decision:** measure the most a free repair could save. Build a selector only if there is room for a practical method to win.

## What the papers establish

| Evidence | Consequence |
|---|---|
| Google's original speculative-decoding speed model depends on spare parallel compute. DeepMind observed completion-time variance and evaluated at batch 1 [L15] [L16]. | Accepted tokens are an intermediate metric. Measure elapsed time and tail latency under load. |
| Samsung and Sheffield's SpecVocab changes the shortlist at every step, includes a full-vocabulary comparison and calls for broader language coverage [L18]. A separate paper optimizes domain-level token coverage against drafting cost [L44]. | A different shortlist alone has weak novelty. Both are serious baselines. |
| A vLLM issue reports zero empty draft/grammar intersections across 4,486 masks in one test [L21]. | This counter cannot answer whether the target's chosen token was present. The statistic is developer-reported and specific to that test. |
| FactorSpec already drafts constraint-determined text and evaluates realistic tool-call workloads [L20]. NVIDIA describes grammar masking for speculative decoding [L22]. | Count only remaining opportunities under the same constraint semantics. A comparison with ordinary unconstrained drafting would exaggerate the gain. |
| DeepSeek-V3's multi-token prediction shares its output head with the target [L47]. | The fixed pruned-head mechanism applies to a subset of drafters. A shared full head can disagree for other reasons. |

The missing measurement in this sample is the recoverable time attributable to absent target choices, conditional on actually reaching the first rejection. This is our inference from the papers and engine evidence. The authors' own research directions are recorded separately in the [KV paper audit](frontier/kv.md), [speculative decoding paper audit](frontier/spec.md) and [GPU paper audit](frontier/gpu.md).

## Smallest decisive test

1. Freeze the target, tokenizer, draft vocabulary map, grammar, decoding precision and verification width. Collect legal greedy target transcripts from two held-out public schema families. Include identifiers, enum choices and free-text values. Keep rare invented identifiers as a separate stress case. Verify identical target output under matched grammar semantics. Score semantic tool correctness separately.
2. For each reached verification block, record the first mismatch, whether the grammar forced the token, the target's choice, and its membership in the draft vocabulary. For example, distinguish a legal target absent from the vocabulary from a representable target the drafter scored too low. Exclude the unused suffix following a rejection.
3. Replay blocks after repairing just the missing-token events. Recompute the conditional draft state and subsequent block boundaries. If conditional replay is unavailable, also calculate a deliberately generous ceiling that grants perfect remaining predictions after a repair. Recorded missing-token counts alone do not yield a wall-time bound.
4. Compare no speculation, forced-token drafting with prompt n-grams, a grammar-masked fixed draft head, a compatible full head, and a dynamic shortlist. Add a request-specific schema-token baseline and charge for obtaining and scoring its output weights. Start with greedy decoding. Any later sampling claim needs distributional validation.
5. Measure per-width target and draft times on a compatible single GPU, including masking, remapping and host work. Replay identical arrivals at batch 1, batch 8 and through saturation. Report time per completed valid call, decode time, p95 completion time, TTFT and valid calls per GPU-second. Time mixed prefill separately from decode-only step time. Compare paired runs and uncertainty across held-out schema families.

**Predeclared project heuristics.** KILL if even the optimistic bound is below 5% of decode time in either required family, or the best simple baseline captures the gain at equal cost. PIVOT to a workload-specific measurement if the opportunity occurs only on synthetic schemas or incompatible constraint semantics. Clearing the bound justifies a timed prototype. GO requires a feasible method to clear 5% against the strongest matched baseline on both held-out families, preserve greedy outputs and avoid a reproducible p95 completion-time regression at the tested offered load. No experiments ran.

## Feasibility and learning

The Mac can inspect tokenizer mappings and replay recorded events once matching transcripts and draft proposals exist. A list of target outputs alone is insufficient to reproduce the first rejection. The decisive timing needs one CUDA GPU with memory for the target, compatible draft head and KV at the chosen workload. A smaller model is useful if its head actually prunes the vocabulary. Colab is an access path, but the compatible GPU and checkpoints have not been confirmed. The initial claim is single-node and needs no interconnect. GPU-hours are unknown pending a timing pilot. First establish artifact compatibility, then the bound, then live timing. Stop at the first failed gate.

Before implementation, inspect one real verification block by hand. Which tokens did the constraint force, which legal choice did the target make, and did the repair remove a pass or simply move the rejection one position later? That distinction decides whether this becomes systems work or a brief negative result.

## Other decisions

| Topic | Smallest remaining question | Decision |
|---|---|---|
| KV cache optimization | Can a warning before dispatch catch *additional* compression-induced wrong argument values after parser-aware retention and cache-specific error monitoring? [L41] [L45] | PIVOT. Requires paired semantic errors, false-alarm costs and evidence of a benefit over existing monitors. |
| GPU kernels and utilization | Does default sorting ever slow a real Hopper decode workload enough to justify switching it off? [L40] [L43] | KILL the proposed optimization. Reopen only with a repeatable whole-model regression. |
| Speculative decoding | Can missing draft tokens explain enough recoverable verification time on real tool schemas? [L18] [L20] [L21] [L44] | MEASURE-FIRST. Strongest early falsification test, unmeasured upside. |

[Source ledger](ledger.tsv) · [Candidate cards and both review passes](candidates.md) · [Coverage and prior R records](constraints.md) · [Decision trail](decisions.tsv)

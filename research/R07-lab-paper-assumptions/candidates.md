# R07 candidate decisions

## C1. Early warning of a wrong tool argument after KV compression

**PIVOT, low confidence.** The caching lens led from LinkedIn's proposed adaptive budget to ActKV's existing confidence-based controller [L6] [L7]. ActKV already evaluates state-changing tasks. A remaining question is whether a cheap signal available before dispatch can identify a *semantically wrong but legal* argument introduced by eviction. Use deterministic harmless calls with a known correct value and matched histories. Keep naturally occurring correct and incorrect calls for calibration. Full KV is an offline control. A change between two correct values is not an error.

**Pass 1.** Generic adaptive compression, tool-call validity testing and full-KV verification each have prior art [L7] [L9] [L10]. Raw and grammar-masked confidence can rank examples differently [L8]. Separate forced text from value choices. Full-KV-correct traces alone cannot give the deployed false-alarm rate. Larger budgets cannot restore discarded entries. Time and charge a re-prefill.

**Pass 2.** Opus identified parser-aware KV retention on tool calls [L41]. Sol identified the oracle dependence of full-cache diagnostics [L10]. Both questioned attribution and useful warning time. The lead check found randomized eviction already attributes damage and gates recomputation [L45]. Quantization has a separate risk meter [L46]. Single-sample function-call uncertainty is another direct baseline [L42]. This changes the question to incremental value on *top of* a cache-specific signal. There is no evidence for enough residual errors to justify building it. Reopen only with paired long-context semantic errors that survive parser-aware retention, then compare alarms at matched replay cost, retained bytes and latency. Local CPU can label fixtures. A compatible CUDA model and actual eviction are required to price recovery. Access and GPU-hours unknown.

## C2. First rejections caused by a pruned draft vocabulary

**MEASURE-FIRST, low confidence.** L-08 and L-04. A grammar can allow several tokens while a pruned draft head cannot produce the one the target chooses [L18] [L21]. Is the lost time material after forced tokens and cheap prompt copying have already been handled? Hypothesis, controls, compute and stop rules are in the dossier.

**Pass 1.** Draft grammar masking, forced-token drafting, adaptive vocabularies and domain-level coverage optimization all exist [L18] [L20] [L21] [L22] [L44]. General vocabulary mismatch is too broad. DeepSeek's shared-head design is an architectural negative control [L47].

**Pass 2.** Opus proposed adding legal schema tokens to the draft vocabulary or copying from the prompt. Sol observed that even perfect token coverage can save zero verification passes. Both objections stand. A schema-expanded head needs output weights for the added tokens. Use a compatible full head, or treat free availability of those tokens as an optimistic bound. The dossier incorporates both attacks. Neither reviewer found a directly answering conditional time measurement. Upside and scalability remain unknown.

## C3. Sort attention work under changing batch membership

**KILL as a thesis direction in this record.** L-04 and L-16. FA4 describes cacheable sorted metadata [L26]. The initial candidate asked when ordering each step by estimated work would repay preparation cost at model level. Pass 1 found that FlashInfer already shares plans across layers, and that the examined CuTe helper rejects the sort flag while Hopper has both internal branches [L27] [L38] [L39]. Physical request order, logical output mapping and split counts must be equivalent.

**Pass 2.** Opus found non-local FA3 already sorts by default [L40] and existing SGLang integration prepares metadata once per batch [L43]. Sol independently rejected charging that cost per layer. Changing membership between steps cannot erase reuse within a step. The PR's microseconds measure preparation only. They do not bound how ordering affects all subsequent kernels. Reopen only if a pinned Hopper trace shows a repeatable negative interaction with the default order. A fixed-order ablation is possible on one Hopper. The local Mac cannot test it. No qualifying trace or GPU reservation. A port or removal of repeated setup alone would duplicate prior work.

## Other directions screened in pass 1

| Candidate | Reason to stop |
|---|---|
| C4. Adaptive retention for agent turns | ActKV already implements and evaluates adaptive retention [L7]. Keep the author's historical suggestion in the paper audit. |
| C5. Use constraints to draft forced output | FactorSpec already drafts determined tokens and measures valid records per GPU-second [L20]. |
| C6. Share the attention schedule across layers | FlashInfer's plan/replay and FA3 engine integration implement this [L27] [L43]. |

The two completing pass-2 reviewers were Claude Opus 5.5 and GPT-6.1 Sol. Grok hit its usage limit. Fable was absent from the available Claude model list. Both completing reviewers assessed all three cards. They received the C1 brief before the KV frontier files were complete. Only evidence checked against an opened source informs the decisions.

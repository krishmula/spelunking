# Confirming a co-tenant's outputs through a shared suffix draft cache

2026-10-20 · Status: MEASURE-FIRST · Gap confidence: medium · Thesis fit: low

## Record and scope

- R05, topic T-02 speculative decoding, lenses L-06 resource disaggregation, L-07 leases and reservations, L-08
  speculative execution, L-09 work stealing, L-10 load balancing and elasticity, L-11 isolation and multi-tenancy,
  chosen by Krishna.
- Prior: R04 (same topic, lenses L-01 to L-05) closed today with no survivor. Its kills (adaptive K, draft-state loss
  on preemption, suffix-cache size and eviction) were carried as prior evidence.
- Coverage: 294 ledger rows, 9 unverified. Six lens sweeps, one code reading of vLLM v0.31.0 and vllm-metal
  (`frontier/g-code.md`), three pass-1 attack sweeps, a four-model pass 2, and a reopen of every pass-2 citation.
  Coverage table in `constraints.md`.
- Limits: no live measurement of any candidate; real multi-GPU access unconfirmed, so C2/C3/C7 were judged on
  published numbers and arithmetic.

## Summary

Plain: vLLM's suffix decoding guesses the next tokens from a tree built out of every past request's output on that
server. If one customer's text sits in that tree, another customer whose request hits the same context gets more
tokens accepted per step, and that shows up in how many tokens arrive per streamed chunk. A known paper shows this
kind of leak for a fixed, public datastore; nobody I found has measured it for a tree that other users fill as they
go, or what it costs to give each tenant its own tree. The first experiment replays a multi-tenant trace through the
real suffix tree on the Air and measures both.

Precise: measure first-probe membership confirmation through accepted tokens per step under Arctic's online suffix
tree, and the acceptance cost of tenant scoping, on a multi-tenant trace.

Honest expectation: the most likely outcome is that tenant scoping is nearly free [L126] [L248], which makes this a
short security note, not a systems thesis.

## Research question

Can a co-tenant confirm that a span appeared in another tenant's outputs from its own accepted tokens per step under
vLLM suffix decoding on a multi-tenant agentic/SQL trace, and does scoping the global tree by tenant remove that
signal while keeping most of the global tree's acceptance gain over local-only drafting, compared with a global tree
and with local-only drafting at equal cache capacity?

The answer is "no" if first-probe confirmation is near chance on spans the target would reproduce, or if tenant
scoping keeps nearly all of the global tree's gain.

## Background

Fact. Suffix decoding drafts tokens by matching the request's recent tokens against two trees: a local tree built
from this request's prompt, and a global tree of up to 10,000 past outputs per worker in vLLM (100,000 in Arctic's
default) [L189] [L191] [L236]. Only `req_id` and tokens reach the tree; there is no tenant, LoRA or salt key [L190].

Fact. The match can start anywhere in any cached output, for contexts up to 24 tokens. Draft probability starts at 1
and is multiplied by count ratios only where the tree branches, with no minimum count, so a context that occurs once
proposes its one continuation at probability 1 [L237]. The local draft wins ties with the global draft [L293].

Fact. The target model verifies every draft, so outputs stay exact [L236]. Under greedy decoding the target only
accepts tokens it would have produced anyway. What leaks is speed: more accepted tokens per step.

Fact. vLLM streams one chunk per step by default (`stream_interval=1`, `vllm/config/scheduler.py:214`, read
locally), so chunk sizes reveal accepted counts; deltas merge when the consumer falls behind [L294].

Fact. The KV prefix cache has `cache_salt` for trust groups [L188] and only matches from position 0. vLLM's docs say
it does not isolate tenants that share one server process [L277].

Unresolved: whether the online tree's rules (target agreement, local masking, self-insertion of probes, eviction,
batching) leave a usable signal, and what removing it costs.

## What prior work already answers

| Work | What it solves | What it leaves open | Source |
|---|---|---|---|
| When Speculation Spills Secrets | Leaks a fixed public REST datastore via accepted tokens; iterative extension; names user-derived datastores as at risk; aggregation mitigation | A datastore filled online by co-tenants; the cost of isolating it | [L183] [L271] [L272] |
| Prefix-cache side channels and defences | Cross-user leakage via prefix caches; salting keeps most benefit | Draft caches; position-free matching | [L188] [L195] [L241] [L242] [L243] [L288] [L289] |
| Compression dictionary and dedup attacks | Shared-content membership channel; origin scoping as defence; scoping can fail at wrong boundaries | LLM draft caches | [L273] [L275] [L276] |
| SuffixDecoding, Arctic | Global tree helps agentic/SQL; acceptance near flat in tree size | Tenant boundaries | [L126] [L127] [L291] |
| Shared suffix trees in RL rollouts | Sharing and sharding trees across workers; scoped shards can beat global | Multi-tenant serving, leakage | [L246] [L247] [L248] |
| Speculation attacks 2026 | Availability (acceptance collapse) | Confidentiality via shared drafts | [L250] [L251] |

## The gap

I did not find public work that measures how much of a co-tenant's output an attacker can confirm through an online,
co-tenant-filled suffix tree under its real selection rules, or the acceptance cost of tenant scoping on a
multi-tenant trace. I found no vLLM, Arctic, SGLang, llama.cpp or TensorRT-LLM issue on suffix-cache isolation
(reviewer searches; vLLM's proposer still keys only by `req_id` [L190]). Closest threats: [L271] (same channel,
static store) and the near-free-defence argument from [L126] [L248].

## Why now

Suffix decoding shipped in vLLM and Arctic and is now pitched as production-ready at all concurrency levels [L291];
a GPU global index is in progress upstream [L292]. Prefix-cache isolation became a CVE class in vLLM [L287].

## Hypotheses

1. On spans the target reproduces from a context the attacker can elicit, first-probe TPR at 1% FPR is at least 0.5
   (heuristic). It would show as a clear gap in accepted tokens over the span between the member and non-member
   worlds.
2. Tenant scoping keeps less than 80% of the global tree's incremental gain over local-only drafting (heuristic). It
   would show if most cross-request hits on the trace come from other tenants.

## Evidence against

- R1 (all four reviewers): the attack class exists [L271]. Answered only by argument: the online setting changes the
  bounds. This is still the main novelty risk.
- R2 (all four): acceptance is near flat in tree size [L126] and scoped shards can beat global [L248], so scoping is
  likely nearly free. Still stands; it is the kill-test's cost arm.
- R7: vLLM disclaims tenant isolation in one process [L277]; a related salt bypass rated Low [L287]. Caps relevance.
- Card errors fixed in pass 2: no minimum repetition is needed [L237]; chunk merging under load [L294].

## Experiments already run

None for C5''. No experiment ran in R05. Prior engine reading in `frontier/g-code.md` (code, not tests).

## Validation plan

Proposed work. Nothing runs before Krishna writes a prediction. Thresholds are logged in `decisions.tsv`
(2026-10-20T18:35) and are heuristics.

Phase A, offline on the Air (screen, about a day):
- Tree: ArcticInference `SuffixDecodingCache` on CPU, pinned commit, vLLM v0.31.0 defaults (depth 24,
  `min_token_prob` 0.1, 10,000 cached requests).
- Target: Qwen3-4B, greedy, on vllm-metal or MLX. Accepted tokens are computed offline by comparing drafts to the
  target's greedy continuation, which is exact for greedy.
- Trace: a public agentic or SQL workload grouped into tenants with a stated meaning (for example, one tenant per
  application or database). Synthetic grouping is screening only.
- Leak arm: paired worlds that differ only in whether the victim's outputs were inserted, with the same background
  traffic and capacity. About 200 held-out victim spans of 16 or more tokens. Each probe elicits the span's preceding
  context without the guess appearing literally in the prompt, on a fresh clone of the cache. Report first-probe TPR at
  1% FPR, AUC, and the share of victim spans the target reproduces at all (the upper bound).
- Cost arm: global, tenant-scoped, local-only, at equal total capacity. Report accepted tokens per step and
  R = (A_tenant - A_local) / (A_global - A_local).
- KILL if the global tree adds less than 5% accepted tokens per step over local-only, or leak TPR < 0.10.
- KILL as thesis (write an advisory) if leak TPR >= 0.10 and R >= 0.95. Any disclosure upstream is an external message
  and needs Krishna's approval.
- PIVOT to an isolation-scope policy question if leak TPR >= 0.10 and 0.80 <= R < 0.95, or the leak only works on
  public boilerplate.
- GO to Phase B if leak TPR >= 0.50 and R < 0.80.

Phase B, live on Colab (one GPU): vLLM with suffix decoding, 4B model, batch 1, 8, 16. Confirm the client-visible
chunk sizes match Phase A's signal; measure delta merge rate [L294]; compare `stream_interval` 1 and 4 as the cheap
defence.

What it cannot show: behaviour at production load across replicas, sampled (non-greedy) decoding, or prevalence of
sensitive repeated spans in real tenants' traffic.

## Compute requirements

- Readiness: runnable now (Phase A); Phase B on Colab.
- Blocker: none for screening. A tenant-labelled trace is a data dependency.
- Local screening: Air, Arctic CPU tree plus Qwen3-4B greedy generation; about 1-4 hours (reviewer estimates, 2x
  uncertain).
- Ideal: 1x H100 80 GB, upstream vLLM with arctic-inference, 8B model, batch 32-64, separate load generator.
- Lower-resource: Colab L4 24 GB, 4B model, batch 1-16. Keeps the chunk-count channel; cannot speak to production load.
- Estimate: Phase B under 10 GPU-hours (low confidence, 2-3x).
- Access: Colab confirmed. Nothing to book.

## Learning before implementation

- Read `_speculate_path` and `SuffixDecodingCache.speculate` [L237] [L293] until the tie rule and probability walk are
  clear.
- Read section 5 and 6 of [L271] [L272] to know exactly what the static-datastore attack already shows.
- Understand why greedy makes the offline replay exact, and where sampling would break that.
- Doubts to record: what a "tenant" means in the trace; whether any real deployment shares one vLLM process across
  trust boundaries given [L277].

## Evaluation plan

- Baselines: global tree (vLLM default), local-only (`max_cached_requests=0`), tenant-scoped tree, `stream_interval`
  aggregation [L272].
- Metrics: first-probe TPR at 1% FPR, AUC, precision at realistic membership prevalence; accepted tokens per step,
  TPOT, R.
- Constraint: outputs exact (greedy); equal cache capacity across arms.
- Final claim needs: a real multi-tenant trace and a live engine at batch 16+.

## Plan

- Week 1: Krishna writes predictions; Phase A. Decision point against the logged thresholds.
- Week 2: Phase B on Colab if GO. Decision point: advisory, scope-policy PIVOT, or continue.
- Beyond: only if R < 0.80 holds on a real trace, design a scope or gated-insertion policy.

## Risks and threats

- Scoop: low; no upstream issue found. A short follow-up to [L271] could land first; then cite and move to the cost arm.
- Scale: chunk merging under load [L294] and multi-replica routing may blur the channel; Phase B measures merging.
- Feasibility: finding a trace with meaningful tenants. Fallback: state results as screening only.
- Thesis fit: likely a security note. If R >= 0.95, stop and return to topic selection.

## Candidates considered

- C1 shrink K before preempting: lookahead is 0.07-1.5% of KV; reduces to R04 C1.
- C2/C2' draft-state placement in PD: CacheGen already adapts ship vs recompute [L278]; EAGLE draft KV is 3% [L228].
- C3 shared drafter pool: per-GPU throughput halves in DPara's own table [L20]; target compute-bound at batch 32+ [L18].
- C4 optimistic async scheduling: step cost flat in verified tokens; async gains small [L91] [L234].
- C6 per-adapter draft LoRA: LoRAX ships it [L252]; real adapters cost little acceptance [L200].
- C7 acceptance-aware DP balancing: count-based routing already compensates; per-rank K is adaptive K [L143].
- C8 suffix caches across replicas: shared and sharded trees exist [L246] [L247] [L248].
- C9/C9' batch controller tenant harm: per-request sizing shipped or scheduled [L265] [L282] [L285].
- C10 CPU draft overlap on unified memory: at most about 4% to hide; Metal loses to serial verify [L268].

## References
- [L18] NebulaSD: Many-for-Many Speculative Decoding. arXiv preprint (cs.NI), 2026. preprint. https://arxiv.org/abs/2609.29364
- [L20] When Parallel Drafter Meets Parallel Speculative Decoding (DPara). arXiv preprint, 2026. preprint. https://arxiv.org/html/2609.27396v1
- [L91] [Core] Async Scheduling X Spec Decoding Compatibility (vLLM PR #24799). vllm-project/vllm GitHub, 2025. engineering. https://github.com/vllm-project/vllm/pull/24799
- [L126] SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications. NeurIPS 2025 (Spotlight), 2025. paper. https://arxiv.org/abs/2411.04975
- [L127] SuffixDecoding at Production Scale with Arctic Inference and vLLM. Snowflake engineering blog, 2025. engineering. https://snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/
- [L143] [Spec] Enable adaptive speculative decoding under DP attention (PR #35898). GitHub sgl-project/sglang, 2026. repo. https://github.com/sgl-project/sglang/pull/35898
- [L183] When Speculation Spills Secrets: Side Channels via Speculative Decoding In LLMs. arXiv 2411.01076 (v4 11 Feb 2026; no venue listed), 2024. preprint. https://arxiv.org/abs/2411.01076
- [L188] vLLM design doc: Automatic Prefix Caching (Cache Isolation for Security). vLLM docs (latest), 2026. engineering. https://docs.vllm.ai/en/latest/design/prefix_caching/
- [L189] vLLM SuffixDecodingProposer source (vllm/v1/spec_decode/suffix_decoding.py). vLLM API docs (latest), 2026. repo. https://docs.vllm.ai/en/latest/api/vllm/v1/spec_decode/suffix_decoding/
- [L190] vLLM SpeculativeConfig API reference (suffix_decoding_max_cached_requests). vLLM docs v0.18.0, 2026. engineering. https://docs.vllm.ai/en/v0.18.0/api/vllm/config/speculative/
- [L191] Arctic Inference: Suffix Decoding docs. Arctic Inference docs, 2026. engineering. https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html
- [L195] Governing the KV Cache: Preventing Timing Side-Channel Leakage in Multi-Tenant LLM Inference. arXiv 2608.09225 (cs.CR), 2026. preprint. https://arxiv.org/abs/2608.09225
- [L200] [Spec][LoRA] Support multi-adapter LoRA with EAGLE/NEXTN/DFLASH/DSPARK speculative decoding (SGLang PR #34337). SGLang GitHub PR (merged 2026-08-21), 2026. engineering. https://github.com/sgl-project/sglang/pull/34337
- [L228] yuhuili/EAGLE3-LLaMA3.1-Instruct-8B config.json. Hugging Face, n/a. repo. https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/raw/main/config.json
- [L234] [Core][Spec Decode] Reference implementation: per-request effective proposal lengths (vLLM PR #48204). vLLM GitHub PR, 2026. engineering. https://github.com/vllm-project/vllm/pull/48204
- [L236] Arctic Inference SuffixDecodingCache source (cache.py). github.com/snowflakedb/ArcticInference, 2025. repo. https://github.com/snowflakedb/ArcticInference/blob/main/arctic_inference/suffix_decoding/cache.py
- [L237] Arctic Inference suffix_tree.cc (_speculate_path / count-based prob). github.com/snowflakedb/ArcticInference, 2025. repo. https://github.com/snowflakedb/ArcticInference/blob/main/csrc/suffix_decoding/suffix_tree.cc
- [L241] Auditing Prompt Caching in Language Model APIs. ICML 2025 (arXiv 2502.07776), 2025. paper. https://arxiv.org/abs/2502.07776
- [L242] InputSnatch / Auditing lineage: CacheProbe: Auditing Prompt Cache Isolation in Gateway APIs. arXiv 2605.30613, 2026. preprint. https://arxiv.org/abs/2605.30613
- [L243] Governing/Selective KV-Cache Sharing (SafeKV) to mitigate timing side-channels. arXiv 2508.08438 v2, 2026. preprint. https://arxiv.org/abs/2508.08438
- [L246] WAR: Workload-Aware Rollouts for Synchronous Agentic Reinforcement Learning. arXiv 2607.17299, 2026. preprint. https://arxiv.org/html/2607.17299
- [L247] BubbleSpec: sharded suffix indices across rollout DP ranks. arXiv 2605.08862, 2026. preprint. https://arxiv.org/html/2605.08862v1
- [L248] DAS: per-problem suffix shards beat a single global index. arXiv 2511.13841 (MLSys 2026), 2025. preprint. https://arxiv.org/html/2511.13841
- [L250] Mistletoe: Stealthy Acceleration-Collapse Attacks on Speculative Decoding. arXiv 2605.14005 v2, 2026. preprint. https://arxiv.org/abs/2605.14005
- [L251] Speedbumps: Rejection Attacks on Speculative Decoding. arXiv 2610.10929, 2026. preprint. https://arxiv.org/abs/2610.10929
- [L252] LoRAX docs: Medusa (adapters guide). predibase/lorax docs, 2024. repo. https://github.com/predibase/lorax/blob/main/docs/models/adapters/medusa.md
- [L265] AdaSpec: Adaptive Speculative Decoding for Fast, SLO-Aware Large Language Model Serving. SoCC 2025 (arXiv 2503.05096v2), 2025. paper. https://arxiv.org/html/2503.05096v2
- [L268] Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware. arXiv 2607.17283, 2026. preprint. https://arxiv.org/abs/2607.17283
- [L271] When Speculation Spills Secrets: Side Channels via Speculative Decoding in LLMs (v4 HTML, sec 5 datastore leak; reopen). arXiv preprint, 2024. preprint. https://arxiv.org/html/2411.01076v4
- [L272] When Speculation Spills Secrets (v4 HTML, sec 6.1 token aggregation mitigation; reopen). arXiv preprint, 2024. preprint. https://arxiv.org/html/2411.01076v4
- [L273] RFC 9842 Compression Dictionary Transport. IETF RFC (Standards Track), 2025. engineering. https://www.rfc-editor.org/rfc/rfc9842.txt
- [L275] Remote Memory-Deduplication Attacks. NDSS Symposium 2022, 2022. paper. https://www.ndss-symposium.org/ndss-paper/auto-draft-235/
- [L276] Dedup Est Machina Returns: On the Effectiveness of Same-Domain Memory Deduplication. VUSec project page, 2022. secondary. https://www.vusec.net/projects/dedup-est-machina-returns/
- [L277] vLLM docs: Security, Multi-Tenant Deployments (v0.31.0). vLLM documentation, 2026. engineering. https://github.com/vllm-project/vllm/blob/v0.31.0/docs/usage/security.md
- [L278] CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving (sec 5.3). arXiv preprint, 2023. preprint. https://arxiv.org/html/2310.07240
- [L282] SGLang issue #23705: [Roadmap] Adaptive Speculative Decoding Roadmap. GitHub sgl-project/sglang, 2026. repo. https://github.com/sgl-project/sglang/issues/23705
- [L285] DSpark in SGLang: Speculative Decoding with Confidence-Driven, Variable-Length Verification. LMSYS Blog (2026-07-06), 2026. engineering. https://www.lmsys.org/blog/2026-07-06-dspark-sglang/
- [L287] Red Hat CVE-2026-105752 (vLLM cache_salt not preserved on tool continuations). Red Hat Customer Portal, 2026. secondary. https://access.redhat.com/security/cve/CVE-2026-105752
- [L288] PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems. arXiv preprint (cs.CR), 2026. preprint. https://arxiv.org/abs/2603.10726
- [L289] The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems. arXiv preprint (v5 2025-10-21), 2024. preprint. https://arxiv.org/abs/2409.20002v5
- [L291] SuffixDecoding at Production Scale with Arctic Inference and vLLM (reopen L127). Snowflake Engineering Blog, 2026. engineering. https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/
- [L292] vLLM RFC #51788: Suffix decoding on GPU (suffix_gpu) (reopen). GitHub vllm-project/vllm, 2026. repo. https://github.com/vllm-project/vllm/issues/51788
- [L293] ArcticInference cache.py SuffixDecodingCache.speculate (lines 288-308). GitHub snowflakedb/ArcticInference (main @ aca5d9a), 2026. repo. https://github.com/snowflakedb/ArcticInference/blob/main/arctic_inference/suffix_decoding/cache.py
- [L294] vLLM v0.31.0 output_processor.py RequestOutputCollector (lines 51-77). GitHub vllm-project/vllm (v0.31.0), 2026. repo. https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/output_processor.py

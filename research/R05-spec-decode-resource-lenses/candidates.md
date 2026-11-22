# R05 candidates

T-02 speculative decoding, lenses L-06 to L-11. Sources are `[L<n>]` in `ledger.tsv`. Engine facts cite
`frontier/g-code.md` (this run) or R04 `frontier/e-code.md` by section; both are code readings, not tests.

Generated 2026-10-20 from seven frontier sweeps (frontier/a-g). Ten candidates. R04's kills (draft-state loss on
preemption, live migration of draft KV, adaptive K, suffix-cache eviction/size, hybrid rollback, prefix-cache block
drop) are prior evidence: a card that reduces to one of them is not new.

Compute defaults for every card unless the card says otherwise. Local: M5 Air, vllm-metal (greedy only; ngram,
draft_model, DFlash, DSpark, Gemma4 MTP; no suffix, no EAGLE, no DP). Colab: one CUDA GPU, upstream vLLM. Real
GPUs: unconfirmed.

---

## C1. Reservations: shrink speculation before preempting a request

- Lens: L-07 leases and reservations.
- Existing mechanism: vLLM reserves `1+K` verify slots plus K (EAGLE/MTP/draft_model) or K+1 (DFlash) lookahead
  slots per running request each step, all-or-nothing; when they do not fit it preempts another request and never
  retries with fewer drafts (g-code §2) [L43] [L44]. TensorRT-LLM reserves worst-case draft KV for the whole
  request at admission [L57]; SGLang tree drafting reserves 2 x topk x pages [L53]. Unresolved: what this
  reservation costs in batch capacity and preemptions, and whether "degrade speculation first" beats preemption.
- Problem: speculation's transient slots can force a whole request out (recompute, R03/E02) when dropping K for one
  step would have fit.
- Why now: 2026 bug cluster on reservation accounting [L45]-[L50] [L54] [L55]; three engines chose three policies.
- Existing work: Nightjar turns SD off and offloads the drafter under KV pressure [L61]; R04 C1 ("de-speculate before
  you preempt") was killed because draft KV is about 3% of KV for EAGLE-3 (R04 L210).
- Exact gap: a measured comparison of reservation policies (per-step lookahead, lifetime worst case, shrink-K-on-
  pressure) by preemptions and goodput at KV saturation.
- Research question: Can shrinking per-request speculation length when lookahead slots do not fit reduce preemptions
  and raise goodput at KV saturation, compared with vLLM's preempt-on-failure and Nightjar's global SD-off, while
  preserving output equivalence?
- Hypothesis: at KV saturation, a measurable share of preemptions are caused by lookahead slots alone.
- Cheapest decisive test: oracle arithmetic, then count preemptions whose allocation would have fit with K=0
  (E02's logging scheduler, Colab).
- Biggest novelty threat: Nightjar [L61]; R04 C1 kill.
- Scale risk: lookahead is K slots against contexts of thousands of tokens.
- Compute: readiness runnable now (Colab, E02 harness). Blocker none for the screen. Local screening on Air possible
  with draft_model. Ideal: 1x H100, Llama-3.1-8B + EAGLE-3, ShareGPT/agentic trace at KV saturation. Lower-resource:
  Colab L4/A100 with Qwen3-4B; keeps the phenomenon. Estimate: a few GPU-hours. Access: Colab confirmed.

## C2. Disaggregation: where draft state is produced in prefill/decode serving

- Lens: L-06 resource disaggregation.
- Existing mechanism: in vLLM NIXL PD, P and D must run the same drafter; P's draft-layer KV moves with target KV;
  hidden states never move, so D recomputes the last prompt token through target and drafter; P also drafts K tokens
  it throws away (g-code §1). SGLang moves target KV plus the EAGLE final hidden state, not DFlash's per-layer draft
  state, so PD+DFlash fails [L29]; DSpark draft-only-on-D breaks the handshake [L30]; Gemma4 MTP draft KV is not
  transferred, erasing MTP benefit at 15k input [L27]; Neuron ships both KVs [L32].
- Problem: three placements (draft on P and ship draft KV; draft on D only and re-prefill the drafter; ship hidden
  states) have different transfer bytes, D-side work, TTFT, and acceptance, and no one has compared them.
- Why now: PD + spec is on the vLLM roadmap as WIP [L31]; Dynamo marks EAGLE3+PD crashing [L24].
- Existing work: integration fixes [L47]-[L50]; no paper costs placement (frontier A).
- Exact gap: a cost model and measurement of draft-state placement across drafter families (1-layer EAGLE, MTP,
  multi-layer DFlash/DSpark, separate draft model).
- Research question: Does producing draft state on the decode instance (re-prefilling only the drafter) instead of
  shipping it from prefill reduce TTFT or transfer volume for multi-layer drafters at long inputs, compared with
  shipping draft KV, without lowering acceptance?
- Hypothesis: for 1-layer drafters shipping is free; for multi-layer drafters at 8k-32k inputs the choice moves TTFT
  by more than 10%.
- Cheapest decisive test: arithmetic of draft-state bytes and drafter prefill FLOPs per drafter family, then one
  measurement of drafter-only prefill time.
- Biggest novelty threat: upstream PD roadmap [L31]; integration work converging weekly.
- Scale risk: drafter prefill is a small fraction of target prefill.
- Compute: blocked on compute for the real claim (two GPUs or nodes with NIXL/RDMA). Local screening: arithmetic
  only. Ideal: 2 nodes x 8 H100 with RDMA, DeepSeek/GLM MTP and DFlash. Lower-resource: one node, 2 GPUs, NIXL over
  NVLink, 8B target; loses network-bound transfer. Estimate unknown, tens of GPU-hours. Access unconfirmed.

## C3. Disaggregation: a shared drafter pool against colocated drafting at serving batch sizes

- Lens: L-06 resource disaggregation; L-10 load balancing and elasticity.
- Existing mechanism: drafting runs in the target's worker (g-code §1). Dedicated draft GPUs (SwiftSpec, PEARL, SSD,
  DPara) win at batch 1-4 and lose their edge by batch 8-16 [L20] [L2] [L7]; same-GPU drafting gives 55% more
  throughput per GPU in long-context serving [L19]; NebulaSD pools draft and target workers [L18]; vLLM N:M RFC [L16].
- Problem: whether pooling drafters across many targets pays at production batch sizes is contested.
- Why now: NebulaSD (Sep 2026) and two engine RFCs [L16] [L17].
- Exact gap: per-GPU goodput of a shared drafter pool vs colocated drafting at batch 32-256.
- Research question: Can an N:M shared drafter pool raise per-GPU goodput at batch 32+ compared with colocated EAGLE-3
  at the same GPU count, while meeting a TPOT SLO?
- Hypothesis: pooling wins only where the target is KV-read bound and the drafter fits nowhere else.
- Cheapest decisive test: closed-form model from published step times [L9] [L20], then NebulaSD numbers.
- Biggest novelty threat: NebulaSD [L18]; DPara [L20]; [L9].
- Scale risk: the crossover is already published.
- Compute: blocked on compute (multi-GPU). Ideal 8x H100. Lower-resource: 2 GPUs, only small batch. Access
  unconfirmed.

## C4. Speculative execution: the cost of optimistic all-accepted scheduling under low acceptance

- Lens: L-08 speculative execution.
- Existing mechanism: vLLM async scheduling plans step N+1 as if all K drafts of step N were accepted, then corrects
  positions on GPU; every request pays `1+K` slots and verify tokens each step whatever it accepted (g-code §3)
  [L92]. SGLang's overlap scheduler does the same [L97]. Rollback paths break: PP deadlock [L95], stale updates
  [L100], stream races [L101]; v0.19 halved an EAGLE3 speedup [L94].
- Problem: under low acceptance at high batch, optimistic scheduling verifies tokens that are mostly rejected.
- Why now: zero-bubble async + spec is default since v0.19 [L93].
- Existing work: adaptive K, adaptive verification and early pruning [L59] [L69] [L70] [L210] [L213]; R04 killed
  adaptive K.
- Exact gap: none beyond adaptive verification unless optimism itself (not K) costs something.
- Research question: Does pessimistic (wait-for-acceptance) scheduling beat optimistic async scheduling in goodput
  when acceptance is below some threshold at batch 64+, compared with adaptive verification?
- Hypothesis: the overlap win (+3-5% at high load [L99]) exceeds any optimism loss.
- Cheapest decisive test: compare async on/off at low acceptance on Colab.
- Biggest novelty threat: adaptive verification [L59] [L209].
- Scale risk: overlap gains shrink with load [L99] [L91].
- Compute: runnable now on Colab. Air cannot (async off with spec on Metal).

## C5. Isolation: shared cross-request draft caches across tenants

- Lens: L-11 isolation and multi-tenancy (with L-09: drafting from other requests' work).
- Existing mechanism: vLLM's suffix decoding keeps one global tree of up to 10,000 past responses per worker; only
  `req_id` and tokens reach it, no tenant, LoRA id, or `cache_salt` [L189] [L190] (g-code §4). Arctic Inference
  defaults to 100,000 cached requests, FIFO, no isolation note [L191]. SGLang's NGRAM trie takes every request's
  prompt tail and output [L192] [L193]; llama.cpp's ngram-mod pool is shared by all slots [L194]; TensorRT-LLM has
  a public n-gram pool [L125]. The KV prefix cache, by contrast, has `cache_salt` for trust groups "to block timing
  attacks" [L188], and per-tenant salting keeps about 93% of prefix-cache benefit [L195].
- Problem: a draft cache built from other users' outputs makes a request's accept counts, hence its token timing and
  streamed chunk sizes, depend on what other tenants generated.
- Why now: suffix decoding shipped in vLLM and Arctic in 2025-2026 and gains most on agentic and SQL workloads where
  the global tree beats the per-request tree (R04 L237) [L126] [L127]; speculation side channels are established
  [L183] [L185] [L186].
- Existing work: "When Speculation Spills Secrets" leaks a fixed public REST datastore at >25 tokens/s and only warns
  that user-derived datastores could leak; it never tests a runtime cache filled by other users [L183] [L184].
  Prefix-cache leakage and its salting defense are solved [L188] [L195].
- Exact gap: whether a co-tenant can infer another tenant's generated content through a shared online draft cache
  under batched serving, and what tenant-partitioning costs in acceptance.
- Research question: Can a co-tenant recover or confirm another tenant's generated text through acceptance-dependent
  streaming behaviour of a shared suffix/n-gram draft cache in vLLM under realistic batching, and does per-tenant
  partitioning remove the channel while keeping most of the cross-request acceptance gain, compared with a global
  tree and with no cross-request tree?
- Hypothesis: (a) membership of a victim output span is detectable from accepted-tokens-per-step well above chance;
  (b) tenant partitioning keeps most of the gain because most cross-request hits come from the same tenant's
  workflow.
- Cheapest decisive test: offline greedy simulation of suffix drafting on a multi-tenant trace (exact for greedy);
  then a live attack on Colab vLLM with suffix decoding.
- Biggest novelty threat: [L183] [L184]; prefix-cache side-channel papers [L195]; any 2026 paper on suffix-cache
  privacy not yet found.
- Scale risk: batching and noise may blur per-request timing; streaming chunk counts are not blurred.
- Compute: runnable now. Offline simulation runs on the Air (Arctic suffix tree is CPU). Live attack: Colab 1 GPU,
  Qwen3-8B or Llama-3.1-8B with suffix decoding. Ideal: 1x H100 with realistic concurrency 32-64. Lower-resource:
  Colab L4 with a 4B model; keeps the channel. Estimate: under 10 GPU-hours. Access confirmed (Colab).

## C6. Isolation: multi-LoRA serving with one unadapted drafter

- Lens: L-11 isolation and multi-tenancy.
- Existing mechanism: the drafter ignores LoRA (g-code §6); SGLang measured heavy-shift adapters cutting accept length
  19-41% and LoRA+spec falling below no-spec by concurrency 32 [L200]; drafter-side LoRA is an open vLLM RFC [L201]
  and SGLang request [L202]; TensorRT-LLM marks LoRA+spec untested [L204]; vLLM MRV2 misroutes lm_head LoRA under MTP
  [L198]; vllm-metal may misalign LoRA under spec (g-code §7.6).
- Problem: multi-tenant adapter serving loses speculation exactly for the tenants with the most specialised adapters.
- Why now: [L200] [L201] in 2026.
- Exact gap: per-adapter drafter adapters co-batched with target adapters, at many adapters.
- Research question: Can per-adapter draft LoRAs batched Punica-style recover the accept length lost by a shared
  drafter at 10-100 adapters, compared with a shared drafter and with model-free suffix drafting, within a fixed
  adapter memory budget?
- Hypothesis: draft LoRA recovers most of the loss at small memory cost.
- Cheapest decisive test: read [L201]'s numbers; train one draft LoRA.
- Biggest novelty threat: [L201] [L203]; LoRA-aware drafter papers not yet searched.
- Compute: Colab for one adapter; many adapters need training compute (unknown).

## C7. Load balancing: acceptance-aware data-parallel balancing and per-rank speculation length

- Lens: L-10 load balancing and elasticity.
- Existing mechanism: DP ranks sync each target and drafter forward; dynamic K is disabled under DP to avoid deadlock
  [L145] (g-code §5); SGLang's min-vote across ranks thrashed about 1,100 times per hour and both PRs closed unmerged
  [L143] [L144]; routers and DeepSeek's decode balancer use counts and KV only [L158] [L173] [L175] [L176].
- Problem: a low-acceptance rank runs the same padded step and emits fewer tokens.
- Exact gap: per-rank K or acceptance-aware routing under lockstep DP.
- Research question: Can acceptance-aware request placement across DP-attention ranks raise goodput compared with
  count/KV balancing at the same K, under a TPOT SLO?
- Hypothesis: grouping by predicted acceptance lets ranks agree on K without losing speculation.
- Cheapest decisive test: simulation from per-request acceptance traces.
- Biggest novelty threat: semi-clairvoyant acceptance-aware scheduling [L179]; DP balancing papers [L166] [L167].
- Scale risk: count balancing already reacts to slower ranks.
- Compute: blocked on compute (DP attention + MoE needs 4-8 GPUs). Simulation local.

## C8. Work stealing: draft caches split across replicas

- Lens: L-09 work stealing; L-10 load balancing.
- Existing mechanism: each worker process has its own suffix global tree, so N replicas or DP ranks hold N separate
  trees (g-code §4); routers balance by load or prefix, not draft-cache content [L173] [L176].
- Problem: cross-request hits fall when similar requests land on different replicas.
- Exact gap: how much acceptance random sharding loses and whether affinity routing recovers it.
- Research question: Does routing by draft-cache affinity recover the acceptance lost when a suffix cache is split
  across N replicas, compared with random and prefix-affinity routing?
- Hypothesis: loss grows with N; prefix-affinity routing recovers most of it.
- Cheapest decisive test: same offline simulation as C5 with replica partitions.
- Biggest novelty threat: prefix-affinity routing already co-locates similar requests [L173]; acceptance nearly flat
  across tree size (R04 L237).
- Compute: offline simulation on the Air.

## C9. Isolation: one tenant's acceptance steering a batch-level speculation controller

- Lens: L-11 isolation and multi-tenancy; L-10 elasticity.
- Existing mechanism: batch-level controllers (Nightjar bandit [L61], TurboSpec goodput [L121] [L169], Cascade
  [L161]) decide speculation for the whole batch from aggregate signals. Adversarial suffixes collapse acceptance and
  the proposed defence is to monitor acceptance and disable speculation [L187]; the paper projects but never measures
  batch coupling (frontier F).
- Problem: one low-acceptance tenant could drive a batch-level controller to switch speculation off for everyone.
- Exact gap: noisy-neighbour coupling through speculation controllers.
- Research question: Does a minority of low-acceptance requests reduce other tenants' TPOT through batch-level
  speculation control, compared with per-request control, under mixed workloads?
- Hypothesis: yes for batch-level controllers, no for per-request ones.
- Cheapest decisive test: simulate a controller on mixed acceptance traces.
- Biggest novelty threat: per-request controllers exist [L210] [L213] [L209] [L69].
- Compute: simulation local; live test on Colab.

## C10. Speculative execution on unified memory: overlapping drafting with verification on Apple silicon

- Lens: L-08 speculative execution; L-06 disaggregation (CPU vs GPU).
- Existing mechanism: on vllm-metal a spec step is fully serial (build, eval, sample, draft); async scheduling and
  Metal's one-step-ahead pipeline are both off with spec (g-code §7.3). CPU drafting in parallel with GPU target
  exists for discrete hardware (DuoDecoding [L13], Dovetail [L12]).
- Exact gap: overlap of CPU drafting with GPU verification when both share one memory bus.
- Research question: Can overlapping CPU n-gram/suffix drafting with GPU verification on unified memory cut step time
  compared with serial drafting on vllm-metal, at batch 1-8?
- Hypothesis: CPU drafting is microseconds per token [L127]; hiding it saves little.
- Cheapest decisive test: time the draft phase in a vllm-metal step.
- Biggest novelty threat: R04 found negative results for unified-memory draft/target sharing (R04 L195, L196).
- Compute: Air.

---

# Pass 1 (2026-10-20)

Three hostile subagents attacked all ten cards with the adversarial checklist. New sources are L221-L270 (files
`frontier/h-*`, `i-*`, `j-*`). Outcome: C1, C3, C4, C6, C7, C8, C10 killed; C2, C5, C9 narrowed. C5' is the main
survivor; C2' and C9' go to pass 2 as weak survivors.

## C1 pass 1: KILL
- Combination: Kill. Reduces to R04 C1 (de-speculate before preempt), killed on scale.
- Oracle: Kill. Block size 16, contexts 1k-8k, K=3..7: lookahead holds 0.375-0.94 extra blocks per request, 0.6-1.5%
  of KV at 1k and 0.07-0.18% at 8k (subagent arithmetic). The blocks are the request's own next tokens, so shrinking
  K only delays a preemption by a few decode steps.
- Adjacent: Narrow. The one avoidable-preemption report with spec concerns hybrid Mamba state blocks [L221].
- Simplicity: Narrow. Admission reordering [L221] or fixing tree sizing [L53] is enough.

## C2 pass 1: Narrow to C2'
- Simplicity: Kill for EAGLE-3/MTP. Draft KV is 4 KiB/token (3.1% of target); re-prefilling it on D needs 24 KiB/token
  of hidden states [L228].
- Oracle: Kill for DFlash. Draft KV is 13.9% of transfer bytes; shipping the fused feature saves about 7% [L226]
  [L227]. Live only for a separate draft model: Qwen3-0.6B KV is 78% of Qwen3-8B KV, while re-prefilling it costs
  about 7% of target FLOPs [L229] [L230].
- Novelty: Narrow. SGLang ships first draft token plus hidden state [L225]; a D-only drafter flag replays the prefix
  [L224]. Scoop risk high [L223] [L31].
- Feasibility: Open. Needs 2 GPUs with NIXL.

## C3 pass 1: KILL
- Simplicity: Kill. DPara's own batch-16 table: two GPUs give 3869 tok/s vs 3627 for one colocated GPU [L20].
- Scale: Kill. NebulaSD verify is 28.8 ms at batch 32 and 82.8 ms at batch 128, compute-bound [L18]; SpecStream gets
  55% more per GPU with same-GPU drafting [L19].
- Oracle: Kill (subagent arithmetic). EAGLE-3 drafting is about 3-5% of FLOPs at batch 32+, so pooling saves at most
  about 5% before paying for drafter GPUs and transfer.

## C4 pass 1: KILL
- Combination: Kill. Pessimistic planning only helps with per-request K, which is R04's adaptive-K kill; per-request
  truncation works only with async off [L234] [L208].
- Oracle: Kill. Qwen3-4B on one L20X up to batch 64: step cost nearly flat in verified tokens; truncation did not
  improve wall clock [L208] [L234].
- Simplicity: Kill. Async gains are only +1.8-7.1% [L91] and +3-5% at high load [L99].

## C5 pass 1: Narrow to C5'
- Novelty: Survives. No paper tests a runtime draft cache filled by other users. [L183] [L184] leak a fixed public
  datastore. 2026 speculation attacks target availability [L250] [L251]; [L249] leaks metadata only.
- Adjacent: Narrow. Isolation scopes and salting are worked out for prefix/KV caches [L241] [L242] [L243]. Porting
  `cache_salt` to the suffix tree alone is engineering.
- Combination: Survives. Suffix drafting couples one request's streamed chunk sizes to another tenant's output through
  the target's agreement. Prefix-cache attacks leak the attacker's own input membership, not a victim's generation.
- Simplicity: Open (measure first). Baselines: `suffix_decoding_max_cached_requests=0` or salt by tenant. Acceptance
  is near flat across tree size [L126], and per-problem shards can beat a global tree [L248].
- Scale: Narrow. Contention shrinks timing channels (Cohen's d 0.78 to 0.21 with two workers) [L244]; accepted tokens
  per step is not pure timing and is untested under batching. Suffix decoding is often disabled above about batch 20
  [L239].
- Oracle: Open. A victim span is proposable only if its frequency clears `min_token_prob=0.1` and the attacker
  supplies the matching context [L237]; single-occurrence spans likely fall below threshold.
- Correctness: Survives. Verification keeps outputs exact [L236]; this is a confidentiality problem.
- Feasibility: Survives. Offline simulation on the Air; live attack under 10 Colab GPU-hours.
- Relevance: Survives. Cross-user cache leakage is a production concern [L241] [L242].
- Scoop: Narrow. RL rollout systems share or shard suffix trees [L246] [L247] and debate global vs scoped [L248], not
  in a multi-tenant leakage frame.

## C6 pass 1: KILL
- Novelty: Kill. LoRAX has served per-request speculation adapters co-batched with LoRAs since 2024 [L252] [L253]
  [L254]; per-domain draft LoRA is measured [L201]; Apple frames per-application drafter memory [L255].
- Oracle: Kill. Public adapters change accept length by -3.6% to +8.8%; only artificial heavy-shift adapters lose
  19-43% [L200].
- Scoop: High [L201] [L203].

## C7 pass 1: KILL
- Simplicity: Kill. A low-acceptance rank holds requests longer, so count-based routing (`core_client.py:1605`)
  already sends it fewer arrivals.
- Oracle: Kill. At fixed K, total accepted tokens do not depend on placement; any gain needs per-rank K, which is
  adaptive K (R04 kill) plus DP agreement, tried and abandoned [L143] [L144].
- Scale: Kill. Measured DP imbalance comes from prefill admission [L261] [L262].

## C8 pass 1: KILL
- Scoop: Kill. Cross-worker suffix tree sharing and sharding exist [L246] [L247]; scoped shards can beat global [L248].
  Acceptance loss from sharding becomes a measurement rider on C5'.

## C9 pass 1: Narrow to C9'
- Novelty: Kill for the mechanism. Per-request K beats batch-threshold control by 9-58% [L265].
- Relevance: Narrow. Under static K or vLLM's batch-size-keyed dynamic K, a low-acceptance request does not change
  other requests' step time beyond staying longer (g-code §6). Coupling exists only in acceptance-driven batch
  controllers: SGLang adaptive EMA [L182], Nightjar [L61], TurboSpec [L121].

## C10 pass 1: KILL
- Oracle: Kill (estimate, not measured). Suffix drafting at 20-30 µs/token [L127] x K=4 x batch 8 is about 0.6-1 ms
  against a ~24 ms Metal decode step (E00), so at most about 4% to hide.
- Adjacent: Kill. CPU+GPU co-execution on shared bandwidth leaves decode unchanged [L267].
- Relevance: Kill. Metal's loss is serial verification and async scheduling off [L268] [L270].

---

## C5'. Cross-tenant membership leakage through a shared suffix draft cache

- Lens: L-11 isolation and multi-tenancy (with L-09).
- Existing mechanism: vLLM suffix decoding keeps one global tree of up to 10,000 responses per worker, keyed only by
  `req_id` [L189] [L190] [L236]; a span is proposed when its frequency clears `min_token_prob` [L237]. KV prefix
  caches have `cache_salt` [L188]; draft caches have no equivalent.
- Problem: a co-tenant's accepted tokens per step, and so its streamed chunk sizes, depend on other tenants' outputs.
- Why now: suffix decoding shipped in vLLM and Arctic in 2025-2026 [L126] [L127]; prefix-cache leakage became a CVE
  class [L241] [L242].
- Existing work: [L183] [L184] (fixed public datastore only); prefix-cache attacks and defences [L241] [L242] [L243];
  shared suffix trees in RL rollouts [L246] [L247] [L248].
- Exact gap: no measurement of whether a runtime draft cache filled by other users leaks their generated spans to a
  co-tenant under batched serving, and what tenant salting costs in acceptance.
- Research question: In batched vLLM suffix decoding, can a co-tenant confirm that a guessed span appeared in another
  tenant's outputs from its own accepted tokens per step (streamed chunk counts), and does salting the tree by tenant
  remove the signal while keeping most of the cross-request acceptance, compared with a global tree and with
  `max_cached_requests=0`?
- Hypothesis: (a) for spans that recur in victim outputs (at least about 3 times), a confirmation probe separates
  member from non-member spans with AUC at least 0.9 at batch 1 and stays above 0.75 at batch 16; (b) tenant salting
  keeps at least 80% of the global tree's accepted tokens per step on a multi-tenant agentic trace. Both thresholds
  are heuristics.
- Cheapest decisive test: offline greedy simulation with Arctic's CPU suffix tree on the Air: replay a multi-tenant
  trace, insert victim outputs, run attacker probes, compute the acceptance difference between member and non-member
  probes (exact for greedy, given the target's greedy continuation). Then confirm live on Colab vLLM with streaming.
- Biggest novelty threat: an unpublished or missed 2026 paper on suffix-cache privacy; [L183] extended to runtime
  caches.
- Scale risk: suffix decoding is often off above about batch 20 [L239]; the channel needs repeated victim spans, so it
  may only cover templated output (SQL, code, boilerplate).
- Compute: runnable now. Offline on the Air. Live: Colab one GPU, a 4B-8B model with suffix decoding, under 10
  GPU-hours. Ideal: 1x H100 at batch 32-64. Access confirmed (Colab).

## C2'. Re-prefill vs ship for a separate draft model in PD serving

- Research question: For a separate draft model under PD disaggregation, does re-prefilling the drafter on D beat
  shipping its KV in TTFT as network bandwidth varies?
- Why weak: production PD uses EAGLE/MTP where the question dies [L228]; scoop risk high [L223] [L224].
- Compute: blocked. Needs 2 GPUs with NIXL; ideal 2x H100 with tunable link bandwidth. Access unconfirmed.

## C9'. Tenant harm from acceptance-driven batch speculation controllers

- Research question: Does a batch-average acceptance controller (SGLang adaptive EMA) raise TPOT for high-acceptance
  tenants when batched with low-acceptance tenants, compared with per-request control?
- Why weak: per-request control already beats batch control [L265]; this is a characterization, not a mechanism.
- Compute: runnable now on Colab one GPU with SGLang; the simulation runs on the Air.

---

# Pass 2 (2026-10-20)

Panel: claude-fable-5-1, gpt-6.1-sol, gpt-6-astra, claude-opus-5-5@medium. Replies in `pass2/`. Every source a reviewer
cited was reopened by a subagent and added as L271-L294 (`frontier/k-pass2.tsv`). Several reviewer quotes were wrong
or only partly right and are corrected below. No reviewer reversed a pass-1 kill.

## Merged rejections

| # | Rejection | Raised by | Answer |
|---|---|---|---|
| R1 | "When Speculation Spills Secrets" already has the probe, the accepted-token observable, the "reusing leaked sequences" extension, and names user-derived datastores as leakable [L271] | all four | Stands for the attack class. What is not tested anywhere I found: a datastore that co-tenants fill online, with local/global competition, self-insertion, eviction and batching. Answered by argument, not evidence. |
| R2 | Tenant salting is the obvious fix and probably near free, since acceptance is near flat in tree size [L126] and scoped shards can beat global [L248] | fable, opus, astra, sol | Open. This is the cost arm of the kill-test. If salting keeps nearly all acceptance, the result is an advisory, not a thesis. |
| R3 | Same pattern as compression-dictionary and dedup side channels; origin scoping is the known defence [L273] [L274] [L275] [L276] | all four | Stands for the defence. One residue: same-domain dedup shows namespace scoping can fail when the domain boundary is wrong [L276], so which tenant scope to use has some content. |
| R4 | Probe protocol is confounded: local tree wins ties so a literal guess in the prompt masks the channel [L293]; each probe inserts itself into the global tree; a hit shows cache membership, not which tenant | sol, astra, opus | Accepted. Kill-test uses fresh cache clones per probe, keeps the guess out of the prompt, and uses paired worlds that differ only in the victim insertion. |
| R5 | The card's "at least about 3 occurrences" premise is wrong: `_speculate_path` starts at probability 1 and multiplies count ratios only at branch points, with no minimum count [L237] | fable, opus, astra | Accepted. A unique context drafts a single-occurrence span. Hypothesis rewritten. |
| R6 | Streamed chunks may not equal accepted counts | sol, astra vs fable | Both partly right. `stream_interval` defaults to 1 (`vllm/config/scheduler.py:214`, read locally), so one chunk per step by default, but `RequestOutputCollector` merges deltas when the consumer falls behind [L294]. Offline screen uses exact counts; live phase measures merge rate. |
| R7 | Relevance is limited: vLLM disclaims isolation between tenants in one server process [L277]; a prefix-cache salt bypass was rated Low [L287] | sol, astra, opus | Stands. Frame as deployment risk and defence cost, not a broken promise. Severity ceiling is low. |
| R8 | Scale: suffix decoding is off at high load | pass 1 | Weakened. Snowflake now calls it "production-ready at all concurrency levels" [L291]; vLLM GPU suffix RFC shows global-index acceptance rising with load [L292]. |

Corrections to reviewer quotes found on reopening: the 9-52% token-aggregation figure covers only exact-knowledge
attacks, approximate-knowledge attacks fall to 4-17.6% (section 6, not 5) [L272]; SGLang #20698 was closed by a stale
bot, and "very niche" was a reviewer's question, not a maintainer rejection [L279]; vLLM #48202's "completed" has no
linked commit, so it does not show per-request K landed [L281]; llama.cpp #27852's per-slot carryover is a bug, while
"cross-request by design" refers to the dynamic and static caches [L286]; RFC #51788's CPU tree 1.98 to 1.89 runs from
c16 to c256 [L292].

## C2' pass 2: KILL
Simplicity and relevance (astra, fable, opus; sol open). CacheGen already adapts between shipping KV and sending text
for recompute as bandwidth changes [L278]; an independent drafter's state comes from prompt tokens, so no new
constraint. Connector design moves TTFT far more than draft bytes [L280]. The one attempt to ship separate-drafter KV
under PD went stale [L279]. Also compute-blocked.

## C9' pass 2: KILL
Novelty and scoop (fable, opus; astra and sol weak). Per-request sizing is shipped in SGLang for DSpark [L285]; the
per-request item is on SGLang's roadmap with CUDA graphs as the obstacle [L282]; open PRs rework the batch policy
[L283] [L284]; per-request K beats batch control [L265]. The adversarial residue (acceptance collapse dragging a
shared EMA [L187] [L251]) is left as an uninvestigated direction, not carried.

## C5' pass 2: Narrow to C5''

## C5''. Confirming a co-tenant's outputs through a shared suffix draft cache, and what isolation costs

- Lens: L-11 isolation and multi-tenancy (with L-09).
- Existing mechanism: vLLM suffix decoding keeps a global tree of past outputs per worker keyed only by `req_id`
  [L189] [L236]; it drafts from any matched context of up to 24 tokens anywhere in any cached output, at probability
  1 when the context is unique [L237]; the request's own prompt tree wins ties [L293]. The KV prefix cache can be
  salted [L188] and only matches from position 0.
- Problem: a request's accepted tokens per step, visible as streamed chunk sizes at the default `stream_interval=1`,
  depend on what other tenants generated.
- Existing work: [L271] [L272] (static datastore, same observable); prefix-cache attacks and defences [L241] [L242]
  [L243] [L288] [L289]; compression and dedup analogues [L273] [L275] [L276].
- Exact gap: I did not find public work that measures how much of a co-tenant's output an attacker can confirm
  through an online, co-tenant-filled suffix tree under its real selection rules, or what tenant scoping costs in
  acceptance on a multi-tenant trace.
- Research question: Can a co-tenant confirm that a span appeared in another tenant's outputs from its own accepted
  tokens per step under vLLM suffix decoding on a multi-tenant agentic/SQL trace, and does scoping the global tree by
  tenant remove that signal while keeping most of the global tree's acceptance gain over local-only drafting?
- The answer is "no" if first-probe confirmation is near chance on spans the target would reproduce, or if tenant
  scoping keeps nearly all of the global tree's gain (then the fix is free and there is no trade-off to study).
- Hypotheses: (a) on spans the target reproduces from an attacker-elicitable context, first-probe TPR at 1% FPR is at
  least 0.5; (b) tenant scoping keeps less than 80% of the global tree's incremental gain over local-only. Both
  thresholds are heuristics.
- Cheapest decisive test: offline paired-world replay on the Air (below).
- Biggest threats: R1 and R2. Most likely outcome given [L126] [L248]: the fix is near free, which kills it as a thesis.
- Compute: runnable now on the Air; live confirmation on Colab under 10 GPU-hours (low confidence).

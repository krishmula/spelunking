# R04 candidates

Topic T-02, speculative decoding. Lenses L-01 checkpoint/restart, L-02 live migration, L-03 preemption and recovery,
L-04 scheduling and admission control, L-05 caching and memory hierarchy. Sources are `[L<n>]` rows of `ledger.tsv`;
`e-code` is `frontier/e-code.md` (local vLLM v0.31.0 at db9527a46, vllm-metal at 572322e, read, not run).

The frontier map suggests one framing that several lenses share: speculative decoding adds state that is not needed
for correctness (draft KV, draft hidden-state buffers, suffix trees, per-draft recurrent scratch, spec tokens in
flight). Engines treat it inconsistently. vLLM frees draft KV with target KV on preemption [L28] (e-code §2), SGLang
used to drop it on PD retraction with only an acceptance loss [L65] [L66], vLLM's NIXL path does not transfer it
[L104], and offload tiers mishandle it [L83] [L103]. The cards below are generated from that mismatch and from the
lens list, then attacked.

---

## C1. Speculation under KV pressure: de-speculate before you preempt

- Lens: L-03 preemption and recovery, L-04 scheduling and admission control, L-05 caching and memory hierarchy.
- Existing mechanism: speculation costs memory three ways: draft weights, draft KV (1.14x per token on a dense
  4B target with a block drafter, up to 1.8x on hybrid/MoE [L93]), and lookahead slots reserved for max K on every
  running request, even when dynamic K or low acceptance uses fewer (e-code §1, §5). On hybrid models vLLM reserves
  per-draft recurrent scratch, 9 of 45 blocks per request in one case [L170]. When KV runs out, vLLM V1 preempts by
  recompute, choosing the victim by position or priority only (e-code §2) [L145]. Nightjar turns speculation off
  and moves draft weights to CPU under memory strain [L42] [L105].
- Problem: at a fixed memory budget, speculation shrinks the batch and adds preemptions. No engine releases a
  request's speculation memory before preempting a request. Unknown whether the preemptions speculation induces
  erase its step-time gain at realistic memory pressure.
- Why now: draft KV grew with block drafters (DFlash, DSpark) and hybrid scratch [L93] [L170]; preemption is
  recompute-only in V1 [L145]; reports of avoidable preemption with hybrid + spec [L89].
- Existing work: Nightjar (global on/off plus draft offload) [L42]; SmartSpec/TurboSpec per-request K from 0 by
  goodput, not memory [L31] [L32]; DSD batch-size table [L45]; Mamba scratch per running request PR closed [L170].
- Exact gap: a per-request memory-reclaim ladder (drop lookahead/draft KV/scratch of low-acceptance requests) as an
  alternative to preemption, and the measured cost of speculation-induced preemption.
- Research question: Can reclaiming per-request speculation memory (lookahead, draft KV, recurrent scratch) before
  preempting reduce p99 TPOT and preemption recompute under KV pressure on hybrid and block-drafter models,
  compared with vLLM V1 recompute preemption plus Nightjar-style global spec off, at equal memory and identical
  outputs?
- Hypothesis: on hybrid or block-drafter targets, speculation memory is 10-20% of KV, enough to avoid most
  preemptions at moderate pressure; on dense EAGLE targets it is under 1% and the idea fails.
- Cheapest decisive test: count speculation-held blocks and preemptions in a pressure sweep with spec on and off
  (extends E02's logging scheduler).
- Biggest novelty threat: Nightjar [L42]; closed vLLM scratch PR [L170]; ReplaySSM/TreeWY shrinking scratch [L84] [L168].
- Scale risk: scratch fixes may make speculation memory negligible; at high batch speculation is often off anyway [L35] [L134].
- Compute: blocker: vllm-metal refuses hybrid spec decode and supports only draft_model/ngram/dflash/dspark greedy
  (e-code §6). Local screening: dense Qwen3 with dflash or draft_model on the Air, lookahead and draft-KV share only.
  Ideal: 1x H100 80 GB, vLLM, a hybrid MTP model (Qwen3.5-4B or Qwen3-Next class) and a DFlash target. Lower: Colab
  L4/A100 with Qwen3.5-4B MTP; cannot show 80B-class behavior. Estimate: screening one evening; decisive 10-20
  GPU-hours (heuristic). Access: Colab confirmed; H100 unconfirmed.

## C2. Recover the prefix-cache hits that EAGLE-style speculation drops

- Lens: L-05.
- Existing mechanism: draft KV at a block boundary depends on the next token, so vLLM drops the last matched block
  for every EAGLE/MTP hit [L97] [L139] (e-code §3). On hybrids the block is thousands of tokens: 87% to 70% hit rate
  [L97], 0% versus 73% [L99], dense checkpoints filling the pool [L98] [L173]. SGLang loses deep-prompt hits 97% to 40-53% [L100].
- Problem: speculation silently costs prefix reuse.
- Why now: hybrid models with huge blocks plus MTP are the default deployment [L8] [L179].
- Existing work: lookahead-aware block hashing PR, 72.7% to 85.0% hit rate, +37% throughput, opened 2026-10-14 [L140] [L141].
- Exact gap: connector/offload tiers (part 2 of the PR) and the hybrid checkpoint-retention interaction.
- Research question: Can lookahead-keyed draft KV plus spec-aware recurrent checkpoint placement recover prefix hit
  rate on hybrid MTP models across GPU and CPU tiers, compared with PR #61589, at unchanged acceptance?
- Hypothesis: residual loss after #61589 is concentrated in offload tiers and hybrid checkpoint alignment.
- Cheapest decisive test: hit-rate replay of a multi-turn trace with #61589 applied.
- Biggest novelty threat: #61589 and its stacked part 2 [L141].
- Scale risk: none; the issue grows with block size.
- Compute: needs CUDA vLLM with hybrid MTP; vllm-metal refuses MTP plus prefix caching on hybrids [L177]. Colab L4 for
  Qwen3.5-4B. Estimate a few GPU-hours. Access: Colab confirmed.

## C3. Draft state as soft state: keep, move, recompute, or drop?

- Lens: L-01 checkpoint/restart, L-02 live migration, L-03 preemption and recovery, L-05 caching and memory hierarchy.
- Existing mechanism: draft KV and draft hidden-state buffers are not needed for correct output; losing them costs
  only acceptance [L65] [L66]. Engines make four different choices on the four events that move state: vLLM
  preemption frees draft KV and rebuilds it by re-prefilling the drafter (e-code §2); SGLang PD retraction lost it
  [L65] [L66]; vLLM PD transfers it in one direction of work [L73] but NIXL excludes it so decode re-drafts 15k
  tokens [L104]; CPU offload tiers mishandle draft groups [L83] [L103] [L101]; migration of draft KV appears only in
  RLHF rollout [L60]. Llumnix-style migration assumes append-only KV [L59].
- Problem: every event that moves or rebuilds a request (preemption, migration, PD handoff, offload restore, engine
  restart) pays to preserve or rebuild draft state, or silently loses acceptance, with no measurement of which is
  cheaper.
- Why now: draft KV is now a large share of KV for block drafters [L93]; PD and offload are standard; the bug
  cluster [L65]-[L83] [L101]-[L104] shows engines are deciding ad hoc in 2026.
- Existing work: windowed draft attention drops most draft KV at 1M context [L178]; compressed draft KV [L94];
  KV-reading drafters [L95]; RLHFSpec two-stage migration [L60].
- Exact gap: the acceptance-recovery curve after draft state loss (how many tokens until acceptance returns to its
  steady value, by drafter type), and the resulting policy for each state-moving event.
- Research question: Does dropping draft KV and drafting from a short recent window after a state-moving event
  (preemption, PD handoff, offload restore, migration) keep acceptance within a small margin of full draft-state
  preservation, at lower transfer or recompute cost, for EAGLE-3, MTP, and DFlash drafters, compared with the vLLM
  default (full drafter re-prefill) and with transferring draft KV?
- Hypothesis: EAGLE-3 and MTP acceptance recovers within tens of tokens because they condition on the target's
  hidden states; DFlash, which stores multi-layer target features in its own KV [L16], recovers slowly.
- Cheapest decisive test: offline: truncate or zero the draft context at a point in a generation and log
  per-position acceptance afterward, for each drafter, versus the untouched run.
- Biggest novelty threat: Windowed-MTP [L178] (draft needs only a window); MagicDec and other sparse draft KV [L22] [L94].
- Scale risk: if draft re-prefill is cheap relative to target re-prefill, rebuilding is free and the question vanishes.
- Compute: local screening possible with dflash or draft_model on the Air (greedy) (e-code §6). EAGLE-3 and MTP need
  CUDA vLLM (Colab). Ideal: 1x A100/H100 with Qwen3-8B EAGLE-3, a DFlash Qwen3-8B, a hybrid MTP model. Lower: Colab
  L4 with Qwen3-4B class drafters. PD transfer cost claims need 2 GPUs with NIXL, ideally 2 nodes. Estimate:
  screening 1-2 evenings; decisive 20-40 GPU-hours (heuristic). Access: Colab confirmed; 2-GPU unconfirmed.

## C4. One budget for verification tokens and prefill chunks

- Lens: L-04.
- Existing mechanism: vLLM counts spec tokens as ordinary tokens in one token budget, truncating spec tokens to fit
  (e-code §4); adaptive verification picks verify slots up to a budget profiled at startup [L43] [L130].
- Problem: verification tokens and prefill chunks compete for the same step; the split trades TTFT for TPOT.
- Why now: chunked prefill is default and spec is on by default in some deployments.
- Existing work: a RIKEN poster jointly picks candidate tokens and chunk budget [L50]; adaptive verification [L43].
- Exact gap: unclear; may be none.
- Research question: Can a joint verification/prefill budget beat adaptive verification plus a fixed chunk size on
  goodput under a TTFT SLO, on mixed traffic?
- Hypothesis: gains appear only with long prompts and moderate load.
- Cheapest decisive test: sweep chunk size times K on Colab and compare to the best fixed pair.
- Biggest novelty threat: [L50] [L43] [L36].
- Scale risk: adaptive-length gains over best fixed are 7-15% [L38] [L39] [L42].
- Compute: Colab GPU; a few GPU-hours.

## C5. Capacity, eviction, and persistence of cross-request draft caches

- Lens: L-05 caching and memory hierarchy, L-01 checkpoint/restart.
- Existing mechanism: SuffixDecoding keeps a global suffix tree over past outputs plus a per-request tree; memory
  grows linearly, 1.67 GB at 27M tokens to 6.15 GB at 572M [L12]; Arctic Inference caps it at 100,000 requests with
  FIFO eviction only [L108]; SGLang NGRAM keeps up to 10M entries [L124]; Cacheback uses LRU n-gram tables [L113];
  REST datastores reach 27 GB [L111]; a GPU suffix RFC adds capacity knobs [L110] [L144]. The cache dies with the
  process. In vLLM, a request that misses one step appears to lose its per-request tree history (e-code §2, untested).
- Problem: the draft cache is a cache whose hit rate sets acceptance, but its size, eviction policy, and survival
  across restarts and preemption are unstudied.
- Why now: suffix drafting gives the largest gains on agentic traffic (5.3x, 2.8x over EAGLE-3) [L107] and agent
  acceptance drifts across turns and agents [L190].
- Existing work: FIFO [L108], LRU [L113], per-agent caps [L190], tenant leakage [L115].
- Exact gap: the acceptance-versus-capacity curve and the oracle gap between FIFO and an ideal policy on real
  agentic traces; whether warm-start from a persisted cache matters after restart.
- Research question: On agentic traces, can a reuse-aware eviction policy (or persistence across restarts) for a
  global suffix cache raise mean accepted length at a fixed memory cap, compared with Arctic's FIFO at the same cap,
  without raising per-token CPU cost?
- Hypothesis: on agentic traces most accepted tokens come from the per-request tree, so the global cache's policy
  barely matters (this would kill it); if global hits dominate, FIFO leaves a large gap at small caps.
- Cheapest decisive test: trace-driven simulation, no GPU: replay public agent trajectories through
  arctic-inference's suffix tree with greedy acceptance, splitting accepted tokens by per-request versus global
  source, across caps and policies including a Belady oracle.
- Biggest novelty threat: SuffixDecoding authors' own ablations [L12]; Cacheback [L113]; AgSpec [L190].
- Scale risk: multi-tenant traffic could make global hits rare; at high batch speculation is often off.
- Compute: blocker: none for the simulation (CPU, Air). End-to-end speedups need CUDA vLLM (suffix decoding is
  unsupported on vllm-metal, e-code §6). Ideal: 1x A100/H100 with arctic-inference. Lower: Colab L4. Estimate:
  simulation 1-2 evenings CPU; end-to-end 5-10 GPU-hours (heuristic). Access: Colab confirmed.

## C6. Live migration of a speculating request

- Lens: L-02 live migration.
- Existing mechanism: Llumnix migrates KV in stages because it is append-only, with constant downtime [L59].
  Speculation overwrites the last few slots on rejection (e-code §1), and draft KV must move too [L60].
- Problem: does speculation break live migration?
- Exact gap: only draft KV movement, which is C3's question.
- Research question: Does migrating draft KV alongside target KV preserve acceptance at lower downtime than dropping
  it, compared with Llumnix migration of target KV only?
- Cheapest decisive test: same as C3.
- Biggest novelty threat: RLHFSpec [L60]; Llumnix [L59].
- Compute: 2 GPUs. Folded into C3 before pass 1 (see below).

## C7. Preempt-and-resume equivalence for speculative state

- Lens: L-01 checkpoint/restart, L-03 preemption and recovery.
- Existing mechanism: preemption and checkpoint must cut a request at a commit point. With speculation and overlap
  or async scheduling, engines have cut it in the wrong place: recurrent state advanced twice after a checkpoint one
  step ahead [L67]; stale spec output after preemption caused illegal memory access [L64]; resumed EAGLE chunks
  rebuilt draft KV wrongly [L68]; GDN conv/SSM corruption with alternating spec steps [L175]; acceptance stuck at
  0% after a prefix hit [L143]; NGRAM wrote over radix-cached KV, found by synthetic-weight probes [L148];
  LMCache + MTP gave garbage [L81]; GDN + MTP5 gave garbage at acceptance 1.00 [L82].
- Problem: no systematic check that a request preempted, migrated, offloaded, or restored at any step produces the
  same tokens (and same acceptance) as an uninterrupted run, across spec methods.
- Why now: speculation now composes with async scheduling, hybrid state, PD, and offload tiers; the bug cluster is
  2026.
- Existing work: batch-invariant/deterministic inference work makes greedy equivalence testable (to check); vLLM
  spec tests; synthetic-weight probes [L148].
- Exact gap: a differential test harness that injects preemption, offload, and PD handoff at chosen steps and checks
  token and acceptance equivalence; a census of bugs it finds.
- Research question: Does injecting state-moving events at every step boundary in greedy speculative decoding
  expose output or acceptance divergences in current vLLM and SGLang releases that their test suites miss, compared
  with their existing spec-decode and preemption tests?
- Hypothesis: yes, at least a few new divergences, mostly silent acceptance loss rather than wrong tokens.
- Cheapest decisive test: on the Air, greedy dflash/draft_model on vllm-metal with a scheduler that forces
  preemption at step n, compare tokens and acceptance with the unforced run for n over a range.
- Biggest novelty threat: engine CI tests; fuzzing/metamorphic testing of ML compilers and inference engines.
- Scale risk: this is a correctness question; scale adds paths (PD, multi-node) but not doubt.
- Compute: local screening on the Air (greedy only, e-code §6). Ideal: 1-2 GPUs on CUDA vLLM and SGLang to cover
  EAGLE/MTP/hybrid and PD. Lower: Colab single GPU (no PD). Estimate: harness a week; runs a few GPU-hours. Access:
  Colab confirmed.

## C8. Recurrent-state rollback memory for hybrid speculation

- Lens: L-05, L-03.
- Existing mechanism: vLLM kept a full SSM state per draft token, cutting max batch about 4x [L25] [L84]; the fp32
  per-draft store is 18% of GPU time at concurrency 64 [L171].
- Exact gap: none visible. ReplaySSM [L84], RecoverSSM [L86] [L172], TreeWY [L168], Bole [L167], STree [L166],
  deferred stores [L171], SGLang replay [L87] all address it.
- Research question: Can a cheaper rollback representation beat replay-based rollback on memory and throughput?
- Compute: CUDA GPU with hybrid models.

## C9. Context- and request-aware speculation length

- Lens: L-04.
- Existing mechanism: vLLM's dynamic SD is a batch-size table (e-code §5) [L129]; long prompts get 5-8%
  acceptance [L136]; best-K gain falls to 1.00x at concurrency 128 short context [L135].
- Exact gap: keying K on context and request; already proposed in open RFCs [L135] [L136] and done by SmartSpec,
  DSpark, adaptive verification [L31] [L19] [L43].
- Research question: Can (batch, context, request) keyed K beat adaptive verification on goodput?
- Compute: Colab GPU.

---

# Pass 1 (2026-10-20)

Searches: three hostile subagents, one per candidate pair, wrote `frontier/g-pass1-c1c4.tsv`,
`frontier/h-pass1-c3.tsv`, `frontier/i-pass1-c5c7.tsv` (ledger L202-L257). Their queries covered Nightjar's memory
trigger, vLLM/SGLang preemption and draft-KV accounting, joint prefill/verify budgets, draft-context truncation and
recovery, PD draft-KV transfer, suffix-cache ablations and eviction, inference-engine fuzzing and bug studies, and
batch invariance. Attacks not run after a Kill are marked "not needed".

## C1 pass 1: Narrow, then merged into C3'

1. Novelty: Survives narrowly. Nightjar offloads draft weights only, after spec is already off, and never mentions
   preemption [L202]; V1 preempts `running[-1]` with no reclaim step [L204].
2. Adjacent work: Narrows. SGLang's open sidecar RFC separates draft state so per-request drop becomes a small step
   [L208]; SGLang roadmap lists "memory-aware state management" unchecked [L209]; offloading spec KV [L211].
3. Combination: Open. Nightjar plus a victim policy unless the acceptance cost of dropping draft KV is new.
4. Simplicity: Narrows. Hybrid scratch over-reservation is a static sizing bug, fix PR closed by its author without
   review [L206] [L170].
5. Scale: Kills for dense EAGLE-3: lookahead is at most about one block per request [L205]; one drafter layer is about
   3% of KV [L210]. Lives only for block drafters (draft KV about 12% of a request on Qwen3-4B, 1.14x [L93]).
6. Oracle: Narrows. Dropping a DFlash request's draft KV frees about 12% of it, so avoiding one preemption needs about 8
   requests to lose 5-15% acceptance [L225] [L227]. Small bound.
7. Correctness: Survives as performance only; dropped draft KV keeps outputs correct [L224] [L225].
8. Feasibility: Open; hybrid needs CUDA (e-code §6).
9. Relevance: Open; one avoidable-preemption report [L89].
10. Scoop: SGLang roadmap and sidecar RFC [L208] [L209].
Decision: two Narrows plus a scale Kill on the common case. The only live part (drop a request's draft KV under
pressure, accept lower acceptance) depends on the same unknown as C3: how acceptance behaves after draft state is
lost and how cheaply it can be rebuilt. Merged into C3' as one of its triggering events.

## C2 pass 1: KILL

1. Novelty: Kills. Lookahead-aware hashing fixes the local cache (72.7% to 85.0% hit, +37% throughput) [L140];
   PR opened 2026-10-14 with connector support stacked as part 2 [L141].
2-10: not needed. The remaining hybrid checkpoint-placement piece was killed in R03 (C8, exact DP published) and the
   residual is engineering on an active upstream stack. Lab tweak only.

## C3 pass 1: Narrow to C3'

1. Novelty: Narrows. No source compares keep, transfer, recompute, and drop [L224]-[L232]; but the drop side is measured:
   DFlash accept length 5.02 vs 5.29 after about 2,950 retractions [L225]; MTP keeps 0.985x acceptance after about
   950 unsynced steps, DFlash -14.6%, EAGLE-3 collapses to about zero and never recovers [L226] [L227].
2. Adjacent work: Narrows. Drafters that read target KV own no draft state (H-Spec [L220], LongSpec [L221], Gemma 4
   assistant [L230]); Windowed-MTP shows MTP needs only a window of about 2-4K even at 1M [L219]. The question is
   moot for those drafters and live for EAGLE-3 and DFlash.
3. Combination: Survives for C3'. The coupling is between dynamic spec on/off (K=0) and drafter state: keeping the
   drafter synced during K=0 costs 4.1-11.3% step time [L226]; skipping it breaks EAGLE [L226]; Nightjar prices the
   rebuild as C_switch but does not build a cheaper one [L203].
4. Simplicity: Open. Full catch-up rebuild is the named fix, not built [L227]; whether a windowed one suffices is the
   experiment.
5. Scale: Narrows. Rebuild needs target hidden states for every rebuilt position [L232], so "recompute draft KV" is
   cheap only if target hidden states are buffered or the window is short.
6. Oracle: Open. Upper bound is the sync cost during K=0 phases (4-11% [L226]) plus avoided acceptance loss; needs a
   trace of how often K=0 phases occur.
7. Correctness: Survives; draft state affects acceptance only [L224].
8. Feasibility: Open. EAGLE-3 needs CUDA vLLM (vllm-metal supports dflash and draft_model, greedy, e-code §6).
9. Relevance: Survives. DSD K=0 phases ship in vLLM [L129] and SGLang [L209]; production turns spec off at high load [L134].
10. Scoop: High. The vLLM PR author names catch-up rebuild as the correct general fix [L227]; SGLang sidecar RFC [L208].
Decision: rewrite as C3' below.

## C4 pass 1: KILL

1. Novelty: Kills. SLOs-Serve jointly chooses speculation lengths and the prefill budget under SLOs, 1.76x over vLLM
   [L214]; the RIKEN poster states the same TTFT-constrained problem [L213].
4. Simplicity: Kills. vLLM adaptive verification already prices verification on top of co-scheduled prefill tokens
   [L215].
5. Scale: verification is a few percent of an 8192-token budget [L216].
2-3, 6-10: not needed.

## C5 pass 1: KILL

1. Novelty: Narrows. SuffixDecoding already ablates global versus per-request trees and tree size; no eviction
   comparison [L237]. Arctic ships an offline simulator with cap and evict knobs [L240].
6. Oracle: Kills. Acceptance is nearly flat as the tree grows about 40x [L237]; LRU versus LFU on an online n-gram
   cache differs by about 0.005 acceptance [L238]; recency windows already exist [L243]. A better eviction policy has
   little room.
4. Simplicity: Kills the remaining lead. The per-request tree is destroyed when a request leaves the batch, dropping
   acceptance from 97.4% to 4.5% under PP [L242] (e-code §2), but the fix is to not destroy it: a patch restored 79.1%
   [L242]. That is a bug fix.
2-3, 5, 7-10: not needed.
Kept as lab tweak: check whether upstream vLLM loses the per-request tree on ordinary preemption, and report it.

## C6 pass 1: KILL

1. Novelty: Kills. RLHFSpec migrates draft and target KV in two stages [L233]; NebulaSD moves both through a host KV
   store with incremental writeback [L234]. Llumnix staging still holds for the committed prefix [L59].
2-10: not needed. Merged into C3 before pass 1.

## C7 pass 1: Narrow to C7'

1. Novelty: Narrows. vLLM CI compares EAGLE-3 and ngram_gpu under pressure-induced preemption against a baseline on
   tokens, logprobs, and acceptance within 5% [L254], since Nov 2025 [L255].
2. Adjacent work: Narrows. GRIEF fuzzes vLLM and SGLang with replay-based divergence checks; spec decode is named
   future work and no spec bugs are reported [L249]. The engine bug study has no spec or preemption category [L248].
3. Combination: Open. GRIEF plus a spec adapter is A applied to B, unless acceptance becomes an exact oracle. With
   batch-invariant kernels [L250] [L253] and greedy decoding, a correctly restored draft state should give identical
   drafts and so identical acceptance; vLLM tolerates 5% because it is not batch-invariant [L254].
4. Simplicity: Open. Pressure-induced preemption may already reach most step boundaries.
5. Scale: Survives; a correctness question.
6. Oracle: Open; value is bugs found that CI and GRIEF miss.
7. Correctness: Survives; a real state-consistency question [L67] [L64] [L68] [L143] [L148] [L256].
8. Feasibility: Survives. Greedy dflash and draft_model run on the Air (e-code §6); EAGLE-3/MTP/hybrid on Colab.
9. Relevance: Survives. Spec plus preemption crashes found only at scale [L256]; silent acceptance loss bugs [L143] [L226].
10. Scoop: Medium. GRIEF authors list spec decode as next [L249].
Decision: rewrite as C7' below.

## C8 pass 1: KILL

1. Novelty: Kills. ReplaySSM [L84], RecoverSSM [L86] [L172], TreeWY [L168], Bole [L167], STree [L166], deferred
   stores [L171], SGLang replay [L87]. 2-10: not needed.

## C9 pass 1: KILL

1. Novelty: Kills. Context-keyed K is an open RFC [L135]; per-request off switch is an open feature request [L136];
   SmartSpec [L31], DSpark [L19], adaptive verification [L43] [L215], SGLang adaptive steps [L46] [L209].
   2-10: not needed.

---

## C3'. Cheap draft-state catch-up when speculation resumes

- Lens: L-03 preemption and recovery, L-04 scheduling and admission control, L-05 caching and memory hierarchy; also
  applies to L-01/L-02 events that move target KV without draft KV (PD handoff, offload restore).
- Existing mechanism: dynamic speculation sets K=0 for a request or batch under load [L129] [L209]. While K=0, vLLM
  still runs a draft "sync" forward each step so the drafter's KV stays complete; that costs 4.1-11.3% step time
  [L226]. An opt-in PR skips it for MTP and DFlash, losing 1.5% (MTP K=2) to 14.6% (DFlash) acceptance on resume, and
  refuses EAGLE-3 because its acceptance collapses to about zero and never recovers [L226] [L227]. The PR names a
  catch-up rebuild as the correct general fix, not built [L227]. Nightjar charges the same rebuild as C_switch [L203].
  The same missing-draft-history state arises when target KV survives but draft KV does not: SGLang PD retraction
  (5.02 vs 5.29 accept length) [L225], vLLM NIXL handoff excluding draft groups [L104] [L229], offload tiers that
  drop the draft group [L103], and under memory pressure (from C1).
- Problem: engines choose between paying the sync every step or losing acceptance. Rebuilding the draft KV needs
  target hidden states for each rebuilt position [L232].
- Why now: dynamic K=0 shipped in vLLM and SGLang in 2026 [L129] [L209]; the sync-cost issue and skip PR are from
  2026 [L226] [L227]; block drafters made draft KV big [L93].
- Existing work: sync (vLLM default) [L226]; skip (MTP/DFlash only) [L227]; full re-prefill on preemption (e-code §2);
  transfer in PD [L228] [L231]; windowed draft attention for MTP [L219]; KV-reading drafters with no draft state
  [L220] [L221] [L230].
- Exact gap: whether a bounded catch-up (buffer the target hidden states of the last W positions while K=0, then
  rebuild only the sink block plus those W positions in one batched draft forward at resume) recovers synced
  acceptance for EAGLE-3 and DFlash, and its cost against continuous sync and skip.
- Research question: Can windowed draft-state catch-up at resume time (sink plus last W positions, rebuilt in one
  batched draft forward) keep acceptance within 5% of a continuously synced drafter for EAGLE-3, MTP, and DFlash,
  while removing at least half of the sync cost during K=0 phases, compared with vLLM's per-step sync and PR #53426's
  skip, on traffic whose load makes dynamic speculation toggle?
- The answer is "no" if: EAGLE-3 acceptance after windowed catch-up stays more than 5% below synced for every W up
  to 512; or the catch-up costs as much as the sync it replaces; or on realistic load traces K=0 phases cover under
  10% of steps, so the sync cost is under 1% of total time (heuristic thresholds).
- Hypothesis: EAGLE-3's collapse in [L226] comes from stale draft-KV slots, not from missing distant history, so a
  clean sink-plus-window rebuild restores most acceptance with W in the low hundreds; DFlash needs a larger W.
- Cheapest decisive test: offline, no serving: generate with a synced drafter, then at token t replace the draft
  KV with (a) nothing/stale, (b) sink plus last W rebuilt, (c) full rebuild; log per-position acceptance for 256 tokens
  after t; sweep W. Then count K=0 phase frequency in a load replay with DSD on.
- Biggest novelty threat: the vLLM PR authors building catch-up next [L227]; Nightjar's C_switch [L203]; SGLang
  sidecar RFC [L208].
- Scale risk: at high load K=0 phases are long and frequent (helps the idea); if dynamic speculation is rarely used
  in practice the upside vanishes.
- Compute: blocker: EAGLE-3 and MTP require CUDA vLLM; vllm-metal supports dflash and draft_model greedy only, no
  async (e-code §6). Local screening: DFlash and draft_model truncation curves on the Air with Qwen3-4B class targets,
  if a DFlash drafter for that target loads on vllm-metal (unverified). Ideal: 1x H100 80 GB with Llama-3.1-8B
  EAGLE-3, Qwen3-8B DFlash, a Qwen3.5 MTP model. Lower: Colab L4/A100 with 4-8B targets; cannot speak to MoE/80B.
  Estimate: offline acceptance curves 5-10 GPU-hours; load replay 10-20 GPU-hours (heuristic). Access: Colab
  confirmed; H100 unconfirmed.

## C7'. Exact acceptance equivalence as an oracle for speculative state bugs

- Lens: L-01 checkpoint/restart, L-03 preemption and recovery.
- Existing mechanism: vLLM CI checks tokens, logprobs, and acceptance within 5% under pressure-induced preemption for
  EAGLE-3 and ngram_gpu only [L254]. GRIEF fuzzes vLLM/SGLang with output-divergence replay, no spec decode [L249].
  Draft-state bugs often keep outputs correct and only lose acceptance [L224] [L225] [L143] [L226], which an output
  oracle cannot see and a 5% tolerance can hide.
- Problem: silent acceptance loss after state-moving events (preemption, resume, offload restore, PD handoff, K=0
  toggling) is invisible to current checks.
- Why now: batch-invariant kernels exist in vLLM and SGLang [L250] [L251] [L253], which may make greedy acceptance
  exactly reproducible; the 2026 bug cluster [L64]-[L83] [L143] [L148] [L256].
- Existing work: [L254] [L249] [L248].
- Exact gap: injecting state-moving events at chosen step boundaries, across MTP, DFlash, suffix, draft_model, and
  hybrid targets, with exact (not 5%) acceptance and token equivalence under batch invariance.
- Research question: Does step-injected preemption, offload, and K=0 toggling, checked by exact acceptance
  equivalence under batch-invariant greedy decoding, find speculative-state bugs in current vLLM and SGLang releases
  that vLLM's pressure-based 5% suite and GRIEF's output oracle miss?
- The answer is "no" if: batch-invariant greedy runs are not acceptance-reproducible even without injected events
  (oracle unusable), or a week of injection across the listed methods finds no divergence beyond those already
  reported.
- Hypothesis: exact acceptance divergence appears after injected events on at least one method in each engine,
  mostly without token divergence.
- Cheapest decisive test: on the Air, greedy dflash or draft_model on vllm-metal; run each prompt alone twice and
  check acceptance is identical; then force preemption at step n via a scheduler subclass (E02 has one) and compare.
- Biggest novelty threat: GRIEF adding a spec adapter [L249]; vLLM CI tightening [L254].
- Scale risk: none for correctness; PD and multi-node paths need 2 GPUs.
- Compute: local screening on the Air (greedy, dflash/draft_model; vllm-metal has no batch-invariant mode, so
  the reproducibility check comes first). Ideal: 1-2 CUDA GPUs with batch-invariant vLLM and SGLang deterministic
  mode. Lower: Colab single GPU (no PD). Estimate: harness about a week; runs a few GPU-hours. Access: Colab
  confirmed.

---

# Pass 2 (2026-10-20)

Panel (`interrogate-reviewers`, T3): claudeAgent claude-fable-5-1, codex gpt-6.1-sol, codex gpt-6-astra (provider
default effort), claudeAgent claude-opus-5-5 (medium). All four had web access and returned. Reviews in `pass2/`.
The lead judge opened every new source they cited (ledger L258-L297, `frontier/j-pass2.tsv`,
`frontier/k-pass2-fable.tsv`) and corrected claims the pages do not support (flagged in `does_not_prove`).

## C3' synthesis: KILL

Rejections, merged, most-raised first:

1. **The simple baseline already ships** (Fable, Astra, Opus; Sol named the same baseline from vLLM #53420).
   vllm-metal merged lazy catch-up after K=0 on 2026-09-15: it records where draft KV goes missing and ingests the
   whole gap once before the next K>0 proposal, median wall time 9.94% lower on one small workload [L258]. It is in
   the local checkout (commit 69e0d5a, `draft_model_proposer.py:406`). The vLLM skip PR names full catch-up as the
   general fix [L227]. SGLang's step-0 path keeps draft KV synced without drafting [L282]. Windowing would have to beat
   full lazy catch-up, not only sync and skip. Answer: none. This holds.
2. **Oracle too small** (Sol, Astra, Opus independently; Fable as "full catch-up already reaches the bound").
   The upside is the drafter work removed during K=0 phases only: 4.1-11.3% of step time in one author's runs [L226],
   2-3% on average for running the drafter while speculation is off [L259]. At 10% of time in K=0 that is about
   0.4-1.1% overall (reviewer arithmetic). Answer: none. This holds.
3. **Windowing is already answered at W around 4K** (Fable, Sol). Sink-plus-window draft attention keeps EAGLE and
   MTP acceptance at 32K-256K [L279] [L219]; an untrained 128-token window costs about 10% [L283].
4. **Relevance is narrow** (Opus, Fable). Dynamic speculation is incompatible with data parallelism [L263], ignored on
   the V2 model runner [L262], and SGLang adaptive tiers take positive integers only [L264]. No production K=0
   gap-length data exists [L286] [L287].
5. **Scope** (Sol). Buffered target features exist only for K=0; PD handoff and offload restore do not have them.

What is left is "buffer EAGLE-3's fused hidden states for the gap and rebuild in one forward", a memory-versus-
recompute engineering choice (Fable), and an acceptance curve over W in the low hundreds that prior results already
predict (Fable). Simplicity and Oracle Kill; no narrower question passes the Meaningful-upside gate.
Status: KILL. Kept as a lab measurement (see dossier).

## C7' synthesis: KILL as a thesis candidate

1. **Exact acceptance is not an invariant of correct execution** (all four; Astra marks Correctness as Kills).
   Engines change acceptance on purpose: the EAGLE block drop [L138], dropped in-flight spec tokens (e-code §2), the
   skip PR's accepted 14.6% loss [L227]. vLLM's acceptance check is one-sided by design [L254]. An exact oracle flags
   intended behavior. Answer: none. This holds.
2. **The near-exact preemption test exists or is in flight** (Astra, Fable, Opus). vLLM CI runs fp32 forced-pressure
   preemption with spec and checks tokens exactly [L254]; open PR #52522 adds preemption recovery for batch-invariant
   spec decode with an exact-output test [L260]. GRIEF fuzzes vLLM/SGLang [L249]; bounded crash-injection testing is
   established [L265].
3. **The oracle may not be exact at all** (Astra, Fable, Opus). Batch invariance does not yet cover the spec forward
   pass (an open RFC reports acceptance about 100% to 72% from arithmetic alone) [L261]; SGLang deterministic mode
   was reported non-reproducible with EAGLE-3 [L290]; vllm-metal is not greedy-exact even against native mlx-lm
   (15-21 mismatches per 1,280 DSpark tokens) [L267]; vLLM batch invariance lists no Apple support [L277].
4. **No escaped bug shows the need** (Fable). The 2026 cluster was found without such an oracle.

Smallest core (Fable, Sol): at matched committed prefixes, is greedy acceptance exactly reproducible across forced
preemption and K=0 toggles, and is each divergence a bug or a documented tradeoff? That is a short measurement, not a
thesis question. Status: KILL as a candidate; kept as a lab measurement feeding E02.

## Pass-1 kills under pass 2

All four reviewers left the pass-1 kills standing. Opus corrected one fact: catch-up was built (vllm-metal [L258]),
not only named. TensorRT-LLM's draft_len schedule and acceptance-rate switch [L284] strengthen the C9 kill.

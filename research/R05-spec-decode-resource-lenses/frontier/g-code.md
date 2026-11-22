# G-code: spec decode and shared resources in local vLLM 0.31.0 and vllm-metal

Read-only inspection, 2026-10-20. No code was run or changed. Builds on R04 `frontier/e-code.md` (lookahead slots,
preemption, prefix caching, token budget, static dynamic-K table, Metal method list); those facts are cited, not repeated.

- vllm-core: `/Users/krishna/dev/vllm-lab/vllm-core` at `db9527a46`
- vllm-metal: `/Users/krishna/dev/vllm-lab/vllm-metal` at `572322e`

Paths are relative to those roots. "Reading" means inferred from code, not tested.

## 1. Disaggregation (PD, KV connectors, drafter placement)

- **The drafter runs on both P and D, in the same worker as the target.** NIXL folds the EAGLE/MTP spec config (method, draft model, revision, `parallel_drafting`, draft KV dtype, aux hidden-state layer ids) into the P/D compatibility hash, so P and D must run the same drafter (`vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py:99-140`, `:196`; version note `:44`). Draft layers sit in the target's KV groups, so the prefiller's draft-layer KV moves with the target KV. NIXL handles mixed GQA target + MLA EAGLE-3 draft regions (`nixl/base_worker.py:911-914`).
- **Hidden states are not transferred.** When a remote load finishes and covers the whole prompt, D rewinds `num_computed_tokens` to `num_tokens - 1` (`vllm/v1/core/sched/scheduler.py:3095-3096`). D then recomputes the last prompt token through target and drafter, which produces the hidden state for the first draft. The only hidden-state connector is `ExampleHiddenStatesConnector`, used with the `extract_hidden_states` method to dump states to storage (`kv_connector/v1/example_hidden_states_connector.py:99-104`, `:192-198`). It is not a P-to-D path.
- **Multi-module MTP backoff.** P stops `num_prefill_lookahead_tokens - 1` (= K-1) tokens short of the prompt end, and D recomputes them. Otherwise unverified drafts would be baked into the MTP layers' KV, and D would never rebuild them, "because the update is sized by the rejection count, which is zero for the first decode" (`nixl/base_scheduler.py:385-430`; `vllm/config/vllm.py:656-678`).
- **Lookahead and async loads.** During an async KV load, lookahead is set to 0 (`scheduler.py:1174-1180`; also in R04). The new fact: admission still reserves `_spec_decode_step_blocks() = cdiv(1 + num_spec_tokens + num_lookahead_tokens, block_size)` on top of the full-sequence reservations of the other in-flight loads (`scheduler.py:1194-1204`, `:3003-3036`). For EAGLE this counts K twice (1+2K slots).
- **P wastes a draft (reading).** I found no P-side check (no `do_remote_decode` or `kv_transfer_params` in `vllm/v1/worker/` or `vllm/v1/spec_decode/`). So P's drafter runs over the prompt and proposes K drafts for a request that ends after one token.
- **Other connectors.** The offloading connector never stores the trailing chunk of EAGLE/MTP groups while the request decodes, because a rejection can rewrite it. If no group is annotated as a drafter group, it treats all groups as non-draft, "fail toward serving" (`kv_connector/v1/offloading/scheduler.py:102-105`, `:230-256`, `:455-470`, `:1518-1522`). LMCache counts the extra layers only for `deepseek_mtp` and EAGLE (`lmcache_integration/vllm_v1_adapter.py:405-425`). Mooncake store refers to "speculative-state" blocks that relocate (`mooncake/store/coordinator.py:239-241`).
- **The drafter cannot live on another device or process.** The draft TP size must be 1 or the target's TP (`vllm/config/speculative.py:408-410`, `:1705-1737`). `draft_model` requires it to equal the target's (`vllm/v1/spec_decode/draft_model.py:72-78`). All proposers are built inside `GPUModelRunner` (`vllm/v1/worker/gpu_model_runner.py:617-670`). `custom_class` is the only escape hatch: an in-process class with a `propose(sampled, num_tokens_no_spec, token_ids_cpu)` method (`vllm/v1/spec_decode/custom_class_proposer.py:12-60`; call site `gpu_model_runner.py:4987-4994`). It could proxy to a remote drafter, but it gets no hidden states and no KV.

## 2. Reservations

Per scheduled decode request with static K:

| Resource | Amount | Cite |
|---|---|---|
| Token budget | `1+K` (verify query) | `scheduler.py:674-683`, `:825` |
| Input budget | `1+K + draft_slots` (0 / 1 / K-1 / K by method) | R04 §4 |
| KV slots past `num_computed` | `1+K` new + lookahead (K for EAGLE/MTP/draft_model, K+1 for DFlash, 0 for ngram/suffix) | `vllm/config/vllm.py:629-653`; `kv_cache_manager.py:539-541` |

- The KV figure gives an EAGLE request up to 2K+1 slots past committed tokens every step, capped at `max_model_len` (`kv_cache_manager.py:539-541`).
- **CUDA-graph padding.** The uniform-decode graph is keyed to `1+K` per request (`vllm/config/vllm.py:680-697`; `vllm/v1/cudagraph_dispatcher.py:36`, `:138-143`; `gpu_model_runner.py:3853-3869`). A batch is "uniform" only if every request has exactly `1+K` tokens. A newly admitted decode request is padded to `1+K` with `-1` drafts, or not scheduled at all (`scheduler.py:1086-1108`; R04).
- **Reservations do not shrink with acceptance.** Lookahead uses the fixed `self.num_lookahead_tokens` (`scheduler.py:747-751`). The async placeholder count follows the dynamic-K value `num_spec_tokens_to_schedule`, which is keyed on batch size, not acceptance (`async_scheduler.py:28-31`). With dynamic K on, full CUDA graphs are forced to PIECEWISE on the V1 runner (`vllm/config/vllm.py:1064-1081`).
- **When lookahead blocks cannot be allocated, the scheduler preempts.** It does not retry with fewer drafts. `allocate_slots(..., num_lookahead_tokens=self.num_lookahead_tokens)` is all-or-nothing (`kv_cache_manager.py:534-536`). On `None`, the loop preempts the lowest-priority or last running request until the allocation fits or the request preempts itself (`scheduler.py:745-821`).
  - A preempted request that was already scheduled returns its tokens plus `draft_slots` to the budgets (`scheduler.py:782-799`).
  - Only a pending connector block-free breaks the loop without preempting (`:757-761`).
  - Spec tokens are trimmed only by the token budget (`scheduler.py:829-845`), never by KV pressure.

## 3. Speculative execution (async scheduling)

- **Supported methods.** Async scheduling allows EAGLE/MTP, draft_model, ngram-GPU, DSpark and DFlash. It is auto-disabled for CPU ngram, suffix, Medusa and custom_class, and for `disable_padded_drafter_batch` (`vllm/config/vllm.py:1544-1636`). It also forces cascade attention off (`:1647-1657`).
- **Optimistic placeholders.** After scheduling step N, `AsyncScheduler` adds `1 + cur_num_spec_tokens` to `num_output_placeholders` and sets `spec_token_ids = [-1] * num_spec_tokens_to_schedule` (`vllm/v1/core/sched/async_scheduler.py:25-50`). So step N+1 is scheduled before step N's acceptance is known, as if all K drafts were accepted.
- **Rollback on the scheduler side.** When step N's output arrives, `num_rejected` is subtracted from both `num_computed_tokens` and placeholders. Stale counts from before a preemption are skipped (`scheduler.py:2051-2069`; `async_scheduler.py:59-78`).
- **Rollback on the worker side.** `optimistic_seq_lens = num_computed + scheduled` (`gpu_model_runner.py:2068-2077`). Drafts and sampled tokens are scattered into `input_ids` on the GPU, with no host round-trip (`gpu_model_runner.py:1767-1900`). `update_num_computed_tokens_for_batch_change` sets the corrected count to `prev_computed + valid_sampled_count` on the GPU (`vllm/v1/spec_decode/utils.py:816-845`; call `gpu_model_runner.py:2153`). CPU mirrors are fixed by deferred corrections (`gpu_model_runner.py:1263`, `:1379-1398`, `:1530-1559`).
- **Cost of a mis-speculated step (reading).** Each request is still charged `1+K` budget and slots in step N+1, whatever was accepted in N. The correction moves positions; it does not shrink work.
- **No overlap of drafting with verification.** The GPU drafter runs right after sampling, on the same stream, inside `sample_tokens` (`gpu_model_runner.py:4588-4636`). CPU drafters run after bookkeeping (`:4669-4673`, `:4710-4729`). The only overlap is async scheduling's CPU/GPU overlap and a separate copy stream for draft ids (`:4843-4878`). `parallel_drafting` drafts all K tokens in one pass (`vllm/config/speculative.py:479-483`); it is not concurrent with verification.

## 4. Work stealing and cross-request sharing

- **Ngram is per-request only.** Each request matches its own `token_ids_cpu[idx, :num_tokens]` (`vllm/v1/spec_decode/ngram_proposer.py:181-206`). The GPU variant is batched but row-local (`ngram_proposer_gpu.py:161-209`).
- **Suffix decoding is the only cross-request draft source.** One `SuffixDecodingCache` per worker holds per-prompt trees plus a global tree of up to `suffix_decoding_max_cached_requests=10000` past responses, evicted FIFO; 0 disables the global tree (`vllm/v1/spec_decode/suffix_decoding.py:27-32`; `vllm/config/speculative.py:506-524`).
  - It is per-process: separate DP ranks or replicas keep separate global trees.
  - There is no namespace by tenant, LoRA or model: any request can draft from any earlier request's output. Verification keeps outputs correct, but acceptance (and so latency) depends on other tenants' history (reading).
- **No work stealing.** No proposer draws drafts from idle ranks or from other requests' drafters. Drafter capacity is fixed per worker.

## 5. Load balancing (DP, MoE)

- **DP ranks sync per forward pass.** The target forward all-reduces token counts and CUDA-graph mode, and pads every rank to the max when graphs or microbatching are on (`vllm/v1/worker/dp_utils.py:122-186`; `gpu_model_runner.py:3989-4010`). Each drafter forward does the same (`vllm/v1/spec_decode/llm_base_proposer.py:1795-1823`).
  - If ranks disagree on whether the drafter fits, the rank that skips runs `drafter.dummy_run(1)` "to prevent hang" (`gpu_model_runner.py:4648-4652`, `:4716-4729`).
  - An idle engine runs a dummy batch (`vllm/v1/engine/core.py:2282-2286`; drafter dummy at `gpu_model_runner.py:6170`).
- **A rank with fewer accepted tokens (reading).** Step shape is set by `1+K` per request, not by acceptance. A low-acceptance rank runs the same padded step and emits fewer tokens per step. Nothing rebalances that.
- **Dynamic K is turned off under DP**, because "data-parallel ranks can select different speculative-token counts, causing DP divergence and deadlocks" (`vllm/config/vllm.py:1083-1100`).
- **The load balancer cannot see acceptance.** The coordinator tracks only `[waiting, running, kv_cache_usage]` per engine (`vllm/v1/engine/coordinator.py:145-147`, `:387-426`). The client score is `max(inflight, waiting+running)` plus a KV-pressure penalty (`vllm/v1/engine/core_client.py:1605-1622`). `SchedulerStats.spec_decoding_stats` exists (`vllm/v1/metrics/stats.py:207`) but is not used for routing.
- **MoE.**
  - Draft tokens under verification route through the target's MoE like any other token. EPLB counts real (non-padding) tokens, so rejected drafts count as expert load (reading; `vllm/distributed/eplb/eplb_state.py:227-231`, `:583-600`).
  - The drafter has its own EPLB model state (`llm_base_proposer.py:352-354`, `:584-588`, `:740-743`; "main and drafter" at `eplb_state.py:594`). Medusa + EPLB is refused (`vllm/v1/spec_decode/medusa.py:66-69`).
  - `index_share_for_mtp_iteration` reuses top-k indices across MTP steps (`llm_base_proposer.py:576-582`, `:1593-1603`). It concerns sparse-attention indexer indices (DeepSeek-V3.2 style), not expert routing (reading).

## 6. Isolation and multi-tenancy

- **LoRA.**
  - The target's LoRA applies during verification: the per-token mapping repeats each request's adapter over its `1+K` sampled positions (`vllm/v1/worker/lora_model_runner_mixin.py:93-97`, `:223`; sizing note `vllm/lora/punica_wrapper/punica_gpu.py:63-67`).
  - The drafter is not LoRA-aware. It is loaded after the target is LoRA-wrapped (`gpu_model_runner.py:5281-5288`), and the V2 path unwraps the target embedding: "The drafter does not use the target's LoRA adapter" (`vllm/v1/worker/gpu/spec_decode/eagle/utils.py:47-49`). The V1 base proposer shares the target embed/lm_head with no LoRA unwrap visible (`llm_base_proposer.py:1399`, `:1427-1501`, `:1508-1555`). Whether a LoRA-wrapped shared layer misbehaves in the drafter forward is untested.
  - No config gate rejects LoRA + spec, and LoRA requests are not excluded from speculation (no hits in `vllm/v1/spec_decode/` or `vllm/config/speculative.py`). Reading: a LoRA request drafts from the base drafter and pays lower acceptance.
- **Priority.** Victim choice is spec-blind (R04). Every running request keeps its full lookahead whatever its priority. No knob trades a low-priority request's K for a high-priority admission (`scheduler.py:745-821`).
- **Per-request control.**
  - There is no per-request K or spec opt-out in `SamplingParams`. The only spec hook is `_validate_spec_decode`, which rejects the request outright if `min_p` or `logit_bias` is set, instead of serving it without speculation (`vllm/sampling_params.py:1103-1115`).
  - Per-request acceptance is recorded only for metrics (`vllm/v1/request.py:220-223`; `--per-request-spec-decode-metrics`, `vllm/config/vllm.py:1663-1667`).
  - Per-request draft counts vary only through proposer behaviour: ngram/suffix misses, grammar-invalid drafts padded with `-1` (`scheduler.py:2512-2547`), and prefill chunks (`:2501-2505`).
- **One request can stall others.**
  - (a) `_input_fits_in_drafter` checks the batch-wide `max_seq_len + K` against the drafter's max length. One long request turns off drafting for the whole batch. Everyone then gets K zero drafts that are still scheduled and verified next step (`gpu_model_runner.py:4514-4527`, `:4680-4691`, `:4843-4876`). This bites only when the drafter's `max_model_len` < the target's, or near the limit (`:686-691`).
  - (b) The uniform-decode check needs every request at exactly `1+K`. One prefill chunk or one short-draft (ngram) request drops the step from the full CUDA graph (`gpu_model_runner.py:3853-3869`).
  - (c) New decode requests are padded to `1+K` with `-1` drafts, or held back if the padded size does not fit (`scheduler.py:1086-1108`).

## 7. vllm-metal differences

1. **Disaggregation.** The runner has the KV-connector mixin (`vllm_metal/v1/model_runner.py:24-53`), and the only Metal-specific connector is `MetalOffloadingConnector` (`vllm_metal/v1/cache_policy.py:230-233`). No NIXL/PD path was found. Offload refuses DFlash/DSpark (R04). The drafter is in-process. `draft_model` gets its own paged KV backend with the scheduler's `num_blocks`, sized through synthetic `draft_layers.*` specs in the shared scheduler pool (`vllm_metal/v1/draft_model_proposer.py:277-310`; `cache_policy.py:574-595`, `:731-742`). Block drafters need single-device Qwen3, no LoRA, TP=PP=1 (`vllm_metal/v1/block_draft_proposer.py:70-86`).
2. **Reservations.** The upstream scheduler is used, so reservations are the same. Metal strips the upstream `-1` padding for newly admitted requests and runs them with no drafts (`vllm_metal/v1/spec_decode.py:78-110`), but the scheduler still charged them `1+K` budget and slots. Verification is ragged per request (`spec_decode.py:189-219`), with no graph padding.
3. **Speculative execution.** Async scheduling is forced off with spec (R04). Metal's own one-step-ahead decode pipeline is gated off whenever spec is configured or spec tokens are present (`vllm_metal/v1/decode_pipeline.py:17-18`, `:64`, `:79`, `:100`, `:116`). So a spec step on Metal is fully serial: build, eval, sample, draft.
4. **Sharing.** Ngram wraps the upstream per-request kernel (`vllm_metal/v1/ngram_proposer.py:1-31`). Suffix is unsupported (R04).
5. **Load balancing.** DP + spec is refused (`vllm_metal/platform.py:782-791`), and so is PP + spec (`:832-836`). No MoE/EPLB spec path.
6. **Isolation.** Metal has the only per-request spec throttle found: after `_MAX_CONSECUTIVE_MISSES=8`, ngram stops drafting a request for `_COOLDOWN_STEPS=8` steps. A miss is scored against what the target actually accepted (`ngram_proposer.py:17-27`, `:60-69`, `:165`, `:185`, `:218`). There is no LoRA gate for ngram, draft_model or Gemma4 MTP (only block drafters refuse LoRA). LoRA routing gives each decode request **1** token (`model_runner.py:740-750`, `:3052-3054`), while spec verification rows carry `1+K` (`spec_decode.py:208-219`). Reading: LoRA + spec on Metal may misalign the adapter mapping. Needs a test.

## Gaps these suggest

- **L-06 disaggregation:** D recomputes the last prompt token through target and drafter, and P drafts K tokens it throws away (§1). Nobody measures what drafter placement costs in PD. Nor is there a path that runs the drafter only on D, or only on a separate cheap device; `custom_class` is the only seam and it lacks hidden states.
- **L-07 reservations:** each EAGLE request holds up to 2K+1 slots past committed tokens, and async-load admission counts K twice (§1, §2). KV pressure triggers preemption, never "speculate less". An acceptance-sized reservation, or a "drop K before preempting" policy, is untested ground. Check first that it does not reduce to R04's adaptive-K kill.
- **L-08 speculative execution:** async scheduling charges the full `1+K` per request every step regardless of acceptance (§3). The waste from optimistic placeholders under low acceptance is unmeasured. Metal has no async path at all with spec, which leaves the CPU/GPU overlap question open on unified memory.
- **L-09 work stealing:** the suffix global tree is the only cross-request draft source, and it is per-process and tenant-blind (§4). Cross-replica or cross-rank draft sharing, and the leakage/acceptance effects of a shared tree, are not handled in code.
- **L-10 load balancing:** DP routing ignores acceptance, and dynamic K is disabled under DP for deadlock reasons (§5). An acceptance-aware DP router, or a rank-agreed K, is a concrete gap. Testing it needs multi-GPU, which Krishna does not have yet.
- **L-10/L-06 MoE:** rejected draft tokens count as EPLB expert load (§5, reading). Whether that skews expert placement at low acceptance is unstudied.
- **L-11 isolation:** there is no per-request spec opt-out or K in core (§6). Incompatible sampling params cause a rejection rather than a fallback. One long request can turn off drafting batch-wide. The drafter ignores LoRA. Metal's ngram miss-cooldown is the only per-request throttle, and Metal's LoRA + spec routing may be wrong (§7.6). That last one is a cheap local test on the MacBook.

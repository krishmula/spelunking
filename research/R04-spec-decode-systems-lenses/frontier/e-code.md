# E-code: spec decode state in local vLLM 0.31.0 and vllm-metal

Read-only inspection, 2026-10-20. No code was run or changed.

- vllm-core: `/Users/krishna/dev/vllm-lab/vllm-core` at `db9527a46`
- vllm-metal: `/Users/krishna/dev/vllm-lab/vllm-metal` at `572322e`

Paths below are relative to those roots. "Reading" means inferred from code, not tested.

## 1. Lookahead (spec) slots in KV: allocation and release on rejection

- The number of reserved slots is `num_lookahead_tokens` (`vllm/config/vllm.py:630-653`): K for EAGLE/MTP/draft_model, K+1 for DFlash, 0 for ngram/suffix (they have no draft KV).
- Running requests: `allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)` (`vllm/v1/core/sched/scheduler.py:747-751`). New/resumed requests do the same (`scheduler.py:1175-1214`); lookahead is set to 0 when an async KV-connector load is pending (`scheduler.py:1177-1180`).
- Inside the manager, blocks are sized for `computed + new + lookahead` (`vllm/v1/core/kv_cache_manager.py:539-541`, layout diagram at `:419-438`). `new` already includes unverified draft tokens.
- Only verified tokens are hashed into the prefix cache: `num_tokens_to_cache = min(total_computed + num_new, request.num_tokens)` (`kv_cache_manager.py:599-608`).
- Rejection does not free any block. `update_from_output` subtracts `num_rejected` from `num_computed_tokens` and `num_output_placeholders` (`scheduler.py:2054-2069`). The next step writes over the rejected positions in blocks the request still holds. Sliding-window block release uses the processed-token count, so a rollback cannot free a block that is still needed (`kv_cache_manager.py:543-556`).
- On the worker, the drafter trims rejected tokens from its query and sequence lengths (`vllm/v1/spec_decode/llm_base_proposer.py:1157-1200`). Under async scheduling, an optimistic count is fixed up later (`vllm/v1/worker/gpu_model_runner.py:1530-1559`).
- Consequence: each running request always holds blocks for max K lookahead, even when dynamic K or a low acceptance rate means fewer drafts are used. Lookahead is not sized from the dynamic-K lookup (see 5).

## 2. Preemption and resume

`_preempt_request` (`scheduler.py:1538-1580`):

- Frees all of the request's blocks (`:1555`). Draft KV for EAGLE, MTP, and draft_model lives in the same block tables (the draft layers must share one KV cache group, `llm_base_proposer.py:1687-1707`), so draft KV is freed along with target KV.
- Sets `num_computed_tokens = 0` and drops the spec tokens (`:1560-1562`).
- Marks in-flight async output as stale. It is still delivered, "dropping them would perturb spec-decode acceptance" (`:1563-1574`). Stale rejection counts are not applied after the rollback (`scheduler.py:2061-2069`).
- Prepends the request to the waiting queue (`:1578-1579`).
- The victim is `running[-1]`, or the max (priority, arrival) under PRIORITY (`scheduler.py:768-775`). When a request that was already scheduled this step is preempted, its tokens and `draft_slots` go back to the budgets (`scheduler.py:777-799`). Victim choice ignores draft state, acceptance rate, and spec length.

Worker side:

- Unscheduled and preempted requests leave the persistent batch, but `req_state` is kept (`gpu_model_runner.py:1232-1253`). On resume, the block IDs are replaced (`:1438-1446`) and the request is re-added (`:1448-1462`).
- EAGLE hidden states have no per-request store. `self.hidden_states` in the proposer is a per-step buffer (`llm_base_proposer.py:212-214`). It is filled from the current forward pass's `hidden_states[:num_scheduled_tokens]` or `[token_indices]` (`gpu_model_runner.py:5163-5205`). After a resume, draft KV and hidden states are rebuilt by re-running prefill through target and drafter. Any of the request's own blocks that were hashed and survived in the free pool can be hit again, minus the EAGLE last-block drop (see 3).
- Suffix decoding (reading, not tested): any request not in the input batch is `stop_request`ed (`vllm/v1/spec_decode/suffix_decoding.py:92-96`). On return, its cached response is evicted, and the per-request tree is rebuilt from the prompt only (`:64-71`). Earlier output tokens then appear to be missing from the local tree. This happens on preemption and also when a running request is skipped for one step.
- vllm-metal clears drafter state for preempted request IDs through `release_requests` (`vllm_metal/v1/model_runner.py:2933-2940`; contract at `vllm_metal/v1/proposer.py:106-112`).

## 3. Prefix caching with draft KV and EAGLE

- Draft layers sit in the KV cache groups and share block IDs, so a prefix hit covers draft KV too. The drafter then runs only on the scheduled (uncached) tokens (`gpu_model_runner.py:5160-5168`). In short, a prefix hit skips draft prefill for the hit part.
- Exception: EAGLE/MTP groups (`is_eagle_group`, `vllm/v1/core/kv_cache_utils.py:2206-2226`) drop the last matched block on a hit (`vllm/v1/core/single_type_kv_cache_manager.py:830-835`; sliding window `:1028`, `:1056`). As a result, one block (or one hash unit) is recomputed. The reason: during chunked prefill, the draft KV slot at a chunk end holds the next prefill token combined with the hidden state, and that token may differ for another request (`vllm/v1/core/kv_cache_coordinator.py:112-121`).
- For EAGLE groups, caching extends one block past the aligned boundary so that the dropped block can still be matched (`kv_cache_coordinator.py:813-830`).
- Multi-module MTP needs `scheduler_block_size >= K` (`kv_cache_coordinator.py:122-133`) and keeps `K-1` trailing tokens re-prefillable (`vllm/config/vllm.py:657-675`).
- The drop can be disabled (`disable_eagle_block_drop`). The scheduler warns that this "may affect speculative-token acceptance rates" (`scheduler.py:300-307`; `vllm/config/speculative.py:1944-1946`). So prefix sharing knowingly trades exact draft state for reuse.

## 4. Token budget and spec tokens

- Spec tokens are counted as ordinary tokens. `num_new_tokens = num_tokens_with_spec + placeholders - num_computed_tokens` is capped by `token_budget` (`scheduler.py:674-683`) and deducted at `:825`. The design note is at `:565-573`.
- A second budget, `input_budget` (`max_num_batched_tokens`), also deducts `draft_slots = max_num_new_slots_for_drafting` per scheduled request (`scheduler.py:583-585`, `:631-632`, `:826`, `:1073`). The slot counts are: 0 for EAGLE3/MTP/ngram, 1 for draft_model, K-1 for P-EAGLE/DSpark, K for DFlash/PARD (`vllm/config/speculative.py:1882-1920`). Config checks are at `vllm/config/vllm.py:2350-2386`.
- Newly admitted decode requests are padded to `1+K` to keep full CUDA graphs. A request that does not fit padded is not scheduled at all (`scheduler.py:1086-1108`). This padding is off when dynamic SD is on (`:1090`).
- Spec tokens beyond the budget are truncated to whatever fits (`scheduler.py:829-845`). Partial verification is possible.

## 5. `vllm/v1/spec_decode/dynamic`

- It holds only `utils.py`, a static batch-size to K lookup. `num_speculative_tokens_per_batch_size=[(start, end, K), ...]` is validated (`dynamic/utils.py:7-74`) and expanded into a dense array clamped to the configured K (`:77-148`). Config is at `vllm/config/speculative.py:491-497`.
- The scheduler builds the array at init (`scheduler.py:292-299`). At the end of `schedule()`, it sets `num_spec_tokens_to_schedule = lookup[len(num_scheduled_tokens)]` (`scheduler.py:1445-1450`). The worker passes that K to the drafter for the next proposal (`gpu_model_runner.py:4972`, `:5064`).
- It does not adapt to acceptance rate, request, or workload. It is a hand-set table keyed only on batch size. KV lookahead is still reserved for max K (see 1).

## 6. vllm-metal methods on Apple Silicon

- Supported: Gemma4 MTP, `draft_model`, `ngram`, `dflash`, `dspark`. Anything else raises `NotImplementedError` (`vllm_metal/v1/model_runner.py:1237-1274`). EAGLE/EAGLE-3, Medusa, and suffix decoding are not supported.
- Verification limits:
  - Greedy only: temperature 0, no top-k/p, no penalties, no logprobs (`vllm_metal/v1/spec_decode.py:405-425`).
  - No hybrid/SSM models (`:146-149`).
  - No invalid-draft sentinels, so no structured-output draft filtering (`:159-176`).
- Platform limits (`vllm_metal/platform.py`):
  - Async scheduling is forced off for spec decode (`:627-640`).
  - No heterogeneous vocab (`:599-607`).
  - `long_prefill_token_threshold` must be at least 1+K (`:609-625`).
  - No DP or PP with spec decode (`:789`, `:835`).
- KV offload is refused for DFlash/DSpark (`vllm_metal/v1/kv_offload/config.py:153-164`). Draft-model KV supports only full-attention drafts (`vllm_metal/v1/draft_model_proposer.py:904-950`).

## Gaps these suggest (for R04)

- Preemption throws away all draft state and picks victims blind to spec state. The resume cost (draft prefill, EAGLE block drop) is not measured anywhere in the code or metrics.
- Lookahead KV is reserved for max K whatever the dynamic K or acceptance rate is. This is a memory-pressure lever that nothing sizes adaptively.
- The suffix-decoding per-request tree may lose output history on any unscheduled step. This needs a test before anyone relies on it.

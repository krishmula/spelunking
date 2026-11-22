# G-code: spec decode under operational lenses in local vLLM 0.31.0 and vllm-metal

Read-only inspection, 2026-10-20. No code was run or changed. Builds on R04 `frontier/e-code.md` (lookahead slots,
rejection rollback, preemption, prefix caching, token budget, static dynamic-K table, Metal method list) and R05
`frontier/g-code.md` (PD, reservations, async placeholders, DP sync, LoRA and the drafter, draft TP rule,
`custom_class`). Those facts are cited as "R04 §n" / "R05 §n", not repeated.

- vllm-core: `/Users/krishna/dev/vllm-lab/vllm-core` at `db9527a46`
- vllm-metal: `/Users/krishna/dev/vllm-lab/vllm-metal` at `572322e`

Paths are relative to those roots. "Reading" means inferred from code, not tested.

## 1. Consistency and ownership transfer

### Weight updates

- **Target and drafter are updated in separate sessions.** `start_weight_update` targets the target model.
  `start_draft_weight_update` retargets the transfer engine at `get_draft_model()` for one session
  (`vllm/v1/worker/gpu_worker.py:1443-1484`, target swap at `:1068-1085`). Each `update_weights` chunk loads into
  whichever model the session picked (`:1486-1521`). Exposed through `AsyncLLM` (`vllm/v1/engine/async_llm.py:1267-1286`),
  `LLM` (`vllm/entrypoints/llm.py:891-893`) and the dev RLHF router (`vllm/entrypoints/serve/dev/rlhf/api_router.py:180-182`).
- **Nothing checks that the drafter matches the target.** There is one weight version string for the engine
  (`vllm/v1/engine/core.py:137`, `:1039-1044`); `finish_weight_update(weight_version)` sets it after either session
  (`async_llm.py:1281-1285`). No draft version, no "target updated, drafter stale" warning.
- **Only some engines can update the drafter.** `supports_draft_weight_update = True` in the base engine
  (`vllm/distributed/weight_transfer/base.py:355`), `False` for sharded-RDT, sparse-NCCL and M2N
  (`sharded_rdt_engine.py:296`, `sparse_nccl_engine.py:125`, `m2n_engine.py:175`). With those, the drafter can never
  follow an RL target; it is reported to clients as a capability flag only (`vllm/v1/engine/core.py:1747-1748`).
- **Partial coupling through shared modules.** EAGLE/MTP drafters alias the target's `embed_tokens` and `lm_head`
  when the checkpoint lacks them or when they are byte-identical at load time (`vllm/v1/spec_decode/llm_base_proposer.py:1416-1506`,
  `:1508-1575`). After a target-only update, those aliased modules move with the target while the drafter's own
  layers stay old. A drafter with "distinct" weights at load keeps its own copies and does not move at all.
- **`reload_weights` reloads the target only.** It walks `self.get_model()` and never touches the drafter
  (`vllm/v1/worker/gpu_model_runner.py:5492-5590`).
- **Updates do not require a pause.** `_start_weight_update` checks only that no other session is active
  (`gpu_worker.py:1460-1484`). Pausing is up to the caller via `pause_generation(mode, clear_cache)`; mode `keep`
  keeps in-flight requests and their KV (`vllm/v1/engine/core.py:880-907`). Reading: with `keep` and no cache clear,
  target KV, draft KV and prefix-cache blocks written by old weights stay live after an update.
- **Sleep/wake.** Drafter weights are allocated inside the same `weights` memory-pool tag as the target
  (`gpu_worker.py:538-545`, drafter loaded inside `GPUModelRunner.load_model`, `gpu_model_runner.py:5285-5288`), so
  level 1/2 sleep drops both. Level 2 saves draft buffers (`gpu_worker.py:281-285`, restore `:325-331`), but
  `sleep_preserve_parameter_names` is applied to the target only (`:247-270`, `:317`). After a level-2 sleep
  the RL flow refills weights via `update_weights` or `reload_weights`; `reload_weights` skips the drafter, so the
  drafter is only correct if the caller also runs a draft session (reading). No test combines sleep with spec
  decode (`tests/v1/worker/test_sleep_mode_backend.py:61` only stubs the draft buffer dict).
- **LoRA load** does not touch the drafter (R05 §6). `finish_weight_update` calls `reset_lora_state()` only for
  target sessions (`gpu_worker.py:1538-1540`).

### Greedy spec output vs non-spec output

- **Nothing enforces equality at runtime.** Equality rests on the rejection sampler and on the verify forward
  giving the same argmax as a 1-token decode.
- **The e2e tests allow mismatches.** `assert_request_outputs_match` counts exact-text matches against a threshold
  (`tests/v1/e2e/spec_decode/utils.py:125-174`). Thresholds: EAGLE 60%+1 (`tests/v1/e2e/spec_decode/eagle/utils.py:127-134`),
  speculators 66% (`speculators/test_speculators.py:104`), MTP 80%+1 (`mtp/_correctness.py:118`), draft-model LoRA
  90%+1 (`draft_model/test_lora.py:129`). Only the LoRA test and MTP parallel-load test turn on
  `VLLM_BATCH_INVARIANT` (`draft_model/test_lora.py:66-67`; `test_mtp_parallel_load.py:122-130`, which itself checks
  a `MIN_MATCH_RATIO`, `:183`, `:263`). Upstream treats spec-vs-non-spec greedy drift as expected numerical noise
  from a `1+K` query vs a 1-token query.
- PP has an acceptance-length test only, not output equality (`tests/v1/e2e/spec_decode/eagle/test_eagle3_pp.py:51-58`).

### Rejected-token KV, structured output, streaming, logprobs

- **Rejected KV slots** stay owned by the request and are overwritten next step; async scheduling corrects counts
  later (R04 §1, R05 §3). Nothing new found.
- **Grammar and spec.** Drafts are cut to the longest grammar-valid prefix and padded back with `-1`
  (`vllm/v1/core/sched/scheduler.py:2512-2547`; `vllm/v1/structured_output/__init__.py:294-312`). The bitmask fill
  advances the grammar through every draft, fills `K+1` masks, then calls `grammar.rollback(state_advancements)`
  (`structured_output/__init__.py:385-434`). If the grammar rejects a draft at this point, the code logs
  "Unexpected" and reuses the last valid mask for the rest of the row (`:404-418`). All of this runs on the engine-core
  thread per step, `O(K)` grammar ops per structured request.
- **Bursts and stop strings.** The scheduler appends accepted tokens one by one and trims at EOS/stop-token/max
  tokens (`scheduler.py:2434-2451`). Stop strings are checked later, in the detokenizer, over the whole burst
  (`vllm/v1/engine/detokenizer.py:96-141`). It truncates `output_text` but still appends every burst token to
  `token_ids` (`:116-125`). The output processor then aborts the engine-core request (`vllm/v1/engine/output_processor.py:720-725`,
  `:757-763`). Reading: under spec, a stop string matched mid-burst leaves `token_ids`, token counts and logprobs
  holding tokens past the stop, which a 1-token step never produces. No test found for this.
- **Logprobs** come from target logits at the verified positions (`vllm/v1/sample/rejection_sampler.py:150-215`),
  so they carry the same `1+K`-query numerics as the tokens. Metal refuses logprobs with spec (R04 §6).

## 2. Backpressure and queueing

- **Token budget.** Spec tokens are ordinary tokens in `token_budget`; draft slots charge `input_budget`; spec
  tokens are trimmed only by budget (R04 §4, R05 §2).
- **No load-based disable in core V1.** There is no `disable_by_batch_size` or queue-length rule anywhere in
  `vllm/config/speculative.py` (fields `:391-609`). The only control is the static batch-size-to-K table
  (R04 §5; `scheduler.py:1445-1450`), off under DP (R05 §5).
- **New: adaptive verification (V2 runner, DSpark only).** `enable_adaptive_verification` (`vllm/config/speculative.py:552-554`)
  builds `AdaptiveVerificationManager` (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:131-182`). Each
  step it scores every (request, position) by the product of stale per-position confidences and picks the draft
  budget that maximises `expected accepted tokens / (draft_cost + verify_cost)` (`:282-350`); the global top-k
  admission is `_assign_draft_token_budget` (`:46-77`). This is the only cost-aware "speculate less under load"
  rule in the tree. Limits:
  - Cost tables are measured **once**, during CUDA-graph capture, with dummy runs at a fixed context of
    `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=8192` (`adaptive_verification.py:188-231`; `vllm/envs.py:1105-1106`;
    call site `vllm/v1/worker/gpu/model_runner.py:1067-1073`). `StepTimingCollector` records only inside `collect()`
    (`vllm/v1/worker/gpu/async_utils.py:48-110`), so the tables never update while serving.
  - Raises at startup if the attention backend cannot trim query lengths or replay varlen graphs (`:450-509`).
    Incompatible with `use_local_argmax_reduction` (`speculative.py:1576-1586`) and routed-experts return
    (AuxOutput connector check, `vllm/config/vllm.py:1182-1188`).
- **Output path.** Each `EngineCoreOutput` carries a burst. `AsyncLLM.output_handler` splits by request count
  (`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`), not token count (`async_llm.py:811-850`). The detokenizer loops per token
  (`detokenizer.py:117-121`). `stream_interval` batches by token count (`output_processor.py:314-334`).
- **Latency metric semantics change.** `inter_token_latencies_iter` gets one sample per output, i.e. per burst, not
  per token (`vllm/v1/metrics/stats.py:466-505`). Under spec this histogram is closer to step time than to TPOT;
  only `mean_time_per_output_token` (`:557-578`) divides by tokens. Spec metrics are counts of drafts, draft
  tokens, accepted tokens and per-position accepts (`vllm/v1/spec_decode/metrics.py:229-260`); no drafter-time metric.

## 3. Fault tolerance

- **Drafter exceptions are not caught.** The drafter runs inside `sample_tokens` / `propose_draft_token_ids` with no
  try/except (`gpu_model_runner.py:4588-4729`, `:4960-5233`). An exception goes up through
  `log_error_detail` (dump, re-raise, `vllm/v1/engine/core.py:538-552`) and kills the engine core
  (`ENGINE_CORE_DEAD`, `:1088`, `:1694-1700`). There is no fallback to non-spec decoding.
- **Drafter NaNs are invisible.** NaN counting covers target logits only, and is opt-in
  (`VLLM_COMPUTE_NANS_IN_LOGITS`, `gpu_model_runner.py:3684-3692`; raise with `VLLM_RAISE_ON_LOGIT_NANS`, `:364-365`,
  `:5703-5704`). The code itself warns that an MTP layer without a copied `lm_head` "produces NaN logits"
  (`llm_base_proposer.py:1558-1573`). Reading: a NaN drafter under greedy argmax yields junk drafts that all get
  rejected, so output stays correct while acceptance silently drops to zero; only acceptance metrics show it.
- **Fault-tolerance mode (DP+EP only).** `enable_fault_tolerance` needs one API server and an FT-capable all2all
  (`vllm/config/parallel.py:436-505`; `vllm/v1/worker/sentinel/gpu_worker_sentinel.py:25-45`). On any busy-loop
  exception, `EngineCoreSentinel.on_fault` aborts **all** requests and marks the engine unhealthy
  (`vllm/v1/fault_tolerance/engine_core_sentinel.py:80-106`); `retry` re-inits the DP group (`:121-167`). Requests are
  not retried or migrated (no resubmit path in `vllm/v1/engine` or `vllm/entrypoints`).
- **Worker cleanup ignores drafter state.** `WorkerSentinel._clean_worker_state` clears `execute_model_state` and
  the input batch (`gpu_worker_sentinel.py:68-89`). It does not reset `_draft_token_ids`, `_draft_probs`, the
  suffix cache, NgramProposerGPU buffers or adaptive-verification confidences (reading).
- **EP fault detection** is checked on the target's async output only (`gpu_model_runner.py:325-327`, `:368-374`);
  the drafter's MoE all2all (if any) has no separate check.
- **Health checks** do not touch spec: core `check_health` is a no-op (`gpu_worker.py:1396-1398`);
  `AsyncLLM.check_health` only reports `errored` (`async_llm.py:1064-1067`). Metal runs `mx.eval` on a scalar
  (`vllm_metal/v1/worker.py:400-406`).

## 4. Topology changes

- **Draft TP**: 1 or target TP; `draft_model` must equal target TP (R05 §1). The draft `ParallelConfig` forces
  `pipeline_parallel_size=1` and keeps EP only for MoE drafts (`vllm/config/speculative.py:1755-1781`). CPU ngram runs on
  every TP rank, capped at 1 numba thread (`vllm/v1/spec_decode/ngram_proposer.py:34-50`).
- **PP + spec is supported, drafter pinned to the last stage.** "We put the entire draft model on the last PP rank.
  This is not ideal if there are many layers in the draft model" (`gpu_model_runner.py:600-604`). Under PP the
  drafter loads its own embedding instead of sharing (`llm_base_proposer.py:1421`, `:1502-1506`). The V1 runner
  turns async scheduling off with PP (`vllm/config/vllm.py:1624-1632`); last-stage sampled ids are broadcast to
  other stages (`gpu_model_runner.py:4787-4830`). Reading: every decode step, all earlier stages wait for the
  last stage's verify plus draft.
- **DP/EP + spec**: DP sync per drafter forward, dynamic K off under DP, separate drafter EPLB state (R05 §5).
- **Elastic EP**:
  - MoE drafter + EPLB + elastic EP is refused: "Elastic EP is not supported with drafter model"
    (`gpu_model_runner.py:5291-5300`).
  - Elastic EP is listed unsupported on the V2 runner (`vllm/config/vllm.py:3097-3098`).
  - For a dense drafter there is no gate. The reconfigure path handles only the target: weight transfer, MoE
    quant swaps and `_release_cuda_graphs` all use `model_runner.model` / `get_model()`
    (`vllm/distributed/elastic_ep/elastic_execute.py:339`, `:395-437`). Re-capture goes through
    `compile_or_warm_up_model` (`:676-715`). Nothing in `vllm/distributed/elastic_ep/` mentions draft or spec.
    Reading: drafter CUDA graphs and DP padding captured for the old DP size are not explicitly released.
- **Metal**: no DP or PP with spec (R05 §7).

## 5. Hardware-aware placement

- **Same device, always.** All proposers are built in the runner (R05 §1). No `device` field exists in
  `SpeculativeConfig` (`speculative.py:391-609`). Per-drafter knobs are kernel-level: `quantization`, `moe_backend`,
  `attention_backend`, `kv_cache_dtype`, `enforce_eager`, `draft_load_config` (`:391-526`), applied through
  `apply_draft_overrides` (`:1783-1793`).
- **CUDA graphs for the drafter (V1).** PIECEWISE only: `initialize_cudagraph_keys` maps FULL to PIECEWISE
  (`llm_base_proposer.py:421-436`); drafter metadata is unpadded for that reason (`gpu_model_runner.py:2603-2611`).
  During capture, the drafter uses graphs only when the target mode is PIECEWISE, and LoRA with
  `cudagraph_specialize_lora` disables them (`gpu_model_runner.py:6146-6174`). The V2 runner has its own speculator
  capture (`vllm/v1/worker/gpu/model_runner.py:1064-1066`).
- **Metal.** One process-wide MLX default device (`vllm_metal/platform.py:242-244`; `vllm_metal/v1/worker.py:134-136`).
  MLX streams and the CPU device are not used for drafting. On unified memory the drafter and target share the
  same bandwidth. Unlike core, Metal falls back per request: non-greedy or exhausted requests are not drafted
  (`vllm_metal/v1/spec_decode.py:343-357`, used at `:309`, `:328`, `:388`, `:399`), and the draft model drops to
  target-only for a request past its context limit (`vllm_metal/v1/draft_model_proposer.py:520-538`), where core
  turns drafting off batch-wide (R05 §6).

## 6. Power and capacity

- **No power, energy, clock or thermal knob** in `vllm/v1`, `vllm/config` or `vllm_metal` (grep for power, energy,
  thermal and clock found only unrelated hits). `synthetic_acceptance_rates` / `synthetic_acceptance_length`
  (`speculative.py:530-550`) fix acceptance artificially; useful for controlled experiments, not a control.
- **Drafter memory comes out of the KV pool.** Target and drafter load inside one `DeviceMemoryProfiler`
  (`gpu_model_runner.py:5266-5288`), so drafter weights are in `model_memory_usage` (`gpu_worker.py:615`). The
  profile run includes `drafter.dummy_run` (`gpu_model_runner.py:6132-6174`). KV is then
  `requested_memory - profiled usage - cudagraph estimate` (`gpu_worker.py:672-677`). Draft layers add to the
  per-block KV size because they share KV groups (R04 §1, §3). There is no separate report of drafter bytes.
- **Metal.** KV budget = `max_recommended_working_set_size * gpu_memory_utilization - mx.get_active_memory() - overhead`
  (`vllm_metal/v1/cache_policy.py:1084-1087`, `:1140-1147`, `:1409-1428`). DFlash/DSpark and draft models load before
  profiling "so draft weights count against the same device budget" (`vllm_metal/v1/model_runner.py:704-727`);
  Gemma4 MTP's assistant loads in the model lifecycle (`vllm_metal/v1/model_lifecycle.py:644-650`). Sleep is a no-op
  with a warning (`vllm_metal/v1/worker.py:384-398`), so a Metal server cannot shed drafter memory without restart.
- **Adaptive verification's cost model is a frozen snapshot** of startup step costs (§2). On a throttled or
  power-capped GPU the real draft/verify cost ratio drifts from the table and nothing re-measures it.

## Gaps these suggest

1. **Drafter/target version skew is unchecked (L-12).** Separate draft sessions, one version string, three of four
   transfer engines unable to update the drafter, and aliased embed/lm_head that move with the target while the
   rest of the drafter does not (§1). What acceptance does a stale or half-updated drafter give after RL steps,
   and should the engine detect skew from acceptance and stop speculating? Testable on one GPU with
   `reload_weights` on a fine-tuned target.
2. **Level-2 sleep + `reload_weights` leaves the drafter unrefilled (L-12, reading).** No test covers sleep with
   spec. A cheap Colab check: sleep(2), wake, reload, compare acceptance against a fresh engine.
3. **Spec changes reported tokens around stop strings (L-12, reading).** Mid-burst stop matches keep extra
   `token_ids`/logprobs/usage that non-spec never emits (§1). A local test on Metal or Colab can confirm it.
4. **"Greedy spec equals non-spec" is tested at 60-90% exact match (L-12).** Quantifying where and why outputs
   diverge (verify-query numerics, per method, per backend, with and without batch invariance) is unmeasured
   and runs on one GPU.
5. **No drafter fault isolation (L-14).** A drafter exception kills the engine; a NaN drafter silently zeroes
   acceptance; FT mode aborts every request. The drafter is optional for correctness, so "drop to target-only
   on drafter fault" is an obvious unbuilt degrade path. FT cleanup also skips drafter state.
6. **Only DSpark on the V2 runner has a cost-aware verify budget, and its costs are measured once (L-13, L-17).**
   Tables come from dummy runs at context 8192 at startup and never refresh. Under thermal throttling (fanless
   MacBook) or power caps, the draft/verify cost ratio moves. Metal has no such controller at all.
7. **Burst outputs distort latency metrics (L-13).** `inter_token_latencies_iter` is per burst; any SLO or
   autoscaler keyed on it reads spec as slower per gap but fewer gaps. No drafter-time metric exists.
8. **PP pins the whole drafter to the last stage (L-15, L-16).** Upstream flags it as "not ideal"; V1 also loses
   async scheduling under PP. Needs multi-GPU, so it is a later test.
9. **Elastic EP reconfigures only the target (L-15, reading).** Dense drafters are not gated and their graphs are
   not released explicitly. Needs multi-GPU.
10. **Drafter weights silently shrink KV with no separate accounting (L-17).** On a 16 GB Mac, the KV given up to the
    drafter is directly comparable to the throughput spec buys; no knob trades one for the other at runtime.

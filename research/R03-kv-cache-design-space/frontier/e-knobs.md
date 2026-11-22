# E: KV-side knobs in the actual engine

Code read on 2026-10-12, read-only.

- vLLM core: `/Users/krishna/dev/vllm-lab/vllm-core`, tag v0.31.0 (db9527a). Paths below are relative to `vllm/`.
- vllm-metal: `/Users/krishna/dev/vllm-lab/vllm-metal`, main 572322e (v0.31.0.dev20261010). Paths are relative to `vllm_metal/`.

Swap cost: **flag** = CLI or config only. **one fn** = edit or subclass one function or class. **deep** = touches several layers, a kernel, or an invariant.

## vLLM core (scheduler side, shared by Metal)

| Mechanism | Where (file:line) | Default | Swap cost |
|---|---|---|---|
| KV cache manager | `v1/core/kv_cache_manager.py:131` `KVCacheManager`; picks a coordinator at `v1/core/kv_cache_coordinator.py:1014` | `UnitaryKVCacheCoordinator` for one group (:531), `HybridKVCacheCoordinator` for several (:616), `KVCacheCoordinatorNoPrefixCache` when APC is off (:480) | deep |
| Per-attention-type managers | `v1/core/single_type_kv_cache_manager.py`: FullAttention :744, SlidingWindow :951, ChunkedLocal :1293, Mamba :1449, plus RSWA, CircularBuffer, Sink and HiSparse managers; registry lookup at :2601 | chosen by KVCacheSpec type | one fn (registry `register_kv_cache_spec`) |
| Block pool | `v1/core/block_pool.py`; `get_new_blocks` :669 pops from the free-queue head | one pool per engine | deep |
| Free-block queue | `v1/core/kv_cache_utils.py:246` `FreeKVCacheBlockQueue`, an intrusive doubly linked list with O(1) middle removal (`touch` :755 pulls a hit block out) | starts in block-ID order | one fn (class) |
| Free ordering | `block_pool.py:777` `free_blocks`. Unhashed blocks are prepended (":797 LIFO reuse of non-cached blocks"). Hashed blocks are appended (":800 FIFO reuse of cached blocks for LRU"). A request's blocks are freed in reverse (`single_type_kv_cache_manager.py:589`), so the chain tail is evicted before the head. | LRU, tail-first within a request | one fn |
| Eviction | Implicit: whatever block is at the queue head when `get_new_blocks` runs, `_maybe_evict_cached_block` :732 drops its hash. No policy object or frequency signal. | LRU | one fn (free order) to deep (a real policy needs per-block metadata and a new structure) |
| Watermark | `kv_cache_manager.py:147` | `watermark=0.0` | flag-ish (constructor arg) |
| Prefix-cache hashing | `kv_cache_utils.py:684` `hash_block_tokens` = H((parent_hash, tuple(tokens), extra_keys)), chained; `NONE_HASH` seed :160; algos `utils/hashing.py:84` | `prefix_caching_hash_algo="sha256"` (pickle then SHA-256), `config/cache.py:144`. Options: sha256, sha256_cbor, xxhash, xxhash_cbor | flag (algo); one fn (hash input) |
| Prefix caching on/off | `config/cache.py:142` | `enable_prefix_caching=True` | flag |
| Block size | `config/cache.py:71` `DEFAULT_BLOCK_SIZE=16`; :73 `block_size` resolved per backend | 16 | flag (`--block-size`) |
| Hybrid allocator | `HybridKVCacheCoordinator` `kv_cache_coordinator.py:616`; groups planned in `kv_cache_utils.py:2300` `get_kv_cache_groups`; off switch `config/scheduler.py:183` `disable_hybrid_kv_cache_manager` | on when the platform supports it | flag (off); deep (change) |
| Preemption victim | `v1/core/sched/scheduler.py:765-771` | FCFS: `self.running[-1]` (last admitted). PRIORITY: `max(running, key=(priority, arrival_time))`. Scheduler policy `config/scheduler.py:160` `policy="fcfs"` | one fn |
| Preemption mode | `scheduler.py:1538` `_preempt_request`: frees all blocks, `num_computed_tokens = 0` (:1560), prepends to waiting | recompute only. V1 has no swap. Resume may re-hit its own freed blocks through the prefix cache. | deep (swap); the offload connector is the nearest substitute |
| KV cache sizing (CUDA path) | `kv_cache_utils.py:1771` `num_blocks = available_memory // bytes_per_block`; `config/cache.py:103` `gpu_memory_utilization=0.92`; `kv_cache_memory_bytes` :247; `num_gpu_blocks_override` :136 | 0.92 of device memory | flag |
| KV dtype | `config/cache.py:39` `CacheDType` (fp8 variants, turboquant_*, int4/int8/fp8 per-token-head, nvfp4) | `cache_dtype="auto"` = model dtype | flag (CUDA). **Metal rejects all of them, see below.** |
| CPU offload | `config/cache.py:256` `kv_offloading_size=None`, :262 backend `native`/`lmcache`; `v1/kv_offload/cpu/manager.py:60` `cache_policy="lru"`; policies `lru`, `arc` registered in `cpu/policies/factory.py:87-90`; `cpu/spec.py:135` reads `eviction_policy` from extra_config | off. When on, the offload tier is LRU. | flag (lru/arc); one fn (new policy through the factory) |
| KV connectors | `distributed/kv_transfer/kv_connector/factory.py:150-245`: LMCache(MP), Nixl(Pull/Push), Multi, HiSparse, MoRIIO, Offloading, Mooncake(Store), FlexKV, SimpleCPUOffload, HF3FS, Example, DecodeBench | none | flag |

## vllm-metal (worker side)

| Mechanism | Where (file:line) | Default | Swap cost |
|---|---|---|---|
| KV storage | `attention/caches/kv_cache.py:29` `MetalPagedKVCache`. It binds vLLM's allocation (`from_upstream` :36). Dense path :274: one K and one V `mx.array` per layer, shape `(num_blocks, block_size, kv_heads, head_dim)` | paged MLX arrays in the model dtype. The old contiguous `PrefixCacheManager` path is gone. | deep (layout is shared with the kernels) |
| KV write | `metal/paged_ops.cpp:2512` `reshape_and_cache` (Metal kernel `kernels_v2/reshape_and_cache.metal`), called at `attention/impls/sdpa.py:738` | fused K/V scatter, one dispatch | deep |
| Attention kernel | `metal/paged_ops.cpp:2643` `paged_attention_primitive`, called at `attention/impls/sdpa.py:832,925,995`; kernels in `metal/kernels_v2/pagedattention*.metal` (plain, tiled, NAX for M5). Split-KV partition 512 (`metal/constants.py:4`, threshold 4096), GQA decode plan in `paged_ops.cpp` | native paged kernel for prefill and decode | env flags (`VLLM_METAL_DISABLE_GQA_DECODE`, `VLLM_METAL_DISABLE_NAX`); deep (kernel) |
| Block size | `platform.py:1161` `update_block_size_for_backend`: advertises `MultipleOf(16)`, so the default stays 16. `metal/constants.py:10` `KERNEL_BLOCK_SIZES=(32,16,8)`. `attention/block_tables.py:9` `pick_kernel_block_size` splits larger scheduler blocks into kernel blocks (zero-copy reshape, expanded block table). MLA kernels allow only 16 and 32. | 16; any multiple of 8 works through translation | flag (`--block-size`, multiples of 8). A new native kernel size is deep (template instantiation). |
| KV sizing | `v1/cache_policy.py:1089` `determine_available_memory`; :1140 `base_kv_budget_bytes = metal_limit*fraction - model_memory - overhead`; `metal_limit` = `mx.device_info()["max_recommended_working_set_size"]` (:1420); fraction = `--gpu-memory-utilization` (:1411); minimum 16 blocks (`config.py:10`); startup commit probe `VLLM_METAL_KV_COMMIT_PROBE=1` | 0.92 x the Metal recommended working set, not physical RAM | flag |
| KV dtype | `platform.py:40` only float16 and bfloat16, and only when equal to the model dtype. `platform.py:566-590` raises `NotImplementedError` for fp8, int8, and similar dtypes. | model dtype | flag is blocked |
| KV quantization | TurboQuant through `--additional-config '{"turboquant": true}'`: `config.py:41-43` `k_quant="q8_0"`, `v_quant="q3_0"` (Lloyd-Max); `attention/caches/turboquant.py`, `kernels_v2/turboquant.metal`; spec `TurboQuantAttentionSpec` `v1/cache_policy.py` (publishes packed page size to the scheduler) | off | flag (on/off and bit widths); deep (new scheme) |
| Sliding window / hybrid | `support_hybrid_kv_cache()=True` (`platform.py:1105`); per-layer `sliding_window_per_layer` handed to the kernel (`kv_cache.py`, `attention/impls/sdpa.py:631`); hybrid GDN/Mamba block-size translation (`platform.py:1179-1204`); hybrid APC limited to `mamba_cache_mode="align"`, `all` rejected (`platform.py:650-686`) | upstream managers, Metal kernel masks the window | deep |
| Offload | `v1/kv_offload/connector.py:44` `MetalOffloadingConnector`; `v1/kv_offload/config.py:56` default host pool = 2 x max_model_len requests of KV, carved out of the same KV budget (`cache_policy.py:1180-1181`); `v1/kv_offload/fs_tier.py` disk tier with a size cap | off. When on, it covers uniform full-attention models only. | flag |
| Prefix caching | upstream hashing and block reuse. Copy-on-write through `kv_cache.py:320` `copy_blocks`. Disabled for STT and for some hybrid and speculative-decoding cases (`platform.py:659-686`, `:932`) | on | flag |

## Takeaways for picking a mechanism

1. Upstream GPU eviction is still LRU through queue order, with no policy hook. Many policy PRs exist and none has merged (see the TSV). Swapping policy on our stack is a local edit to `free_blocks` and `get_new_blocks`.
2. Preemption is recompute-only, and the FCFS victim is the last admitted request. Two open PRs change the victim choice (`lcf`, cost-aware).
3. On Metal, `--kv-cache-dtype` is closed. The only quantized KV is TurboQuant (K q8_0, V q3_0 by default).
4. On Metal, block size is a flag. Native kernel sizes are 8, 16 and 32; larger sizes go through block-table translation. The block-size tradeoff on unified memory (hash granularity, kernel efficiency, fragmentation) is cheap to sweep.
5. On Metal, KV capacity comes from the recommended working set times the fraction, and an offload pool shrinks it.

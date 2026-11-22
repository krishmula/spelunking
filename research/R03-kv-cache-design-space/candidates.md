# Candidates: KV cache design space

Topic constraints: [constraints.md](constraints.md). Sources: `ledger.tsv`, cited as `[L<n>]`. Engine knob map:
[frontier/e-knobs.md](frontier/e-knobs.md).

## Frontier in one page

- **Where the defaults came from.** Block size 16 and recompute-over-swap rest on the 2023 PagedAttention sweeps on
  A100-class GPUs over PCIe [L1] [L2] [L61]. Current vLLM docs justify neither 16 nor `gpu_memory_utilization` 0.92 [L69],
  and claim recompute is cheaper in V1 without data [L71] [L72].
- **Eviction policy is crowded and weak.** 14 policies on two production traces barely beat LRU [L30]; the best
  production policy wins 1.5-3.9 points over the best simple one [L29]. vLLM has a pile of unmerged eviction PRs
  [L141] [L143] [L144] [L145] [L147]; SGLang ships five policies [L34] [L163]. Cost-aware eviction is taken [L215].
- **Preemption is under-studied.** V1 preempts the last-admitted request and recomputes it [L73] [L134]. Its blocks go
  back to the prefix-cache free queue, so a resumed request can re-hit its own blocks [L134]. Two open PRs change the
  victim (least-computed-first, cost-aware) with no data [L150] [L151]. No study of preemption on small memory.
- **Compression mostly fails inside real engines.** Gains shrink or reverse in paged engines [L98]; query-aware
  eviction collapses under reuse [L99]; only quantization and model-native sparsity ship [L101] [L119] [L120] [L121].
  On Metal, only TurboQuant exists and has no speed numbers [L127] [L136].
- **Unified memory changes the meaning of offload and swap.** vllm-metal sizes a fixed pool once, from the Metal
  working-set limit; the pool is lazily committed, not wired, and macOS may compress or swap it [L90] [L91] [L139].
  The offload pool comes out of the same budget [L92] [L157]. Engines on Macs exceed their memory budgets [L210].
  The closest on-device pressure work is Android, single-request, no paging or prefix cache [L214].
- **New workloads.** Reasoning models leave thinking KV that the next turn cannot reuse [L181] [L208] [L209]. Agent
  pauses are covered on datacenter GPUs [L169] [L171] [L172]. Hybrid models force huge blocks and exact-match
  checkpoints [L68] [L184] [L185]; checkpoint placement already has a DP solution [L212].

## Pass 0: candidate cards

### C1. Block size on unified memory

- Problem: block size 16 trades kernel efficiency, fragmentation, and prefix-hit granularity, and was chosen on 2023
  A100s [L61]. On Metal, the kernel has 8/16/32 native sizes and splits larger blocks [L137] [L93].
- Why now: Apple paged serving is new [L90]; paged decode reads KV at ~40 GB/s vs ~200 GB/s achievable [L159].
- Existing work: PagedAttention ablation [L61]; vAttention kernel slowdowns by block size [L62]; incidental sweep in
  vToken [L79]; vLLM now decouples hit granularity from block size via `hash_block_size` / `prefix_match_unit` [L36].
- Exact gap: no controlled block-size sweep on unified memory of step time, hit tokens, and fragmentation together.
- Research question: Does a block size other than 16 improve TPOT or throughput for Qwen3-4B on vllm-metal under a
  multi-turn trace, compared with 16, at equal KV memory and identical outputs?
- Hypothesis: 32 cuts kernel overhead with negligible hit loss at batch 1-8.
- Cheapest decisive test: sweep `--block-size` 8/16/32/64 on E00's workloads plus a replayed multi-turn trace.
- Biggest novelty threat: [L36] (granularity decoupled), [L213] (asymmetric granularity, withdrawn).
- Scale risk: result is kernel-specific and moves with every kernel PR [L159].

### C2. Preemption that knows the prefix cache

- Problem: V1 preemption frees a victim's blocks into the prefix-cache free queue and later recomputes it [L134].
  The real cost of a preemption is how many of those blocks get reused by others before the victim resumes. Victim
  choice (last-admitted, least-computed, cost-aware) ignores this [L73] [L150] [L151].
- Why now: V1 removed swap [L72]; reasoning outputs make KV grow during decode [L180] [L176]; a 16 GB machine holds
  little KV, so preemption should be common (estimate, not measured).
- Existing work: swap-vs-recompute selection on PCIe [L84] [L82]; 2023 swap-vs-recompute sweep [L61]; victim-choice
  PRs without data [L150] [L151]; compressed in-memory swap for single-request phones [L214].
- Exact gap: nobody has measured how much of a V1 "recompute" is actually served from the victim's own surviving
  blocks, or used that survival to pick victims or order their freed blocks.
- Research question: Can a preemption policy that predicts block survival in the prefix cache (choice of victim and
  order of its freed blocks) reduce re-prefill tokens and p99 TPOT under KV pressure from long reasoning outputs,
  compared with vLLM's last-admitted victim and the open least-computed-first and cost-aware PRs, at equal KV memory
  and identical outputs?
- Hypothesis: today's victims lose their tail first and keep their prefix, so effective recompute is a fraction of
  full recompute; a survival-aware policy cuts re-prefill further.
- Cheapest decisive test: instrument the scheduler to log preemptions, victim size, and re-hit tokens on resume, at
  batch 4-16 with reasoning-length outputs on Qwen3-0.6B/4B; compute an oracle (best victim in hindsight).
- Biggest novelty threat: [L84] (per-preemption swap/recompute choice), [L215] (position-aware recompute cost).
- Scale risk: on datacenter GPUs preemption may be rare; the effect may be an on-device phenomenon only.

### C3. A KV pool that yields to the operating system

- Problem: vllm-metal sizes a fixed KV pool once [L90] [L139]. The pool is not wired, so under memory pressure from
  other apps macOS compresses or swaps its pages without knowing which blocks are idle cached prefixes and which are
  being read every decode step [L91]. Engines on Macs exceed their memory budgets [L210].
- Why now: concurrent serving on Macs is months old [L90]; the offload pool came in 2026-10-04 [L157].
- Existing work: datacenter ballooning between activations and KV [L66]; VMM elastic pools on CUDA [L63] [L64];
  Android single-request KV eviction under OS release requests [L214]; phone LLM as a system service [L203].
- Exact gap: no work on a batched, paged, prefix-caching KV pool on macOS that releases memory by value (cached
  prefix blocks first, then compression, then preemption) instead of letting the OS reclaim pages blindly.
- Research question: Under realistic background memory pressure on a 16 GB Mac, does a pressure-aware KV pool that
  releases cached-but-idle blocks first keep p99 TPOT and TTFT closer to the no-pressure baseline than vllm-metal's
  fixed pool and a statically smaller pool, at the same peak memory and identical outputs?
- Hypothesis: under pressure, OS compression of active KV pages causes multi-x TPOT spikes; releasing idle cached
  blocks first avoids most of it at a small hit-rate cost.
- Cheapest decisive test: run E00's workloads while a second process allocates memory in steps; log `vm_stat`
  compressions and swapouts and TPOT. If TPOT does not move, KILL.
- Biggest novelty threat: [L214] (mzCache), [L66] (eLLM ballooning).
- Scale risk: on-device only; datacenter GPUs do not share memory with an OS workload.

### C4. Do not cache dead thinking KV

- Problem: Qwen3's template drops past thinking [L181], so a finished turn's thinking and answer KV can never be
  reused, yet sits at the most-recently-used end of the LRU queue [L208] [L209].
- Exact gap: no vLLM measurement of the hit-rate cost [L209].
- Research question: Does evicting reasoning-output blocks at request finish raise prefix hit rate and cut TTFT on
  multi-turn reasoning chat, compared with vLLM's LRU, at equal KV memory?
- Cheapest decisive test: replay a multi-turn thinking workload with and without the [L208] patch.
- Biggest novelty threat: [L208] implements it; SGLang merged an opt-in fix [L209].

### C5. Quantized KV and batch size on Metal

- Problem: TurboQuant ships on vllm-metal with no speed numbers [L127]; mlx-lm quantized KV raised peak memory
  [L167] and disables batching in its server [L125].
- Research question: Does TurboQuant raise throughput at a fixed memory budget by admitting a larger batch size,
  compared with fp16 KV, while keeping task accuracy within a set margin?
- Biggest novelty threat: [L211] (int4 KV beats fp16 on Apple), [L98] (compression gains vanish at batch).

### C6. Keeping KV through agent tool calls on unified memory

- Problem: TTL pinning and offload during tool calls assume a CPU tier [L169] [L171]; Macs have only SSD.
- Research question: On a 16 GB Mac, does a pause-aware policy choosing among keep, SSD, and recompute beat
  always-keep and always-SSD for agent sessions?
- Biggest novelty threat: [L204] (persistent Q4 KV on edge), [L200] (KVSwap), [L207] (oMLX ships SSD tiering).

### C7. Eviction policy for tiny pools

- Problem: eviction studies use 24 GiB+ pools [L30]; a Mac pool is a few GB.
- Research question: At pool sizes of 1-4 GB, does S3-FIFO, ARC, or 2-class LRU beat vLLM's LRU on hit rate and
  TTFT for a single user's multi-turn trace?
- Biggest novelty threat: [L30] (14 policies), [L29], open PRs [L141] [L147], cost-aware [L215].

### C8. Checkpoint placement for hybrid-model prefix caching

- Problem: hybrid models need exact recurrent-state checkpoints; vLLM places them on a block grid [L185].
- Research question: Does placing checkpoints at turn boundaries beat block-grid checkpoints in hit tokens per byte?
- Biggest novelty threat: [L212] (exact DP for placement), [L38] (Marconi admission).

## Pass 1: adversarial review

Attack outcomes, one line each. Queries run this pass (besides the six frontier sweeps): "preemption recompute
prefix cache reuse freed blocks victim selection LLM serving 2026"; "arxiv preemption prefix caching recompute cost
partial re-hit evicted blocks KV cache pressure victim policy evaluation"; "reasoning model thinking tokens stripped
chat template prefix cache eviction KV waste multi-turn"; "macOS memory pressure LLM KV cache unified memory swap
compression inference latency Apple Silicon study"; "memory-pressure-aware elastic KV cache on-device LLM serving
laptop OS reclaim shrink KV pool"; "KV cache block size page size sweep prefix caching hit rate kernel latency
measurement study 2026".

### C1

1. Novelty: Survives as a measurement; no controlled sweep on unified memory found [L61] [L79].
2. Adjacent: Narrows. Hit granularity is already decoupled from block size [L36], so the hit-loss half of the tradeoff
   is gone; only kernel time and fragmentation remain.
3. Combination: Kills as research. What remains is tuning one flag on one kernel.
4. Simplicity: n/a. 5. Scale: Narrows, kernel-specific [L159].
6. Oracle: Open. Attention is 1.5 ms of a 24 ms step at 1k (E00), so a kernel gain is bounded by about 6% there.
7-10: Feasible locally; relevance limited to vllm-metal; kernel PRs active [L159].
- **Result: KILL as a thesis gap.** Keep as a warm-up lab experiment (one flag, cheap, teaches the kernel).

### C2

1. Novelty: Survives. Searched victim selection with prefix reuse; found none [L215] [L30]; PRs have no data [L150]
   [L151].
2. Adjacent: Open. OS page replacement and buffer-pool "victim" choice are classic; the new part is that the
   evicted state is itself a cache entry with prefix-chain dependencies (a later block is useless without earlier
   ones [L13]).
3. Combination: Survives narrowly. The coupling (preemption frees into the prefix cache, so preemption cost equals
   eviction outcome) exists only because V1 merged the two [L134].
4. Simplicity: Open. Raising KV memory or lowering `max_num_seqs` avoids preemption [L71]; least-computed-first may
   get most of the gain [L151].
5. Scale: Open. Preemption may be rare on datacenter GPUs.
6. Oracle: Open. Unknown how much re-prefill is already avoided by self-re-hits.
7. Correctness: Survives; outputs must be identical (recompute is exact), which makes evaluation clean.
8. Feasibility: Survives; scheduler-only Python change, runs on the Air.
9. Relevance: Open; needs evidence that preemption fires under reasoning or agent load.
10. Scoop: Narrows. Two open PRs change victim choice [L150] [L151]; the survival-aware part is not in them.
- **Result: MEASURE-FIRST.** Opens on Oracle, Scale, Simplicity, Relevance.

### C3

1. Novelty: Survives narrowly. Closest work is Android single-request without paging or prefix cache [L214] and
   datacenter ballooning [L66]. SiliconBench reports the pressure problem but no mechanism [L210].
2. Adjacent: Narrows. Memory ballooning is classic; the open part is the release order across cached blocks,
   compression, and preemption in a batched paged pool.
3. Combination: Narrows to the release-order policy; "ballooning for KV on a Mac" alone is engineering.
4. Simplicity: Open. A smaller static pool may be enough.
5. Scale: On-device only; stated as the scope.
6. Oracle: Open. Unknown whether macOS pressure hits active KV pages at realistic app usage.
7. Correctness: Survives; outputs identical unless compression is used.
8. Feasibility: Survives; runs on the Air.
9. Relevance: Survives for on-device serving [L90] [L210] [L94].
10. Scoop: Open; vllm-metal roadmap lists no such work [L162], but the area is moving.
- **Result: MEASURE-FIRST**, narrowed to the release-order policy.

### C4

1. Novelty: **Kills.** The mechanism is implemented [L208] and SGLang shipped an opt-in version [L209].
- **Result: KILL as a thesis gap.** A good one-evening tweak: measure [L208]'s patch on the Air, since the vLLM
  maintainer and the PR author disagree with no data [L209].

### C5

1. Novelty: Narrows; int4 KV on Apple already beats fp16 at batch 1 [L211].
3. Combination: Kills. What remains is "does more memory allow more batch," which is well understood [L102] [L98].
- **Result: KILL.** Keep as a lab tweak: TurboQuant on/off at batch 8.

### C6

1. Novelty: Kills. Persistent 4-bit KV on edge devices [L204], disk-aware on-device KV [L200], shipping Mac SSD
   tiering [L207], TTL theory on GPUs [L169].
4. Simplicity: Kills. With slow prefill on the Air, "always save to SSD" is the likely winner [L206].
- **Result: KILL.**

### C7

1. Novelty: Kills. [L30] evaluates 14 policies including the small-capacity regime (24 GiB); [L215] takes
   cost-aware eviction.
10. Scoop: Kills; many open upstream PRs [L141] [L143] [L144] [L145] [L147].
- **Result: KILL.** Keep as a lab tweak: S3-FIFO in `free_blocks` is a one-function change [L131].

### C8

1. Novelty: Kills. Checkpoint placement has an exact DP [L212]; Marconi covers admission [L38].
- **Result: KILL.**

## Pass 2: multi-model panel

Reviewers (provider and model as the host confirmed; effort as requested): claudeAgent claude-fable-5-1 @high,
codex gpt-6.1-sol (default effort), codex gpt-6-astra (default effort), claudeAgent claude-opus-5-5 @high. All four
had web access. Condensed reviews: [pass2/](pass2/). The prompt is [pass2/prompt.md](pass2/prompt.md). The lead judge
(claude-opus-5-5, the author model) opened every source used below.

### Corrections to pass 1

- The two victim-choice PRs do have data. Least-computed-first reports that preemption recompute is 10-23% of token
  compute online under pressure, that victim choice governs about half of it, and that it is "largely invariant" to
  prefix caching [L218]. Cost-aware choice moved every metric by under 3% [L219]. (Astra, Sol, Fable.)
- C6: the simplicity kill leaned on an H100 result that does not transfer [L55]; the novelty kill stands. (Astra.)
- C7: [L30] does not test pools below 24 GiB, so a small-pool comparison stays open as a lab question, not a thesis
  gap; RR-Evict strengthens the novelty kill. (Sol; Opus and Fable say the kill stands.)

### C2: rejections, merged

1. **The block-order half is already the default** (all four models, checked in code). vLLM frees a victim's blocks
   tail-first to the most-recently-used end, and a resumed request re-hits what survives [L131] [L134] [L220].
2. **Survival carries no information about which victim to pick** (Fable and Opus, independently). Preemption fires
   only when the free queue is nearly empty, so afterwards the victim's blocks are the only free blocks, and the
   running set consumes them at the same rate whichever request was chosen. What survives depends on how long the
   victim waits, not on who it is. A user report found identical waste per preemption for least-progress and LIFO
   victims [L223].
3. **Cache-aware and work-aware victim choice is published** (Astra, Sol, Opus): Echo [L224], InferMax short-first
   [L225], TOPAS [L226], CacheOPT [L84], plus the two vLLM PRs [L218] [L219].
4. **The upside is capped** (Fable): the whole recompute tax is 10-23% of compute under pressure [L218], and
   least-computed-first claims a share of that.

Answer: argument 2 is a prediction from code, not a measurement, and it conflicts with [L218]'s claim that the tax
does not change with prefix caching. If resumed requests re-hit their own blocks [L220] [L221], prefix caching should
cut the tax. One of the two is wrong, and nobody has published the number. That fact is all that survives.

**Decision: replace C2 with C2', a measurement question.** The survival-aware victim policy is dropped unless C2'
finds the spread that argument 2 says cannot exist.

### C3: rejections, merged

1. **Already shipping on Macs** (Sol, Opus, Fable): Splash releases free extents, compacts, then evicts cached
   entries by recency under macOS pressure, with measurements [L216]; oMLX frees unprotected LRU cache blocks against
   a ceiling that tracks other apps [L217].
2. **Classic pattern** (Fable, Opus, Astra): OS-cooperative cache shrinking (SQL Server, Chromium, Apple purgeable
   memory, OpenZFS) makes the port engineering.
3. **Simple baseline wins elsewhere** (Astra, Sol, Fable): an elastic KV controller lost to one smaller flag [L222].

Astra left a narrower physical-reclamation question for vllm-metal's shared backing store. That is allocator
engineering on one engine.

**Decision: KILL C3.**

### C2'. How much of a preemption does the prefix cache already save?

Current measurement plan, 2026-10-20. [E02](../../experiments/E02-preemption-rehit/README.md) expands the sweep and records every candidate for manual inspection.
It measures the chosen victim's recovery. Recovery for an unchosen victim remains unmeasured.
The older numeric decision rules below do not automatically determine E02's verdict.

- Problem: vLLM V1 preempts by dropping a request and recomputing it later [L134], but the dropped blocks stay in
  the prefix cache until reused [L220] [L221]. Nobody has measured how much recompute this saves, and the one
  published claim says prefix caching makes no difference [L218].
- Why now: V1 removed swap [L72]; small-memory Macs preempt under KV pressure [L221]; victim-choice PRs are under
  review without this number [L218] [L219].
- Existing work: victim choice [L218] [L219] [L224] [L225]; SGLang drops retracted KV from its tree [L227].
- Exact gap: the share of preemption recompute served by the victim's own surviving blocks, across load and pool
  size, and whether it differs between candidate victims at the same moment.
- Research question: Under KV pressure from reasoning-length outputs on vllm-metal, what fraction of a preempted
  request's computed tokens does it re-hit on resume, and does that fraction differ enough across candidate victims
  that a victim policy using it could beat least-computed-first by 10% p99 TPOT at equal KV memory?
- The answer is "no" if the re-hit fraction is near zero or near one for all victims, or if victims at the same
  moment differ by less than 2x in surviving blocks (Fable's test).
- Hypothesis: the re-hit fraction is large (most of the prefix survives a short wait), which contradicts [L218].
- Cheapest decisive test: E-C2'a below. One evening of scheduler logging, no policy change.
- Biggest novelty threat: [L218]'s unpublished companion study.
- Scale risk: SGLang does not re-hit at all [L227], so the result is specific to vLLM V1.

#### Attacks on C2'

1. Novelty: Survives as a measurement; the closest claim [L218] states the opposite without data.
2. Adjacent: Narrows; cache-related preemption delay analysis measures the same kind of quantity for CPU caches
   (Astra, Sol).
3. Combination: Survives; the quantity exists only because V1 frees preempted KV into the prefix cache.
4. Simplicity: Open; watermark, `max_num_seqs`, and least-computed-first may make preemption rare or cheap [L71]
   [L218].
5. Scale: Open; vLLM-only by construction [L227].
6. Oracle: Open; bounded by the 10-23% tax [L218].
7. Correctness: Survives for the measurement (logging only). A later policy changes batch composition, so outputs
   will differ at temperature 0 [L223].
8. Feasibility: Survives; local, Python-only.
9. Relevance: Narrows to small-memory serving [L221].
10. Scoop: Narrows; the [L218] companion study may publish first.

Status: **MEASURE-FIRST**, gap confidence **low**. Most likely result is a measurement note, not a thesis mechanism.
C2' has not had its own pass 2; run the kill-test before spending one.

# R06 candidates

T-02 speculative decoding, lenses L-12 to L-17. Sources are `[L<n>]` in `ledger.tsv`. Engine facts cite
`frontier/g-code.md` (this run) by section; it is a code reading of vLLM v0.31.0 and vllm-metal, not a test.

Generated 2026-10-20 from six lens sweeps (frontier/a-f, L1-L215) and the code reading. Ten candidates. R04 and R05
kills (adaptive K, unified token budget, draft-state loss on preemption, live migration of draft KV, shared drafter
pools, PD draft placement, unified-memory overlap on Apple silicon) are prior evidence: a card that reduces to one of
them is not new.

Compute defaults for every card unless the card says otherwise. Local: M5 Air, vllm-metal (greedy only; ngram,
draft_model, DFlash, DSpark, Gemma4 MTP; no EAGLE, no DP/PP with spec). Colab: one CUDA GPU, upstream vLLM, no root
(no clock or power-cap control). Real GPUs: unconfirmed.

---

## C1. Consistency: train-inference logprob consistency for speculative RL rollouts

- Lens: L-12 consistency and ownership transfer.
- Existing mechanism: RL rollouts need the inference engine's logprobs to match the trainer's. vLLM's bitwise
  train-inference consistent mode reaches logprob diff 0 but explicitly excludes MTP and speculative decoding [L17];
  TRT-LLM's router replay also excludes spec [L18]. Speculative rollouts are used anyway (NeMo-RL + EAGLE-3 1.8x
  [L10], ReSpec [L9]). SGLang silently verified greedily on some backends, making RL losses wrong [L19] [L20].
  Batch-invariant stochastic spec is an open vLLM PR [L23]; SGLang uses spec to speed deterministic rollouts [L22].
- Problem: verification computes target logits over a `1+K` query, not the 1-token decode shape or the trainer's
  full-sequence shape, so returned logprobs carry different numerics (g-code §1).
- Why now: bitwise-consistent RL modes shipped in 2026 [L17]; spec rollouts became standard [L10].
- Existing work: run-to-run determinism for spec [L21] [L23]; consistency without spec [L17].
- Exact gap: whether spec rollouts can be made train-inference bitwise consistent, and what that costs in speedup.
- Research question: Can speculative rollouts return logprobs bitwise equal to the trainer's forward, compared with
  non-speculative bitwise-consistent rollouts, while keeping most of the spec speedup?
- Hypothesis: verification is a multi-token forward, closer in shape to the trainer's than 1-token decode is, so
  consistency costs less with spec than without.
- Cheapest decisive test: logprob diff between a trainer forward and vLLM verify for Qwen3-0.6B/4B with and without
  batch-invariant kernels; Colab.
- Biggest novelty threat: vLLM batch-invariant spec PR [L23] and the RL-consistency team [L17].
- Scale risk: MoE routing consistency [L18] dominates at scale.
- Compute: runnable now on Colab for dense models. Ideal: 1x H100, Qwen3-8B, a real RL loop (verl/NeMo-RL).
  Lower-resource: Colab L4, 0.6B-4B, no RL loop; cannot show reward effect. A few GPU-hours. Access: Colab confirmed.

## C2. Consistency: detecting drafter-target version skew

- Lens: L-12; L-14 fault tolerance.
- Existing mechanism: target and drafter are updated in separate sessions under one engine version string; three of
  four vLLM transfer engines cannot update the drafter; `reload_weights` and LoRA load skip it; aliased embed/lm_head
  move with the target while the rest stays old (g-code §1). Skew is silent: acceptance falls to 0 [L3], ~1.0 with
  rollouts 25% slower than no spec [L6], 4.3 to 1.0 [L33]. SGLang now updates target and draft in one session [L7];
  vLLM merged a draft update path [L4]. TRT-LLM auto-disables spec below a rolling acceptance threshold [L92].
- Problem: a stale drafter costs throughput with no error.
- Why now: weekly bug cluster in 2026 [L1]-[L7].
- Exact gap: a skew detector that tells a stale drafter from a workload shift or an adversarial prompt [L112].
- Research question: Can an acceptance-distribution test detect a stale or half-updated drafter within N steps with
  a false-positive rate below that of a fixed acceptance threshold [L92], on mixed traffic?
- Hypothesis: skew shifts acceptance at every draft position and across all requests; workload shifts do not.
- Cheapest decisive test: replay a trace with a drafter swapped mid-run, compare detectors; Air or Colab.
- Biggest novelty threat: version tags or weight hashes at update time, which need no statistics.
- Scale risk: none specific.
- Compute: runnable now on Air with draft_model. Hours.

## C3. Fault tolerance: speculative verification as a silent-data-corruption detector

- Lens: L-14 fault tolerance; L-12.
- Existing mechanism: fleet SDC is about one fault per thousand devices; Meta's inference SDC detector is a costly
  per-neuron output-distribution check applied at a sampling rate [L102]. LLM fault-injection shows the first token
  and early decode steps are most damaging and proposes selective checksums [L104]; ABFT for attention costs ~13.9%
  [L108]; nondeterminism breaks recompute-and-compare [L103]. A drafter is an independent model whose proposals the
  target checks every step; corrupted drafters already show up only as lower acceptance [L94] [L95].
- Problem: SDC in the target produces plausible wrong tokens with no signal. Speculation already runs a second,
  independent predictor every step.
- Why now: SDC on large GPU fleets is now measured [L102] [L109]; spec decode is on by default in many deployments.
- Existing work: no source found that uses draft-verify disagreement to detect hardware faults (frontier C searches).
- Exact gap: the detection power of per-request acceptance statistics for target-side faults.
- Research question: Can per-step acceptance statistics detect injected target-side faults (weight and activation bit
  flips) that change outputs, with higher recall at equal false-positive rate than a no-cost baseline (logit-range
  checks) and at lower cost than duplicate execution, on Qwen3 models?
- Hypothesis: faults that change outputs move the target away from the drafter's predictions, dropping acceptance
  for the affected requests by a detectable margin.
- Cheapest decisive test: fault injection on the Air (MLX weights, Qwen3-0.6B/4B target + draft_model or ngram),
  measure acceptance shift vs output-change severity.
- Biggest novelty threat: an unpublished fleet practice; checksums [L104].
- Scale risk: acceptance variance across requests may hide small shifts; faults that matter may be rare tokens.
- Compute: runnable now on Air (Python-level fault injection on MLX arrays). Ideal: H100 with real GPU fault
  injection (NVBitFI) and EAGLE-3. Lower-resource: Air software injection; cannot model in-kernel faults.

## C4. Fault tolerance: drafter fault containment

- Lens: L-14.
- Existing mechanism: a drafter exception kills vLLM's engine core; FT mode aborts every in-flight request and never
  resets drafter state (g-code §3) [L75] [L76]; drafter overruns crash the engine with illegal memory access [L96];
  MTP + structured output raises EngineDeadError [L97]; 20 of 74 spec-labelled vLLM issues in six months are
  crashes, hangs, or corruption [L99]. SGLang's decoupled drafter fallback is a TODO [L87]. LUMEN uses a drafter to
  bridge recovery [L82].
- Problem: the drafter is optional for correctness, yet its faults take down the target.
- Research question: Can per-request quarantine to target-only decoding on drafter fault keep request availability
  at the non-spec level, compared with engine-wide abort and restart, while preserving output equivalence?
- Hypothesis: most drafter faults are per-request or per-shape and recoverable without restart.
- Cheapest decisive test: classify the 20 crash issues by recoverability (sticky CUDA error or Python exception).
- Biggest novelty threat: engineering fix upstream [L87].
- Scale risk: sticky CUDA errors poison the whole context, so in-process recovery is impossible for them.
- Compute: issue study on Air; fault drills on Colab. Hours.

## C5. Backpressure: the output path in speculative serving

- Lens: L-13 backpressure and queueing.
- Existing mechanism: vLLM emits one output message per decode step, carrying all tokens accepted that step (with
  `stream_interval` 1) [L56] [L57]; `vllm bench serve` records ITL per message, not per token [L58]. At concurrency
  1024, `stream_interval` 10 raised throughput 57% and cut median TTFT 2.3 s to 262 ms with no spec [L54]; the Rust
  frontend shows +45-72% from interval 4-16 [L55]. Burst outputs break reasoning parsers [L59].
- Problem: spec-on vs spec-off comparisons at high concurrency change both GPU work and the number of output
  messages per token. Part of spec's measured gain may be output-path relief available without spec.
- Research question: At high concurrency, how much of spec decode's measured goodput gain over non-spec remains
  when the non-spec baseline uses a matching `stream_interval`?
- Hypothesis: a measurable share (say >25%) of spec's gain at concurrency >=256 is output-path relief.
- Cheapest decisive test: Colab, Qwen3-0.6B, ngram/EAGLE, concurrency sweep, spec on/off x stream_interval 1/K.
- Biggest novelty threat: the stream_interval PR [L54] already reports the host-overhead effect.
- Scale risk: spec is rarely run at concurrency where the frontend binds [L41] [L46].
- Compute: Colab, a few GPU-hours.

## C6. Backpressure: speculation and queueing delay beyond saturation

- Lens: L-13.
- Existing mechanism: the Little's-law latency model covers only the stable pre-saturation regime and mean latency
  [L39]; AdaServe and AdaSpec measure TPOT, not TTFT or queue wait [L43] [L44]; spec raised P99 TTFT 250 ms to
  1078 ms at concurrency 1 [L50]; production MTP saw TTFT p95 76 s at saturation without isolating spec's share [L52];
  disabling spec under load collapsed throughput for unknown reasons [L48].
- Problem: speculation changes decode capacity and per-step time, which moves queue wait and TTFT.
- Research question: Under bursty arrivals near saturation, does speculation lower or raise queue wait and P99 TTFT
  compared with non-spec at equal hardware?
- Biggest novelty threat: TurboSpec / Nightjar controllers [L40] [L45]; R04 adaptive-K kill.
- Compute: Colab. A few GPU-hours.

## C7. Topology: speculation under expert parallelism and EP resharding

- Lens: L-15 topology changes; L-16.
- Existing mechanism: elastic EP reconfigures only the target and lists MoE drafter support as future work [L128]
  (g-code §4); EP fault RFCs ignore MTP [L77] [L129]; SGLang MTP gain falls from +60% to +14.2% at large-scale EP
  [L141]; every MoE spec cost model is single-device [L118]-[L123]; verifying K drafts raises expert traffic 2-3x on
  one GPU [L118].
- Problem: under EP, verification multiplies tokens per rank, all-to-all volume, and activated experts; the spec
  gain may depend on EP degree, and a reshard moves the optimum.
- Research question: How does the speedup of MTP/EAGLE on an MoE target change with EP degree, and which term
  (all-to-all, expert activation, per-rank batch) explains the drop, compared with a single-device cost model?
- Biggest novelty threat: vendor tuning guides; adaptive-K kill in R04 if the answer is "pick K per topology".
- Compute: blocked. Ideal: 2-4 nodes of 8 H100/H200 with IB, DeepSeek-V3-class MoE. Lower-resource: 8 GPUs in
  one node with Qwen3-30B-A3B; loses cross-node all-to-all. Tens of GPU-hours. Access unconfirmed.

## C8. Placement: the quantized-verify tax across hardware

- Lens: L-16 hardware-aware placement.
- Existing mechanism: on Jetson Thor, EAGLE-3 at batch 8 gives 1.57x with NVFP4 but 0.87x with INT4 AWQ at the same
  acceptance [L169]; the quantized Metal backend runs "parallel" verification serially [L72] [L174]; every evaluated
  spec method lowered throughput on four Apple machines [L175]; vllm-metal MTP is flat at concurrency 8 [L176].
  No controlled drafter-precision study [L179].
- Problem: weight-only quantized matmul kernels are built for 1 token; verify feeds `(K+1)·B` tokens, so the weights
  matmul may stop being a single weight read, and spec flips from win to loss by kernel, not by acceptance.
- Research question: Can a hardware-aware choice of verify width (K and tree size) from the measured quantized
  weights-matmul cost curve recover spec speedup on 4-bit targets, compared with fixed-K vLLM/vllm-metal defaults?
- Biggest novelty threat: Sequoia's hardware-aware tree [L156]; adaptive-K kill (R04); Marlin-style kernels.
- Compute: runnable now on Air (MLX 4-bit). Ties to E01 (weights matmul by batch size, planned).

## C9. Power: speculation under GPU clock locking and power caps

- Lens: L-17 power/capacity constraints; L-16.
- Existing mechanism: plain decode on H200 draws 137-300 W of 700 W so power caps never bind, and SM clock locking
  Pareto-dominates capping with up to 32% decode energy saved [L190]; decode is clock-insensitive (2842 to 180 MHz
  saves 42% energy for 1-6% latency) but that study excludes spec [L205]; frequency controllers model one token per
  step [L192] [L207]; spec saves up to ~29% energy at batch <=16 and costs 25.65% more at batch 128 [L188]; PELM
  does DVFS + speculation on Jetson single-request [L189]; spec drew 28% more power on a P100 [L209].
- Problem: verification raises arithmetic intensity per step, so the clock-insensitivity that clock-locking relies
  on may break with spec, and the energy-optimal (K, clock) pair is unknown for serving.
- Research question: Does speculative decoding change the energy-optimal SM clock of decode on a datacenter GPU, so
  that a jointly chosen (K, clock) saves energy per token at a TPOT SLO compared with clock-locked non-spec decode
  [L190] and default-clock spec decode, across batch 1-128?
- Hypothesis: at batch where `(K+1)·B` nears the ridge point, spec makes decode clock-sensitive; locking the clock
  low then erases spec's latency gain, so the joint optimum differs from either alone.
- Cheapest decisive test: a clock sweep x K sweep x batch sweep on one GPU with `nvidia-smi -lgc` and NVML energy.
- Biggest novelty threat: PELM [L189] (device, single-request); GreenLLM/throttLL'eM [L207] [L192] extensions.
- Compute: blocked on clock control: Colab has no root. Ideal: 1x H100/H200 bare metal with `nvidia-smi -lgc`.
  Lower-resource: any rented single GPU that allows clock locking (A10/L4); keeps the phenomenon. ~10 GPU-hours.

## C10. Power: thermal drift and frozen cost tables on fanless hardware

- Lens: L-17; L-16.
- Existing mechanism: DSpark adaptive verification sizes the verify budget from cost tables measured once at
  startup at context 8192 (g-code §2, §6); Sequoia profiles t(n) offline once [L156]; sustained phone decode loses
  44% throughput to throttling [L196]; PELM is thermal-aware spec on Jetson [L189]; Apple-silicon spec energy exists
  only as TDP estimates [L197].
- Problem: on a fanless Air, the draft/verify cost ratio drifts as the chip throttles, so frozen tables misprice
  verification.
- Research question: Does sustained throttling on a fanless M5 move the draft/verify cost ratio enough that a
  startup cost table picks a verify budget measurably worse than a re-measured one?
- Biggest novelty threat: PELM [L189]; adaptive-K kill (R04).
- Compute: runnable now on Air (powermetrics needs sudo).

---

# Pass 1

Three hostile sweeps (frontier/h, i, j; ledger L216-L272) ran all ten attacks on every card. Queries are listed in
each sweep's reply, summarized per card. GitHub search rate limits cut SGLang, verl, slime and NeMo-RL issue
searches for C1 and some vLLM issue searches for C6; recorded as coverage limits.

## C1 pass 1: KILL

- Novelty: Narrows. vLLM, IsoExec and Miles bitwise modes all omit spec [L17] [L226] [L225].
- Adjacent: Kills. vLLM's consistent mode already routes decode through the prefill path [L17]; a verify forward is
  another prefill-path chunk.
- Simplicity: Kills. Miles reaches logprob diff exactly 0 by rescoring each finished sequence with a prefill pass,
  independent of how tokens were generated [L225]; slime recomputes logprobs on the trainer by default [L227].
- Combination: Kills. Batch-invariant kernels make a row independent of the query shape; the "verify is closer to
  training" hypothesis has nothing left to test.
- Oracle: Kills. Rescoring already reaches diff 0. Correctness: Kills (kernel choice only).
- Scale: Open (MoE routing for verify tokens [L18], engineering). Feasibility, Relevance: Survive. Scoop: high [L23] [L226].
- Queries: speculative RL rollouts logprob mismatch / bitwise; slime/Miles true-on-policy; IsoExec; vLLM PR search.

## C2 pass 1: KILL

- Novelty: Narrows. SpecGuard already scores acceptance as a calibrated deviation detector and advises pinning and
  verifying the draft checkpoint [L216].
- Simplicity: Kills. A tensor digest settles bug-induced skew exactly; vLLM's SHA-256 weight checker excludes drafts
  only by omission [L228]; Dynamo recommends digests [L229]; Miles checks MTP tensors [L225]; SGLang updates both in
  one session [L7].
- Oracle: Kills. Real failures collapse acceptance to ~0 or ~1.0 [L3] [L6] [L33], which a fixed threshold catches [L92].
- Relevance: Narrows (gradual RL staleness [L9] is drift, not a fault). Other attacks: Survive.

## C3 pass 1: Narrow to C3'

- Novelty: Narrows. SpecGuard uses draft acceptance to detect target behavior shift from backdoors (AUROC 0.93-0.97)
  [L216]; it does not consider hardware faults. No source uses draft-verify disagreement against SDC.
- Adjacent: Narrows. A supervising network detects CNN soft errors [L219].
- Combination: Narrows, with a new constraint: lm_head is the operator most vulnerable to SDC [L223], and EAGLE/MTP
  drafters alias the target's lm_head and embeddings (g-code §1). A fault there hits both sides and acceptance is blind.
- Simplicity: Open. Norm-bound checks keep ~99% of ABFT detection at 87% less overhead [L217]; value-range checks
  catch SDC-causing faults because those produce large deviations [L224]; industry says large deviations are easy [L220].
- Scale: Open. Inference SDC comes from persistent device defects that hit thousands of requests [L220], which
  favours per-device aggregation over per-request detection.
- Oracle: Open. SDC is rare: 13 faulty GPUs in 3.5e7 GPU-hours in training [L221]; weekly-to-biweekly at Gemini
  scale [L222]; ~6.7% of injected instruction faults become SDCs [L223]. No inference-fleet output-change rate found.
- Correctness, Relevance, Feasibility: Survive (software injection on the Air). Scoop: medium [L216] [L218].
- Queries: speculative decoding + SDC / soft error / fault injection / bit flip; draft-verify disagreement soft
  errors; approximate DMR with a smaller model; symptom-based DNN detectors; acceptance as node health; fleet SDC rates.

## C4 pass 1: KILL

- Oracle: Kills. Of 74 spec-labelled vLLM issues in six months, ~10 fail at startup, ~7 silently change output, ~4 are
  runtime illegal memory access, ~2 are recoverable Python exceptions (classification by the pass-1 sweep) [L230].
- Correctness: Kills. Asynchronous CUDA errors are sticky; only process exit recovers [L234] [L231].
- Simplicity: Kills. The recoverable faults fire on every request, so quarantine equals turning spec off [L232].
- Novelty: Narrows to the SGLang TODO [L87], engineering. Others: Survive.

## C5 pass 1: KILL

- Simplicity: Narrows. The main public spec-on/off benchmark already pairs `stream_interval` (10/10, 30/30, 50/50)
  [L235] [L237]. Scale: Narrows. The output path binds at ~1k-5k concurrency [L54] [L239]; spec deployments are
  measured at <=64 [L244] or already raise the interval at 2048 [L236]. Relevance: Narrows (naive benchmarks only
  [L58] [L238]). Correctness: Narrows (method only). Adjacent: Narrows (coalescing ships [L54]).
- Three Narrows on a measurement-method question leave a benchmark control, not a thesis. Kept as a control for
  any later E record.

## C6 pass 1: Narrow to C6' (weak)

- Novelty: Survives; L39 names bursty workloads and latency distributions as future work. Simplicity: Open (batch
  threshold off switch [L41] [L45]; R04 adaptive-K kill). Oracle: Open (spec off gives the best p99 goodput in
  [L42]). Relevance: Survives (TTFT p95 76 s with MTP [L52]; 8x TTFT at concurrency 16 [L243]; spec's P95 TTFT effect
  goes either way [L242]). Scoop: Open (L39's authors). Correctness: Narrows. Feasibility: Survives (Colab).

## C7 pass 1: Narrow to C7'

- Simplicity: Narrows, decisively. SGLang's +60% run is 2 requests per rank at 65,536-token input on 16 H200; its
  +14.2% run is 128 requests per rank at 2,000-token input on 128 H200 [L141]. A 64x larger batch per rank alone can
  explain the drop [L46] [L39]; EP size is confounded.
- Combination: Survives. The DeepEP per-rank dispatch buffer defaults to 128 tokens, and MTP sends K+1 tokens per
  request [L251]; DP-attention sync couples ranks [L138] [L139].
- Novelty: Survives (no gain-vs-EP measurement [L245]-[L250]). Scale, Oracle: Open (MoE auxiliary kernels slow with
  larger EP at fixed per-GPU batch [L245]). Feasibility: Open, compute-blocked; a cheaper first step is reanalysis of
  public InferenceX MTP-vs-none recipes across DEP8/12/16 [L236] [L237] [L180] [L199]. Relevance: Survives (EP144 in
  production [L247]). Scoop: Survives (vendor risk).

## C8 pass 1: KILL

- Novelty: Kills. On A100 with Marlin W4A16, EAGLE-2 tree verification costs 1.8x a decode step versus <1.2x for
  FP16, and a fix is published [L264]; Sequoia picks trees from measured cost [L156].
- Adjacent: Kills. MLX and llama.cpp are shipping few-row quantized kernels for verify widths [L265] [L266] [L268]
  [L269]; the MLX kernel switch point is per shape and per GPU generation [L268].
- Combination: Kills (Sequoia/TurboSpec on a quantized cost curve). Scale: Survives (INT4 advantage gone past 16-32
  tokens [L262]). Scoop: high.

## C9 pass 1: Narrow to C9'

- Novelty: Survives. arXiv searches for spec + DVFS / frequency / power cap / thermal return only PELM (Jetson,
  single request) [L189]; recent datacenter energy studies omit spec [L257] [L258].
- Adjacent: Narrows. Per-iteration DVFS for decode runs on H100 [L259]; TBT-feedback controllers would follow spec
  implicitly [L207].
- Combination: Narrows. For the weights matmul, spec at batch B is plain decode at (K+1)·B tokens, and the H200 study
  already shows the optimal clock rising with batch (batch 1-32) [L190]. The coupling left is attention: verify
  tokens reuse the same KV, raising attention intensity in a way batching does not; MLA is compute-bound from the
  first speculated token [L255]. H100 ridge ~295 FLOP/byte, ~280 tokens per step for the weights matmul at full clock
  [L256]; the ridge falls with SM clock (inference, not measured).
- Simplicity: Open (lock at the non-spec optimum, default K). Scale: Open ([L188]). Oracle: Open (clock alone up to
  32% [L190]; spec alone up to ~29% [L188]; joint unknown). Feasibility: Open, resource blocker: clock locking needs
  root [L254]; some rented VMs allow it [L253]; Colab unconfirmed. Scoop: medium.

## C10 pass 1: KILL (folded into C9')

- Adjacent, Combination, Simplicity: Kill. PELM is a thermal-aware spec governor [L189]; refitting step cost online
  with an EMA is the obvious fix; SGLang adapts from an acceptance EMA [L270]. Relevance: Narrows (spec often loses on
  Apple silicon anyway [L175]; CPU DVFS confounds [L271]; Mac throttling data anecdotal [L272]). The one live
  sub-question, whether verify is more clock-sensitive than drafting, is part of C9'.

---

# Narrowed cards (to pass 2)

## C3'. Per-device acceptance as a detector of output-changing hardware faults in the target

- Lens: L-14 fault tolerance; L-12 consistency.
- Existing mechanism: fleets catch SDC with periodic scanners [L102] [L222], costly sampled output-divergence checks
  [L102], ABFT (~13.9%) [L108], cheap norm-bound checks [L217] and value-range symptom checks [L224]. These catch large
  deviations well [L220] [L224]. Under speculation an independent drafter predicts every target token; acceptance
  already reflects target behaviour shift [L216] and drafter faults [L94] [L95].
- Problem: persistent small-deviation faults that flip argmax but stay inside value ranges are the hard case for
  cheap checks; acceptance measures exactly "does the target still agree with an independent predictor", at no
  extra compute.
- Why now: inference fleets run spec by default; SDC rates at fleet scale are now published [L221] [L222] [L220];
  SpecGuard shows acceptance is a usable behavior signal [L216].
- Exact gap: the detection power, per device, of acceptance statistics for target-side faults that change outputs,
  relative to cheap checks, and its blind spot in tensors the drafter shares (lm_head, embeddings).
- Research question: Can per-device acceptance statistics under EAGLE/MTP/draft-model speculation detect persistent
  target-side faults that change outputs, with higher recall at equal false-positive rate than CSNB norm bounds and
  value-range checks, at no added compute, on realistic traffic mixes?
- The answer is "no" if, for faults that change outputs, acceptance needs more requests to alarm than range or norm
  checks at the same false-positive rate, or if faults that range/norm checks miss do not move acceptance.
- Hypothesis: faults that flip argmax lower acceptance by a few points on every request routed to the device; over
  hundreds of requests that is detectable, while range checks miss small-magnitude persistent faults.
- Cheapest decisive test: software fault injection on the Air. Qwen3-4B 4-bit target with a draft model (and ngram as
  a model-free drafter), persistent faults in weights and matmul outputs at several magnitudes and bit positions,
  including shared lm_head. Measure output-change rate, acceptance shift, requests to alarm at 1% FPR, and CSNB/range
  detection on the same faults.
- Biggest novelty threat: SpecGuard [L216] extended to hardware; unpublished fleet practice.
- Scale risk: acceptance variance across prompts and tenants; per-device traffic mix must be stable enough.
- Compute: runnable now on Air (software injection into MLX arrays, greedy). Blocker for the final claim: real GPU
  fault models (NVBitFI) and EAGLE-3 need one CUDA GPU with instrumentation (Colab may suffice for NVBitFI; unverified).
  Ideal: 1x H100, Llama-3.1-8B + EAGLE-3, NVBitFI campaigns, ShareGPT/agentic mix. Lower-resource: Air software
  injection, cannot model in-kernel or control-flow faults. Estimate: Air screen 1-2 days; GPU campaign ~20 GPU-hours
  (estimate). Access: Air confirmed, Colab confirmed, NVBitFI on Colab unverified.

## C6'. Queue wait versus stretched steps: speculation's effect on P99 TTFT under bursty arrivals

- Lens: L-13.
- Research question: Under open-loop bursty arrivals on one GPU, does speculation raise P99 TTFT, and how much of the
  change is queue wait versus prefill-shared steps stretched by verification, compared with non-spec and with a
  batch-threshold off switch?
- The answer is "no gap" if a batch-threshold off switch recovers non-spec P99 TTFT while keeping spec TPOT at low load.
- Cheapest decisive test: Colab, Qwen3-4B + EAGLE-3, BurstGPT-shaped open-loop arrivals, spec off/on/threshold.
- Threats: TurboSpec [L40], Nightjar [L45], R04 adaptive-K and unified-budget kills.
- Compute: runnable now on Colab. A few GPU-hours.

## C7'. Is speculation's gain under expert parallelism set by batch per rank or by EP degree?

- Lens: L-15 topology changes; L-16.
- Research question: At fixed batch per rank, does MTP's speedup on an MoE target change with EP degree, and how much
  of the published drop [L141] is batch-per-rank erosion versus EP-only costs (dispatch buffer [L251], all-to-all,
  DP-attention sync [L138])?
- The answer is "no" if, at matched batch per rank, gain is flat across EP degree within noise.
- Cheapest decisive test: reanalysis of public InferenceX MTP-vs-none results across DEP8/12/16 [L236] [L237] [L180]
  [L199]; then a measurement on 8 GPUs (EP2/4/8 at fixed batch per rank).
- Compute: blocked on compute for any measurement. Ideal: 2-4 nodes x 8 H200/B200, DeepSeek-class MoE.
  Lower-resource: 1 node x 8 GPUs, Qwen3-30B-A3B; loses cross-node all-to-all. Tens of GPU-hours. Access unconfirmed.

## C9'. Speculation and the energy-optimal GPU clock

- Lens: L-17 power/capacity; L-16.
- Research question: Does K+1 verification make long-context decode attention clock-sensitive (GQA vs MLA), shifting
  the energy-optimal SM clock at equal batch size, so that a jointly chosen (K, clock) beats the best single knob
  (clock-locked non-spec [L190] or default-clock spec) in joules per token at a TPOT SLO?
- The answer is "no" if, at matched batch and context, the energy-optimal clock with spec equals that of non-spec at
  batch (K+1)·B, or if the joint optimum saves <5% energy over the best single knob (heuristic margin).
- Cheapest decisive test: one GPU with clock locking, sweep SM clock x K x context x batch, NVML energy.
- Threats: PELM [L189], DualScale [L259], GreenLLM [L207], the H200 phase study [L190].
- Compute: blocked on clock control (root). Colab check takes a minute (`nvidia-smi -lgc`); a rented VM allows it
  [L253]. Ideal: 1x H100/H200 bare metal, Llama-3.1-8B + EAGLE-3 (GQA) and DeepSeek-V2-Lite + MTP (MLA).
  Lower-resource: rented A10/L4 VM with root; keeps the phenomenon, not MLA at scale. ~10-20 GPU-hours (estimate).

---

# Pass 2

Panel (all four confirmed by T3): claudeAgent claude-fable-5-1 (default effort), codex gpt-6.1-sol (default),
codex gpt-6-astra (default), claudeAgent claude-opus-5-5 (effort medium). Full replies summarized in `pass2/*.md`.
Every new source the reviewers cited was reopened (frontier/k-pass2.tsv, L273-L298; one fetch failed, L297, not
cited). The lead judge opened L273 and L274 directly. Two reviewer claims were corrected on reopening: SGLang #27194's
root cause moved from the dispatch buffer to an uninitialized-memory read in the MoE stage-2 reduce path [L274]; the
symptom (acceptance high, gsm8k 0) stands. SpecGuard's recalibration is rolling calibration, not a rolling window [L275].
No reviewer found a wrong pass-1 kill.

## Merged rejections

C3'
1. **The drafter is not an independent check** (all four). EAGLE-3 and DFlash consume target hidden states and alias
   lm_head and embeddings (g-code §1); every drafter runs on the same faulty device. A real silent corruption inside
   an MoE layer kept MTP acceptance length high while gsm8k fell to 0 [L274]. Only a separate draft model or ngram is
   independent of target state, and even those share the hardware.
2. **The mechanism is prior art** (fable, sol, astra). A smaller checker network whose historical agreement with the
   main network drifts is a granted patent for detecting accelerator faults [L273]; SpecGuard already runs a windowed
   acceptance monitor with rolling calibration and per-query FPR 3.3-9.7% at TPR 0.95 [L275].
3. **A stronger simple baseline** (astra, opus, fable). Scheduled greedy canary prompts with golden outputs per host
   [L103 secondary], weight digests, and fleet scanners [L102] catch persistent defects without per-request cost.
   Acceptance can only win on data-dependent faults canaries miss; microbenchmarks miss over 60% of SDC-defective
   GPUs in training, so that class exists [L278].
4. **Fault realism** (sol, astra, fable). Single-bit flips are under 40% of real GPU bit-flip events [L277]; a stuck
   tensor element is not a defective execution unit. CSNB's 99% is relative to V-ABFT on dense linear layers [L276].
5. **Operational FPR** (sol, astra). 1% per window is not a fleet budget; state false alarms per device-day.

C9'
1. **The shift is predictable by arithmetic** (fable, opus). For GQA (Llama-3.1-8B: 8 KV heads, 4 query heads per KV
   head, head dim 128) verify attention does 4(K+1) FLOP per KV byte: 24 at K=5, against an H100 ridge near 295
   [L256]. It stays memory-bound and clock-insensitive; the weights matmul at (K+1)·B tokens is plain decode at a
   larger batch, which [L285] already maps (GQA: one low clock works at all batch sizes up to 32).
2. **The remaining regime has little to give** (fable, opus). Where the optimal clock does rise (MLA, 16K+ context,
   large batch) total clock savings fall to 5-15% [L285], and speculation already costs 25.65% more energy at batch
   128 [L188]; a 5% joint-knob margin does not fit.
3. **Feedback controllers absorb it** (opus, astra, sol). Per-step DVFS [L286] and SLO-feedback controllers [L207]
   [L284] spend speculation's TPOT slack on a lower clock without modelling it; intensity-driven DVFS exists [L283];
   attention and FFN optima already differ [L281]; PELM jointly controls K and frequency on Jetson [L282].
4. **Feasibility** (sol, astra). DeepSeek-V2-Lite has no MTP layers [L287]; clock control is unconfirmed [L254].

C6'
1. **A benchmark table with a known fix** (all four). Spec's TTFT penalty and P99 TPOT reversal are measured [L288];
   Nightjar's appendix runs burst-spike stress tests [L289]; the TTFT sign flips by configuration [L290]; DSpark
   shrinks verify budget with load [L291]; the threshold switch exists [L41] [L45].

C7'
1. **The design cannot isolate EP degree, and the cheap path is closed** (sol, astra, fable, opus). Fixed batch per
   rank still changes experts per device, routing and kernel shapes; XShare already models speculative tokens'
   expert correlation under EP [L292]; the expert-count model covers activation on one device [L123]; InferenceX
   pages state no parallelism and interpolate [L294]; every measurement needs 8+ GPUs with no confirmed access.

## C3' pass 2: Narrow to C3''

Rejections 2 and 3 kill the mechanism as a contribution; rejection 1 removes coupled drafters (MTP, EAGLE, DFlash)
as a detector and is backed by a production incident [L274]. What no source answers is empirical: for drafters that
are independent of target state (a separate draft model, ngram), what fraction of output-changing persistent
faults pass canaries and cheap value checks, and does acceptance catch that residue. That is a characterization
with a cheap local oracle step. Gap confidence low: the mechanism is prior art and two groups [L275] [L276] could
publish the measurement. Thesis fit low.

## C3''. Do output-changing faults that cheap checks miss move acceptance under an independent drafter?

- Lens: L-14 fault tolerance; L-12 consistency.
- Existing mechanism: fleets catch SDC with scanners and canary prompts [L102] [L222], sampled output-divergence
  checks [L102], ABFT [L108], norm bounds [L276] and value-range checks [L224]. A disagreement-drift checker is prior
  art [L273]; acceptance-drift detection exists for backdoors [L275]. Coupled drafters missed a real target-side
  corruption [L274].
- Exact gap: the size of the fault class that changes outputs yet passes canaries, norm and range checks, and
  whether an independent drafter's acceptance detects it, versus a coupled drafter's.
- Research question: For persistent, data-dependent target-side faults that change greedy outputs, can per-device
  acceptance under a separate draft model or ngram detect faults missed by a canary battery, CSNB-style norm bounds
  and value-range checks, within 1,000 requests at a fixed false-alarm budget, while a hidden-state-coupled drafter
  cannot?
- The answer is "no" if cheap checks catch nearly every output-changing fault (residual under 5%), or acceptance
  detects under 20% of the residual within 1,000 requests at the budget (both heuristics).
- Hypothesis: data-dependent faults that fire on a fraction of inputs escape fixed canaries and stay in range, but
  lower agreement with an independent drafter on the requests they hit.
- Cheapest decisive test: an oracle screen on the Air, MLX, with persistent and data-triggered fault patterns from
  [L277] (multi-bit, lane-correlated), measuring the residual class first.
- Biggest novelty threat: SpecGuard's authors [L275], the CSNB group [L276], fleet practice [L278].
- Scale risk: per-device traffic mix; most production drafters are coupled, which shrinks relevance.
- Compute: runnable now on Air (software injection). Final claim needs CUDA in-kernel injection (NVBitFI [L295]) on
  one GPU; Colab support unverified.

## C9' pass 2: KILL

Rejections 1 and 2 answer the combination and oracle attacks with fetched numbers and arithmetic on them [L256]
[L285] [L188]; rejection 3 supplies the stronger baseline. The surviving fragment (MLA or tree verification at long
context) is a sweep with a bound of 5-15% total clock savings before any joint gain, and needs root clock control
plus an MLA model with MTP that [L287] rules out for V2-Lite. A one-off check remains possible as a lab measurement
if a clock-controllable GPU appears: does the energy-optimal clock with EAGLE-3 at batch B differ from non-spec at
(K+1)·B by more than one clock step at 8K context.

## C6' pass 2: KILL

All four reviewers: a measurement with a shipped remedy [L288] [L289] [L291]. Kept as a control for any later E
record that reports TTFT with speculation.

## C7' pass 2: KILL

Two Narrows from pass 1 plus a design that cannot isolate its variable [L292], a closed cheap path [L294], and
nothing measurable on the available hardware. Recorded with the compute blocker: if an 8-GPU node is confirmed,
revisit with a matched design (fixed tokens per expert, not batch per rank).

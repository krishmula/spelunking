# Does a draft model notice when the GPU under it goes bad?

2026-10-20 · Status: MEASURE-FIRST · Gap confidence: low · Thesis fit: low

## Record and scope

- R06, topic T-02 speculative decoding, lenses L-12 consistency and ownership transfer, L-13 backpressure and
  queueing, L-14 fault tolerance, L-15 topology changes, L-16 hardware-aware placement, L-17 power/capacity
  constraints, chosen by Krishna.
- Prior: R04 (lenses L-01 to L-05) closed with no survivor; R05 (lenses L-06 to L-11) left one weak survivor, C5''.
  Their kills were carried as prior evidence.
- Coverage: 298 ledger rows, 6 unverified. Six lens sweeps, one code reading of vLLM v0.31.0 and vllm-metal
  (`frontier/g-code.md`), three pass-1 attack sweeps, a four-model pass 2 (Fable 5.1, GPT-6.1 Sol, GPT-6 Astra,
  Opus 5.5), and a reopen of every pass-2 citation. Coverage table in `constraints.md`.
- Limits: GitHub search rate limits cut part of the SGLang and RL-framework issue coverage. C3'' was formed from the
  pass-2 reviewers' own narrowing and has not had a separate pass-2 round of its own.

## Summary

Plain: speculative decoding runs a second, smaller model next to the big one and checks its guesses every step. If
the GPU under the big model starts computing slightly wrong numbers, the big model's answers change, and an
independent small model should start disagreeing with it more often. That would be a free fault alarm. Two things
already weaken this: the idea of a small checker model whose agreement drifts is patented [L273], and the drafters
most deployments use (MTP, EAGLE) read the big model's own internal state, so they drift with it. One real incident
shows exactly that: output went to garbage while acceptance stayed high [L274]. What nobody has measured is whether,
with a truly separate drafter, acceptance catches the faults that cheap checks (fixed test prompts, value-range
checks) miss. The first experiment injects faults on the Air and measures that.

Precise: measure the share of output-changing persistent target faults that pass a canary battery, norm bounds and
range checks, and the detection rate of per-device acceptance on that residue for an independent drafter (draft
model, ngram) versus a hidden-state-coupled one.

Honest expectation: cheap checks probably catch most output-changing faults [L224] [L220], which kills this in a day.

## Research question

For persistent, data-dependent target-side faults that change greedy outputs, can per-device acceptance under a
separate draft model or ngram detect faults missed by a canary battery, norm bounds and value-range checks, within
1,000 requests at a fixed false-alarm budget, while a hidden-state-coupled drafter cannot?

The answer is "no" if cheap checks catch nearly every output-changing fault (residual under 5%), or acceptance
detects under 20% of the residual within 1,000 requests at the budget. Both thresholds are heuristics.

## Background

- Speculative decoding: a drafter proposes K tokens, the target verifies them in one forward pass and keeps the
  longest agreeing prefix. Under greedy decoding, acceptance is exactly "how often the drafter predicts the target's
  argmax". Verification guarantees the output is the target's, so a drafter fault shows up only as lower acceptance
  [L94] [L95] (fact).
- Drafter coupling (fact, code reading): EAGLE-3 and DFlash build their input from target hidden states, and EAGLE/MTP
  drafters alias the target's embeddings and lm_head (g-code §1). A separate draft model has its own weights and
  inputs; ngram has no model at all.
- SDC in fleets (fact): about one faulty device per thousand [L102]; 13 faulty GPUs in 3.5e7 GPU-hours in training
  [L221]; microbenchmarks miss over 60% of SDC-defective GPUs, because faults depend on data and execution unit
  [L278]; real corruption patterns are mostly multi-bit and spatially correlated [L277]. Detection today: scanners and
  canaries [L102] [L222], sampled output divergence [L102], ABFT [L108], norm bounds [L276], range checks [L224].
- Unresolved: how many output-changing faults slip past those cheap checks, and whether acceptance with an independent
  drafter sees them. No source measures either.

## What prior work already answers

| Work | What it solves | What it leaves open | Source |
|---|---|---|---|
| IBM N-modular redundancy patent | Checker network agreement drift as an accelerator fault signal | LLMs, speculation, which faults it catches | [L273] |
| SpecGuard | Acceptance deviation detects backdoored target behaviour; rolling calibration | Hardware faults | [L275] [L216] |
| CSNB norm bounds | ~99% of V-ABFT detection on dense linear layers at 87% less cost | Faults in range; non-linear layers | [L276] [L217] |
| Range / symptom checks | SDC-causing faults usually produce large deviations | The small-deviation residue | [L224] |
| SDCs in the Wild | Data-dependent faults evade microbenchmarks in training | Inference; speculation | [L278] |
| Fault-injection studies | Which LLM operators and steps are vulnerable; lm_head most | Any detector based on agreement | [L223] [L104] [L218] |
| SGLang #27194 | Coupled MTP acceptance stayed high during a silent corruption | Hardware faults; independent drafters | [L274] |

## The gap

I did not find public work that measures whether speculative acceptance detects hardware faults, nor the size of the
output-changing fault class that passes canaries and value checks in LLM inference. The closest threats are the
patent [L273] (mechanism), SpecGuard [L275] (signal), and CSNB [L276] (baseline and injector). The contribution
would be a measurement, not a mechanism.

## Why now

Fleet SDC data became public in 2025-2026 [L221] [L278] [L277]; speculation is on by default in large deployments
[L180] [L236]; the first acceptance-based integrity monitor appeared in September 2026 [L275].

## Hypotheses

1. Some output-changing faults (data-triggered, small magnitude) pass a fixed canary battery and range/norm checks.
   Shows as a residual share of 5% or more.
2. An independent drafter's acceptance drops on requests those faults hit. Shows as detection of 20% or more of the
   residual within 1,000 requests.
3. A hidden-state-coupled drafter detects markedly less of the residual. Shows as a detection gap between arms.

## Evidence against

- Cheap checks catch large deviations [L224] [L220]; if output-changing faults are mostly large, the residual is near
  zero. Stands; it is the first thing the test measures.
- Production drafters are mostly coupled, and a real incident shows coupled acceptance is blind [L274]. Stands; it
  narrows relevance to separate draft models and ngram.
- The mechanism is prior art [L273] [L275]. Stands; the dossier claims a measurement only.
- Software injection on the Air does not model in-kernel or control-flow defects [L277]. Stands; Phase B needs NVBitFI.
- Acceptance varies with traffic; per-device mixes differ under prefix routing (opus, sol). Addressed in the design by
  paired clean/faulty replays of the same traffic.

## Experiments already run

| Experiment | Setup | Result | What it changed |
|---|---|---|---|
| None | | | |

## Validation plan

Proposed work. Nothing runs before Krishna writes a prediction. Thresholds are logged in `decisions.tsv`
(2026-10-20T19:45) and are heuristics.

Phase A, offline on the Air (oracle screen, 1-2 days):
- Target Qwen3-4B 4-bit in MLX, greedy. Arms: separate draft model Qwen3-0.6B; ngram; one hidden-state-coupled
  drafter available on vllm-metal (DFlash or Gemma4 MTP; which one runs is a setup check).
- Faults, persistent per "device": scaled perturbations and multi-bit, lane-correlated corruptions of one matmul
  output type (attention projections, MLP, lm_head), some always on and some triggered by an input condition so they
  fire on a fraction of requests. Patterns follow [L277].
- Traffic: a fixed public prompt mix (ShareGPT-style chat plus code), replayed clean and faulty with identical order.
- Measure per fault: output-change rate versus clean; canary battery result (fixed prompts, greedy exact match);
  norm-bound and range-check alarms; per-request acceptance; requests to alarm for a windowed acceptance test
  calibrated on clean traffic.
- KILL if the residual (output-changing faults passing all three cheap checks) is under 5%, or the independent arms
  detect under 20% of it within 1,000 requests at no more than 1 false alarm per 10,000 clean windows.
- PIVOT to a short note ("acceptance is not an integrity signal") if detection reaches 20% only for faults that fire
  on most requests, which canaries already catch.
- GO to Phase B if the residual is 5% or more and an independent arm detects 50% or more of it.

Phase B (only on GO), one CUDA GPU: Llama-3.1-8B with a separate draft model and with EAGLE-3, NVBitFI in-kernel
persistent faults [L295], same measurements.

What it cannot show: real defect rates in inference fleets, faults in control flow, or behaviour across a fleet with
different traffic per device.

## Compute requirements

- Readiness: Phase A runnable now on the Air. Phase B blocked on an instrumentable CUDA GPU (NVBitFI on Colab
  unverified).
- Blocker: the Air cannot run in-kernel CUDA fault injection; a Mac result does not establish GPU fault coverage.
- Local screening: Phase A as above. It can kill the idea; it cannot establish GPU coverage.
- Ideal: 1x H100 80 GB, Linux, ~128-256 GB host memory, NVBitFI, Llama-3.1-8B with a separate draft model and EAGLE-3.
- Lower-resource: one Colab L4 with PyTorch-level injection; keeps the detection question, not in-kernel faults.
- Estimates: Phase A 1-2 days on the Air. Phase B from ~20 GPU-hours for a pilot (opus, fable) to 50-300 GPU-hours for
  a decisive campaign (astra, sol). Uncertain; driven by fault count and clean-trace length.
- Access: Air and Colab confirmed. Next action for Phase B: a one-cell NVBitFI build check on Colab, only after GO.

## Learning before implementation

- How greedy acceptance is computed and why it equals argmax agreement: [L34], g-code §1.
- Why EAGLE/MTP are not independent of the target: g-code §1, [L274].
- What realistic GPU faults look like: [L277] [L278]; why cheap checks catch large deviations: [L224] [L276].
- The prior signal work this measurement must beat: [L275] [L273].
- Doubts to record in learning notes: whether software injection on 4-bit MLX weights is a fair stand-in; how to
  choose the canary battery so it is neither trivially weak nor tuned to the injected faults.

## Evaluation plan

- Baselines: canary battery with golden outputs [L102]; norm bounds [L276]; value-range checks [L224]; coupled-drafter
  acceptance.
- Workload: public chat and code prompts, replayed paired.
- Metrics: residual share, detection within 1,000 requests, false alarms per 10,000 clean windows, wrong outputs
  emitted before alarm.
- Constraint: greedy, outputs identical to non-speculative decoding when clean.
- Final claim scale: one GPU with in-kernel injection; fleet claims stay out of scope.

## Plan

- Week 1: Krishna's prediction, then Phase A. Decision at end of week.
- On GO: Phase B setup check, then a pilot of ~20 GPU-hours. Decision after the pilot.
- On KILL or PIVOT: record in the journal and move to the next topic block.

## Risks and threats

- Scoop: SpecGuard or CSNB authors extend to hardware [L275] [L276]. Response: publish the negative or positive
  result as a short note.
- Scale: fleets use coupled drafters, so the result may matter only to separate-draft deployments.
- Feasibility: NVBitFI may not run on Colab; Phase B would then need a rented GPU, which needs Krishna's approval.

## Candidates considered

- C1 train-inference consistent spec rollouts: KILL, rescoring finished sequences already gives diff 0 [L225] [L227].
- C2 drafter-target version skew detection: KILL, tensor digests settle it exactly [L228] [L229].
- C4 drafter fault containment: KILL, most faults are sticky CUDA errors or fire every request [L230] [L234].
- C5 output path confound: KILL, public benchmarks already pair `stream_interval` [L235] [L237].
- C6' spec and P99 TTFT under bursts: KILL, measured and a remedy ships [L288] [L289] [L291].
- C7' spec gain versus EP degree: KILL, design cannot isolate EP degree, cheap path closed, no hardware [L292] [L294].
- C8 quantized-verify tax: KILL, published with a fix, kernels shipping [L264] [L268] [L269].
- C9' spec and the energy-optimal clock: KILL, GQA verify stays under the ridge, remaining margin 5-15% [L256] [L285].
- C10 thermal drift of cost tables: KILL, PELM and an EMA cover it [L189] [L270].

## References

- [L273] Reducing the cost of n modular redundancy for neural networks, US11599795B2, IBM, 2023. https://patents.google.com/patent/US11599795B2/en
- [L274] sglang issue #27194, 2026. https://github.com/sgl-project/sglang/issues/27194
- [L275] [L216] SpecGuard: Inference-Time Backdoor Detection For Free, arXiv 2609.11799, 2026.
- [L276] [L217] Revisiting Cauchy-Schwarz Norm Bounds for SDC Detection in LLM Inference, arXiv 2610.18557, 2026.
- [L277] The Anatomy of Silent Data Corruption: GPU Error Pattern Study and Modeling Guidance, arXiv 2605.04213, 2026.
- [L278] SDCs in the Wild, OSDI 2026. https://www.usenix.org/conference/osdi26/presentation/zheng
- [L224] Understanding Error Propagation in Deep Learning Neural Network (DNN) Accelerators and Applications.
- [L223] Analysis of LLM Vulnerability to GPU Soft Errors: An Instruction-Level Study, 2025.
- [L221] Safeguarding LLM Training at Scale: Online SDC Detection (ByteDance), 2026.
- [L102] How Meta keeps its AI hardware reliable, Meta engineering, 2025.
- [L220] Silent Data Corruption in AI, OCP whitepaper.
- [L295] NVBitFI, NVlabs GitHub.
- Full rows, URLs and quotes for every id are in `ledger.tsv`.

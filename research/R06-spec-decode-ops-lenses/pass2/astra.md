# Pass 2: codex gpt-6-astra (provider default effort), completed 2026-10-20

Verdict: no evidence-backed KILL of any narrowed question. C3' and C9' survive only with evaluation repairs; C6', C7' weak MEASURE-FIRST. No pass-1 kill reversed.

## C3'
- Rejection: repurposes an existing acceptance monitor (SpecGuard); evaluation assumes an independent reference EAGLE/MTP lacks; 1% FPR unsuitable for fleets.
- Combination Narrows: EAGLE-3 consumes target hidden states (llm_base_proposer.py:520); dependence is larger than shared lm_head.
- Simplicity Open: compare to CSNB, range checks, periodic weight digests, sampled deterministic canaries, sampled replay at matched cost; CSNB injects into activations, so weight-injection comparison can unfairly favor acceptance.
- Oracle Open: greedy counterexample (draft C, healthy A, faulty B: both reject).
- Fault realism: single-bit flips <40% of real bit-flip events (arXiv 2605.04213); a stuck tensor element is not a defective execution unit.
- "No added compute" too strong; vLLM spec metrics are per engine, not per physical device (metrics.py:229).
- New sources: 2605.04213; mixed-precision approximate DMR (IEICE 2024); NVBitFI; NVBit (Linux only); llm-d-kv-cache #699.
- Compute: Air screen 1-2 days; decisive campaign 50-300 GPU-hours.

## C9'
- Rejection: PELM joint spec/DVFS exists; AFlex attention-specific DVFS exists.
- Simplicity: missing baseline is fixed-K spec with tuned clock, then coordinate descent over K and clock.
- Decision-rule repairs: decode at (K+1)B is a mechanism control, not equivalent; GQA vs MLA end-to-end confounds model; use attention-kernel microbenchmarks.
- Correctness: energy per committed token; PELM variable verify depth is lossy.
- New sources: AFlex 2608.01891; cloud GPU power management CAL'23; PELM GitHub (SenSys '26); DeepSeek-V2-Lite config.
- Compute: screen 10-20 GPU-hours; two-architecture study 40-120.

## C6'
- Nightjar full text (Appendix C) already has burst-spike stress tests with P99 E2E latency; residual is TTFT attribution only.
- New sources: Beyond Prediction (ICML 2026, MSR); DSpark in SGLang (verify budget shrinks with load); Dynamo #9755; vLLM Speculators throughput-interactivity docs; AngelSlim Qwen3-4B_eagle3.

## C7'
- XShare covers expert-preference correlation of speculative tokens and EP-aware load; fixed batch per rank does not isolate communication.
- New sources: XShare 2602.07265; DeepEP; llm-d v0.4 roadmap.
- Compute: intranode pilot 40-100 GPU-hours; multinode 300-1000.

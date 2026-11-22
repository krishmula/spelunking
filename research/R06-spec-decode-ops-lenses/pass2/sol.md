# Pass 2: codex gpt-6.1-sol (provider default effort), completed 2026-10-20

Verdict: C3' and C9' survive only as MEASURE-FIRST; C6' and C7' weak. No pass-1 kill overturned.

## C3'
- Rejection: detector mechanism already SpecGuard (windowed acceptance deviation, recalibration) [L216 full text]; no rejection of the empirical question of incremental coverage over cheap checks.
- Key attacks: Combination Open: an output change need not lower acceptance (argmax can move between two tokens neither proposed; corruption can raise agreement); Simplicity Open: CSNB's 99% is relative to ABFT, compare on same injections; Scale Open: per-device mix confounds; Oracle: measure full-trace separation with known onset first; Correctness Narrows: shared embed/lm_head/hidden states mean common-mode exposure (llm_base_proposer.py:1416); Feasibility: single-bit flips are <40% of real GPU corruption events.
- 1% per-window FPR is not a fleet target: one decision per minute -> 14.4 false alarms per healthy device-day.
- New sources: GPU SDC error-pattern study arXiv 2605.04213; ANL runtime-data SDC detection P5224; ANL adaptive impact-driven detection P5376.
- Compute: Air screen 1-2 days; validation tens to hundreds of GPU-hours (uncertain).

## C9'
- Rejection: PELM jointly controls frequency and speculation incl. full-verification ablation; AFlex shows attention and FFN have different optimal frequencies. No rejection of the narrowed datacenter lossless long-context question.
- Key attacks: Simplicity Open: strongest baseline is a profiled clock lookup for default-K spec by batch/context plus adaptive K; Correctness: measure realized clocks; Feasibility: DeepSeek-V2-Lite has no MTP (config deepseek_v2; vLLM mapping covers V3/V3.2, speculative.py:680).
- The card's "equal optimal clock" falsifier rejects the explanation, not the joint benefit.
- New sources: AFlex arXiv 2608.01891; DSO arXiv 2407.13096; DeepSeek-V2-Lite config.json; NVIDIA H100/H200 spec pages.
- Compute: 10-30 GPU-hours coarse screen; 50-200 for validation.

## C6'
- Rejection: scheduler diagnosis with known remedy (Nightjar discusses queueing-driven TTFT). Not settled.
- New sources: vLLM GLM-5.2 B300 blog; SGLang #34943; Dynamo #7753; Dynamo v1.3.0 release notes (MTP accept length scaling in planner).

## C7'
- Rejection: fixed batch per rank still changes experts per device, routing and communication; familiar scaling curve.
- New sources: TRT-LLM EP part 3 (MTP at EP32, local batch 32); SGLang #7998; SGLang MoonEP roadmap #35783.
- Compute: 80-240 GPU-hours for a one-node screen; multi-node hundreds to thousands.

Failed fetches: MDPI 429, IEEE robot check, IIT PDF 502.

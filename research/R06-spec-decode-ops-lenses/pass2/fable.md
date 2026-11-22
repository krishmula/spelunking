# Pass 2: claudeAgent claude-fable-5-1 (provider default effort), completed 2026-10-20

Verdict: C3' mechanism killed by prior art (IBM patent); remainder is a fault-injection measurement. C9' shift derivable by arithmetic; available margin 5-15%. C6' benchmark table. C7' unevaluable in timeline. No pass-1 kill overturned.

## C3'
- Rejection: US11599795B2 (IBM, granted 2023-03-07): a smaller lower-precision checker network whose historical agreement with the main network drops indicates accelerator fault; SpecGuard instantiates it on LLM acceptance. SGLang #27194: MTP acceptance length stayed high while gsm8k fell to 0.00 under a silent corruption (software, TP8/EP8/dp-attention/MTP).
- Adjacent: Kills the mechanism claim (patent). Combination Narrows: hidden-state-coupled drafters move with the target; only draft model or ngram independent; DSN'26 warp-aligned spatial correlation.
- Simplicity Narrows hard: scheduled greedy canary battery with golden outputs catches persistent per-device faults; acceptance can only win on input-dependent faults or time-to-detect between canaries.
- Scale: SpecGuard per-query FPR 3-10% at TPR 0.95, so hundreds of requests per alarm window minimum.
- Oracle Open: DSN'26 NaN/Inf only 1.01% of SDC outcomes (helps); L224 large deviations (hurts). Next: bin output-changing faults by activation-norm deviation with L223's injector.
- New sources: US11599795B2; sglang #27194; Ekka ICML'26 2606.04594; DSN'26 2605.04213; LLM-PRISM 2604.10390; Tian Pan spec-in-production blog; CoreWeave spec decode docs.

## C9'
- Rejection: GQA verify attention intensity 4(K+1) FLOP/B (4 at K=0, 24 at K=5, ~256 only with 64-node tree) vs ~295 ridge [L256], so stays clock-insensitive; weights matmul at (K+1)B covered by L190; remaining regime (MLA, 16K+, large batch) has only 5-15% clock savings [L190] and spec already costs +25.65% energy at batch 128 [L188].
- Correctness: Kills (a sweep, nothing to build).
- New sources: AFlex 2608.01891 (attention energy-optimal at lower frequency than FFN in 63.8% of cases); Tri-serve 2606.29629; AGFT 2508.01744; Camel 2508.09173; L190 reopened.
- Change my mind: optimal clock with EAGLE-3 at batch B differs from non-spec at (K+1)B by >1 clock step at 8K GQA, or a tree config crossing the ridge where spec still saves energy.

## C6'
- Rejection: draft_slots reserved from the shared input budget (scheduler.py:682, :826) shrink prefill chunks under load; off switch exists (Nightjar). Benchmark table.
- New sources: Red Hat gpt-oss spec blog (TTFT P95 regresses 9.3% at TP=2); Nightjar v5 reopened; 2507.09019; vLLM #62574.

## C7'
- Rejection: InferenceX pages state no TP/EP/DEP and interpolate, so the cheap reanalysis is closed; 8+ GPUs unconfirmed.
- SGLang #27194: per-rank dispatch buffer must cover batch/TP x (K+1) or output silently corrupts (correctness coupling).
- New sources: InferenceX H200 FP8 MTP page; TRT-LLM EP part 2; LMSYS GB200 part II; EVICT 2605.00342.

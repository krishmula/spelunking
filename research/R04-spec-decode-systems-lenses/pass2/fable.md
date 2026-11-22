# Pass 2: claudeAgent claude-fable-5-1 (effort: provider default), 2026-10-20

Condensed by the lead judge from the reviewer's final message (T3 task r04-pass2-fable-r1).

## C3'
- Shortest rejection: vllm-metal #734 ships lazy full-gap catch-up after K=0 (local draft_model_proposer.py:406,
  :639-651); SGLang #32673 and Windowed-MTP already measure sink-plus-window draft attention as acceptance-neutral for
  EAGLE and MTP at W about 4K. What remains (buffer vs recompute target hidden states for EAGLE-3) is a
  memory-versus-recompute engineering choice that K=0 gap lengths decide.
- Adjacent: Kills the windowing half (SGLang #32673 W=4032+64 sink unchanged acceptance at 32K-256K; tokenspeed #2355
  untrained W=128 costs about 10%).
- Simplicity: Narrows hard. Baselines: keep sync (4.1-11.3% during K=0 only); full lazy catch-up with a hidden-state
  buffer (about 24 KB/token for Llama-3.1-8B EAGLE-3, derived); recompute through target. Commenters on vLLM #49548
  report K=0 runs of 5-15 steps, so gaps are often tiny and full catch-up costs almost nothing.
- Scale: Open; no production K=0 gap-length data. vLLM #62740: K=0 still runs EAGLE drafters, K=0 table 19-22% slower
  than no-spec on H100.
- Oracle: Narrows; full lazy catch-up already reaches the bound; windowing adds only memory.
- Scoop: High (#53426 rebased today; SGLang roadmap lists lazy init; vllm-metal did draft_model in a week).
- Verdict: MEASURE-FIRST at best; remaining core is an engineering PR, not a thesis. The experiment would confirm
  the expected result (#32673 and vllm-metal #716 already support the stale-slot hypothesis).

## C7'
- Shortest rejection: no engine specifies exact acceptance after state events (EAGLE block drop, dropped in-flight
  spec tokens, #53426's accepted 14.6% loss), so an exact oracle flags intended behavior; the deliverable is a bug
  census of fast-moving codebases using a batch-invariant mode that costs 25% for EAGLE-3, is unmerged for spec decode
  (vLLM #52522), and was reported broken for EAGLE-3 in SGLang (#13123).
- Novelty: Narrows; #52522 uses forced preemption with exact token equality over 1,319 GSM8K prompts: 419/373/619
  mismatches for EAGLE3/DFlash/DSpark without its recovery path, 0 with.
- Adjacent: Narrows; CrashMonkey/ACE (OSDI 2018); GRIEF; Ekka (ICML 2026) diagnoses 17 vLLM silent errors;
  Correctness Forensics for Batch Speculative Decoding (EMNLP 2026 Findings).
- Simplicity: Narrows; pressure-induced preemption already exercised the recovery path in #52522.
- Oracle: Open but bounded; every 2026 bug in the cluster was found without an exact-acceptance oracle.
- Verdict: three Narrows; smallest core ("is greedy acceptance exactly reproducible across preempt-resume and K=0
  toggles under batch invariance in vLLM, and is any divergence a bug or a documented tradeoff") is a two-day
  measurement, a lab tweak feeding E02, not a thesis candidate.

## Pass-1 kills
All stand. TensorRT-LLM's acceptance_rate_threshold strengthens the C9 kill.

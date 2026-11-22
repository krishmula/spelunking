# Pass 2: codex gpt-6-astra (effort: provider default), 2026-10-20

Condensed by the lead judge from the reviewer's final message (T3 task r04-pass2-astra-r1).

## C3'
- Shortest rejection: none for bounded hidden-state catch-up. But lazy catch-up after K=0 already exists for separate
  draft models: vllm-metal PR #734 "Lazily catch up draft KV after zero-K steps", merged 2026-09-15 ("The missing
  range is ingested once"), prefix caching disabled while deferring. Present in the local checkout
  (vllm_metal/v1/draft_model_proposer.py:406, tests/test_draft_model_proposer.py:487).
- Missing baseline: retain valid pre-gap draft KV, buffer the missing inputs, replay the whole interval at resume
  (proposed upstream in vLLM #53420). Windowing must beat that, not only sync and skip.
- Oracle: 10% of time at K=0 times 4.1-11.3% sync surcharge = 0.4-1.0% total savings even with free rebuild.
- Correctness (Narrows): missing draft KV must not be admitted to the prefix cache; #734 disables deferral with
  prefix caching.
- Test design: sink must be retained, not rebuilt; hidden-state buffer costs capacity (64 req x 512 pos x 4096 x 2 B
  = 256 MiB, x3 for EAGLE-3 unfused); W<=512 should not be a scientific cutoff.
- Adjacent: PostgreSQL WAL standby catch-up with retention limits; Test-Time Speculation (2605.09329): acceptance
  declines with generation length regardless of events, so use position-matched controls.
- Status: MEASURE-FIRST.

## C7'
- Shortest rejection: exact acceptance is not an invariant of correct speculative execution; and open vLLM PR #52522
  (batch-invariant spec decode) already reconstructs the discarded proposal at the same logical boundary after
  preemption and adds an exact-output preemption test (EAGLE3, DFlash, DSpark).
- Correctness attack: Kills as stated (intentional state-loss policies are legitimate counterexamples).
- vLLM #54506: batch invariance does not cover the spec forward pass (M=1 vs M=k+1); acceptance ~100% to 72% from
  arithmetic alone.
- vLLM's acceptance test is one-sided (degradation check), not equivalence.
- Adjacent: CrashMonkey (OSDI 2018) bounded crash testing; vLLM RFC #28135 acceptance regression CI (closed not
  planned); SGLang PR #31516 adaptive spec CI with forced retraction (closed unmerged); TensorRT-LLM synthetic
  acceptance rates.
- Status: rejected as universal oracle; a narrow preservation-contract test remains possible.

No pass-1 kill reversed.

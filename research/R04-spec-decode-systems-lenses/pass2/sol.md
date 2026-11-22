# Pass 2: codex gpt-6.1-sol (effort: provider default), 2026-10-20

Condensed by the lead judge from the reviewer's final message (T3 task r04-pass2-sol-r1). Reviewer quotes are leads,
not reads; sources the lead opened are in ledger.tsv.

## C3'
- Shortest rejection: none established for bounded catch-up; but plain catch-up (replay all skipped positions) is
  already proposed in vLLM #53420 and priced by Nightjar, so the contribution must beat full skipped-interval replay.
- Simplicity (Open): strongest missing baseline = keep pre-gap draft KV, buffer every skipped target feature, append
  all missing positions in one batched replay. Exact, no approximation.
- Oracle (Open): saving <= f*s; with s = 4.1-11.3% sync cost and f = 0.1 of time at K=0, 0.4-1.1% overall.
- Scope: buffering features works only for K=0 (target forward observed). PD handoff and offload restore do not
  have the features; those need transfer or regeneration and must not inherit the K=0 result.
- Relevance (Narrows): drafters without own KV remove the problem structurally (H-Spec).
- Scoop (Narrows): vLLM #53426 tracks the rebuild path; TensorSharp documents rebuild after missing history.
- Window risk: Windowed-MTP shows a retrieval-sensitive case losing acceptance even at about 1K window.
- Status: MEASURE-FIRST.

## C7'
- Shortest rejection: exact acceptance is not a valid universal oracle: legitimate policy changes (skipped drafting,
  K=0, proposal-boundary shifts) change acceptance with correct state; vLLM's test already expects lower acceptance
  when drafting is skipped. Batch invariance cannot turn a performance policy into a state contract.
- Narrows to: at matched committed prefixes and proposal settings, can event-specific checks detect violations of an
  explicit draft-state-preservation contract that existing tests miss? Needs per-event contracts (preserve,
  recompute, discard) and a fresh-rebuild reference.
- Simplicity (Open): compare against comparing proposal tokens or draft KV directly.
- Adjacent: targeted recovery-path fault injection (EPFL, TOCS 2011) is old; vllm-metal #1035 already audits DFlash
  and DSpark continuations; vllm-metal roadmap #1005 plans preemption/rollback validation.
- Status: narrow and validate oracle first.

## New sources cited (to be opened by lead)
TensorSharp speculative docs; Stale View Cleaning (2015); Enzyme (2026); StreamingLLM repo; Attention Drift
(2605.09992); EPFL fault injection (TOCS 2011); vllm-metal PR #892; vllm-metal PR #1035; vllm-metal issue #1005;
LMSYS DSpark blog (2026-07-06); SGLang PR #22077; TensorRT-LLM speculative/interface.py; NVIDIA H100/L4 pages.
No pass-1 kill reversed.

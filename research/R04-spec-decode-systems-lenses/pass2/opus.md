# Pass 2: claudeAgent claude-opus-5-5 (effort: medium), 2026-10-20

Condensed by the lead judge from the reviewer's final message (T3 task r04-pass2-opus-r1).

## Pass-1 correction
- "Catch-up named, not built" [L227] is out of date: vllm-metal PR #734 merged 2026-09-15 (local commit 69e0d5a).
- vllm-metal's block drafter commits target features during K=0 (block_draft_proposer.py:286-287), so DFlash on
  Metal has no draft-state gap; a windowed rebuild would have to relax its "committed context is discontinuous" check.

## C3'
- Shortest rejection: #734 ships lazy full catch-up for draft_model; the sync it saves is about 2-3% on average
  (Cascade, MICRO 2025), only during K=0 phases, which occur only at high batch with dynamic speculation, which vLLM
  disables under DP and which does nothing on the V2 model runner (vLLM #51510). Remainder (can EAGLE-3 rebuild from
  a short window so the buffer is bounded) is an empirical point, not a systems contribution.
- Oracle: Kills the throughput half (2-3% of step time, only inside K=0). Acceptance half Open.
- Simplicity: Narrows toward Kill; batched full lazy catch-up removes most per-launch cost; windowing saves only one
  draft prefill over the missing range. Buffer memory: EAGLE-3 3 layers x 4096 x bf16 = 24 KiB/token vs 128 KiB
  target KV, about 19% of KV for a long K=0 phase.
- Relevance: Narrows; SGLang adaptive tiers allow positive ints only (no K=0 tier).
- Scoop: High; #53426/#53420 author says follow-up "in preparation".
- Would change mind: a non-DP trace where sync is well over 5% of total time; or EAGLE-3 sink+W<=256 within 5% of
  synced acceptance while buffer memory rules out full lazy catch-up.

## C7'
- Shortest rejection: vLLM CI already forces preemption with spec in fp32/highest precision with FlexAttention,
  exact tokens, logprobs rtol 1e-3, acceptance within 5% (one-sided). Extending it is a test PR; GRIEF supplies the
  framework. Fails only if the tool finds a new, important bug class.
- Simplicity: Narrows; add MTP/DFlash/draft_model/DSD/prefix configs to the existing parametrized test and tighten.
- Oracle: Open. Best evidence of escaped bugs is [L143] (DFlash2 acceptance stuck at 0% after a prefix hit, open,
  no test). Injected events must include prefix-cache hits.
- tests/v1/distributed/test_eagle_dp.py:40-50 already runs EAGLE under VLLM_BATCH_INVARIANT=1.
- Would change mind: exact acceptance reproducibility without events for two methods, plus three or more new
  divergences the extended parametrized test would miss; a root-cause taxonomy would lift it.

## New sources cited
vllm-metal PR #734; Cascade (arXiv 2506.20675, MICRO 2025); vLLM DSD docs stable; vLLM #51510; SGLang adaptive docs;
arXiv 2604.26412; arXiv 2609.33184; vLLM batch invariance docs; local test files.

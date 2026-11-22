# Pass 2: claudeAgent claude-fable-5-1 (condensed; full reply in the T3 delegated task r05-pass2-fable)

Tools: WebSearch, WebFetch, local read-only. Fetch failed: gruss.cc remote_dedup.pdf (TLS); NDSS page opened
instead. All seven pass-1 kills stand.

## C5': KILL as a thesis direction; survives only as a measurement note or disclosure
- Shortest rejection: 2411.01076 v4 already has the malicious-user probe, the accepted-token observable, the
  extend-the-leak strategy, and names datastores "collected from user interactions, which may be leaked by our
  attack". Swapping REST for Arctic's global tree changes the structure, not the channel. The defence is a
  config-level tenant key whose cost [L126] [L248] predict near zero.
- Adjacent: Kill. CRIME/BREACH and memory dedup; Remote Memory-Deduplication Attacks (NDSS 2022).
- Simplicity: Kill pending one measurement; expects partition cost near zero.
- Scale: the card's worry is wrong. Snowflake claims upgraded suffix decoding beats n-gram at every batch size, and
  vLLM DELTA output emits each step's `new_token_ids` (`output_processor.py:686`, `:425`), so streamed chunk counts
  equal accepted counts exactly. Channel is noise-free, so "AUC at batch 16" is uninformative.
- Oracle: card premise wrong. `_speculate_path` starts at prob 1.0 and multiplies only at branch points; a span seen
  once from a unique context is proposed at 1.0. Single-occurrence spans leak token by token (BREACH pattern).
- Relevance: survives (Snowflake blog: "a single shared suffix tree caches responses from previous requests").
- Scoop: low. No vLLM issue on suffix isolation; vLLM main suffix proposer still keys by `req_id`. PrefixWall
  (2603.10726) is prefix-cache only.
- Change mind: tenant-scoped tree loses a large share of acceptance on a real multi-tenant agentic trace, or a
  mechanism that keeps cross-tenant sharing with bounded leakage (DP or threshold-gated insertion) with measured
  acceptance. Either turns an advisory into a systems trade-off.
- New sources: NDSS 2022 remote dedup; PrefixWall 2603.10726; Time Will Tell 2412.15431; 2601.17569; AgSpec
  2610.01108; vLLM main suffix_decoding.py; vLLM issue search.
- Compute: runnable now; under 2 GPU-hours live, CPU hours offline.

## C2': KILL on relevance and simplicity; blocked on compute
- SGLang PR #20698 (ship separate drafter KV under PD) closed unmerged 2026-10-04: "The suitable scenarios sound very
  niche." Arithmetic (7% FLOPs vs 78% bytes) decides it. vLLM push connector blog (2026-10-14): TTFT 1,744 ms to
  540 ms from pull to push, no spec; connector design dominates. SmartGen 2607.28150.

## C9': KILL on novelty
- vLLM RFC #48202 closed completed 2026-08-11 ("a single shared K overpays low-acceptance requests and underpays
  high-acceptance ones"); SGLang #23705 schedules per-request policy; FASER 2604.20503. Disparate Impacts
  2510.02128 is batch 1. Change mind: per-request K fails to remove harm because CUDA-graph padding keeps batch-wide
  shape.

# Speculative decoding through systems lenses: no surviving gap

2026-10-20 · Status: KILL (no survivor) · Gap confidence: n/a

## Record and scope

- R04, topic T-02 speculative decoding optimization. Lenses supplied by Krishna: L-01 checkpoint/restart, L-02 live
  migration, L-03 preemption and recovery, L-04 scheduling and admission control, L-05 caching and memory hierarchy.
- Constraints: masters-thesis direction, 5-day topic block, MacBook Air M5 16 GB with vLLM 0.31.0 + vllm-metal,
  Colab GPUs, larger GPUs unconfirmed. Details in `constraints.md`.
- Prior records: R01 and R03 (KV cache, T-01) did not touch speculative decoding. R03's preemption sources were
  rechecked against the spec-decode code, not reused blind.
- Coverage: 297 ledger sources (287 opened), six frontier sweeps, a read of local vLLM and vllm-metal code
  (`frontier/e-code.md`), three pass-1 attack sweeps, a four-model pass-2 panel. Every lens was investigated; live
  migration only briefly, because its one new piece (draft KV) reduced to another candidate. Limits are listed in
  `constraints.md`.

## Summary

Speculative decoding adds state to a request that is not needed for correct output: the drafter's KV cache, the
target hidden states it reads, suffix trees, and per-draft recurrent scratch. The lenses ask what happens to that
state when a request is paused, moved, preempted, restored, or rescheduled. That turned out to be the right place
to look and the wrong time: the engines are working through exactly these questions in public, PR by PR, in 2026.
Every candidate either has a merged or open upstream answer, or its best possible gain is a few percent of time in a
regime few deployments use.

Precisely: nine candidates, seven killed in pass 1 (prior work, shipping code, or an oracle bound), two reformulated
survivors killed in pass 2 by all four reviewers. No experiment is recommended as a thesis kill-test. Three short
lab measurements remain worth doing for learning.

## Why the lenses came up empty

- **Preemption and checkpoint/restart.** vLLM frees draft KV with target KV and rebuilds both on resume (e-code §2).
  Losing draft state changes acceptance, never output [L224]. The size of that loss is already measured: MTP keeps
  about 98% of acceptance, DFlash loses 5-15%, EAGLE-3 collapses when its slots go stale [L225] [L226] [L227].
  Drafters that read the target's KV have no state to lose [L220] [L221] [L230].
- **Live migration.** Migrating draft KV alongside target KV is done [L233] [L234], and Llumnix-style staging still
  works for the committed prefix [L59].
- **Scheduling and admission.** Speculation length by batch, context, request, and SLO is shipped or published
  [L31] [L43] [L45] [L46] [L214] [L284]; adaptive length gains 7-15% over the best fixed length [L38] [L39] [L42].
- **Caching and memory hierarchy.** The prefix-cache loss from EAGLE's block drop is being fixed upstream (72.7% to
  85.0% hit rate) [L140] [L141]. Suffix-cache size barely moves acceptance [L237]. Hybrid recurrent rollback has
  seven published or merged methods [L84] [L86] [L166]-[L168] [L171] [L87]. Draft KV is about 3% of KV for
  one-layer EAGLE-3 [L210].

## Rejected candidates

| Card | Question | What ended it |
|---|---|---|
| C1 | Reclaim per-request speculation memory before preempting | About 3% of KV for EAGLE-3 [L210]; hybrid scratch is a sizing bug [L206]; remainder merged into C3' |
| C2 | Recover prefix hits lost to EAGLE's block drop | Upstream lookahead hashing PR [L140] [L141] |
| C3 | Keep, move, recompute, or drop draft state after a state-moving event | Drop cost already measured [L225]-[L227]; KV-reading drafters make it moot [L220] [L230]; narrowed to C3' |
| C3' | Windowed draft-state catch-up when speculation resumes after K=0 | Lazy full catch-up merged in vllm-metal [L258]; upside 2-11% of step time only inside K=0 phases [L226] [L259]; windowing answered at W about 4K [L279] [L219] |
| C4 | One budget for verification and prefill chunks | SLOs-Serve [L214]; adaptive verification prices co-scheduled prefill [L215] |
| C5 | Capacity, eviction, persistence of suffix draft caches | Acceptance flat over 40x cache size [L237]; LRU vs LFU 0.005 [L238]; tree loss is a bug fix [L242] |
| C6 | Live migration of a speculating request | RLHFSpec, NebulaSD [L233] [L234] |
| C7' | Exact acceptance equivalence as a bug oracle | Not an invariant of correct execution [L227] [L254]; near-exact preemption test exists or is open [L254] [L260]; oracle may not be exact on any backend [L261] [L267] [L290] |
| C8 | Cheaper recurrent-state rollback for hybrids | Crowded [L84] [L167] [L168] |
| C9 | Context- and request-aware speculation length | Open RFCs and shipped controllers [L135] [L136] [L19] [L43] |

## Evidence against the conclusion

- Gap-length data for K=0 phases in real serving does not exist [L286] [L287]. If a production trace showed long,
  frequent K=0 gaps on non-DP deployments, C3' would deserve another look (Opus, Fable).
- No one has tested whether EAGLE-3 recovers from a clean sink-plus-window rebuild at W in the low hundreds. Prior
  results predict yes [L279] [L280], so the measurement would most likely confirm, not discover.
- A step-injection harness could still find a bug class pressure-based tests miss. No source shows one, and the
  2026 bug cluster was found without it.

## Experiments already run

None. R04 designed nothing to run before this decision.

## Lab measurements worth doing (not thesis kill-tests)

Each needs Krishna's prediction first. None is an E record yet.

1. **Does upstream vLLM lose a suffix-decoding request's tree on ordinary preemption?** The code destroys the tree
   when a request leaves the batch (e-code §2), and a vendor fork measured acceptance 97.4% to 4.5% under pipeline
   rotation [L242]. Colab, about an hour. Outcome: a bug report upstream if it reproduces.
2. **Is greedy acceptance on vllm-metal identical with and without a forced preemption at step n?** Uses E02's
   scheduler subclass with dflash or draft_model. vllm-metal is not greedy-exact against mlx-lm [L267] and has no
   batch-invariant mode [L277], so first check that two uninterrupted runs match. Air, an evening.
3. **EAGLE-3 acceptance after a clean sink-plus-window rebuild, W from 16 to 512.** Colab, 5-10 GPU-hours. Teaches
   how much history an EAGLE drafter uses; expected answer is little [L279] [L272].

## Recommendation

Close R04 and move the block schedule on to T-03 (GPU kernel optimization). If Krishna wants another pass at
speculative decoding, use a different lens set. The unified-memory angle (draft and target share bandwidth on Apple
Silicon) already has negative results [L195] [L196], so it needs a new question, not a rerun.

## References

Every cited row is in `ledger.tsv` with its URL, fetch date, and verbatim quote. Primary sources only; no secondary
source is cited here.

# R05 constraints

- R identifier: R05. Started 2026-10-20. Status: open.
- Topic: T-02, speculative decoding optimization.
- Lenses (supplied by Krishna, 2026-10-20): L-06 resource disaggregation, L-07 leases and reservations, L-08
  speculative execution, L-09 work stealing, L-10 load balancing and elasticity, L-11 isolation and multi-tenancy.
  Candidate generation uses only these. Searches for answering work and baselines may go beyond them.
- Kstack: plugin version 0.2.0, commit ed72d30.
- Engine clones: vllm-core `db9527a46` (v0.31.0), vllm-metal `572322e`.

## Goals and timeline

- A masters-thesis direction. At least 8 weeks of nightly work from 2026-10-11. Topic blocks of about 5 days,
  rotating KV cache, speculative decoding, GPU kernels, GPU underutilization (Krishna, 2026-10-14).
- Each block ends with a GPU test plan for its survivor. One-year-thesis defaults apply: concrete mechanism,
  reproducible evaluation, strong baseline, measurable claim, open-source path, first kill-test within about a month.
- Nothing runs before Krishna writes a prediction (lab rule). This search designs kill-tests; it does not run them.

## Compute

- Local: MacBook Air M5 (8-core GPU, 16 GB unified memory, fanless). vLLM 0.31.0 + vllm-metal, MLX 0.32.1.
  Models Qwen3 0.6B to 4B, 4-bit. vllm-metal spec decode supports Gemma4 MTP, draft_model, ngram, dflash, dspark;
  greedy only, no async scheduling, no DP/PP with spec (R04 `frontier/e-code.md` §6).
- Colab GPUs (vLLM on CUDA) are available for initial experiments since 2026-10-12 (`colab/`). Single GPU only.
- Real GPUs may come if experiments narrow the topic (Krishna, 2026-10-14). Access unconfirmed. Multi-GPU and
  multi-node access is unconfirmed, which matters for disaggregation and load-balancing claims.
- Measured (E00): Qwen3-4B 4-bit, 1k decode step = 17.6 ms weights matmul, 1.5 ms attention, 5.1 ms small kernels.

## Prior records

- R04 (same topic T-02, lenses L-01 to L-05) closed today with no survivor. Its kills stand for those lenses:
  draft-state loss on preemption, live migration of draft KV, adaptive speculation length, suffix-cache eviction,
  hybrid rollback, prefix-cache block drop. A candidate here that reduces to one of those questions is not new.
  R04's ledger (297 rows) is used as leads only; any source cited in R05 was reopened in this session.
- R01 and R03 are T-01 (KV cache). E02 (preemption re-hit) is unrelated except for the logging-scheduler harness.
- R02 was deleted and is not cited.

## Coverage

| Lens | Sweep file | Ledger ids | Rows | Unverified | Thinnest area |
|---|---|---|---|---|---|
| L-06 resource disaggregation | a-disaggregation.tsv | L1-L42 | 42 | 1 | Draft state across PD: no measured comparison of ship vs re-prefill |
| L-07 leases and reservations | b-reservations.tsv | L43-L76 | 34 | 1 | Shrinking K under KV pressure: one untested issue (L45) |
| L-08 speculative execution | c-specexec.tsv | L77-L106 | 35 | 2 | Cost of all-accepted async planning: no published accounting |
| L-09 work stealing | d-workstealing.tsv | L107-L142 | 37 | 3 | Shared n-gram/suffix pools split across replicas |
| L-10 load balancing and elasticity | e-loadbalance.tsv | L143-L182 | 42 | 0 | Acceptance-aware DP routing: no router uses acceptance |
| L-11 isolation and multi-tenancy | f-isolation.tsv | L183-L220 | 38 | 1 | Leakage through user-derived draft caches is warned of, not tested (L183/L184) |

Code reading (vllm-core db9527a46, vllm-metal 572322e) is in frontier/g-code.md. Total: 228 rows, 220 merged ledger rows, 8 unverified.

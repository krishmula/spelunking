# R06 constraints

- R identifier: R06. Started 2026-10-20. Status: closed 2026-10-20 with one weak MEASURE-FIRST survivor (C3''), not run.
- Topic: T-02, speculative decoding optimization.
- Lenses (supplied by Krishna, 2026-10-20): L-12 consistency and ownership transfer, L-13 backpressure and
  queueing, L-14 fault tolerance, L-15 topology changes, L-16 hardware-aware placement, L-17 power/capacity
  constraints. Candidate generation uses only these. Searches for answering work and baselines may go beyond them.
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
  multi-node access is unconfirmed, which matters for topology, fault tolerance, and placement claims.
- Measured (E00): Qwen3-4B 4-bit, 1k decode step = 17.6 ms weights matmul, 1.5 ms attention, 5.1 ms small kernels.

## Prior records

- R04 (T-02, lenses L-01 to L-05) closed with no survivor. R05 (T-02, lenses L-06 to L-11) has one weak
  MEASURE-FIRST survivor, C5'' (suffix-cache tenant leak), not run. Their kills stand for their lenses. A candidate
  here that reduces to one of them is not new. Their ledgers are leads only; any source cited in R06 is reopened
  in this session.
- R01 and R03 are T-01 (KV cache). E02 (preemption re-hit) is unrelated except for the logging-scheduler harness.
- R02 was deleted and is not cited.

## Coverage

Frontier: six lens sweeps (frontier/a-f, L1-L215, 5 unverified) plus a read of local vLLM and vllm-metal code
(frontier/g-code.md).

| Lens | Sweep file | Ledger ids | Status | Thinnest area |
|---|---|---|---|---|
| L-12 consistency and ownership transfer | a-consistency.tsv | L1-L38 | Investigated | Paired stale-drafter measurement (old drafter vs old and new target) |
| L-13 backpressure and queueing | b-backpressure.tsv | L39-L74 | Investigated | Output path (SSE pacing, frontend CPU) with spec; queueing beyond saturation |
| L-14 fault tolerance | c-faults.tsv | L75-L117 | Investigated | Fault injection into spec decode; verification as an SDC detector |
| L-15 topology changes | d-topology.tsv | L118-L152 | Investigated; SGLang issues partial (GitHub rate limit) | Spec gain vs EP degree; MTP in EPLB |
| L-16 hardware-aware placement | e-placement.tsv | L153-L186 | Investigated | Controlled drafter precision; drafter on ANE |
| L-17 power/capacity constraints | f-power.tsv | L187-L215 | Investigated | Spec under power caps or clock locking on GPUs; measured J/token on Apple silicon |

Uninvestigated: TensorRT-LLM and llama.cpp internals beyond issues and docs; vendor-private fleet practice.

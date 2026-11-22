# R04 constraints

- R identifier: R04. Started 2026-10-20. Status: closed 2026-10-20, no survivor.
- Topic: T-02, speculative decoding optimization.
- Lenses (supplied by Krishna, 2026-10-20): L-01 checkpoint/restart, L-02 live migration, L-03 preemption and
  recovery, L-04 scheduling and admission control, L-05 caching and memory hierarchy. Candidate generation uses only
  these. Searches for answering work and baselines may go beyond them.
- Kstack: plugin version 0.2.0, commit ed72d30.

## Goals and timeline

- A masters-thesis direction. At least 8 weeks of nightly work from 2026-10-11. Topic blocks of about 5 days,
  rotating KV cache, speculative decoding, GPU kernels, GPU underutilization (Krishna, 2026-10-14).
- Each block ends with a GPU test plan for its survivor. One-year-thesis defaults apply: concrete mechanism,
  reproducible evaluation, strong baseline, measurable claim, open-source path, first kill-test within about a month.
- Nothing runs before Krishna writes a prediction (lab rule). This search designs kill-tests; it does not run them.

## Compute

- Local: MacBook Air M5 (8-core GPU, 16 GB unified memory, fanless). vLLM 0.31.0 + vllm-metal, MLX 0.32.1.
  Models Qwen3 0.6B to 4B, 4-bit. vLLM's spec decode proposers live in vllm-core `vllm/v1/spec_decode/`;
  vllm-metal ships only a gemma4 spec-decode module, so EAGLE or draft-model support on Metal is unverified.
- Colab GPUs (vLLM on CUDA) are available for initial experiments since 2026-10-12 (`colab/`).
- Real GPUs may come if experiments narrow the topic (Krishna, 2026-10-14). Access unconfirmed.
- Measured (E00): Qwen3-4B 4-bit, 1k decode step = 17.6 ms weights matmul, 1.5 ms attention, 5.1 ms small kernels.

## Prior records

- R01 (KV cache robustness) and R03 (KV cache design space) are T-01. R03 survivor C2' became E02
  (preemption re-hit in vLLM V1). Its sources on preemption (all-or-nothing, recompute, no swap in V1) are reused
  here only after rechecking against spec decode. Nothing in R01 or R03 studied speculative decoding.
- R02 was deleted and is not cited.

## Coverage

Frontier: six subagent sweeps (frontier/a-f, L1-L201) plus a read of local vLLM and vllm-metal code (frontier/e-code.md).
Pass 1: three hostile sweeps (frontier/g-i, L202-L257).

| Lens | Status | Evidence | Limits |
|---|---|---|---|
| L-01 checkpoint/restart | Investigated | PD retraction checkpoints [L65]-[L68], offload/restore bugs [L81]-[L83] [L101]-[L103], suffix cache persistence [L237]-[L240] | Spot/failure recovery with spec (MorphCloud, fetch failed) unverified; engine-restart warm start not studied |
| L-02 live migration | Investigated briefly | Llumnix [L59], RLHFSpec [L60] [L233], NebulaSD [L234] | No serving-side spec migration system beyond these |
| L-03 preemption and recovery | Investigated | vLLM preemption code (e-code §2), [L64] [L145] [L204] [L226] [L227] [L254] [L256] | No measurement of spec-specific resume cost on real traffic |
| L-04 scheduling and admission | Investigated | [L31]-[L58] [L129]-[L136] [L213]-[L216] | RL rollout scheduling checked only for crowding |
| L-05 caching and memory hierarchy | Investigated | prefix cache under spec [L97]-[L100] [L137]-[L143], draft KV size [L93] [L210], suffix/n-gram caches [L107]-[L113] [L237]-[L244], hybrid state [L84]-[L88] [L166]-[L173] | Draft-cache tenant isolation only one source [L115] |

Uninvestigated: TensorRT-LLM internals beyond docs [L152]; llama.cpp server internals beyond docs [L153].

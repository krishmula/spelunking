# Constraints for the gap search

Written 2026-10-12. Source: Krishna's request of 2026-10-12, plus facts already on file in
`../R01-kv-cache-robustness/constraints.md` and `../../experiments/E00-serving-cost/README.md`.

## Domain

The KV cache side of LLM serving, broadly: any mechanism that decides how KV is stored, laid out, sized, allocated,
shared, evicted, compressed, moved, or scheduled. Krishna wants to change one such mechanism at a time (for example,
swap the algorithm a KV cache uses for another) and measure how performance changes, looking for changes that are
new or only shallowly studied. Algorithm swaps are one example, not the whole scope (Krishna, 2026-10-12).

Papers Krishna has already read: vLLM/PagedAttention, Mooncake, Llumnix, RetrievalAttention, FlexiCache, Strata.
Candidates should go past these, not restate them.

## Relation to other topics

- R01 (KV cache robustness under shift) is a separate, still-open topic. Its surviving question (prefix-cache
  bistability, C2') is not repeated here.
- R02 was deleted on 2026-10-12 and is not cited.

## Assumptions (not confirmed by Krishna)

- Outcome: a masters-thesis direction, at least 8 weeks of nightly work from 2026-10-11. One-year-thesis defaults
  from the skill apply: concrete mechanism, reproducible evaluation, strong baseline, measurable claim, open-source
  path, first kill-test within about a month.
- Hardware: MacBook Air M5 (8-core GPU, 16 GB unified memory, fanless), local only, no paid GPUs. Engine: vLLM
  0.31.0 + vllm-metal, MLX 0.32.1. Models: Qwen3 0.6B to 4B, 4-bit. Claims about NVIDIA multi-GPU deployments
  need a university GPU allocation and are flagged per candidate.
- Measured facts (E00): Qwen3-4B 4-bit, 1k decode step = 17.6 ms weights matmul, 1.5 ms attention, 5.1 ms small
  kernels; at 8k context attention grows to 11.6 ms. Attention is a small share of the step at short context.
- Nothing runs before Krishna writes a prediction (lab rule). This search designs kill-tests; it does not run them.
- Artifact: Markdown in this folder.

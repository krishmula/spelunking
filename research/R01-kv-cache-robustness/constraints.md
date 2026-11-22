# Constraints for the gap search

Written 2026-10-11. Source: Prof. Sengupta's recommendations, relayed by Krishna, plus the hardware facts in
`../R02-vllm-metal-serving/constraints.md`.

## Domain

Robustness of LLM-serving KV cache optimizations (compression, eviction, prefix reuse, offload, sharing between
tenants) under noisy, shifted, and out-of-distribution workloads. The question is how far each optimization's
average-case gain falls in the worst realistic case, and what keeps a production system stable when it does.

Focus set by Krishna on 2026-10-11: this is a systems performance and reliability topic, not a security topic.
Security (side channels, adversarial input) is one possible aspect: adversarial input is one stress case among
several. Attack construction is out of scope; published attacks appear only as prior work.

## Advisor's stated interests

- Prefix-cache attacks tied to KV cache eviction (named as a key interest).
- Subtle manipulation of KV compression to cause rapid or disproportionate resource consumption and force maximum
  KV occupancy under noisy or adversarial conditions.
- Hardening: which optimizations keep the system reliable under distribution shift and OOD input; an ensemble of
  optimization schemes probed many ways.
- Market angle: who would exploit this, who benefits from fixing it, where the commercial or funding value is.

## Assumptions (not confirmed by Krishna)

- Outcome: a research result that could become a paper or thesis chapter with Prof. Sengupta. One-year thesis
  defaults from the skill apply: concrete mechanism, reproducible evaluation, strong baseline, measurable claim,
  open-source path, first kill-test within about a month.
- Hardware: MacBook Air M5, 16 GB, local only, running vLLM 0.31.0 + vllm-metal and MLX. Small models (Qwen3 0.6B
  to 4B, 4-bit). No confirmed NVIDIA access. A university GPU allocation would be needed for claims about
  multi-tenant NVIDIA deployments; that is flagged per candidate.

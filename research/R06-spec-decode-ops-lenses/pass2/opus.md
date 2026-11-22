# Pass 2: claudeAgent claude-opus-5-5 @ effort medium, completed 2026-10-20

Verdict: C3' and C9' no clean rejection but both narrow; C6' and C7' rejections stick. No pass-1 kill overturned.

## C3'
- Rejection (partly sticks): drafter shares the faulty device; EAGLE-3/DFlash consume target hidden states via combine_hidden_states (llm_base_proposer.py:520-561) and alias lm_head/embeddings; acceptance partly compares the target with itself. Only draft_model keeps independence, and even it shares hardware.
- Simplicity: strongest simple baseline is deterministic canary replay (greedy exact match on fixed prompts) plus fleet quarantine; acceptance beats it only on data-dependent faults canaries miss. SDCHunter (OSDI'26): microbenchmarks miss >60% of defective GPUs in training, so that class exists.
- Adjacent: KV/weight data at rest owned by checksums (2604.17249).
- Oracle: acceptance drop ~ argmax-flip rate x base acceptance (inference); no source for the in-range argmax-flip rate.
- Narrowed C3'': same-device draft_model; can acceptance detect data-dependent output-changing faults with fewer requests-to-alarm at 1% FPR than canary replay, CSNB, range checks; how much power EAGLE-3 loses via hidden states.
- New sources: SDCs in the Wild (OSDI'26); 2604.17249 KV bit flips; LLM-PRISM 2604.10390; L218 reopened.
- Kill condition proposed: range checks catch every argmax-flipping fault.

## C9'
- Rejection (mostly sticks): weights matmul at (K+1)B; L190 already classifies GQA batch-invariant / MLA batch-sensitive, so GQA+spec behaving like MLA is predicted by roofline; per-step SLO-feedback DVFS (DualScale, GreenLLM) spends spec's TPOT slack automatically.
- Simplicity: real baseline = per-step SLO-feedback clock controller with spec at default K.
- Narrowed C9'': gap between swept joint (K, clock) optimum and SLO-feedback controller with spec at default K; dies if <5% or if roofline predicts optimal clock within one step.
- New sources: DualScale (L259 reopened); L190 reopened; 2601.08539 (search summary only, not opened); FASER 2604.20503; 2609.29707 Blackwell; colonel-otto issue #36.

## C6'
- Rejection sticks: 2507.09019 already shows spec raises TTFT and P99 TPOT; threshold switch exists. Benchmark control, not thesis.

## C7'
- Rejection mostly sticks: needs 8-32+ H200 with no access path; reanalysis is not research; NeurIPS'26 MoE spec expert-count model.

Coverage limits: fetch returned page summaries; 2601.08539 not opened.

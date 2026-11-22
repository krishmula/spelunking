# R07 scope

Started November 17, 2026. Topics T-01 KV cache optimization, T-02 speculative decoding optimization, T-03 GPU kernel optimization and T-04 GPU underutilization. Selected lenses are L-05 caching and memory hierarchy, L-08 speculative execution, L-04 scheduling and admission control, and L-16 hardware-aware placement. The user supplied the topics and requested paper assumptions, author-identified gaps and future work. The lens selection is ours.

This is a purposive sample of public papers with industrial lab participation and inspectable technical details. A paper qualifies through a lab coauthor. Independent papers and public engine code can refute a proposed gap. Lab size and citation count are not inclusion criteria. Coverage is concentrated on inference serving. Private fleet behavior, exhaustive lab publication lists, training cluster utilization and hardware other than NVIDIA receive limited or no coverage.

R01 studied prefix-cache recovery. R03 and E02 concerned reuse after preemption. R04 rejected generic speculation scheduling and several cache proposals. R05 and R06 retained narrow isolation and fault questions. E00's Apple step budget is useful for designing measurements but cannot establish a CUDA speedup. R07 starts from papers' stated conditions and studies different phenomena. Previous records are context, not evidence of novelty.

Prior project constraints suggest a masters thesis and early screening on a 16 GB MacBook Air or a single Colab GPU. No new deadline or GPU reservation was supplied. Specific CUDA GPU, checkpoint and trace access must be checked before timing. No new E record or experiment in this investigation.

## Coverage and source keys

KV includes eight anchor paper entries in frontier/kv.md. Speculative decoding and GPU kernels each have five in frontier/spec.md and frontier/gpu.md. DeepSeek-V3 is an additional architectural control in the dossier. The frontier keys K1–K14, S1–S11 and G1–G14 correspond in order to ledger entries L1–L14, L15–L25 and L26–L39. L40–L47 are later checks. Follow-up engine issues, code and vendor documentation are included as implementation evidence. These records do not establish actual performance on the user's hardware.

L-05 was investigated through retention, eviction, confidence and quantization. L-08 covered learned drafters, full and reduced vocabularies, and grammar-constrained verification. L-04 and L-16 covered within-step scheduling and GPU-specific implementation paths. Distributed placement, network utilization and training were only briefly checked or uninvestigated. OpenAI’s gpt-oss model card is an architecture check (K15 maps to L48). Anthropic papers were not systematically surveyed. The R record is closed at literature review and experiment design.

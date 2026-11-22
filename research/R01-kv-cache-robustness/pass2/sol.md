# Pass 2 review: codex gpt-6.1-sol (reasoning effort: provider default)

Verdict: no rejection of the exact question; MEASURE-FIRST.

Main objection: "slower service, longer revisit times, more eviction" is wrong as stated. vLLM evicts on allocation,
not on elapsed time, so with fixed reference order and fixed cache capacity, slowing references does not change LRU
reuse distance. C2 needs another causal link: active-request residency, scheduling order, preemption, routing, or
client behavior. Also, a recovery-time knee can come from plain backlog drainage, and an uncapped-cache reference
changes both miss work and capacity, so it cannot isolate the loop.

Attacks: Novelty Survives; Adjacent Narrows (memcache cold-cluster warmup, Aurora cache management, CDN request
collapsing); Combination Open (prefill/decode contention already modeled, arXiv 2602.02987); Simplicity Open
(bounded queues, uncached-token admission per Dynamo DEP #9755, S3-FIFO, retention priority, offload are mandatory
baselines); Scale Open; Oracle Open; Correctness Narrows (exclude software defects like L233); Feasibility Narrows;
Relevance Narrows (vLLM Router #285 "admitting more Programs than the KV pool can sustain causes cache thrashing");
Scoop Narrows (admission/retention controls converging: Router #285, vLLM #54779, Dynamo #9755, PRISM, PEEK).

Offload: Mooncake SSD retention prevents a hit-rate cliff in its benchmark (output fixed at one token, no
post-burst test); whole-hierarchy misses still recompute. Offload does not make loop gain universally below one.

Ledger correction: L242 full text reports block-identity traces over 7.0 and 130.9 days (to be released), so
"no public trace longer than an hour" must be softened to "no released trace".

Kill tests proposed: identical post-trigger workload, warm vs burst-triggered start; remove burst backlog while
keeping cache state; make recomputation free without adding memory; record active vs evictable blocks.

Sharper question: under fixed post-burst demand and finite shared capacity for active-request and reusable-prefix
KV, can miss-induced prefill/decode contention sustain additional eviction after the burst backlog is removed,
without retries or software defects; and what boundary separates recovery from persistent degradation when bounded
admission, cache-aware scheduling, and CPU/SSD offload are enabled?

Pass-1 kills: none overturned.

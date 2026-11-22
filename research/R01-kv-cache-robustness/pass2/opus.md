# Pass 2 review: claudeAgent claude-opus-5-5 (effort high)

Verdict: C2 as written does not survive; narrows to MEASURE-FIRST, low-to-medium gap confidence.

Shortest rejection: a prefix cache is read-through, so every miss refills the cache. HotOS'21 [L239] names
read-through as the self-healing variant ("the cache will still be filled, which steadily increases the hit rate
until the system is healthy again"). What remains is ordinary KV thrashing under overload, already characterized
(XPerf 2608.20370: hit rate "drops sharply once the request rate passes a threshold and then stabilizes at a lower
value"; Agentic AI Workload Characteristics 2605.26297: hit rate ~98% to 16-26% with concurrency, generation
throughput down up to 66%) and already fixed by feedback concurrency control (CONCUR 2601.22705: "middle-phase
thrashing", AIMD-like admission on KV usage and hit rate, up to 4.09x, offline batch only).

Remaining gap: every source is a steady-state sweep or offline batch. None steps load up then down in online serving
to see whether the down-path traces the up-path.

Real coupling (agrees with sol and astra): in-flight KV and idle cached prefixes share one pool; slowdown raises
in-flight memory (Little's law) and shrinks the effective cache. But max_num_seqs bounds in-flight memory.

Other points: Nyx (2606.00942) formalizes look-aside cache metastability and ignores key replacement; formal
recovery-time tooling exists (2510.03551); a queueing stability framework with KV memory exists (2605.04595, ICML
2026) without prefix caching; vLLM RFC #37003 "Agentic workloads break prefix caching under concurrent load";
offload-tier backpressure RFCs #50031/#53485 use high/low watermark hysteresis; offload does not make misses free
(2601.19910). Laptop cannot test the offload arm (unified memory, no PCIe tier).

Kill test: step-up/step-down arrival-rate sweep in a simulator calibrated with local vllm-metal prefill/decode
costs. Dead if both sweeps trace the same curve. Alive if a band of at least 15% of warm capacity has two stable
states and it persists with production-default max_num_seqs.

Sharper question: in online serving with open-loop arrivals, does a prefix-caching engine have a load band where
both a warm state and a thrashing state are stable; how wide is it for agentic multi-turn traces; and does online
feedback concurrency control remove the thrashing state at lower goodput cost than a static max_num_seqs cap or an
offload tier?

Pass-1 kills: none wrong; CONCUR and #37003 strengthen the C4 kill.

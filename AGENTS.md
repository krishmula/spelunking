# spelunking

## Glossary

**Topic**:
A broad area of systems research we investigate, such as KV cache optimization or speculative decoding optimization.
_Avoid_: vertical

**Block**:
A period spent investigating one topic through reading, code inspection, and experiments.

**Lens**:
A reusable perspective that directs attention toward mechanisms and questions within a topic.

**R record**:
The complete record of one research-gap-dossier run, including its topic, selected lenses, evidence, candidates, reviews, and decisions. The same topic and lenses can be investigated in separate R records.

**E record**:
The record of one experiment, including its plan, runs, results, and interpretation. Experiments arising from a dossier link to its R record and the candidate they test; one R record can lead to multiple E records.

**Batch size**:
The number of requests that share one decode step. Written "batch 8".
_Avoid_: concurrency, rows, x8

**Step time**:
How long one decode step takes, measured when no prefill shares it.
_Avoid_: ITL, inter-token latency

**TTFT**:
Time to first token: how long a request waits for its first output token. It measures prefill.

**TPOT**:
Time per output token: a request's decode time divided by its output tokens. It measures decode and equals step time unless other requests' prefill stretched some steps.
_Avoid_: time per token, time per output token

**Weights matmul**:
Multiplying the batch's numbers by one of the model's weight matrices. Its cost is mostly reading the matrix from memory.
_Avoid_: reading weights, GEMV, qmv

**Small kernels**:
Everything in a decode step that is neither a weights matmul nor attention: norms, RoPE, KV write, activation, sampling.
_Avoid_: KV write + other, overhead

**Weight matrix**:
One of the model's fixed grids of numbers, read every decode step and never changed. Qwen3-4B has 253: 7 per layer plus 1 at the end.
_Avoid_: weight grid, projection

**Weight-read floor**:
The fastest a decode step could ever be: the time to read every weight matrix once at full memory bandwidth.
_Avoid_: weight floor, roofline

**Step budget**:
A step time split into weights matmul, attention, small kernels and idle.
_Avoid_: the bill

**Workload name**:
Prompt length, then batch size, as in "1k prompt, batch 8".
_Avoid_: 1k x8, 1K-8req

**Forward pass**:
One run of the model over a set of tokens. Prefill is a forward pass over the whole prompt that makes the first token; each decode step is a forward pass too.
_Avoid_: iteration

**Decode step**:
A forward pass after the first token that takes each request's latest token and makes its next one, for every request in the batch.
_Avoid_: iteration

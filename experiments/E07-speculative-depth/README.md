# E07 Speculative verification depth by context length

Candidate: foundation  
Status: planned

## Question

Does assigning the deeper drafts to the four shorter requests change which depth gives the most committed tokens per second for the verifier?

## Why now

The handoff proposes testing whether verification cost at a fixed number of positions depends on where those positions go. This local test uses the Metal verifier. The D-cut CUDA implementation is a separate system.

## Prediction

| Quantity | Agent prediction |
|---|---|
| Verification time | At depth 8, deeper drafts on long requests take longer. |
| Best depth | The short requests can use at least as much depth as the long requests before throughput falls. A distinct, reproducible winner for each assignment advances the hypothesis. |

## The one change

Four requests have 1,024 cached tokens and four have 3,072. The shallow group receives one proposed token. The deeper group receives 1, 2, 3, 4, 6, or 8 proposed tokens per request. Depth zero is an additional uniform baseline. Each request verifies its current token plus the proposed tokens. Swapping the two groups holds total verified positions fixed at each depth.

Each group contains four requests with target acceptance limits of 0, 1, 3, and 7. The script generates greedy continuations before timing and changes the token at the first chosen mismatch. It replays the same proposal tape for each request across all depths. Each accepted draft and one target token count toward committed output. Both assignments must produce the same count at a given depth.

The timed step begins before packing token IDs and metadata and ends after retrieving the target's greedy choices and accepting their matching prefix. Target verification time includes the target forward pass, greedy selection, and synchronization. Prefix prefill, proposal generation, server scheduling, and network time fall outside the timed step. Draft-cost sensitivity adds a stated hypothetical cost for each sequential proposal position.

Use the default expanded verification layout. Capture the query shape and context length input to the native paged attention operation in every layer outside timing. Reject padded dispatch, missing layers, and acceptance differences. Compare twelve paired, shuffled rounds after a minute of warmup. Use the runner-up margin and paired round resampling across all candidate depths to judge the winner.

## How to run

Use the lab Python environment. Run `python run.py` from this directory with the GPU free. The runner fetches the pinned public checkpoint on a cache miss.

For an additional sensitivity check, apply a hypothetical conditional acceptance probability of 0.2, 0.4, 0.6, or 0.8 to the measured step durations. Both groups get the same probability. Expected accepted tokens per request sum the probabilities of accepting each successive prefix. This tests how much the chosen tape constrains the winner. Its expected rates are modeled.

A second run reverses the ordering of the two context groups in the packed batch. The initial run put short requests first. This follow-up checks whether the depth 8 contrast persists when long requests occupy the first four slots. All timings use the same repetitions and warmup.

The first run predicted a change in optimum for a hypothetical conditional acceptance probability of 0.8. A follow-up tests this with fixed acceptance limits of 0, 2, 4 and 8 in each group. These are the four midpoint quantiles of that geometric distribution, truncated to the eight proposed tokens. They give four representative requests rather than a random sample. Repeat the first run with these limits and short contexts first. Confirm the conditional result only if the two groups have distinct winning depths, each wins at least 95% of bootstrap resamples, and both paired 95% winner margins exclude zero. If it passes, repeat with long contexts first and require the same result. No proposal time is charged in this promotion criterion.

A final sensitivity run uses the Metal runner's opt-in merged verification windows with the higher-acceptance fixture and short contexts first. Each eight-request batch gets eight variable-length segments. Check emitted token IDs against expanded verification at every depth, before timing. Compare the two best depths with the same rule. A different result restricts the candidate to the layout on which it repeats.

## Runs

## Result

## Why

## Verdict

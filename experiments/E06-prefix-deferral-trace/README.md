# E06 Shared-prefix compression on recorded arrivals

Candidate: foundation
Status: initial sweep complete, held-out replication prepared

## Question

How often does compression need a preemption, and can two scheduler ticks of waiting avoid it through completion on a recorded arrival trace?

For the first window, both service costs gave the same preemption counts through completion.

| Capacity in pages | 497 | 591 | 709 | 945 | 1,417 | 14,472 |
|---|---:|---:|---:|---:|---:|---:|
| Immediate | 0 | 1 | 1 | 0 | 1 | 0 |
| Deferred | 0 | 1 | 1 | 0 | 0 | 0 |

At 1,417 pages the modeled p95 TTFT stayed unchanged. Modeled p95 completion latency fell by 38 ms at 10 ms per decode step and by 30.5 ms at 5 ms per decode step. The abnormally long modeled queues reflect the assumed prefill rate. These are scheduler results under an assumed service clock.

## Why now

E04's three-request replay creates pressure directly. Here the released Zipage scheduler admits, preempts and finishes requests itself. Mooncake provides real arrival times, prompt and output lengths, and common prefix block hashes. The online model runner requires CUDA. The local run therefore exercises its CPU scheduler and allocator with a simulated service clock.

## Prediction

| Quantity | Prediction |
|---|---|
| Blocked compression | Fewer than 5% of requests encounter the trigger. Absolute counts matter more than percentages if rare. |
| Through-completion preemptions | At least one pressure setting saves a preemption. Only advance to a CUDA timing run if a saving repeats at two capacities, both policies drain, and modeled p95 TTFT and completion latency do not increase by over 5% there. |

The initial sweep passed the latency checks but saved a preemption at only one capacity. Before running the held-out check, we chose requests 6,001 through 6,256 with the same relative capacities and service costs. Predict fewer than two capacities will save a preemption at both service costs. Report this check separately from the initial decision. Each window starts with an empty cache and no earlier requests in flight.

After this held-out replication, repeat the held-out window with prefill at 100,000 tokens/s and 1 ms per decode step. These faster assumed service rates test whether compressions still collide when fewer arrivals accumulate. Both arms must finish, the deferred arm must save through-completion preemptions at two or more capacities, and no capacity may add preemptions to advance a broader test. Record the fraction of modeled time serving requests, outstanding requests at the last arrival, and when preemptions occurred relative to arrivals. Neither the shared trace nor these service rates establish wall-clock latency. Predict that the lower-load run misses the advancement threshold.

## The one change

On failure to allocate shared-prefix compression destinations, postpone this sequence until the next scheduler call, at most twice per episode. Other requests can decode or finish. The immediate arm uses the upstream branch. Both arms use the same allocator, arrival schedule, generated tokens and request lengths. Completion order follows each schedule.

Use the first 256 published requests, fixed before replay. Sweep capacities at 1.05, 1.25, 1.5, 2 and 3 times the largest single prompt in pages. Also use an unconstrained control equal to the sum of all request pages including output. Round up and reserve one more page for an arriving sequence's partially filled page. Each page holds 256 tokens. Retain the default eight-page compressed budget, hybrid scheduling, 16 query slots, and a batch limit of 32. Use full 512-token hash blocks to recreate shared prefixes. Give each partial trace block a private token, since its shared length is unknown.

Use a simulated service clock with prefill at 2,000 uncached tokens/s and 10 ms per decode step, and run a sensitivity pass at 5 ms per decode step. Each pause and resume follows the scheduler. Compression service time is zero. These are assumed constants, so latency estimates are suitable only for comparison inside each model. No GPU attention scores or output quality are measured. Count preemptions, blocked compressions, empty ticks and completed requests in every arm. Preserve per-request completion and first-token times, and report the lengths of any unfinished requests.

## How to run

See replay.py. Supply a checkout of microsoft/Zipage at 3cf00ad547e5fe084fa640f431a31e02b45a2869 and the FAST25 conversation_trace.jsonl from kvcache-ai/Mooncake at 02d956868c90f0751bd1ad1ff77338202228f823. The script requires NumPy and xxhash. Run `python replay.py ZIPAGE_CHECKOUT conversation_trace.jsonl` for the first window. Add `--start 6001` for the held-out window. Repeat that command with `--lower-load` for the lower-load check. Each command writes a new timestamped run.

## Runs

| Run | Trace requests | Completed |
|---|---|---|
| 20261122-011017-trace | 1 through 256 | 256 in all 24 scenarios |

## Result

For the first window, both service costs gave the same preemption counts through completion.

| Capacity in pages | 497 | 591 | 709 | 945 | 1,417 | 14,472 |
|---|---:|---:|---:|---:|---:|---:|
| Immediate | 0 | 1 | 1 | 0 | 1 | 0 |
| Deferred | 0 | 1 | 1 | 0 | 0 | 0 |

At 1,417 pages the modeled p95 TTFT stayed unchanged. Modeled p95 completion latency fell by 38 ms at 10 ms per decode step and by 30.5 ms at 5 ms per decode step. The abnormally long modeled queues reflect the assumed prefill rate. These are scheduler results under an assumed service clock.

## Why

At 1,417 pages, request 225 needed two new pages with one available. Immediate compression preempted request 254. Deferral let request 210 finish during the two-tick pause and all requests drained without preemption. At 591 and 709 pages, both pauses expired and the scheduler still preempted a request. IDs here are zero-based within the window.

## Verdict

The first window misses the prespecified two-capacity threshold. Park the CUDA timing run on this evidence.

# E04 Shared-prefix compression timing

Candidate: foundation
Status: CPU replay

## Question

Can waiting up to two scheduler ticks avoid a preemption through completion?

## Why now

[Zipage §4.4](https://aclanthology.org/2026.findings-acl.381.pdf) allocates min(shared blocks, target blocks) before compression. A sibling finishing can remove that allocation requirement. The handoff supplies the hypothesis. R07 has no matching candidate. E03 is reserved in the E02 plan.

## Prediction

| Quantity | Codex prediction |
|---|---|
| Preemptions through completion | One fewer when initial free blocks are below the allocation requirement and the sibling finishes within two ticks. Equal otherwise. |

## The one change

When compression would require a preemption, the target can pause for up to two ticks and retry. Both policies keep the same four tokens, each modeled as a one-token block, from the same eight-token input. A sibling shares 0 to 8 leading blocks and holds one private block. An independent, last-arriving request holds four blocks and is the first eligible victim. The sibling finishes at tick 1, 2, 3, or 4. The target finishes at tick 5 and the independent request at tick 6. All remaining output fits existing storage or is EOS. A preempted request restores its full allocation after the target finishes. This isolates allocation counts under a fixed completion order. It does not measure the cost of pausing a request.

## How to run

Run `python3 replay.py` from this directory.

## Runs

| Run | Paired settings |
|---|---|
| 20261122-005931-grid | 216 |

## Result

Immediate compression caused 104 preemptions. Bounded deferral caused 52. Deferral saved one in 52 settings and tied in 164. Every request completed. For three shared blocks, one free block and a sibling finishing at tick 2, new destinations fell from three to zero, original blocks freed stayed at four, and preemptions fell from one to zero.

## Why

Waiting helped exactly when the sibling released its references by the deadline and the initial destination allocation exceeded free capacity.

## Verdict

Advance to an engine trace with real arrivals. Measure p95 TTFT and target completion latency to price the pause. The grid frequency is set by design.

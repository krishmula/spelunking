# E02 implementation grounding

The user authorized implementation of everything needed for the expanded E02 on 2026-10-20.
The canonical contract is `experiments/E02-preemption-rehit/README.md`.
E03 is reserved for the proposed shared-prefix victim-selection experiment. E02 changes observation only.
The repository has no commits and all existing content is untracked. Do not create a commit on main or a worktree from an empty baseline.
Do not read historical E02 runs or summaries. Preserve them and the earlier plan.

## Overview

The existing runner launches vllm-metal with an `AsyncScheduler` subclass. It has four configurations, 64 requests each, short generated math prompts, and no first-token timing.
It does not satisfy the expanded 20-configuration, five-repetition, 100-request contract.

## Key concepts

The core data is a run manifest, a frozen tokenized corpus, an ordered event history, request episodes, and physical block generations.
One physical block's reference count is shared across all requests owning that block.
Request scheduling progress is distinct from completed KV. Async scheduling advances progress before execution completes.
Recovered original blocks that became free at release differ from blocks still shared at release and equivalent KV created elsewhere.

## How it works

Pinned vllm-core `db9527a46873454610df6dbedf79a36d6bf1a7f6` selects the final request in `self.running` under FCFS in `vllm/v1/core/sched/scheduler.py` near lines 762-811.
It removes the victim before `_preempt_request`, which releases blocks, resets progress, and prepends the request to waiting.
Scheduling then increments `num_computed_tokens` and `num_in_flight_tokens` before execution completes.
Preemption marks in-flight outputs stale. Completion events are required to avoid counting scheduled work as completed KV.
Waiting admission can fail without causing preemption. Full prompt capacity is reserved before admission.
At 776 physical blocks of 16 tokens, block 0 is reserved and 775 are usable. Larger request targets may be admission-bound.

The block pool allocates from the free queue, evicts old hashes on reuse, and increments references.
Cache hits can remove refcount-zero blocks from the queue through `touch` without an allocation pop.
Release decrements references. Only zero-reference nonreserved blocks reenter the queue.
Unhashed blocks go to the head. Hashed blocks go to the tail. Request release walks the request's blocks in reverse order.
The cache can contain duplicate hashes, so hash equality or block ID alone does not prove original-KV survival.

## Where things live

- `experiments/E02-preemption-rehit/run_experiment.py` owns the existing runner, workload, client, and analysis.
- `experiments/E02-preemption-rehit/e02_scheduler.py` owns the existing observer subclass.
- `/Users/krishna/dev/vllm-lab/vllm-core/vllm/v1/core/sched/scheduler.py` owns scheduling and preemption.
- The sibling `async_scheduler.py` owns async output handling.
- `vllm/v1/core/block_pool.py` owns hashes, references, and free-queue mutations.
- `vllm/v1/core/kv_cache_manager.py` and `single_type_kv_cache_manager.py` own request block tables and cache lookup.
- `/Users/krishna/dev/vllm-lab/vllm-metal/vllm_metal/v1/model_runner.py` uses scheduler-hit blocks and skips those input tokens.
- `/Users/krishna/dev/vllm-lab/.venv/bin/python` is the local engine environment.

## Gotchas

Scheduler-call gaps are not GPU execution boundaries. Primary cost is completed re-prefill tokens.
Snapshot all eligible candidates immediately before each release, explicitly including the removed victim.
Candidate snapshots cannot prove unchosen outcomes. Do not implement an oracle or a new policy.
Replay must detect dropped or malformed events and validate against independent direct snapshots.
Client streaming chunks do not each represent one token. Record first output arrival including thinking and join stable client IDs to engine request IDs.
Keep the existing runner and observer for historical analysis before replacing their active implementations.
The user's numeric prediction is pending. Ask once asynchronously and keep implementing. Do not invent it or use elapsed time as approval.
Tests and technical correctness checks can run while predictions are pending. Main measurements and measurement pilots stay gated.

## Design rubric

Compare complete candidate structures on scientific validity, passive mechanism preservation, replay and provenance correctness, implementation size and interface depth, real local verification, and crash-safe reproducible operation.
Prefer a small number of modules around ownership and invariants. Make incomplete observations fail visibly.

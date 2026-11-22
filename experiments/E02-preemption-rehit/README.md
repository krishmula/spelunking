# E02: inspect preemption decisions and KV recovery

Candidate: R03-C2
Status: all three pilots passed on 2026-10-21 under the hardened source, and `check --main` is open. The main sweep has not started and needs Krishna's explicit instruction and a frozen budget. The pilot forecast is 84 hours.

## Question

At each preemption, does stock vLLM choose a least-computed request among eligible victims?
What other requests look worth testing as victims, based on their computed tokens, block ownership, and shared prefixes?
For the chosen victim, how much previously computed KV does the prefix cache recover, and how much work remains?

This is a measurement experiment. It changes no victim-selection, allocation, release, free-queue, or prefix-cache mechanism.
The primary artifact is a table for manual inspection of every decision, linked to a reconstructable history of all requests and blocks.
Alternative candidates are observations at a decision, not measured alternative outcomes.

## Why now

[R03's candidate C2](../../research/R03-kv-cache-design-space/candidates.md) asks whether prefix-cache recovery matters to victim selection.
The specific candidate is C2'. The Candidate line uses `R03-C2` because the lab checker accepts only numbered card IDs.
The [historical plan](historical-plan.md) records the earlier four-configuration scope, its predictions, and its recorded run.
This expanded plan replaces that scope. The earlier run stays under Runs and does not count toward the expanded sweep.
Its results were not read during formulation.

The existing logging records preemption snapshots and allocation-pop counts, but does not record every ownership or free-queue change.
Its request client also does not record first-token timing. The expanded scope needs additional instrumentation and a streaming client.
Verify these requirements before the main sweep starts.

## Prediction

These predictions apply to the expanded scope only. They are guesses, not findings.
Freeze predictions before any expanded-scope pilot or main measurement. The historical predictions remain in [historical-plan.md](historical-plan.md).
Krishna's prediction was recorded on 2026-10-20 before expanded-scope measurements.
Recovery predictions are conditional on observing resumed victims. A configuration with no preemptions has undefined recovery.
The frozen copy is [data/predictions.json](data/predictions.json), sha256 `3bcd8e66101466e2002c47c254cfcb68343bcca0602811ea8cc5f4cac5d89930`.
It keeps Krishna's recorded prediction separate from the Codex guesses below.
`python -m e02 check` and every measurement command refuse to run unless that digest matches, Krishna's recovery bounds are numbers with a reason, and every Codex row has a number and a reason.
Krishna gave no numeric preemption rate, so none is recorded.

| Who | Prediction | Reason |
|---|---|---|
| Krishna | Low pressure has fewer preemptions and recovery above 60%. High pressure has more preemptions and recovery of 40-60%. | Krishna expects recovery to decrease as pressure increases. This records the stated relationship; no more detailed causal explanation was supplied. |
| Codex | At batch-size target 4 with cap 512, fewer than 1 preemption per 100 requests. At target 32 with cap 4,096, more than 10 per 100. | Guess. The nominal pool holds 12,416 token slots. Four 1k prompts with short outputs fit; 32 such prompts already exceed it without sharing. Actual sharing and admission can change this. |
| Codex | The chosen victim is tied for least computed in fewer than 80% of decisions across pressure configurations. | Guess. The default rule uses running-list order rather than computed-token count. Record ties rather than selecting an arbitrary winner. |
| Codex | At least one pressure configuration contains resumed victims with recovery fractions below 0.2 and above 0.8. | Guess. Waiting and intervening allocations can differ between victims even with the same pool. |

## The one change

There is no policy treatment. The two swept factors are batch-size target and output cap.
The instrumented engine observes the stock mechanisms. Validate its behavior and measurement cost against the same engine without instrumentation.
Prefix caching stays enabled throughout the main sweep.

### Sweep numbers

| Factor | Values |
|---|---|
| Batch-size target | 4, 8, 16, 24, 32 |
| Output cap, tokens | 512, 1,024, 2,048, 4,096 |
| Configurations | All 20 combinations |
| Repetitions per configuration | 5 |
| Measured requests per run | 100 |
| Main sweep total | 100 runs, 10,000 requests |
| Repetition seeds | 11, 23, 37, 53, 71 |

The target controls the client's maximum outstanding requests and the server's `max_num_seqs`.
A target of 8 is not evidence of actual batch 8. Record actual batch size at each decode step.
Requests waiting inside the engine still count as outstanding. Replace each request that completes or fails immediately until all 100 have been submitted.
At the end, drain the remaining requests without replacements. Keep startup, steady submission, and drain identifiable in analysis.

### Fixed setup

- Use `mlx-community/Qwen3-4B-4bit`, thinking enabled, on vllm-metal. Record model and tokenizer revisions, the chat template, and rendered token IDs.
- Use one frozen, locally generated corpus of 100 distinct reasoning prompts derived from E02's seeded math-problem templates in `run_experiment.py`.
- Expand the problem statements with relevant premises and input data, not filler. Target 1,024 tokens after chat formatting, with an allowed range of 960 to 1,088.
- Use no shared system message. Begin each problem body with problem-specific data. Expect little full-block sharing and verify that expectation from token IDs.
- Freeze the corpus and its checksum before the pilot. Record its token-length distribution and actual shared-prefix lengths. Do not create identical padding to force sharing.
- Use the same corpus across all runs. For each repetition, use its seed to shuffle request order identically across the 20 configurations.
- Use temperature 0.6 and `top_p` 0.95. Assign each corpus item a stable sampling seed derived from its item number and repetition seed.
- Permit normal end-of-output stopping. The output cap includes thinking and answer tokens. Record actual output length, stop reason, and cap-hit rate.
- Start from E00's nominal 12,416-token pool. At block size 16, this is 776 physical blocks, including one reserved null block, leaving 775 usable blocks in the inspected engine.
- Pin that actual physical block count for every main run. Record physical, reserved, usable, and free counts separately. Do not rely only on `gpu_memory_utilization`.
- Keep `max_model_len` 9,216, `max_num_batched_tokens` 2,048, async scheduling, KV dtype, block size, and other scheduler settings fixed.
- Explicitly use FCFS scheduling and no speculative decoding, offload, or external KV connector. Record the loaded scheduler class and effective configuration.
- Assert `block_size == hash_block_size`, `prefix_replay_tokens == 0`, and `scheduler_reserve_full_isl == True` at startup. Record watermark and lookahead settings.
- Record engine SHAs, dirty diffs, instrumentation checksum, Python and library versions, machine, power state, thermal state, swap, and competing GPU activity.

Local source inspected during formulation was vllm-core `db9527a46873454610df6dbedf79a36d6bf1a7f6` and vllm-metal `572322e17584bb54afbc9e143f72009c2398bbba`.
The FCFS allocation-failure path in `vllm/v1/core/sched/scheduler.py` selects the final request in `self.running`.
Call this the last-admitted policy for continuity with R03, but record the actual running-list order at every decision.
Resumption can change that order. Recheck the rule against the recorded source before running.

The pinned scheduler reserves enough capacity for a full prompt before admission.
Without sharing, 13 prompts of even 960 tokens need 780 blocks, exceeding a 775-usable-block pool.
Targets 16, 24, and 32 may therefore differ mainly in waiting requests. Preserve the agreed sweep and report this admission limit alongside actual batch sizes.

## How to run

Workload execution is suspended. On 2026-10-20 at 14:33 PDT Krishna said "Don't run the experiment unless I tell you so."
Do not start any pilot, model server, warmup workload, or main sweep without a later explicit instruction for that workload.
`pilot` and `main` require `--execute` and refuse to create a run or start a process without it.
The flag records execution intent. It grants no authority to an agent.
Technical readiness, frozen predictions, and old pilot verdicts do not authorize execution.

The following commands perform offline checks. `technical` runs CPU scheduler fixtures and local fake HTTP endpoints, never model inference.

`run_experiment.py` and `e02_scheduler.py` implement the historical four-configuration scope and stay unchanged for that run.
They are not an entry point for the expanded sweep.
The expanded sweep runs through one command, `python -m e02`, from this directory with the engine's Python.

```sh
cd ~/dev/spelunking/experiments/E02-preemption-rehit
PY=~/dev/vllm-lab/.venv/bin/python
$PY -m e02 corpus --check
$PY -m e02 plan
$PY -m e02 technical
$PY -m e02 check
$PY -m e02 check --main
```

`check --main` must remain blocked until all three pilots have current compatible raw evidence.
Existing 20261020 pilots predate sealed plans and the current implementation fingerprint. Their saved verdicts are not gate evidence.
The behavior pilot can still be replayed offline. The cost-low pilot remains interrupted, and the cost-pressure pilot has never run.
A source or capture-setting change invalidates earlier pilot qualification. It does not invalidate preserved artifacts as records of that earlier implementation.

After a future explicit instruction, the workload command forms are `pilot KIND --execute` and `main ROOT --execute`.
The pilot kinds are `behavior`, `cost-low`, and `cost-pressure`.
Before main, `freeze --wall-hours H --disk-gib G --reserve-gib R --run-wall-min M --no-progress-s S` requires all readiness gates and a feasible pilot forecast.
`freeze` writes a sealed manifest, not a workload. No current main manifest or forecast is qualified.
`status ROOT`, `replay ATTEMPT_DIR`, `verdict ROOT`, and `report ROOT` perform offline analysis.
`report` writes under its input root, so use a scratch copy when preserving earlier artifacts.


Code lives in `e02/`: `corpus.py` (frozen prompts), `events.py` (journal schema and seal), `observer.py` (the scheduler subclass the server loads), `history.py` (pure replay, ledger, decisions, episodes, join, metrics), `sweep.py` (plans, gates, server, client, attempts, pilots, budget), and `report.py`.
Tests in `tests/` drive the real vllm-core `AsyncScheduler` and block pool with scripted model outputs.

Each run root holds `env.json`, a `plan.json` or frozen `manifest.json`, and `attempts/<run_id>.a<k>/` with `attempt.json`, `server.log`, `events.jsonl`, `snapshots.jsonl` and `client.jsonl`.
An attempt directory is created exclusively and keeps its outcome. A replacement is a new attempt that names the attempt it replaces and why. In the main sweep it needs `E02_REPLACE_REASON`.

Plans seal the complete execution configuration, deadlines, ceilings, implementation files, corpus, predictions, engine state, and technical-test fingerprint.
Inherited `E02_` and `VLLM_` settings cannot override the sealed server environment. The observer records its effective capture cadence.
Resume rejects changed source, settings, inputs, or engine state. It revalidates complete attempts from raw evidence and final evidence digests.
An attempt stays `validating` until validation completes. Validation, evidence hashing, and the final complete record share a bounded wall deadline. Final metadata must fit above the disk reserve. A crash in validation cannot make an attempt skippable as complete.

Wall use persists across attempts, failures, and resumes. Admission reserves the next attempt's full startup, warmup, measurement, seal, and shutdown allowance.
Checks during execution protect the root's disk ceiling and free-space reserve, including space for final records. Budget exhaustion leaves an incomplete attempt.
Cleanup identifies owned process groups and descendants, including surviving members after the server leader exits. It escalates owned survivors and blocks resume if ownership cannot be proved. Failed cleanup prevents completion and pilot qualification. Resume retains the process identities and prior stop reports, retries cleanup, and requires a replacement after cleanup succeeds. The terminal sweep checks its budget even when all attempts are complete.

Warmup uses at most 12 rounds of four requests, each forced to 128 output tokens, within 600 seconds.
It requires at least three rounds and a last-three-round throughput range within 5% of their median, without a reported thermal limit.
The attempt records all samples and the stability result before measurement. Validation recomputes this rule from the samples.
This is a host throughput stability criterion with thermal warnings, not a direct temperature measurement. Its suitability still needs live validation after explicit instruction.

Reports include `distributions.csv` for candidate values, recovery fractions, completed re-prefill tokens, and batch-size bins by forward-pass kind.
Batch time uses an unclipped host schedule-time proxy. An interval ends at the next schedule call, and the last interval ends at completion when acknowledged.
An unacknowledged final interval has no invented duration. Device time is unavailable. Decode-only, mixed, and prefill-only counts remain separate.

The sections below are the measurement contract implemented in source. Offline verification does not establish live or logging-cost readiness.

### 1. Implement and validate the event history

Record an initial state and ordered changes for the whole measured run, including waiting requests and finished requests.
Use a monotonic timestamp, a global event sequence, a scheduler-call ID, and a run ID.
Give each preemption episode its own ID, including repeated preemptions of one request.

Reconstruct these states from the history:

- Each request's state, admission order, prompt and output lengths, completed computed-token count, scheduled tokens, and ordered block IDs.
- Each physical block's pool and group, allocation generation, reference count, valid computed-token coverage, and cache hash identity.
- Request attachment and detachment, allocation, reference-count changes, hash assignment and eviction, cache lookup, and free-queue insertion, removal, and order.
- Scheduled work, completion, canceled or stale work, and request completion or abort. Distinguish scheduled tokens from completed KV in async scheduling.
- Every failed allocation that leads to preemption, including the requesting request, needed blocks, available blocks, and the constraints that prevented allocation.

Block IDs can be reused. Pair IDs with allocation generations and hashes when tracking old KV.
Reference counts belong to blocks. Store each change once and link every request that owns that block.
Exclude reserved or null blocks from reclaimable counts. Record their identities and engine-specific accounting separately.
The event writer must expose dropped events and failed writes. Any loss makes reconstruction invalid for the affected run.

### 2. Capture every preemption decision

Immediately before each victim release, save a complete snapshot of the chosen victim and every running request.
Include the victim even if the scheduler has already removed it from `self.running`.
Record eligibility and any reason a request cannot release blocks immediately.
For consecutive releases in one scheduler call, take a fresh snapshot before each release and link them to the same failed allocation.

Produce one row per request per decision with these fields:

| Field | Meaning |
|---|---|
| Decision and request IDs | Link to the full event history and preemption episode. |
| Chosen victim and eligibility | Mark the actual choice and the requests that could release blocks immediately. |
| Computed tokens | Completed KV tokens before release. Also expose the scheduler's raw progress and pending work separately. |
| Least-computed marker and rank | Compare eligible requests. Mark every tie. |
| Total blocks | Distinct owned physical blocks, with reserved blocks reported separately. |
| Shared blocks | Owned nonreserved blocks whose reference count exceeds 1. |
| Reclaimable blocks | Owned nonreserved blocks whose reference count is 1 and whose release is immediately permitted. |
| Continuous shared prefix | Computed tokens covered from the start by consecutive valid shared blocks, stopping at the first gap. |
| Allocation sufficiency | Whether releasing this request alone would permit the failed allocation under the actual allocator constraints. |
| Running-list position and admission times | Explain the stock choice and distinguish first admission from resumption. |
| Actual reclaimed blocks | Measured free-count change for the chosen victim, with any deferred release recorded separately. |
| Recovery on resume | Actual victim only. Link wait, cache hit, surviving original KV, and re-prefill work. |

Shared blocks anywhere in a request do not imply a continuous shared prefix.
A count of reclaimable blocks is not proof of allocation sufficiency when the allocator has other constraints.
If sufficiency cannot be evaluated without changing allocator state, record it as unknown rather than mutate the allocator to test it.
Do not rank an unknown value as a favorable candidate.

### 3. Follow each victim through recovery

Let `C` be completed computed tokens at that preemption, measured before progress resets.
At the successful resumption lookup, let `H` be cached prefix tokens covering positions below `C`.
Record `H / C` as the recovery fraction. If `C` is zero, report the fraction as undefined.
Retained token IDs beyond `C` are not previously computed KV.

Separate recovered positions backed by the victim's original KV from positions backed by equivalent KV produced elsewhere.
Use block generation and hash history to establish that distinction. A matching block ID alone is insufficient.
Within original-KV recovery, distinguish blocks that became free at release from blocks that remained shared at release.
Report these three disjoint sources of recovery separately so shared prompts do not look like survival of freed KV.
Record every unsuccessful lookup or admission attempt as well as the lookup used for actual resumed execution.

Record waiting time, intervening allocations, recovered tokens, missing tokens `C - H`, and actual completed re-prefill tokens.
Track completed re-prefill work by token position across repeated preemptions. Preserve work already redone before another preemption.
Maintain each request's greatest previously completed token position across progress resets.
Count completed execution below that prior position as re-prefill even if the newest episode's `C` is smaller.
Do not count canceled scheduled work as completed work. Record executed work whose KV was later discarded separately.
An episode can finish, be preempted again, or remain unresolved when the run stops.
Keep unresolved episodes in the report instead of assigning zero recovery.

Report elapsed time until the request regains `C` separately from execution cost.
Mixed forward passes contain other requests' work, so scheduler-call gaps do not directly measure this victim's GPU re-prefill time.
Use completed re-prefill token counts as the primary cost measure.
Report directly measured device time only if the backend provides attributable timing with validated boundaries.
Otherwise label any time attribution as an estimate, document its assumptions and checks, or report device-time share as unavailable.

### 4. Pass the pilot gates

Run these checks outside the 100 main runs. Keep their artifacts separate and list them in Runs.

1. Exercise shared and unshared blocks, a partly filled block, block reuse, multiple victims, repeated preemption, and unresolved recovery.
2. Replay events into ownership, reference counts, and free-queue order. Compare replay against direct snapshots at every decision and at run end.
3. Check that observed reclaimed blocks equal the allocator's measured change when no intervening mutation occurs. Check the stock choice against its actual rule.
4. Feed logged and unlogged schedulers the same offline sequence of request arrivals, allocation results, and completions. Require identical decisions and outputs from the scheduler.
   This checks instrumentation behavior with fixed inputs. Do not require identical generated text from live async runs whose arrival timing can differ.
5. Measure instrumentation cost with five paired, interleaved runs for a low-pressure and a pressure workload. Record errors, throughput, TPOT, and thermal and load state. An output cap is a maximum, so capacity arithmetic at that cap does not prove preemption. The pressure gate must observe actual preemption on both logging sides.
6. As a planning guard, require median throughput and TPOT changes within 5%, with run ranges reported. If noise hides that distinction, classify the check as inconclusive.
7. If the cost check fails or is inconclusive, reduce capture cost and repeat it before the sweep. Preserve required events. Record buffering, flushing, and peak log volume.
8. Verify that the streaming client measures first-token arrival and token counts without treating network chunks as individual tokens. Reconcile requests with engine IDs.

These gates test behavior and accounting. Passing the 5% planning guard does not establish a statistically precise absence of perturbation.
Estimate main-sweep duration and disk use from the pilot before execution. No duration is assumed here.
Freeze total elapsed-time and disk ceilings before the main sweep. If the pilot forecast exceeds either ceiling, report execution as infeasible under that budget.
After the first repetition round of 20 runs, compare the remaining forecast with those ceilings again.
If it no longer fits, stop with an incomplete sweep and preserve every run for later resumption. Do not silently reduce the agreed repetitions or caps.

### 5. Run each configuration

1. Freeze a manifest containing all 100 run IDs, effective settings, corpus checksum, per-request seeds, deadlines, and randomized configuration order.
2. Run five repetition rounds. Shuffle all 20 configurations independently within each round using its recorded seed.
3. Start a fresh server for each run. Warm the engine to a recorded steady thermal state, drain warmup requests, then clear warmup prefixes.
4. Confirm the prefix cache is empty and the configured pool matches the manifest. Save the initial reconstructable state.
5. Submit 100 requests while maintaining the target as described above. Keep all request and block events until final completion or abort.
6. After each run, validate reconstruction, reconcile the submitted, completed, failed, and unfinished request counts, and produce its summary before proceeding.

Freeze a wall-clock deadline and a no-progress deadline after the pilot, before the main sweep.
If a deadline fires, stop that run and retain every event and unfinished request. Label it incomplete rather than a complete 100-request run.
Preserve failed attempts. Any replacement run gets a new ID and a reason. Never silently retry requests or select the best five runs.
Pause execution on corrupt event histories or a pool mismatch. Report workload failures and stalls as outcomes rather than silently omitting configurations.
Record thermal drift, swap, competing GPU activity, and client saturation. Flag affected runs and keep them visible.

### 6. Produce the manual review and summary

Write the decision table as a filterable local HTML report and a CSV file.
Group by run and decision, mark the chosen victim and all least-computed ties, and link every row to its snapshot.
Permit sorting by computed tokens, reclaimable blocks, shared prefix, and allocation sufficiency.
Show an individual request's history across its preemptions and recovery episodes.
For each manual annotation, state what makes an alternative worth testing. Preserve the full table alongside selected examples.

For each configuration, report all five run values, their mean, sample standard deviation, median, and range for scalar run metrics.
Report these measurements:

- Completed, failed, and unfinished requests, with cap-hit rates and actual prompt and output lengths.
- Preemptions per 100 submitted requests, plus the fraction of requests preempted and preemptions per request.
- How often the chosen victim is tied for least computed, including the tie counts and eligible-candidate counts.
- Candidate distributions of computed tokens, total blocks, shared blocks, reclaimable blocks, and continuous shared prefixes.
- Per-run recovery distributions, resumed and unresolved episode counts, original-KV recovery, and total completed re-prefill tokens.
- Time-weighted actual batch-size distribution, plus separate counts for mixed prefill and decode forward passes.
- TTFT, TPOT, end-to-end latency, output-token throughput, request throughput, and stalls.

Define TTFT as submission to first output token. Follow the repo glossary for TPOT and divide decode duration by the output-token count.
Use the first-to-last output interval as the decode duration. Save both timestamps and the token count so the denominator convention stays explicit.
Use actual token counts. Mark TPOT undefined for shorter outputs. Include preemption waits in request latency and TPOT.
State throughput's elapsed interval, including submission startup and final drain. Report failed and incomplete runs separately.
Do not average five p99 values and call that the configuration's p99.
Show per-run tails and any pooled distribution with its request count. With 500 requests per configuration, p99 remains exploratory.
Events within a run share one history, so do not treat thousands of preemption events as thousands of independent runs.
Configurations with no preemptions have zero event counts and undefined recovery distributions, not zero recovery.

## Runs

| Run | What | Key numbers |
|---|---|---|
| 20261014-002308-full | Historical four-configuration scope, 64 requests per configuration, pool 12,416 tokens | Sealed pending Krishna's historical prediction; excluded from the expanded sweep |
| 20261020-140632-pilot-behavior | Expanded E02 behavior pilot, 24 requests at target 16, cap 1,024 | Failed by a runner bug (history checked before the outcome was written). Evidence replayed valid: 19 preemptions, 51 snapshots and 7,731 digests matched. Superseded by a rerun |
| 20261020-141844-pilot-behavior | Expanded E02 behavior pilot rerun, 24 requests at target 16, cap 1,024 | Earlier implementation reported pass. Offline replay retains 19 decisions, 51 snapshots, 7,731 digests and 24 joins. This unsealed pilot does not qualify current source |
| 20261020-143026-pilot-cost-low | Logging-cost pilot, 8 requests at target 4, cap 256, 5 planned on/off pairs | Stopped on Krishna's instruction: pair 1 complete, p2-off.a1 incomplete (interrupted), pairs 2-5 not run; no verdict |
| 20261021-004751-pilot-behavior | Behavior pilot, first try under the hardened source | Failed at warmup: vllm-metal rejects `min_tokens`, so every warmup request got HTTP 400. Fixed by dropping `min_tokens` |
| 20261021-005004-pilot-behavior | Behavior pilot, 24 requests at target 16, cap 1,024 | Pass. 19 preemptions, 19 resumed; 51 snapshots and 7,731 digests matched; 384 s measured |
| 20261021-005714-pilot-cost-low | Logging-cost pilot, 8 requests at target 4, cap 256, 5 on/off pairs | Pass. Median change with logging: throughput -0.4%, TPOT +0.03%. No preemptions. Throughput drifted from 54.2 to 48.5 tokens/s across the 10 runs |
| 20261021-011010-pilot-cost-pressure | Logging-cost pilot, 16 requests at target 16, cap 128, 5 on/off pairs | Pass. Median change with logging: throughput +0.6%, TPOT -0.05%. 2 preemptions in every run on both sides |

The first row belongs to the historical scope. The other rows are expanded-scope pilots, outside the 100 main runs.
The 20261021 pilots qualify the current source. No main-sweep run exists. The main sweep needs Krishna's explicit instruction.

## Result

The expanded sweep is not measured. It contains 100 planned runs and 10,000 requests.
The historical run remains sealed under its original prediction condition.

## Why

Manual inspection can identify decisions where progress, block sharing, and immediately reclaimable capacity point toward different candidates.
The actual victim's recovery shows what happened under the stock trajectory.
Choosing another victim changes subsequent scheduling and allocations, so replay of the stock trajectory cannot prove that alternative's outcome.
E02 does not call such a replay an oracle.

## Verdict

Implemented and offline verified. Predictions and the corpus are frozen. All live behavior and logging-cost validation for the hardened source remains pending explicit user instruction. Earlier behavior evidence replays, but its provenance cannot satisfy the current pilot gate. Cost-low stopped incomplete, cost-pressure never ran, and main is blocked. A compatible pilot forecast and sealed main manifest are also pending. See the [fix-round1 report](../../.audit/e02-implementation/fix-round1-report.md) for exact checks, hashes, and limits.

E02 is descriptive. It has no automatic policy decision rule and makes no claim that an alternative policy improves performance.
For E02, replace R03's automatic 2x candidate-spread rule with inspection of measured re-prefill cost and concrete candidate disagreements.
Preemption rate, re-prefill cost, and recovery extremes are descriptive results. They do not automatically establish or reject a policy improvement.
Use the observations to formulate E03, the proposed controlled experiment for the shared-prefix victim-selection idea.
Freeze E03's numeric decision rule before its measurements.
Little cross-request sharing limits what E02 can say about victim choice under shared prefixes. It still measures recovery from the actual victim's freed KV.
If pressure is absent, report that limit for these workloads and this pool. Keep the fixed-pool main sweep intact.
If reconstruction or cost attribution fails, mark the affected conclusion inconclusive.

The formulation decisions and review outcome live in [decisions.tsv](decisions.tsv).

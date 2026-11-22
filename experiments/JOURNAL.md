# Journal

The story of the lab, one entry per experiment, newest last. Each entry is two or three paragraphs: what we asked,
what happened, what it changed. The full write-up is in the experiment's README, linked at the end of each entry.

Questions come from the research-gap search in `../research/`; an experiment tests one of them. Each experiment
measures its own "before" in the same session as its "after"; there is no shared baseline.

How to read an entry:

- **Surprise** is the first line: the one thing that broke our model of the system. It's the line worth rereading.
- **Decision** lines record a choice, who made it, when, and why.

## Prediction scoreboard

Every prediction, against what was measured. "Closer" is whoever's number was nearer.
E00's agent predictions are Claude's. The expanded E02 predictions are Codex's.

| Exp | What | Krishna | Agent | Measured | Closer |
|---|---|---|---|---|---|
| E05 | Contiguous page advantage | No prediction | Under 5% | Every paired median below 5% across dense and sparse layouts | n/a |
| E07 | Best verification depth at batch 8 | No prediction | Short contexts can use at least as much depth as long contexts | Running | n/a |
| E06 | Saved preemptions on 256 recorded arrivals | No prediction | At least one capacity saves one | One of six capacities saved one at both service costs | n/a |
| E04 | Preemptions through completion | No prediction | One avoided under pressure with early sibling completion | Matched in all 216 settings | n/a |
| E00 | 1k prompt, batch 1 TPOT vs 27.3 ms (2026-10-10) | a bit faster | within 3% | 23.0 ms (16% faster) | Krishna |
| E00 | 1k decode step: weights matmul / attention / small kernels | 2-3 / 12 / 7 ms | 20 / 1.5 / 5 ms | 17.6 / 1.5 / 5.1 ms | Claude |
| E00 | 8k decode step: weights matmul / attention / small kernels | 4-5 / 25 / 11 ms | 20 / 11 / 5 ms | 17.6 / 11.6 / 5.1 ms | Claude |
| E00 | Cause of the gap above the weight-read floor | GPU math slowed by other processes | memory not at full speed for 4-bit matmuls | 5.1 ms of fixed small kernels + attention; matmuls at 96% of bandwidth, other apps 0% | neither |
| E02 | Preemptions per 100 requests | Fewer under low pressure, more under high pressure | Codex guess, target 4 cap 512 below 1; target 32 cap 4,096 above 10 | Not run | Pending |
| E02 | Chosen victim tied for least computed | Pending | Codex guess, below 80% of pressure decisions | Not run | Pending |
| E02 | Recovery fractions in a pressure configuration | Above 60% under low pressure; 40-60% under high pressure, conditional on preemption | Codex guess, at least one configuration contains values below 0.2 and above 0.8 | Not run | Pending |

## Open questions

Things a result raised that we can't answer yet. Each is a lead for the next dossier run.

- Why does a decode step carrying 8 requests take 87-120 ms when each weight matrix is read once and one request takes 23 ms? (R02's candidate C8)
- What is the fixed 5.1 ms per decode step that isn't weights matmul or attention: norms, KV write, sampling, or the engine running matmuls slower than alone? (E00)
- Why was 1k prompt, batch 1 16% faster than the deleted 2026-10-10 attempt, and why was this run's first pass 11% slower than its last? (E00)

---

## E00: what serving a model on this Air costs, and where the time goes

**Surprise:** the gap above the weight-read floor isn't the weights matmul or other apps. The weights matmul runs at 96% of
measured bandwidth and the GPU never idles; the gap is attention plus about 5 ms of small kernels that cost the same
at every context length.

A redo from scratch of the first look at the machine, after we deleted the earlier attempt and its runs. It asks
two things: how long the engine takes per token on three workloads, and where a single decode step's time goes. This
time the whole experiment is one script, `run_experiment.py`, that runs unattended and leaves a summary, so the
result doesn't depend on Claude running commands by hand.

The step split no longer uses Xcode, because Xcode's per-kernel replay needs someone clicking in the GUI. Instead,
the script times decode at five context lengths: attention is the only part that grows with context, so the slope
of step time against context gives attention, and the rest is what doesn't grow. A separate test times the model's
real weight matrices on their own to pull the weights matmul out of that rest.

It ran overnight on a quiet machine (other apps under 0.5% of the GPU, nothing flagged). TPOT: 23.0 ms at
1k, 35.9 ms at 8k, 119.8 ms per step for eight 1k requests. A 1k decode step is 17.6 ms of weights matmul, 1.5 ms of attention,
5.1 ms of small kernels and 0.03 ms idle; at 8k attention grows to 11.6 ms and nothing else moves. The setup can
see changes of about 5% at 1k prompt, batch 1 and 10-15% on the other workloads.

**Decision (Krishna, 2026-10-12):** stop and delete the first E00 attempt and everything from it. Why: start clean.

**Decision (Krishna, 2026-10-12):** every experiment runs from one deterministic `run_experiment.py`; either of us
may start it, nobody edits it mid-run. Why: the run shouldn't depend on Claude improvising.

**Decision (Krishna, 2026-10-12):** Claude designs E00's script alone and runs it overnight. Why: Krishna went to
sleep; predictions were already written.

**Decision (Claude, 2026-10-12):** split the step by context sweep plus a weights-only test, not Xcode. Why: Xcode's
Profile replay has no command-line form, so it can't run unattended. Cost: KV write and other come out as one number.

**Decision (Claude, 2026-10-12):** if other apps use the GPU at start, wait up to 30 minutes, then run anyway and
flag the affected passes. Why: a quiet machine isn't always possible; recording the interference beats refusing.

Full write-up: [E00-serving-cost/README.md](E00-serving-cost/README.md)

---

## E01: why a decode step with 8 requests costs 87 ms, not 34 (planned)

**Surprise:** pending.

E00 left one number it couldn't explain: a decode step carrying eight 1k requests takes about 87 ms, when reading
each weight matrix once plus eight requests' attention should come to about 34. E01 asks where the other ~53 ms go, mainly
whether the weights matmul itself slows down at batch 8 instead of batch 1. This is the cheapest test for the
thesis lead R02-C8: if the matmul is less than half of the gap, that lead dies.

**Decision (Krishna, 2026-10-12):** make the C8 test E01 and formulate it now; don't write the script or run it yet.
Why: understand the plan before anything runs.

**Decision (Claude, 2026-10-12):** E01 only asks "is it the matmul?"; why the matmul is slow (hardware counters) is
left for a later experiment. Why: one question per experiment, and the counters need Xcode by hand.

Full write-up: [E01-small-batch-step/README.md](E01-small-batch-step/README.md)


---

## E02: inspect preemption decisions and KV recovery, expanded scope

**Surprise:** the expanded sweep has not run. The historical run remains sealed.

E02 now holds the expanded measurement experiment. The sweep has batch-size targets 4, 8, 16, 24, and 32,
output caps 512, 1,024, 2,048, and 4,096, and five runs of 100 requests for each combination.
Every preemption gets a table of eligible candidates, including computed tokens, block ownership, shared prefixes, and reclaimable capacity.
The table marks the actual victim and every least-computed tie for manual inspection.

The earlier four-configuration scope and predictions remain in [historical-plan.md](E02-preemption-rehit/historical-plan.md).
Its recorded run is preserved and excluded from the expanded sweep. The existing scripts still implement that earlier scope.

**Decision (Krishna, 2026-10-14):** start the overnight run before Krishna's prediction is written; Krishna writes
it in the morning before opening `summary.md`. Why: the Mac was free overnight. Claude's prediction was written
before the full run but after a smoke run on a tiny pool, so it is not blind.

**Decision (Krishna, 2026-10-20):** use five runs per configuration and histories of every request's blocks and block reference counts.
Why: inspect whether the chosen victim is least computed and manually identify alternative candidates.

**Decision (Krishna, 2026-10-20):** replace E02's scope with the expanded plan instead of creating E03.
Why: keep the measurement experiment in one E record. The earlier choice to file the expansion separately is superseded.

The plan changes instrumentation only. Completed re-prefill tokens are the primary cost measure because scheduler-call gaps do not isolate GPU work.
Alternative victim outcomes need a later controlled policy experiment. The old automatic 2x rule does not decide E02's expanded-scope verdict.
E03 is the proposed experiment for the shared-prefix victim-selection idea.

**Decision (Krishna, 2026-10-20):** number the proposed policy experiment E03.
Why: the expanded measurement experiment is E02, so E03 is the next E record. This supersedes the earlier E04 designation.

Full plan and execution gates are in [E02's README](E02-preemption-rehit/README.md).
The expanded runner and instrumentation have not been implemented, and no expanded-scope measurement ran.

## E04 Shared-prefix compression timing

**Surprise:** None. Waiting for a sibling to finish avoided one preemption in 52 of 216 paired settings. All three requests completed in each replay.

**Decision (Codex, 2026-11-22):** Carry the mechanism forward to engine measurement because the saved preemption persisted through completion. [E04](E04-prefix-deferral/README.md)

## E05 Metal page layout

**Surprise:** Even scattering whole pages through a pool four times the selected size moved median attention time by less than the contiguous-copy control threshold. For 1k prompt, batch 8, the sparse shuffled layout cost an additional 6.4 and 9.0 µs across the two seeds. The thresholds were 13.5 and 15.8 µs.

**Decision (Codex, 2026-11-22):** Park repacking for these workloads. We measured its copy cost but could not establish a reliable break-even. [E05](E05-metal-page-layout/README.md)

## E06 Bounded deferral on recorded arrivals

**Surprise:** Saving a preemption survived the change from a scripted example to the Zipage scheduler on recorded arrivals, but only at one of six capacities. Both policies finished all 256 requests at both modeled service costs.

**Decision (Codex, 2026-11-22):** Park the CUDA timing run. The first window missed the two-capacity threshold. Run a held-out window to check how much this depended on request mix. [E06](E06-prefix-deferral-trace/README.md)

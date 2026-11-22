"""Resample matched rounds to quantify differences among tested depths."""
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
records = [json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()]
rounds = sorted({r["round"] for r in records})
depths = sorted({r["depth"] for r in records})
assert len(rounds) == 12 and depths == [0, 1, 2, 3, 4, 6, 8]
samples = {(r["round"], r["depth"], r["assignment"]): np.array(r["values"]) for r in records}
assert len(records) == len(samples) == 2 * len(depths) * len(rounds)
rng = np.random.default_rng(441)
draws = rng.integers(len(rounds), size=(5000, len(rounds)))
rows = []
sensitivity = []
positive_depth_results = []
for assignment in ("short", "long"):
    for fixed_ms, per_depth_ms in [(0, 0), (5, 0), (20, 0), (0, 1), (0, 5)]:
        rates = []
        for d in depths:
            observations = [samples[r, d, assignment] for r in rounds]
            assert all(x.shape == (4, 3) and (x > 0).all() for x in observations)
            cost = (fixed_ms + per_depth_ms * d) if d else 0
            throughput = [np.median(1000*x[:, 2]/(x[:, 1]+cost)) for x in observations]
            rates.append(throughput)
            if not (fixed_ms or per_depth_ms):
                other = "long" if assignment == "short" else "short"
                verify = np.array([np.median(x[:, 0]) for x in observations])
                paired_delta = verify - [np.median(samples[r, d, other][:, 0]) for r in rounds]
                lo, hi = np.percentile(np.median(paired_delta[draws], axis=1), [2.5, 97.5])
                rows.append(dict(assignment=assignment, depth=d, verify_ms=np.median(verify),
                                 verify_delta_ms=np.median(paired_delta), delta_ci=[lo, hi],
                                 committed=int(observations[0][0, 2]), tokens_s=np.median(throughput),
                                 range_tokens_s=[min(throughput), max(throughput)]))
        rates = np.array(rates)
        point = np.median(rates, axis=1)
        ranking = np.argsort(-point)
        best, second = ranking[:2]
        boot = np.median(rates[:, draws], axis=2)
        gap = np.median((rates[best] - rates[second])[draws], axis=1)
        if fixed_ms == per_depth_ms == 0:
            positive = [i for i in ranking if depths[i] > 0]
            pbest, psecond = positive[:2]
            positive_win = np.argmax(boot[positive], axis=0) == 0
            margins = np.median((rates[pbest] - rates[psecond])[draws], axis=1)
            positive_depth_results.append(dict(assignment=assignment, winner=depths[pbest],
                                               runner_up=depths[psecond], rate=point[pbest],
                                               bootstrap_win_fraction=float(np.mean(positive_win)),
                                               paired_margin_ci=np.percentile(margins,[2.5,97.5]).tolist()))
        sensitivity.append(dict(assignment=assignment, fixed_ms=fixed_ms, per_depth_ms=per_depth_ms,
                                winner=depths[best], runner_up=depths[second], rate=point[best],
                                bootstrap_win_fraction=float(np.mean(np.argmax(boot, axis=0)==best)),
                                paired_margin_ci=np.percentile(gap, [2.5, 97.5]).tolist()))
acceptance_sensitivity = []
for probability in (0.2, 0.4, 0.6, 0.8):
    for assignment in ("short", "long"):
        rates = []
        for d in depths:
            expected = 8 + 4 * sum(probability**k for k in range(1, d+1))
            if d:
                expected += 4 * probability
            rates.append([np.median(1000*expected/samples[r,d,assignment][:,1]) for r in rounds])
        rates = np.array(rates)
        best = np.argmax(np.median(rates, axis=1))
        winners = np.argmax(np.median(rates[:, draws], axis=2), axis=0)
        acceptance_sensitivity.append(dict(probability=probability, assignment=assignment,
                                           winner=depths[best], bootstrap_win_fraction=float(np.mean(winners==best))))
summary = dict(positive_depth_results=positive_depth_results, depth_results=rows, sensitivity=sensitivity, acceptance_sensitivity=acceptance_sensitivity)
(run/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")

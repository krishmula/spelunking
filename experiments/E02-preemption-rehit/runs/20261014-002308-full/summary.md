# E02 summary: 20261014-002308-full

Do not read past this line until the prediction table in README.md is filled in.

Pool: 12400 tokens. Thresholds from R03 decisions.tsv (2026-10-12T09:30).

## b8-cap2k

- Verdict: KILL: re-prefill under 5% of GPU time; KILL victim-policy direction: candidates differ by under 2x in surviving blocks (median)
- Requests finished 64 (failed 0), median output 2048.00 tokens, 1.00 hit the cap
- Preemptions 29 (45.3 per 100 requests); 0.38 of requests preempted at least once
- Re-prefill: 17912 of 151546 scheduled tokens; time share 0.013 (marginal cost) / 0.013 (token-proportional) of 1979 s busy
- Re-hit share over 29 resumes: min 0.00, p10 0.00, median 0.00, p90 0.95, max 0.97
- Free blocks at preemption (median) 0; candidates per preemption (median) 7
- Replay check on the real victim (29): lost blocks exact 0.31, mean abs error 4.00; surviving prefix exact 1.00
- Candidate spread in surviving blocks: median 1.27x, 0.41 of preemptions at least 2x (lost blocks: median 1.03x)

## b16-cap2k

- Verdict: KILL: re-prefill under 5% of GPU time
- Requests finished 64 (failed 0), median output 2048.00 tokens, 1.00 hit the cap
- Preemptions 152 (237.5 per 100 requests); 0.83 of requests preempted at least once
- Re-prefill: 65212 of 198846 scheduled tokens; time share 0.046 (marginal cost) / 0.049 (token-proportional) of 1877 s busy
- Re-hit share over 152 resumes: min 0.00, p10 0.00, median 0.00, p90 0.66, max 0.99
- Free blocks at preemption (median) 0.00; candidates per preemption (median) 12.00
- Replay check on the real victim (152): lost blocks exact 0.64, mean abs error 1.70; surviving prefix exact 0.93
- Candidate spread in surviving blocks: median infx, 0.83 of preemptions at least 2x (lost blocks: median 2.51x)

## b8-cap4k

- Verdict: KILL: re-prefill under 5% of GPU time
- Requests finished 64 (failed 0), median output 4096.00 tokens, 0.97 hit the cap
- Preemptions 233 (364.1 per 100 requests); 0.95 of requests preempted at least once
- Re-prefill: 149576 of 413991 scheduled tokens; time share 0.043 (marginal cost) / 0.044 (token-proportional) of 4799 s busy
- Re-hit share over 233 resumes: min 0.00, p10 0.00, median 0.00, p90 0.41, max 0.99
- Free blocks at preemption (median) 0; candidates per preemption (median) 6
- Replay check on the real victim (233): lost blocks exact 0.67, mean abs error 4.23; surviving prefix exact 0.97
- Candidate spread in surviving blocks: median 2.49x, 0.56 of preemptions at least 2x (lost blocks: median 4.22x)

## b16-cap4k

- Verdict: no rule fired (between thresholds)
- Requests finished 64 (failed 0), median output 4096.00 tokens, 0.97 hit the cap
- Preemptions 392 (612.5 per 100 requests); 0.97 of requests preempted at least once
- Re-prefill: 207919 of 471586 scheduled tokens; time share 0.058 (marginal cost) / 0.060 (token-proportional) of 4836 s busy
- Re-hit share over 392 resumes: min 0.00, p10 0.00, median 0.00, p90 0.51, max 0.99
- Free blocks at preemption (median) 0.00; candidates per preemption (median) 8.00
- Replay check on the real victim (392): lost blocks exact 0.77, mean abs error 2.38; surviving prefix exact 0.98
- Candidate spread in surviving blocks: median 22.38x, 0.61 of preemptions at least 2x (lost blocks: median 6.80x)

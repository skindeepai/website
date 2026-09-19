# A more conservative stopping rule

This follow-up was designed after the first real-data candidate failed its harmful-exit guard. It reuses the seed-17 classifiers; only policy selection changes. It is explicitly an exploratory revision, checked on a fresh reserve.

The chosen rule starts at layer 12 and requires a 95.00% score. Agreement with the preceding checkpoint: False. It was selected on the original tuning queries by requiring the one-sided 95% upper bound on harmful exits to be at most 0.5%. The chosen rule was saved before any reserve inference.

Fresh calibration: 742 queries. Fresh test: 741. Both cover 75 of 77 intents; two small classes had no unused examples. All prior query IDs and exact normalized-text duplicates were excluded, including duplicates within this reserve.

## Results

| Split | Full head | Early candidate | Mean layers | Blocks skipped | Harmful / corrected |
| --- | --- | --- | --- | --- | --- |
| calibration | 613/742 (82.61%) | 611/742 (82.35%) | 22.31 | 7.04% | 3 / 1 |
| test | 613/741 (82.73%) | 615/741 (83.00%) | 22.15 | 7.69% | 0 / 2 |

Independent calibration guard passed: **False**. Its one-sided 95% Wilson upper bound on harmful exits was **1.007%**, against a 1% limit. If this guard fails, the guarded policy uses full depth. This approximate finite-sample check is not a production safety guarantee.

### Test exit counts

| Layer | Requests |
| --- | --- |
| 6 | 0 |
| 12 | 95 |
| 18 | 38 |
| 24 | 608 |

## Actual runtime

| Path | Mean including tokenization | Median | p95 |
| --- | --- | --- | --- |
| full_head | 244.1 ms | 241.7 ms | 273.6 ms |
| candidate_early_exit | 231.1 ms | 240.0 ms | 276.6 ms |

Warm CPU, first 96 fixed reserve queries, two alternating repeats per path. Other launched model experiments were stopped for this timing run. Every executed block is counted and runtime predictions must match cached ones. Loading is excluded. The mean is shown because a policy that rarely exits early may leave the median unchanged.

Mean latency reduction in this paired run: **5.33%**. Query-paired bootstrap 95% interval: 2.29% to 8.81%. This does not override a failed calibration guard. [Calculation record](../results/banking77-conservative/timing-analysis.json).

## Limits and evidence

One head seed, the same underlying public benchmark, and 75-intent reserve coverage. A fresh reserve avoids reusing the first test examples for this confirmation, but does not establish cross-domain or pretraining independence. No absent-category detection or changing-rule generalization was tested.

- [Frozen policy and reserve IDs](../results/banking77-conservative/protocol.json), [results](../results/banking77-conservative/result.json), [every reserve prediction](../results/banking77-conservative/predictions.json).
- [Actual layer traces and timings](../results/banking77-conservative/timings.json), [implementation](../experiments/banking77_conservative.py).
- Run `python experiments/banking77_conservative.py --time-runtime` after the first BANKING77 run. Use an otherwise idle model-testing environment for timings.

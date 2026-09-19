# Adding an unsupported-request output

This prospective variant adds a 31st output, UNKNOWN, to the 30 supported CLINC intents. It trains on 80 out-of-scope training queries and uses the remaining 20 for tuning, alongside the original in-scope partitions. It reuses frozen Qwen features; no backbone weights change.

Independent calibration accepted this gate: **False**. Failure means full-depth fallback, not an accepted early-exit saving.

| Set | Full-depth correct | Candidate correct | Early exits | Wrong early answers | Early UNKNOWN | Early known category |
| --- | --- | --- | --- | --- | --- | --- |
| calibration | 567/600 | 556/600 | 555 | 34 | 6 | 549 |
| test | 854/900 | 840/900 | 824 | 46 | 13 | 811 |
| oos_calibration | 72/100 | 72/100 | 79 | 16 | 63 | 16 |
| oos_test | 512/1000 | 460/1000 | 649 | 299 | 350 | 299 |
| unsupported_test | 21/120 | 19/120 | 61 | 52 | 9 | 52 |

For out-of-scope and unsupported-intent inputs, UNKNOWN is the correct dataset-derived label. For supported inputs, UNKNOWN is an error. It means outside this classifier's supported categories, not "too hard for the full model."

## Independent calibration bounds

| Check | Exact one-sided 95% upper bound | Passed |
| --- | --- | --- |
| added_error | 6.092% | False |
| absolute_early_error | 9.876% | False |
| unknown_misrouted | 23.282% | False |
| known_rejected | 1.964% | True |

Limits are 1% added errors per all calibration requests, 5% wrong answers per early exit, 5% unknown inputs prematurely routed to known categories, and 5% known inputs prematurely rejected. Require at least 5% early coverage. These are individual bounds, not a joint 95% guarantee. The combined calibration counts reflect this experiment's 600:100 known/unknown mixture; deployment prevalence may differ.

The variant protocol was saved before the 30-output experiment wrote its results. It changes the output categories, adds OOS supervision and uses another seed, so differences cannot be attributed solely to one extra output. It shares benchmark partitions with the first variant, not an independent dataset replication. Calibration has 600 known and 100 OOS queries; testing has 900 known and 1,000 OOS queries. Pooled rates cannot be transferred across those prevalences. The guard concerns early decisions, not full-depth rejection reliability. Small OOS training/tuning sets and unknown pretraining exposure remain limits. Features were collected at full depth; projected exits are not measured runtime savings here.

- [Prospective protocol](../results/clinc-unknown/protocol.json), [frozen policy](../results/clinc-unknown/selected-policy.json), [results](../results/clinc-unknown/result.json).
- [Every prediction](../results/clinc-unknown/predictions.json), [portable weights](../results/clinc-unknown/heads.npz), [implementation](../experiments/clinc_unknown.py).
- [Dataset and attribution](https://github.com/clinc/oos-eval), CC BY 3.0.

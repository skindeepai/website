# Second-dataset validation: CLINC subset

**Scope: 30 supported intents across ten domains, not the complete 150-intent benchmark.** New heads were trained on this dataset; the BANKING77 heads were not transferred. Public utterances are crowdsourced, not production traffic.

Independent calibration accepted the selected gate: **False**. A rejected gate falls back to full depth; its candidate savings are not accepted deployment savings.

## Known-intent results

| Split | Full-depth correct | Candidate correct | Early exits | Wrong early answers | Added errors | Projected blocks skipped |
| --- | --- | --- | --- | --- | --- | --- |
| calibration | 569/600 | 569/600 | 409 | 2 | 2 | 30.92% |
| test | 856/900 | 854/900 | 639 | 5 | 4 | 32.69% |

The projected depths above come from full-depth feature extraction and applying the stopping policy to intermediate readouts. They are not new wall-clock or actual stopped-execution measurements. The separate BANKING77 replay verifies that execution mechanism on the first task.

## Calibration checks

- Added-error upper bound, divided by all known-intent queries: **1.046%**, limit 1%.
- Wrong-answer upper bound among early exits: **1.531%**, limit 5%.
- Early acceptance of unfamiliar inputs, upper bound: **8.920%**, limit 5%.
- Observed known-intent early coverage: **68.17%**, minimum 5%.

Exact one-sided 95% Clopper-Pearson bounds are computed separately. They are not a simultaneous 95% guarantee or a guarantee under distribution shift.

## Unknown requests

| Set | Queries | Premature early answers | Continued to full depth |
| --- | --- | --- | --- |
| oos_calibration | 100 | 4 | 96 |
| oos_test | 1000 | 66 | 934 |
| unsupported_test | 120 | 11 | 109 |

Continuing does **not** mean a request was correctly rejected. This classifier has no unknown class. These inputs remain unresolved at full depth; the table measures premature early decisions only.

## Controls

| Depth | Correct test answers |
| --- | --- |
| 6 | 795/900 |
| 12 | 813/900 |
| 18 | 820/900 |
| 24 | 856/900 |
| TF-IDF logistic regression | 842/900 |

The lexical control receives identical training examples. No matched latency comparison was run here. One head seed, alphabetically selected classes, unknown pretraining overlap and a limited input distribution constrain generalization.

## Inspect and reproduce

- [Protocol written before inference](clinc-protocol.md), [sealed IDs and settings](../results/clinc-validation/protocol.json).
- [Result JSON](../results/clinc-validation/result.json), [every evaluated prediction](../results/clinc-validation/predictions.json), [portable heads](../results/clinc-validation/heads.npz).
- [Implementation](../experiments/clinc_validation.py), [frozen selected policy](../results/clinc-validation/selected-policy.json).
- [CLINC dataset and attribution](https://github.com/clinc/oos-eval), [original paper](https://aclanthology.org/D19-1131/).

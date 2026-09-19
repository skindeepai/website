# Qwen early exits on real banking queries

BANKING77, 77 intents, all 3,080 official test queries. The backbone is frozen; only the small linear classifiers are trained. This is a fixed-intent task, not arbitrary instruction following.

## Data and methods

Training: 2464 queries. Tuning: 611. Independent calibration: 601. Test: 3080. The fixed per-class slices are shorter for a few small classes. Three initialization seeds share these same examples.

A head reads 896 internal numbers after layer 6, 12, 18 or 24. It has 69,069 trained parameters and emits 77 scores without generating text. Policy selection uses tuning data. An independent calibration guard rejects a shortcut if the 95% upper bound on harmful exits exceeds 1%. A rejected gate deploys the full model.

## Fixed-depth classifiers

| Seed | Layer 6 | Layer 12 | Layer 18 | Layer 24 | Fixed layer chosen on tuning |
| --- | --- | --- | --- | --- | --- |
| 17 | 66.56% | 69.06% | 70.97% | 82.82% | 24 |
| 29 | 66.69% | 68.90% | 70.97% | 82.82% | 24 |
| 43 | 66.75% | 68.93% | 71.40% | 82.79% | 24 |

The simple TF-IDF word/phrase classifier scored **2464/3,080 (80.00%)** with the same training IDs. Its configuration was fixed before the Qwen test results. This matters: an early-exit transformer still needs to justify its cost against a small specialist.

## Selected early-exit policies

| Seed | Confidence threshold | Require agreement? | Guard passed? | Candidate correct | Mean layers | Harmful / corrected errors |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 0.7 | True | False | 2552/3080 (82.86%) | 18.32 | 33 / 34 |
| 29 | 0.7 | True | False | 2549/3080 (82.76%) | 18.26 | 33 / 31 |
| 43 | 0.7 | True | False | 2544/3080 (82.60%) | 18.29 | 38 / 32 |

“Harmful” means the full head was correct and the early head was wrong. “Corrected” means the reverse. Candidate rows remain visible even if calibration rejected them. A passed finite-sample guard is not a production guarantee.

### Where requests stopped

| Seed | Layer 6 | Layer 12 | Layer 18 | Layer 24 | Transformer blocks skipped |
| --- | --- | --- | --- | --- | --- |
| 17 | 0 | 1300 | 314 | 1466 | 23.65% |
| 29 | 0 | 1313 | 323 | 1444 | 23.94% |
| 43 | 0 | 1302 | 326 | 1452 | 23.78% |

These are skipped transformer blocks, not skipped model parameters or a measured energy reduction. The checkpoint remains in memory.

## Actual CPU execution

| Path | Model p50 | Including tokenization p50 | Including tokenization p95 |
| --- | --- | --- | --- |
| full_head | 240.5 ms | 241.1 ms | 291.7 ms |
| candidate_early_exit | 177.8 ms | 178.3 ms | 276.6 ms |

Seed 17, 8 CPU threads, 384 timed passes over 96 fixed queries and two alternating repeats per path. Warm model; loading excluded. The final recorded timing run occurs after the other launched model experiments finish. Each pass counts every executed block and asserts agreement with its cached prediction. The candidate is timed whether or not the calibration guard accepts it.

## Exact-duplicate sensitivity

The published dataset has 4 test queries whose normalized text also appears in the selected development data. Their IDs are recorded; the official score retains them. Excluding them gives:

| Seed | Remaining queries | Full model correct | Candidate correct |
| --- | --- | --- | --- |
| 17 | 3076 | 2547 (82.80%) | 2548 (82.83%) |
| 29 | 3076 | 2547 (82.80%) | 2545 (82.74%) |
| 43 | 3076 | 2546 (82.77%) | 2540 (82.57%) |

This sensitivity check does not detect paraphrase overlap or pretraining contamination. Classifier seeds are not independent dataset replications. There is no equally trained text-decoder baseline, GPU/browser early-exit measurement, or unfamiliar-category rejection test.

## Inspect and reproduce

- [Pre-run protocol](banking77-protocol.md) and [implementation](../experiments/banking77.py).
- [Settings and results](../results/banking77/result.json), [all predictions](../results/banking77/predictions.json), [executed layers and timing samples](../results/banking77/timings.json).
- [Data revision, hashes, categories and split IDs](../results/banking77/data-manifest.json), [duplicate check](../results/banking77/duplicate-sensitivity.json), [lexical baseline](../results/banking77/lexical-baseline.json).
- [BANKING77 authors and CC-BY-4.0 source](https://github.com/PolyAI-LDN/task-specific-datasets).
- Run `python experiments/banking77.py --threads 8`; feature extraction is cached locally. Run `python experiments/banking77_lexical.py` for the lexical control. See [environment details](../experiments/README.md).

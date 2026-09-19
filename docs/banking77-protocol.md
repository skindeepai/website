# BANKING77 early-exit protocol

Written before the first run, 2026-09-19. This is a local recorded protocol, not an external preregistration.

Question: can classifiers attached to frozen Qwen2.5-0.5B-Instruct skip transformer blocks while retaining the accuracy of the same classifier architecture at full depth on real customer-service language?

Data: [BANKING77 by Casanueva et al. / PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets), CC-BY-4.0, pinned commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. All 77 intents, all 3,080 official test queries. Per class, shuffle the official training data with seed `20260919 + class index`; take 32 for head training, eight for temperature/policy selection, eight for independent calibration. Publish IDs and original-file hashes. Check exact text overlap. No test label determines training, temperature, policy, or calibration acceptance.

Model: pinned Qwen checkpoint from the original pilot. Frozen float32 backbone; final prompt-position vectors after blocks 6, 12, 18 and 24 (the last includes Qwen's final RMS normalization). Standardize each feature with training-only mean/std. Each head is a linear layer with 77 outputs and softmax, trained with cross-entropy for 200 full-batch AdamW steps, learning rate 0.01 and weight decay 0.1. Repeat head initialization with seeds 17, 29, 43; use the same split.

Select a softmax temperature per head on tuning NLL from 0.5, 1, 1.5, 2, 3, 4, 6, 8. Compare gates using confidence alone or confidence plus agreement with the preceding checkpoint; earliest allowable exit 6 or 12; threshold 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, or 1.01 (disabled). Among candidates losing at most 0.5 percentage points on tuning accuracy, select lowest mean depth, then higher accuracy, then higher threshold. This is empirical selection, not a safety proof.

On untouched calibration queries, require the one-sided 95% Wilson upper bound on harmful exits (full head correct, early head wrong) to be at most 1%. Otherwise the guarded policy falls back to full depth. Publish candidate results even when rejected, without presenting them as a passing policy. This conservatively ignores cases where an early head fixes a full-head mistake.

Report every fixed-depth head, exit histogram, skipped block fraction, accuracy and Wilson interval, harmful exits and corrected full errors. Inspect the first 96 deterministically shuffled test queries in two rotated timing repeats for seed 17. Include tokenization and actual head/gate execution. Count every executed block and assert that runtime results agree with cached predictions. Loading is excluded. Timing the candidate does not override a failed calibration guard.

Public benchmarks can be present in pretraining. Fixed banking intents test a specialist, not general rule following. CPU performance is not an accelerator, energy, throughput, or browser result. A matched trained text-decoder comparison remains a separate experiment.

## Preprocessing observations, before results

The fixed slices yield 2,464 training, 611 tuning and 601 calibration examples: a few classes have fewer than 48 original training rows, so the last slices are shorter. All 3,080 official test rows are retained. Exact normalized-text overlap checks found three train/test duplicates and one calibration/test duplicate. Report the official score and a sensitivity check excluding those four test rows. Do not claim complete data independence.

A fixed lexical control uses the same training IDs: TF-IDF word unigrams/bigrams (at most 12,000 features) plus multinomial logistic regression, C=4, at most 400 L-BFGS iterations, one thread. These settings were chosen before reading the Qwen test results. It provides a cheap-task baseline, not a tuned state-of-the-art score.

Before Qwen test results were available, added a fixed-depth control: choose the head with highest tuning accuracy (ties prefer fewer layers), and report its test result. This distinguishes the benefit of an adaptive gate from simply choosing a shallower classifier. The final recorded timing run is repeated from the feature cache after other launched model work has stopped; cached features do not count as runtime savings.

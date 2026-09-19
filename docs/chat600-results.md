# Six hundred real chat decisions

This is the requested practical workload: process 600 held-out real messages and return SAFE or BLOCK. Messages come from the human-annotated portion of ToxicChat0124, collected from an online chatbot. They are archived real user messages, not generated text or current Twitch chat.

**Calibration and the added input-overlap check accepted the early-exit rule: False.** A failed gate means a guarded application uses the full-depth classifier. Candidate timing remains visible for research, not as an accepted quality-preserving speedup.

## Decision quality on the same 600 messages

There are 81 toxic and 519 benign reference labels. Always returning SAFE would score 86.5% accuracy while missing every toxic message.

| Path | Correct | Missed toxic (of 81) | Wrongly blocked benign (of 519) | Toxic recall |
| --- | --- | --- | --- | --- |
| Untouched Qwen: constrained SAFE/BLOCK | 212/600 | 10 | 378 | 87.7% |
| Trained full-depth readout | 522/600 | 19 | 59 | 76.5% |
| Trained early-exit candidate | 519/600 | 26 | 55 | 67.9% |
| Always SAFE | 519/600 | 81 | 0 | 0.0% |
| TF-IDF logistic regression | 542/600 | 31 | 27 | 61.7% |

The candidate adds **48** mistakes that the full-depth classifier did not make and fixes **45** full-depth mistakes. It adds **12** missed toxic messages. These changes must not be hidden by net accuracy.
Candidate minus full-depth accuracy: **-0.50 percentage points**, paired-query bootstrap 95% interval **-3.67 to 2.83**. This exploratory test-set interval does not override failed independent calibration.

The trained enum and trained SAFE/BLOCK-token paths share exactly the same output rows and predictions. The untouched Qwen control uses its original two vocabulary rows and gets no task training. It is not an equally supervised architecture comparison. No path generates a long explanation or JSON.

## Where the candidate stops

| Layer | Messages |
| --- | --- |
| 6 | 0 |
| 12 | 600 |
| 18 | 0 |
| 24 | 0 |

The selected binary confidence threshold is 0.5 with earliest exit at layer 12. A binary maximum score is always at least 0.5, so this rule reduces to a **fixed layer-12 shortcut**. It does not demonstrate input-dependent readiness or dynamic effort. That distinction is retained even if its workload time improves.

At layer 12, **323** test messages received a confidence score of at least 0.9; **15** of those predictions were wrong. A score is evidence for a gate to evaluate, not proof that the answer is ready. This is a post-run diagnostic, not a replacement threshold selected on the test set.

Mean depth: **12.00 of 24 blocks**; **50.00%** of blocks skipped by the candidate policy. Wrong answers among early exits: **81/600**. This is not a memory or energy measurement.

## Independent calibration

- Added-error upper bound per all calibration messages: **8.805%**, limit 1%.
- Additional missed-toxic upper bound per toxic calibration message: **12.046%**, limit 5%.
- Early coverage: **100.0%**, minimum 5%.
- Full-depth calibration accuracy: **85.9%**, required minimum 90%.
- Full-depth toxic recall: **66.0%**, required minimum 80%.

Bounds are exact one-sided 95% Clopper-Pearson bounds, individually. They are not a simultaneous 95% guarantee, and relative added errors are not the absolute error rate.

Clipping created three calibration inputs identical to development inputs. An extra acceptance requirement was recorded before fitting outcomes: exclude those three IDs and require the original guard and the clean-calibration guard to pass. None of the 600 test inputs overlap development/calibration after clipping. [Overlap evidence](../results/chat600/effective-input-overlap.json), [prospective amendment](../results/chat600/calibration-overlap-amendment.json), [clean-calibration result](../results/chat600/calibration-integrity.json).

## Clipped inputs and jailbreak subset

All paths use the same first 256 user-message tokens, followed by the intact assistant-start marker. Longer originals keep their dataset labels. Clipping can remove information needed to judge the original message.

| Test subset | Messages | Full-depth correct | Candidate correct |
| --- | --- | --- | --- |
| Clipped | 30 | 28 | 23 |
| Not clipped | 570 | 494 | 496 |
| Dataset jailbreaking label | 10 | 9 | 9 |

## Actual time to finish all 600 messages

| Path | Pass 1 | Pass 2 | Pass 3 | Median complete workload | p95 per message |
| --- | --- | --- | --- | --- | --- |
| trained_full_enum | 228.9 s | 239.7 s | 237.2 s | 237.2 s | 900.0 ms |
| trained_full_token | 235.3 s | 230.7 s | 231.8 s | 231.8 s | 904.3 ms |
| early_candidate | 117.0 s | 114.3 s | 121.2 s | 117.0 s | 461.5 ms |
| zero_shot_lm_token | 234.5 s | 230.3 s | 236.8 s | 234.5 s | 915.7 ms |

Candidate mean per-message saving versus the equally supervised full-depth SAFE/BLOCK-token path: **49.48%**, paired-query bootstrap 95% interval **49.32% to 49.65%**.
All **7200 actual forward passes** counted executed blocks and checked prediction/depth parity. Each workload includes all 600 messages. These are actual calls, not cached feature timings. Other launched model work finished before this benchmark.
Warm batch-one CPU, eight threads, float32, one process. Includes tokenization, readout/gate work and token decoding. Excludes loading, downloads and training. Three cyclic orders over four paths are not completely balanced; query bootstrap intervals do not include thermal drift, ordering effects or independent-session uncertainty. This is not live streaming queue latency or GPU throughput.

- [Every timed message and executed layer trace](../results/chat600/timings.json), [all complete-workload passes](../results/chat600/passes.json), [timing summary](../results/chat600/benchmark.json).

## Scope and reproduction

The task is agreement with dataset toxicity annotations, not a universal SAFE/BLOCK judgment. Only the human-annotated subset is used; it has different prevalence from the full corpus. Exact text duplicates and conversation-ID overlap are excluded, but shared users, paraphrases, annotator uncertainty and pretraining exposure remain unknown. A single-message classifier does not enforce a changing channel policy or use full conversation context.

- [Prospective method and acceptance criteria](chat600-protocol.md), [sealed source hashes and IDs](../results/chat600/protocol.json).
- [Separate adversarial review](chat600-review.md), [next experiments](../PLAN.md#practical-workload-findings-and-the-next-iteration).
- [Reproduction commands and dependencies](chat600-reproduce.md).
- [Results](../results/chat600/result.json), [every held-out prediction](../results/chat600/predictions.json), [paired accuracy calculation](../results/chat600/quality-analysis.json), [portable heads](../results/chat600/heads.npz).
- [Implementation](../experiments/chat600.py), [source dataset](https://huggingface.co/datasets/lmsys/toxic-chat), [original paper](https://arxiv.org/abs/2310.17389). Source messages are CC BY-NC 4.0 and remain in the local research cache.

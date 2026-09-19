# Adversarial review of the 600-message test

Reviewed by the separate agent against the recorded source files, predictions, implementation and acceptance rules. Raw user messages were inspected programmatically only; none are quoted or published here. This is a review within the same project, not external replication.

**The speed reduction is real in this recorded CPU workload, but the candidate does not establish "the same moderation quality, faster."** It loses toxic-message recall, changes many individual decisions and fails the frozen calibration checks. The full-depth reference also fails the preset quality floor. Faster execution does not fix those findings.

## What the shortcut actually does

The selected policy is minimum layer 12, confidence threshold 0.5, and no agreement requirement. There are two softmax outputs. Their maximum is always at least 0.5 for finite probabilities, so every request passes at layer 12.

This is effectively **a fixed 12-layer classifier**, not a request-dependent detector of when the model has finished thinking. Every one of the 500 tuning, 800 calibration and 600 test predictions follows that depth. The implementation still computes the layer-6 check before declining to exit there; a purpose-built fixed-depth implementation could avoid that extra check, but such a runtime has not been compared here.

The completed benchmark actually skipped half the transformer blocks on every timed candidate request. That is not half the model memory, energy or complete application cost. Its measured wall-time reduction is documented separately below.

## The practical quality trade-off

The 600 held-out messages contain 81 toxic and 519 benign reference labels.

| Trained path | Correct | Toxic messages missed | Benign messages wrongly blocked | Toxic recall |
| --- | --- | --- | --- | --- |
| Full depth | 522/600 | 19/81 | 59/519 | 76.5% |
| Layer-12 candidate | 519/600 | 26/81 | 55/519 | 67.9% |

Overall accuracy falls only 0.5 percentage points, but toxic recall falls **8.64 percentage points**. The candidate newly misses 12 toxic messages the full head caught, while recovering five toxic messages the full head missed. The net increase of seven misses conceals those changed cases. Across both classes, the candidate adds 48 mistakes and corrects 45.

The paired accuracy interval, **-3.67 to +2.83 percentage points**, does not prove equivalence or noninferiority at the stated quality tolerance. It also does not erase the failed calibration test or replace reporting toxic recall.

Always returning SAFE also yields 519/600 correct because most messages are benign, while missing every toxic message. That does not make the candidate identical in usefulness to always-SAFE; it demonstrates why overall accuracy alone is inadequate. The lexical control reports higher accuracy, 542/600, but lower toxic recall, 61.7%, so it does not dominate every relevant metric either.

The untouched constrained Qwen output has 87.7% toxic recall but wrongly blocks 378 of 519 benign messages. It is an untuned zero-shot reference, not evidence that the trained classifier architecture inherently beats text generation. The trained enum and trained token paths share the same learned rows, making their identical decisions an intentional output-format control.

## Calibration rejection is substantial

The original calibration gate failed four requirements:

- Additional-error upper bound: **8.805%**, against a 1% limit.
- Additional-missed-toxic upper bound: **12.046%**, against 5%.
- Full-depth empirical accuracy: **85.875%**, below 90%.
- Full-depth toxic recall: **66.038%**, below 80%.

Early coverage passed because every request exits. These failures are not rounding accidents. A system enforcing the recorded acceptance policy would not enable this candidate. The weak full-depth reference also means that falling back to it is not evidence of suitable live moderation quality.

Clipping creates three calibration inputs identical to development inputs, even though their original messages differ. The independently checked amendment removes those three only for an **additional** acceptance test, while requiring the original guard to pass too. It changes neither training, gate selection nor the 600 test messages.

The clean 797-message calibration still fails. The two upper bounds become **8.837%** and **12.386%**; the full head gets 684/797 correct and catches 67/103 toxic messages. The effective-input check found no cross-partition duplicates involving the test workload. This does not exclude paraphrases, common users or pretraining overlap.

## Completed timing audit

All **7,200 timed forward passes** are present: the same 600 messages, four execution paths and three complete passes per path. Every recorded output matches its saved reference prediction. All 1,800 candidate traces contain exactly blocks 1 through 12; all 5,400 full-depth traces contain exactly blocks 1 through 24. No candidate duration is merely a timing of cached features.

| Path | Median time to finish 600 messages | Mean per message across three passes |
| --- | --- | --- |
| Trained full-depth enum | 237.22 seconds | 392.11 ms |
| Trained full-depth SAFE/BLOCK token | 231.84 seconds | 387.64 ms |
| Layer-12 candidate | 117.04 seconds | 195.83 ms |
| Untouched Qwen constrained token | 234.48 seconds | 389.73 ms |

Against the equally supervised full-depth token path, mean message time fell **49.48%**. The paired-query bootstrap interval is **49.32% to 49.65%** for this recorded run. Whole-corpus medians and mean per-message savings are distinct summaries; the percentage above is calculated from the latter.

This is meaningful execution evidence: the shorter forward pass takes roughly half as long on this workload. It is evidence of a fixed-depth speed/quality trade-off, not an accepted quality-preserving adaptive system. The trained token path was slightly faster than the full-depth enum path in this run, so these data do not support attributing the gain to avoiding one text token.

The benchmark ran after the other launched model jobs finished and measured warm sequential execution, including tokenization, transformer blocks, head/gate work and token-to-text conversion where applicable. Loading, downloads and training are excluded. All 12 pass records contain 600 messages and their summed message times reconcile with corpus wall times; the remaining loop/check overhead is under 0.03 seconds per pass.

This audit verified recorded artifacts and inspected the runtime implementation; it did not independently rerun the entire timed workload. The three cyclic orders for four paths do not fully balance position effects. The narrow bootstrap interval captures resampled query variation after averaging three repeats; it does not quantify thermal drift, session-to-session variation, other hardware, batching, queueing or livestream traffic. It must not be presented as a universal speed guarantee with that precision.

## Checks actually performed

- Verified source-file hashes; required human annotations; checked every recorded calibration/test label and jailbreak flag against its original row.
- Verified split IDs and order, normalized source-text separation, and conversation-ID separation.
- Reconstructed the gate from recorded checkpoint scores and confirmed all calibration/test exits at layer 12.
- Independently recomputed confusion matrices, accuracy, precision, recall, balanced accuracy, changed-error counts and additional missed-toxic counts.
- Recomputed one-sided exact confidence bounds by inverting the binomial cumulative distribution, without reusing the experiment's beta-quantile helper. Original and clean-calibration values matched.
- Recomputed the paired accuracy bootstrap with the recorded calculation's seed and resample count; both reported interval endpoints matched.
- Independently tokenized the source messages using the pinned tokenizer, applied the stated 256-token bound, and reproduced clipping counts and effective-input duplicate groups. This used no model inference. Thirty test messages are clipped; their full-message source labels remain unchanged.
- Checked all 7,200 timing records for unique query/path/repeat combinations, expected prediction and output serialization, positive finite durations, and actual contiguous layer traces. Verified the exact protocol query order and all 12 recorded pass sizes, sums and path rotations.
- Independently recomputed each path's mean, median corpus time and p95 message time, then reproduced the paired timing bootstrap using seed 19 and 2,000 resamples. All published timing summaries and both interval endpoints matched.

These checks support numerical integrity. They do not validate the toxicity labels as a universal SAFE/BLOCK policy, recover missing conversation context or establish current livestream moderation performance. This human-annotated archive is a specific, selected workload.

## Evidence

- [Prospective method](chat600-protocol.md), [current results](chat600-results.md), [implementation](../experiments/chat600.py).
- [Recorded protocol and IDs](../results/chat600/protocol.json), [every prediction](../results/chat600/predictions.json), [quality interval](../results/chat600/quality-analysis.json).
- [Effective-input overlap](../results/chat600/effective-input-overlap.json), [prospective extra requirement](../results/chat600/calibration-overlap-amendment.json), [797-message calibration check](../results/chat600/calibration-integrity.json).
- [Completed benchmark](../results/chat600/benchmark.json), [all timed outputs and layer traces](../results/chat600/timings.json), [all 12 corpus passes](../results/chat600/passes.json).

The defensible finding is that a task-trained binary classifier running 12 of 24 blocks processed this 600-message workload in about half the time, while missing more toxic messages and failing the stated acceptance rules. The current evidence establishes a measured fixed-depth speed/quality trade-off, not reliable adaptive readiness or accepted equal-quality acceleration.

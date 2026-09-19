# Adversarial follow-up: stronger evidence, reliability still unproven

Reviewed 2026-09-19 by the separate reviewer agent after the first [adversarial audit](adversarial-review.md). Scope is the local working tree and recorded experiments, not verification of a hosted deployment. The reviewer also implemented the separately written mechanism replay below; this is independent agent work within the same project, not external peer review.

**The mechanism is real and substantially easier to inspect and reproduce. The new tests expose serious limits: no early-exit gate has passed its acceptance rule, changing-rule performance is poor, and adding an UNKNOWN label did not solve unfamiliar requests.** Better documentation does not turn those failures into validated capabilities.

## Revised ratings

Same rubric as the first review: 0 means unsupported or misleading; 10 means independently verified and adequately supported for the stated scope. These are review judgments, not probabilities or research benchmark scores.

| Area | First review | Now | Basis |
| --- | --- | --- | --- |
| Numerical integrity | 9/10 | **9/10** | Additional scores, raw labels, policy decisions and confidence bounds independently recomputed without contradictions. Saved artifacts still do not independently attest to every historical computation. |
| Evidence of actual skipping | 8/10 | **9/10** | Separate code actually reran Qwen, reproduced labels/depths, and checked actual block-index traces. Both the historical installation and published Transformers 4.50.3 replay passed. |
| Real-world validity | 3/10 | **3/10** | Broader tests improve knowledge of the limits, but do not establish reliable general early stopping. Every calibration gate failed; the changing-rule test also failed. |
| Honesty of public claims | 7/10 | **9/10** | Visible copy now describes classifier confidence, fixed categories, failed reliability checks and failed changing-rule validation. Accuracy/timing sample sizes and the full-depth classifier baseline are clearly labeled. |
| Reproducibility | 6/10 | **8/10** | Portable weights, exact source snapshots, dependency/source hashes, and a successful published-release replay are now available. No clean-environment full training/evaluation replication or external replication has been performed. |

The real-world rating concerns the broad ambition of trustworthy decisions with dynamic instructions and unfamiliar inputs. The narrower finding that a trained fixed-category classifier can terminate actual transformer execution has much stronger support. No score is raised simply by redefining the broad ambition as a smaller demonstration.

## The strongest negative findings

### Every reliability gate still fails

The original BANKING77 candidates and conservative reserve already failed their checks. On the second dataset, the 30-intent CLINC candidate got **854/900** supported test requests right versus **856/900** at full depth. It would exit on 639 queries, with five wrong early answers and four new errors relative to full depth.

That is encouraging closed-set accuracy, but calibration rejected it: the added-error upper bound was **1.046%**, exceeding 1%, and the unfamiliar-input early-acceptance bound was **8.920%**, exceeding 5%. On the held-out unfamiliar set it prematurely answered **66/1,000** requests; another **11/120** unsupported-intent examples also received early answers. Continued requests are not successful rejections because this head has no UNKNOWN output. [Full evidence](clinc-results.md).

The candidate's **32.69% projected block reduction** comes from intermediate features collected by full-depth passes. It is not a new measurement of actual stopped execution or latency. A policy honoring the failed guard would use full depth.

### An explicit UNKNOWN output did not provide dependable rejection

The 31-output variant got **840/900** known test requests right versus **854/900** at full depth. It prematurely routed **299/1,000** unfamiliar requests to known categories. On the separate 120 unsupported intents it produced **52 wrong early known-category answers**. The full-depth UNKNOWN head itself recognized only **512/1,000** unfamiliar examples and **21/120** unsupported-intent examples.

Three calibration error checks failed. This variant changes the output categories, OOS supervision and random seed; its difference from the 30-output experiment cannot be attributed solely to adding an output. The known/OOS mixture also changes between calibration and test, so pooled error rates do not transfer unchanged. [Full evidence](clinc-unknown.md).

### Following changed instructions remains a substantive failure

For 100 held-out utterances paired with opposite A/B rules, the full-depth head answered both rules correctly on only **14/100 pairs**, against a pre-recorded 90% criterion. Its prompt-level score was **107/200**. The best observed intermediate result was **20/100 pairs** at layer 12; selecting it after seeing these results would require fresh validation.

The test is narrow: the same intent labels and pairs appear in training, with new utterances and one authored template change at test time. Even that bounded challenge was not solved. It does not support claiming that the current heads obey arbitrary user instructions. [Every prompt and result](changing-rules.md).

### Avoiding one output token has no demonstrated useful timing advantage

The matched control uses the same trained classifier rows for enum and constrained one-token output. Its identical predictions are guaranteed by construction; they do not compare independently trained model quality.

The measured mean difference was **1.60 ms** on approximately 287 ms of work, with a paired-query interval of **-2.94 to 6.13 ms**. This run does not establish a reliable latency advantage from avoiding one token. It uses a custom constrained readout rather than ordinary Qwen generation or a complete JSON response. [Control and limits](matched-output.md).

## What materially improved

The reviewer wrote a separate inference implementation, importing no original experiment implementation and loading no hidden-feature cache. The first fresh-process replay reproduced 48 full/early passes on 24 predetermined BANKING77 queries. A published-release replay reproduced those same queries plus a separately labeled, outcome-selected layer-18 case: **50 passes matched**, with verified actual layer-12, layer-18 and full-depth paths. The additional layer-18 case checks that code path; it is not representative accuracy evidence.

The existing classifier tensors are now available as NPZ numeric arrays loaded without pickle. Array roundtrip and bitwise-equal classifier-logit checks passed. Transformers 4.50.3 ran from an isolated local directory without changing the global installation. Its wheel hash, dependency versions and source hashes are recorded. This demonstrates limited compatibility with an obtainable release, not recovery of the exact historical development source or repetition of training. [Replay details](reproduction.md).

The homepage no longer claims the model knows an answer is ready. The decision page visibly states the fixed-category scope and changing-rule failure. The results table explicitly distinguishes 3,080 accuracy queries from 96 timing queries and labels the baseline as a full-depth trained classifier. Failed gates remain visible. These corrections address the first audit's main copy objections.

## Checks actually performed in this follow-up

- Reviewed CLINC, UNKNOWN, changing-rule and matched-output implementations prospectively, before their result claims, for label leakage and mismatched comparisons.
- Matched every new CLINC and UNKNOWN prediction's label against the original dataset and the supported-category list; checked query IDs/order against the saved protocol and verified source-file hashes.
- Recomputed known/unknown correct counts, early errors, added errors, coverage and mean depths. Reconstructed the 30-output gate's decisions from each recorded checkpoint score.
- Independently inverted the binomial cumulative distribution to verify the one-sided Clopper-Pearson bounds, rather than reusing the experiment's beta-quantile helper. Every bound and guard decision matched.
- Recomputed fixed-depth CLINC accuracies and the saved lexical-control accuracy.
- Checked changing-rule prompt mappings against raw utterance labels, opposite-label pairing and exact-text separation from training; recomputed every layer's prompt, paired and flip scores.
- Checked all 3,080 matched-format outputs against original predictions, the bijective token mapping and portable-weight hash. Recomputed all timing means/medians and the paired bootstrap interval.
- Ran the separate actual-inference replays described above and verified saved source/output hashes. No full-model training or complete new-dataset inference was rerun by the reviewer.
- Read the corrected visible homepage, decision, adaptive and results copy, plus the linked reports. This review does not independently re-audit unrelated coordinate or preference research, hosted publication, accessibility or physical devices.

## What blocks higher ratings

Reliable general early stopping needs a candidate that passes frozen acceptance rules on genuinely untouched evaluation data, with useful coverage and reliable unfamiliar-input behavior. The current failures must guide new training and gate design; repeatedly adjusting thresholds against these same tests would weaken the evidence.

Dynamic instructions need substantially better held-out paired-rule performance, diverse rule templates and independently checked labels. A public benchmark still leaves pretraining contamination and production distribution shift unresolved.

Practical value needs cost/quality comparisons against inexpensive specialists and efficient constrained outputs, plus actual stopped execution on any newly claimed task. Warm batch-one CPU savings alone do not establish batching, accelerator, memory, energy or browser benefits.

Reproducibility would improve further through a clean installable environment, fresh training and complete evaluation from the recorded inputs, and an external person or system reproducing the results. The same-machine agent replays are useful but cannot substitute for those checks.

**Defensible current claim:** trained intermediate classifiers can genuinely stop Qwen computation on a fixed-category task. The recorded candidates sometimes save work while keeping similar aggregate accuracy. Their reliability checks fail, and the broader instruction-following and unfamiliar-request goals remain unsolved.

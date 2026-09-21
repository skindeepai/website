# Accuracy and completion review

The latest output and late-layer results are internally consistent. No incorrect headline count or invented layer saving was found. The useful remaining work is improving and validating the models, not adding qualifications to the main pages.

This is a separate agent's code and artifact review inside the project. It checks retained evidence rather than independently repeating model inference.

## Checks completed

- Recalculated the output study's nine confusion tables and timing means, and the late-layer study's three timing and accuracy summaries. Published numbers match the records.
- Checked all 1,350 timed-call labels against the hashed, human-annotated ToxicChat CSVs. Training and evaluation conversation IDs are disjoint; no normalized full-message duplicate crosses those two splits.
- Reconstructed all 450 late-layer predictions using NumPy, the saved features and portable weights. Feature means and standard deviations come only from training examples. All reported layer traces match the selected depths.
- Read the training, first-token, vocabulary-score, generation, parser and timing implementations. The six original moderation methods already use numerical classifier heads. The browser vocabulary-score path and the new two-row CPU path remain correctly distinguished.

Sources: [output experiment](chat-output-steps.md), [late-layer experiment](chat-late-exit.md), [existing classifier implementations](chat-label-suite.md), [previous broader method audit](chat-smoke-review.md).

## Findings worth acting on

**Separate the sources of the output speedup.** The original two-row versus one-token comparison changes the vocabulary projection, KV-cache setting and generation API together. Its 6.6% end-to-end difference is real in the recorded calls, but that comparison alone cannot attribute the entire gain to a smaller output projection. Both recorded timing passes favor the two-row path, with paired mean differences of 51.1 and 43.3 ms. A new matched readout control addresses this question without retraining or choosing a better-looking quality sample.

**Optimize the useful error trade-off.** The trained classifier's 36/50 total exceeds the vocabulary path's 31/50, but it misses nine toxic messages rather than three. The vocabulary path blocks 16 of 25 benign messages. Neither is a practically strong moderator. The next training comparison should choose thresholds on separate development data at a specified toxic-recall or false-block target, then freeze them before evaluating fresh examples. Do not choose a deployment winner from this reused set.

**Treat late exits as candidates, not completed quality improvements.** Layer 22 corrects three full-depth mistakes while adding two toxic misses. Layer 23 exchanges a false block for a toxic miss. These implementations genuinely skip blocks, but their error changes still need work. A jointly trained earlier head with a conservative continuation gate is more useful to investigate than selecting the best depth from these 50 results.

**Test the application that is actually wanted.** These moderation studies classify individual bounded messages. They do not yet validate the application's recent-history, split-word abuse and multi-user context policy. A separate context-labelled benchmark is needed; tolerant parsing or faster readout cannot supply missing conversational evidence.

## Follow-up control

The [readout-control runner](../experiments/chat_readout_control.py) keeps the same 50 messages and weights, comparing two-row and full-vocabulary readouts with caching disabled, a two-row cached path, and constrained one-token generation. The first contrast isolates vocabulary projection. The generation comparison still includes several pieces of execution work; it does not isolate the generation wrapper alone.

The [independent audit](../experiments/check_chat_readout_control.py) verifies source and dataset hashes, declared execution order, every layer trace, original decision parity, selected scores and all summary arithmetic. Results and its audit report are retained alongside the original experiment rather than replacing it.

| Matched path | Mean ms/message | Readout stage |
|---|---:|---:|
| Two vocabulary rows, no cache | 650.9 | 0.055 ms |
| Full vocabulary, no cache | 695.5 | 41.106 ms |
| Two vocabulary rows, with cache | 656.2 | 0.055 ms |
| Forced one-token generation | 700.0 | Included in generation |

The same-cache two-row path takes **6.42% less total time** than full vocabulary scoring. Its readout saves about 41 ms; this explains most of the observed end-to-end gap in this run. All four paths preserve every original decision: 31/50 correct, three toxic misses and 16 false blocks. All 400 timed calls passed the independent audit, including original source labels and both earlier timing passes' decisions. The largest stored-score difference from the earlier experiment was 0.00000954.

This establishes a useful execution optimization, not a quality improvement. No layers are skipped. The cache and generation paths remain diagnostic controls; cached-state deallocation is outside the manual timer, and the generation wrapper was not separately isolated. The natural next implementation step is a browser export that computes only the needed output scores, followed by parity and timing checks in that browser runtime.

[Control results](../results/chat-readout-control/result.json) · [Raw calls](../results/chat-readout-control/records.json) · [Independent audit](../results/chat-readout-control/audit.json)

# Accuracy and completeness review — 20 September 2026

Three separate agents reviewed the results, implementation, unfinished work and evidence links. Their checks found no incorrect headline arithmetic in the audited experiments. This is an internal code/artifact audit, not an independent laboratory replication or a rerun of every model.

## Changes made

- Corrected stale evidence summaries and an offline-only description of a working browser demo.
- Fixed a generator that erased later research lessons, preserving the output/parser findings during rebuilds.
- Added anchor validation and repaired one archived sitemap link. Current topic/demo/result routes passed review.
- Ran a 400-call matched readout experiment: two scores took 650.9 ms versus 695.5 ms for full-vocabulary scoring, with identical decisions. The **6.4% saving** isolates the readout change under the same cache setting. [Experiment](chat-readout-control.md).
- Reanalysed privacy predictions at whole-name level: only 290/397 fully observed PERSON occurrences had every intersecting token marked. Token recall alone hid partial names and truncation. [Error analysis](privacy-spans.md).

The primary pages and layout remain unchanged. Findings live with the supporting research evidence.

Subsequent fix: the [name detector now checks the whole input](privacy-full-document.md), marking 29 more complete name occurrences in the same 50 documents. Its matching results page now compares full-input coverage with the original prefix behavior.

## Most useful next work

1. **Validate shared-model stopping on SMS spam.** This starts one of the five proposed but unfinished datasets. Freeze duplicate-group splits, compare a sparse baseline, then train the two/four-layer model and select stopping rules on development data. Test legitimate messages wrongly hidden as well as missed spam. Do not choose a winner from the already consumed ToxicChat results.
2. **Compare moderation methods at a useful error target.** Select thresholds on development data under a stated missed-toxic or false-block limit. Total accuracy currently ranks methods with very different error costs. Add labelled recent-chat cases for split-word abuse, quotations and harmless fragments; single-message benchmarks do not test that application.
3. **Strengthen early-exit evidence for rare errors.** The fresh shared-BERT study preserved all 500 decisions, but included only 18 toxic examples. More independently labelled toxic cases and shifted inputs are needed before treating that as a reliable moderation shortcut.
4. **Improve complete outputs, not just scores.** For privacy, process full documents and score whole-name coverage. For image actions, learn when to decline and measure successful tasks. For preferences, compare suggestions using actual blinded human choices. These outcome checks are more valuable than adding another confidence gate to weak predictions.

The original five dataset proposals, contextual live-chat validation, human preference study, image-generation cancellation, ECG/MRI applications and end-to-end GUI tasks are not all completed. They remain investigations, not advertised results.

## Detailed reviews

- [Accuracy, split integrity and timing](adversarial-accuracy-review.md)
- [Completeness, documentation and navigation](adversarial-completeness-review.md)
- [Other methods and practical error analysis](adversarial-methods-review.md)

Raw experiments and their checks remain linked from those reviews. The existing UI needs no additional qualification blocks to carry these findings.

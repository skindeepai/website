# Completeness review, 20 September 2026

Separate agent review of the local checkout at `1c48ddd`, with targeted corrections afterward. This review inspected research ledgers, current result pages, experiment runners, demo documentation and navigation. It did not rerun every historical model or inspect the hosted deployment.

## Findings and corrections

- **The current page routes are intact.** Checks passed for 133 current/archive pages and 3,792 local references, metadata on 112 current pages, and 46 use cases with 59 topic round trips and 76 mapped demo links. An additional HTML audit found no broken local fragments between current pages.
- **Several evidence summaries were stale.** The claim ledger still treated the exact browser sampler as untested, described only the initial six-layer exit, and omitted the later abstention diagnostic. The research ledger omitted fresh shared-BERT validation and combined caching/batching results despite their presence in its source metadata. These summaries now point to the completed evidence.
- **A working demo was called offline-only.** The search notes now link to the live TinyBERT reranker and its runtime checks.
- **Regeneration erased later lessons.** `refresh_preference_followups.py` replaced the research ledger including its manually maintained output/parser lessons. It now preserves the additional-lessons section while regenerating the 29 protocol rows from `content/research-evidence.json`.
- **The 6.6% result is a whole-path comparison.** The two-score runner disables KV caching; the one-token generation runner enables it. The difference also includes generation API overhead and vocabulary scoring. Identical decisions and measured elapsed times remain valid, but that run alone does not isolate the cost of projecting two vocabulary rows. A matched cache/projection ablation is the immediate follow-up.

## Work that is still incomplete

The five datasets proposed in [the dataset plan](real-world-datasets.md)—SMS Spam Collection, CFPB complaints, GoEmotions, PhiUSIIL and smartphone activity recognition—do not have completed experiments in this checkout. Later practical studies instead covered TAB names, SROIE receipt totals and CLINC routing; those are useful additions, not completion of the original five.

Human preference validation, real generator transfer, image-generation cancellation, ECG/MRI applications, contextual live-chat decisions and end-to-end GUI task completion remain proposals or missing studies. Existing related demos illustrate mechanisms; they do not validate these applications. The browser Qwen demo also does not yet implement the new two-row CPU path or the trained Qwen classifier.

## Next investigations, in order

1. **Isolate output costs.** Compare the same backbone call with full versus two-row projection, cache on/off, and one-token generation without an unused cache. Verify label/logit parity and use paired timing. This answers whether the newest speed gain comes from readout, cache construction or generation machinery.
2. **Validate the useful shared-model shortcut on a different dataset.** Start the already proposed SMS benchmark with duplicate-group splits, a sparse baseline and the shared two/four-layer model. Select thresholds on development data and freeze them before 50–100 smoke-test messages. The existing 500-message ToxicChat remainder has only 18 toxic examples, so it cannot tightly establish rare-error behavior.
3. **Test actual application context.** The current moderation studies classify isolated bounded user messages. Build a separately labelled contextual set with split-word abuse, quotations and harmless fragments before claiming equivalence to Steve's recent-chat prompt. Report error changes by case type, not only total accuracy.
4. **Improve the decision before adding another gate.** The selected distilled BERT corrected false blocks but increased toxic misses; deeper Qwen is not always a better teacher. Compare supervised learning and distillation under a development-selected toxic-recall constraint, retaining all seed outcomes. Do not promote a seed based on the already consumed test set.
5. **Measure completed tasks.** For preferences, collect blinded choices between equally budgeted suggestions. For clicks, introduce a learned no-action output using development examples and score both successful valid actions and accepted invalid actions on separate data. More layer savings will not fix unreliable targets.

These are investigations, not newly achieved results. No additional primary-page fine print or visual redesign was needed for the documentation corrections.

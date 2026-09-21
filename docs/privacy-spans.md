# How much of each annotated name was marked?

The earlier name detector found 88.5% of PERSON tokens. Looking at complete annotated occurrences exposes a more useful failure measure: **290 of 397 fully observed name spans had every intersecting token marked (73.0%)**. Another 70 were partly marked and 37 were missed entirely.

This is retrospective analysis of the same 50 TAB documents and frozen classifier outputs. No model, threshold or prediction changed; it is not a fresh test.

| Annotated PERSON occurrence | Count |
|---|---:|
| Fully marked within the processed prefix | 290 |
| Partly marked within the prefix | 70 |
| Entirely unmarked within the prefix | 37 |
| Crosses the 1,200-token boundary | 3 |
| Entirely beyond the boundary | 45 |
| Total distinct occurrences | 445 |

**26 of 50 documents contain an incompletely marked PERSON occurrence**, counting both detection misses and the unprocessed tail. All PERSON occurrences were marked in the other 24. “Marked” means covered by positive token predictions, not safely anonymized: other identifiers, unannotated names and context can still identify someone.

## Counting and checks

An occurrence is a distinct `(start_offset, end_offset)` span, unioned across the human annotators as in the original test. Identical offsets count once; nonidentical overlapping spans remain separate. Repeated mentions at different positions count separately. Thus this is occurrence coverage, not recall of distinct people or exact-boundary entity F1.

An entirely observed occurrence is fully marked when every regex token intersecting it has a positive prediction at the original 0.6 threshold. Whitespace is not scored. Spans extending beyond the final processed token are counted separately, even if their visible part is marked. False positive tokens remain in the original report: **339**. This coverage diagnostic does not reward or evaluate over-redaction as correct anonymization.

The independent calculation checks the source-data and saved-probability hashes, exact document order, every retained token label, and every document's confusion counts. It reproduces the original 1,088 true positives, 141 false negatives and 339 false positives before computing span coverage. The public record contains offsets and numeric predictions, without copying document text.

Run `python experiments/privacy_span_audit.py --check` using the pinned TAB test cache and saved replay NPZ. A first run without `--check` creates the result and refuses to overwrite an existing one. This performs no training or model inference and caps numerical pools at one thread.

## Next improvement

The [full-input follow-up](privacy-full-document.md) now removes the old prefix boundary in the editable demo. It marks 29 additional complete occurrences on these same documents; the original findings above remain the record of the earlier behavior.

Next, compare a contextual entity tagger that can mark complete multi-token names, using overlapping windows if its context length requires them. Select rules on development documents and measure complete-occurrence misses, false masking and whole-document coverage on separate documents. Simple expansion around flagged tokens is worth a development test, but these already inspected 50 documents cannot certify that change. Timing should exclude gold-label alignment and include the entire document.

[Reproducible audit](../experiments/privacy_span_audit.py) · [Every occurrence and source hash](../results/privacy-spans/result.json) · [Original practical study](practical-baselines.md) · [Original numerical replay](practical-replay.md).

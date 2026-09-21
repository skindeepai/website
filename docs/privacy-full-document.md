# Check the whole input for names

The browser name detector now examines every token in its accepted input instead of stopping at 1,200. The trained weights, features and 0.6 cutoff are unchanged. Its existing 30,000-character input limit remains; no input is sent to a server.

On the same 50 TAB documents, the fix fully marked **29 additional name occurrences** without losing any previously fully marked occurrence. It also marked 39 more unrelated tokens. This repairs an input-coverage gap; it does not solve every classification error.

| Same classifier on 50 documents | Old prefix | Entire input |
|---|---:|---:|
| Tokens examined | 39,626 | 45,481 |
| Complete PERSON occurrences marked / 445 | 290 | 319 |
| Partly marked occurrences | 70 | 80 |
| Entirely missed occurrences within examined text | 37 | 46 |
| Occurrences partly or wholly outside examined text | 48 | 0 |
| Unrelated tokens incorrectly marked | 339 | 378 |
| Documents with every annotated PERSON occurrence marked / 50 | 24 | 25 |

Both complete-occurrence counts use **445**, including the old unprocessed tail. The earlier 73.0% figure used only the 397 entirely observed occurrences; comparing that percentage directly with this full-document rate would change the denominator. An occurrence is a distinct annotated offset pair, not a distinct person or proof of anonymization. Details of the counting rule are in the [span audit](privacy-spans.md).

The full path found 1,157/1,328 PERSON tokens and missed 171. Counting the unexamined tail as missed, the old path found 1,088 of those same 1,328. The extra context can also change the old final token's next-word feature. All other overlapping tokens retain exactly the same features and scores. No model or threshold was selected using this follow-up.

## Implementation and checks

The classifier needs only the current word and its immediate neighbors, so full-input scoring needs no new model or approximate overlapping-window merge. `PracticalDemo.privacy(text, model, {fullDocument: true})` evaluates all tokens. The default prefix mode remains available to reproduce the original experiments; the editable privacy demo explicitly requests the full mode. Routing and receipt behavior are unchanged.

Two alternating-order JavaScript passes reproduce every old per-document token and span count before scoring the full input. Both repeats produce identical outputs. Data, model, source and record hashes are retained. All 50 documents fit the existing browser input limit. Tokenization, features and classification averaged 2.8 ms per document for the prefix and 3.0 ms for the full input in Node; these are diagnostic costs, not a browser or LLM speed claim. Loading, annotation alignment and result scoring are outside the timer.

A separate Python implementation reconstructed 85,107 token predictions across both paths and matched every per-document confusion count and annotated-span outcome. Actual Chromium checks cover tail highlighting, Unicode offsets, the retained input limit, routing/receipt behavior, and 14 populated mobile/desktop page states. [Numerical audit](../results/privacy-full-document/audit.json) · [Browser checks](../results/privacy-full-document/browser.json).

These are already inspected test documents, not fresh quality validation. A contextual name model and independent document evaluation remain worthwhile follow-ups. Other identifiers and ambiguous names still prevent treating this as complete anonymization.

## Evidence

- [Protocol, hashes and exact document IDs](../results/privacy-full-document/protocol.json)
- [Summary](../results/privacy-full-document/result.json), [every occurrence](../results/privacy-full-document/records.json) and [timings](../results/privacy-full-document/timing.json)
- [Runner](../experiments/privacy_full_document.cjs) and [frozen classifier implementation](../results/privacy-full-document/core.cjs)
- [Original browser source copies and hashes](../results/privacy-full-document/prior-source.json), including the exact core from the earlier browser protocol
- [Independent Python checker](../experiments/check_privacy_full_document.py) and [browser regression test](../scripts/test_practical_full_input.cjs)
- [Original training and data](practical-baselines.md#find-names-before-redacting-text)
- [Current results](../redaction-results.html) and [live name detector](../practical-demo.html?task=privacy)

The runner protects completed evidence from overwrite. Use a separate output directory for a repeat; no raw document text is republished in these records.

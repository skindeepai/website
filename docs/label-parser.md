# SAFE versus OK after parsing

Applying the supplied parser to the same generated replies gives **SAFE 29/50 correct (58%) and OK 26/50 (52%)**. A more defensive format parser produces exactly the same decisions on these messages. SAFE scores higher in this sample, but the difference is not statistically conclusive (paired exact p = 0.549).

This is a post-hoc rescore of the [existing label-wording experiment](label-wording.md), not another model run. All three repeated passes were rescored and agreed; accuracy counts the 50 distinct messages once. The direct-score experiment is unaffected because it already returns a valid label.

| Evaluation | SAFE correct / 50 | OK correct / 50 |
|---|---:|---:|
| Original: invalid reply counts as an error | 27 | 25 |
| Exact label only, otherwise block | 29 | 26 |
| Supplied synonym/first-match parser, otherwise block | 29 | 26 |
| Enhanced format/negation parser, otherwise block | 29 | 26 |

## What the parser actually changed

The malformed outputs were unrelated continuations such as “Certainly! Here is an introduction…” or “StoryBot writes the following…”. None contained a recoverable decision label. They were capped at eight generated tokens in the original experiment; parsing cannot recover a later verdict that was never generated.

| With either supplied or enhanced parser | SAFE | OK |
|---|---:|---:|
| Successfully parsed replies | 44/50 | 48/50 |
| Correct among parsed replies | 27/44 | 25/48 |
| Unparsed replies blocked by default | 6 | 2 |
| Correct default blocks on toxic messages | 2 | 1 |
| Incorrect default blocks on benign messages | 4 | 1 |
| Toxic messages missed, overall | 3/25 | 0/25 |
| Benign messages blocked, overall | 18/25 | 24/25 |
| Invalid replies with a label recovered by tolerant parsing | 0 | 0 |

The score increase comes entirely from the fallback policy, not improved understanding or successful error correction. OK catches all 25 toxic examples here, but also blocks 24 of the 25 benign examples. Keep `parseOk` separate from the block decision when reporting results.

## Parser behavior

The supplied function is preserved exactly in [censor_decision_parsers.cjs](../experiments/censor_decision_parsers.cjs), with its name changed for export. Its first matching word can produce incorrect approvals:

| Reply | Supplied parser | Enhanced parser |
|---|---|---|
| `not SAFE` / `not OK` | Allow, parsed | Block, parsed |
| `SAFE. Actually BLOCK.` | Allow, parsed | Block, unparsed |
| `The message says "OK" but is abusive` | Allow, parsed | Block, unparsed |
| `Do not block` | Block, parsed | Allow, parsed |
| `{"label":"SAFE"}` | Allow, parsed | Allow, parsed |
| Bare `YES` | Allow, parsed | Block, unparsed: question-dependent meaning |

The enhanced parser accepts explicit aliases, simple punctuation/Markdown, a single JSON verdict field, a verdict prefix, and a few complete negated phrases. Leading verdicts may have an explanation, but contradictory or uncertain wording is rejected. It does not search arbitrary prose for an approval word. Duplicate/multiple JSON verdict fields are rejected rather than choosing one.

These are bounded formatting rules, not general natural-language reasoning. The enhancement is more conservative and can reject a legitimate explanation; it is not proven universally better. All 33 hand-authored diagnostic cases passed, but they are implementation checks, not additional real-user accuracy samples. Parser design happened after the model outputs had been inspected.

## Speed

The saved model times remain SAFE **1,014 ms/message**, OK **937 ms/message**, averaged across three passes. Parsing afterwards cannot shorten text that has already been generated. Both parsers took less than 0.001 ms/reply in a separate local repeated-string microbenchmark; this is negligible next to inference, not a new end-to-end model timing run.

## Evidence

- [Parser implementations](../experiments/censor_decision_parsers.cjs)
- [Rescoring script](../experiments/rescore_label_wording.cjs)
- [Rescoring protocol and diagnostic specifications](../results/label-wording/parser-rescore/protocol.json)
- [Summary](../results/label-wording/parser-rescore/result.json)
- [Every parsed result, including fallback flags](../results/label-wording/parser-rescore/records.json)
- [33 diagnostic cases and both parsers' outputs](../results/label-wording/parser-rescore/diagnostics.json)
- [Source and arithmetic audit](../results/label-wording/parser-rescore/audit.json)

No application parser or website model default was changed. The underlying experiment still uses the isolated-message prompt, not the application's recent-history policy.

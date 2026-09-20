# Independent reproduction of the practical baselines

All three original runs reproduced exactly in separate processes: every saved per-item outcome, aggregate and selected threshold matched. Timings were excluded from comparison because the replay adds diagnostic capture and is not a speed benchmark.

| Task | Reproduced result |
|---|---|
| Court-document names | 1,088 true positives, 141 missed name tokens, 339 false positives across 50 documents / 39,626 tokens |
| Receipt totals | Learned selector 32/49; largest-amount rule 27/49; one missing reference retained |
| Request routing | Raw classifier: 833/900 supported requests correct, 393/1,000 unsupported rejected; rejection rule: 733/900 and 894/1,000 |

The [replay runner](../experiments/practical_replay.py) executes the exact sealed source used for each original task. It redirects output to a new directory and copies each classifier's returned probability matrix; it does not change training, features, thresholds or predictions. Privacy and routing use their archived source because later receipt parsing fixes changed the common runner. The [replay protocol](../results/practical-replay/protocol.json) pins every source, original protocol and original result before execution. Each process caps numerical pools at two threads and ran after the isolated transformer timing work finished.

The replay retains more numerical evidence than the original pilot:

- Privacy: all development/test probability matrices, test token labels and fitted numerical coefficients. [Comparison](../results/practical-replay/privacy/comparison.json), [array-to-document index](../results/practical-replay/privacy/diagnostics.json), [numerical arrays](../results/practical-replay/privacy/numerical-predictions.npz).
- Receipts: candidate amounts, selected amount, largest amount, reference total, every candidate probability and fitted coefficients. [Comparison](../results/practical-replay/receipts/comparison.json), [candidate outcomes](../results/practical-replay/receipts/diagnostics.json), [numerical arrays](../results/practical-replay/receipts/numerical-predictions.npz).
- Routing: complete development/test class probabilities, class order and fitted coefficients. [Comparison](../results/practical-replay/routing/comparison.json), [array index](../results/practical-replay/routing/diagnostics.json), [numerical arrays](../results/practical-replay/routing/numerical-predictions.npz).

The independent audit also reconstructed the source splits, PERSON annotation unions, title-rule baseline, normalized merchant groups, receipt candidates and largest-amount baseline, and all routing rejection decisions. Dataset and executed-source hashes matched. The browser specialist export was separately checked against its saved 100-message report: decisions and preprocessing matched, with 82 correct, 13 toxic messages missed and 5 false blocks.

This establishes computational reproducibility on the same examples. It does not create a new holdout, prove production reliability, validate full anonymization, include receipt OCR, or establish an LLM speedup. Raw input text is omitted from the new diagnostics. The fitted numerical coefficients alone are not standalone deployable models; exact feature construction remains in the pinned source and data. Receipt training originally left liblinear's random seed unspecified; the replay preserves that choice and still reproduced all outcomes.

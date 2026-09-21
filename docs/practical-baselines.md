# Three practical starting points

These small experiments establish ordinary task-specific baselines before adding a language model, early exits or a cascade. They do not establish an LLM speedup. All runs cap numerical worker pools at two threads. The scripts, frozen IDs and per-item outcomes are linked below.

A separate agent reran all three studies from their exact recorded source and reproduced every original outcome and selected threshold. The [independent replay report](practical-replay.md) includes complete numerical predictions, token labels and receipt candidate choices. This is same-data reproducibility, not fresh quality validation.

## Find names before redacting text

The editable demo now checks the [entire input](privacy-full-document.md). The figures below preserve the original bounded experiment.

On the first 1,200 tokens of each of 50 held-out TAB court documents, a trained token classifier found **1,088 of 1,229 person-name tokens**, missing 141 and incorrectly marking 339 other tokens. A fixed title rule (Mr/Mrs/Ms/Dr followed by capitalized words) found 646, missed 583 and incorrectly marked 6. Learned precision/recall were **76.2% / 88.5%**, versus **99.1% / 52.6%** for the rule.

This is name detection, **not complete anonymization**. It does not cover addresses, dates, indirect identifiers, the identity of the person to protect, or all text in a document. Thirteen test documents exceeded the 1,200-token prefix. Token recall is not the chance that a person remains unidentifiable.

A retrospective [complete-name coverage audit](privacy-spans.md) found all tokens marked in 290/397 fully observed PERSON occurrences (73.0%); another 48 occurrences cross or fall beyond the input limit. It uses the same frozen predictions, with no retraining or new accuracy test.

The [Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) contains real ECHR court documents with human annotations. We preserve its official train/dev/test division and verify no document ID overlap. A fixed seed samples 100 training, 25 development and 50 test documents. PERSON annotations are unioned across annotators; this simplifies their differing annotations and is not the official subject-specific masking evaluation. The token classifier uses word, neighboring word, capitalization and prefix/suffix features. Development F2 selects its threshold. No test labels select it.

This initial runner performs annotation alignment during preparation, so the saved seconds are diagnostic and **not inference latency**. A next implementation should separate annotation processing from timed inference, measure full-entity misses and evaluate entire documents.

[Protocol](../results/practical-privacy/protocol.json) · [Results](../results/practical-privacy/result.json) · [Per-document outcomes](../results/practical-privacy/records.json) · [Exact executed source](../results/practical-privacy/implementation.py).

## Read a receipt's total

A trained candidate selector returned the correct total on **32 of 49 scorable held-out receipts**, versus **27/49** when always choosing the largest decimal amount. All 49 gold totals appeared among the candidates. One additional held-out receipt lacked a usable total annotation; it remains recorded and is excluded from accuracy denominators.

Input is **provided transcription and text boxes**, not pixels. We have not measured OCR, scan processing or end-to-end receipt extraction. Candidate features include position, amount and nearby words such as total, cash and change. Logistic regression learns which candidate amounts match the training totals; repeated occurrences of the same gold amount are all positive examples.

The source is a pinned [contestant-maintained, corrected SROIE mirror](https://github.com/zzzDavid/ICDAR-2019-SROIE), derived from the [ICDAR SROIE receipt task](https://arxiv.org/abs/2103.10213). This is not an official SROIE leaderboard evaluation. We take numbered receipts 000–239, group by normalized merchant name, split groups 70/30 with a fixed seed, train on 156 receipts and inspect the first 50 held-out receipts. Name grouping can miss related chains and shared templates. No raw receipt text or images are republished here.

Two mechanical issues were corrected before a completed run: a protocol dictionary key collision and currency-prefixed/empty reference totals. Failed source/protocol versions are retained. The final attempt accepts currency prefixes, records missing references, and retains all 50 chosen IDs. No classifier setting was adjusted after scoring the held-out results.

[Protocol](../results/practical-receipts/protocol.json) · [Results](../results/practical-receipts/result.json) · [Every outcome](../results/practical-receipts/records.json) · [Initial failed source](../results/practical-receipts/initial-implementation.py) · [Currency-handling attempt](../results/practical-receipts/currency-implementation.py).

## Route a request, or decline to route it

A word/phrase classifier recognizes 30 supported intents plus UNKNOWN. On the existing 900 supported and 1,000 unsupported CLINC test requests:

| Decision rule | Supported intent correct / 900 | Unsupported rejected / 1,000 | Supported requests rejected / 900 |
|---|---:|---:|---:|
| Highest category score | 833 | 393 | 16 |
| Also reject low scores | 733 | 894 | 152 |

The rejection rule catches more unsupported requests, but sacrifices correct handling of supported ones. It is not an unconditional improvement. The threshold was selected on development requests to balance supported-intent accuracy and unsupported rejection. It was not selected on these test results.

The [CLINC dataset](https://github.com/clinc/oos-eval) uses human crowdsourcing and imagined scenarios, not production request logs. This pilot reuses the exact 30-intent subset and 80 UNKNOWN training examples from our previous experiment. Word/bigram TF-IDF plus logistic regression is an ordinary baseline. Its 1,900-request batch took roughly 25 ms during concurrent research work; that observation is not a paired comparison with Qwen or a live-service latency benchmark.

[Protocol](../results/practical-routing/protocol.json) · [Results](../results/practical-routing/result.json) · [Every outcome](../results/practical-routing/records.json) · [Exact executed source](../results/practical-routing/implementation.py).

## Reproduction and next tests

`python experiments/practical_baselines.py privacy`, `receipts`, or `routing` runs the selected study and refuses to overwrite completed results. Use a fresh checkout/output directory for reproduction. Privacy and routing have archived copies of their exact executed source because the later receipt parsing correction changed the common runner. Source and dataset hashes are recorded in each protocol.

Before using any candidate in an application, test fresh data, ambiguous/unsupported inputs and the relevant error costs. Add a learned encoder only after comparing it against these simple controls. None of these experiments validates private-data protection, financial processing or autonomous action in production.

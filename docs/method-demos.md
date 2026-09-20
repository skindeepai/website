# Live method and task demos

Every topic and measured method has a direct, labelled entry point. [The live demo directory](../demo-directory.html) groups them; [the coverage map](../content/demo-coverage.json) records all 43 page-to-demo links. Historical investment/archive pages are not recast as working products.

## What actually runs

| Demo | Live computation | Relationship to the research |
|---|---|---|
| [Decision methods](../method-demo.html) | ONNX BERT: fixed layer 2, adaptive 2/4, independent tiny→four-layer fallback, changed training, batch four, float32/INT8 | Real smaller-model implementations; not replicas of the Qwen experiments |
| [Qwen execution](../execution-demo.html) | Actual Qwen 0.5B q4 forward passes, private instruction KV caches, batches of four | Same mechanisms; vocabulary scores and 64-token messages differ from the trained float32 task-head studies |
| [Practical classifiers](../practical-demo.html) | Fitted name-token, receipt-candidate and request-routing classifiers in JavaScript | Re-created from the sealed recipes; original quality summaries reproduced |
| [Search reranker](../rerank-demo.html) | Same pinned float32 two-layer TinyBERT over the keyword top 20 | Actual query–document inference, including optional skip rule |
| [Maze](../maze-live.html) | Exported 48→128→128→4 MLPs, GELU activations, optional legal-action mask | Reproduces the small learned policies; distinct from earlier Qwen replays |
| [Screenshot point](../point-demo.html) | English Tesseract OCR, exact normalized text matching, bounding-box center | A working alternative to heavy GUI-Actor inference; not the recorded pointer model |
| [Preference methods](../preference-method-demo.html) | Browser logistic training, linear/quadratic features, random/uncertainty sampling | Editable small renderer; optional example ratings are explicitly synthetic |

The existing preference, shared-model, direct-output, image-action and search demos remain available. Drawings and arrows remain synthetic. Proposed music/dating/science applications link to the working preference loop with an explicit scope note rather than claiming an implemented domain product.

## Model assets and inputs

`export_method_lab.py` exports the development-selected `distill-9927` four-layer checkpoint and a dynamic per-channel INT8 MatMul conversion of the original BERT full graph. The trained export matched PyTorch on all 100 consumed messages. The INT8 conversion is a new BERT illustration, not the earlier Qwen diagnostic. [Protocol](../results/method-lab/export-protocol.json), [export results](../results/method-lab/export.json), [model hashes and thresholds](../models/method-lab/manifest.json).

The cascade pairs the existing 4.37M two-layer BERT with the 11.1M four-layer BERT. Its illustrative low/high cutoffs are 0.05/0.95, not validated for this new pair. Fallback executes both encoders; it neither shares representations nor claims equivalence to the measured tiny→Qwen cascade. Different-width block counts are not comparable FLOPs.

The practical export re-executes each original recipe against its pinned cached data, captures the trained numerical weights, and preserves all previous artifacts. [Privacy export](../results/practical-browser/privacy/export.json), [receipt export](../results/practical-browser/receipts/export.json), [routing export](../results/practical-browser/routing/export.json). Editable receipt text uses supplied rows or SROIE box CSV; there is no OCR in that classifier demo. Privacy features follow Python title/digit behavior, including Unicode edge cases checked separately; the demo does not establish complete anonymization.

Neural model downloads begin after a run button. The tiny maze weights load on page entry. Typed messages and uploaded images stay local; downloaded comparison reports omit the typed message. Public-data reports retain IDs, labels and predictions. Browser runtimes/tokenizers are downloaded from their documented sources. BERT moderation artifacts use ToxicChat (CC BY-NC 4.0) and are research artifacts for noncommercial use; base BERT and the TinyBERT reranker are Apache-2.0.

## Runtime validation

- **2,400 actual browser method calls:** four passes/paths per mode across all 100 messages. Fixed, adaptive, cascade, training and batch predictions matched their expected exported model decisions; routes, repeated decisions, error arithmetic and actual graph-call counts checked. [Independent artifact audit](../results/browser-demos/method-audit.json).
- **INT8 runtime difference retained:** the browser conversion matched its browser float32 labels on this 100, while Python INT8 differed on one message. The initial strict CPU/WASM equality test failed and remains recorded. This is not evidence of generally lossless conversion. [Original failed assertion](../results/browser-demos/first/failure.json), [complete browser outputs](../results/browser-demos/first/method-quantization.json).
- **Actual Qwen caching and batching:** four real messages through four paths, two reverse-order passes. All repeated and cross-path decisions matched, but only **1/4** reference answers were correct. These verify execution, not useful moderation quality. Caching was faster here; batching alone was slower. [All 32 outcomes and measured times](../results/browser-demos/remainder3/qwen-execution.json). Prefix construction and private cache cloning are charged to cached workloads. The token counter excludes padding and is not total computation; all 24 layers still run.
- **Search:** the first browser attempt exposed Transformers.js 3.8.1 truncating the final separator from long BERT pairs. Explicit longest-first pair truncation before adding CLS/SEP corrected this. The first two sealed queries now reproduce Python's top ten. [Successful first query](../results/browser-demos/remainder3/rerank-1.json), [original mismatching output](../results/browser-demos/remainder/rerank-1.json). A subsequent BigInt conversion error was caught before inference and fixed; its failure record remains under `remainder2/`.
- **Maze and practical numbers:** all **595 episode moves** and **1,024 state decisions** match the saved policies. Thirty-two practical readout fixtures agree within 2.3e-16; 13 whole-input cases check tokenization, features and probabilities, including Unicode name tokens. [Numerical parity](../results/browser-demos/numerical-parity.json), [input fixtures](../results/practical-browser/input-fixtures.json).
- **Actual OCR:** the generated example returns the Continue text location; nonexistent text returns NO TEXT MATCH. [Observed OCR result](../results/browser-demos/remainder3/ocr.json). This is a functional smoke test, not screenshot-benchmark accuracy.
- **UI:** own-text inference, no invented accuracy, export without typed text, real-message results, stopping, preference updates and all 43 deep links passed. Populated results fit seven viewport widths. [UI checks](../results/browser-demos/ui/result.json). The whole site passed **476 page/viewport checks** across 68 pages. [Site regression](../results/ui/result.json).

The final complete runtime suite passed: all 2,400 decision-method calls, both reranking queries, actual OCR, all 32 Qwen path/message outcomes and 49 new-page viewport checks. [Final result](../results/browser-demos/final/result.json), [frozen source hashes](../results/browser-demos/final/protocol.json), [final Qwen records](../results/browser-demos/final/qwen-execution.json). The stricter failed INT8 cross-runtime equality check remains preserved as described above.

Browser checks used Chromium 151 and one WASM thread. They do not establish physical iOS/Safari support, production reliability, independent-hardware speedups or fresh accuracy. Earlier failures remain available; successful checks do not overwrite those records.

## Re-running

Build with `python scripts/refresh_research.py` and `python scripts/build_site.py`. Serve the repository over localhost or HTTPS; model workers cannot be validated by opening `file://` pages directly.

`node scripts/live-demos-test.cjs NEW_NAME` runs actual models and retains results under a new directory. `--rest` skips the six decision-method workloads. `node scripts/live-demos-numerical-test.cjs` checks portable arithmetic and saved maze outcomes. `node scripts/live-demo-ui-test.cjs` checks controls and mapped links. Set `PLAYWRIGHT_MODULE` to an installed Playwright package if it is not on Node's search path. Exports require the original ignored research caches; this is not a clean-checkout training installer.

Implementation references: [pinned Transformers.js model API](https://huggingface.co/docs/transformers.js/v3.8.1/api/models), [Tesseract.js 5.1.1](https://github.com/naptha/tesseract.js/tree/v5.1.1), [original search method](search-next.md), [maze training](maze-smoke.md), [practical baselines](practical-baselines.md).

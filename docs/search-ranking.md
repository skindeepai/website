# Small-model search over real papers

This pilot compares **BM25 word matching, MiniLM embeddings, and a fixed combination of the two**. Every method ranks the same full corpus of 5,183 scientific abstracts. The live demo accepts a new query and actually scores that corpus on the user's device; it does not replay saved search results or generate an answer.

## What is real?

[SciFact](https://github.com/allenai/scifact) contains real scientific abstracts and expert-authored claims with human evidence annotations. The claims were written for research, not collected from production search logs. A relevant paper may **support or refute** a claim. Ranking it does not establish that the claim is true. [Original paper](https://aclanthology.org/2020.emnlp-main.609/).

We use the [BEIR SciFact corpus and queries](https://huggingface.co/datasets/BeIR/scifact/tree/b3b5335604bf5ee3c4447671af975ea25143d4f5), revision `b3b5335604bf5ee3c4447671af975ea25143d4f5`, and [official test relevance judgments](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029), revision `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029`. Python's random seed 45109 chooses 100 of the 300 official test query IDs before any scoring. The first pilot uses no labels to train a model, choose an example pool, tune a weight, or inject a gold document into a shortlist. A separate supervised follow-up below uses official training labels only.

The full collection is still small and domain-specific. Results do not measure searching all scientific literature, the web, or a production corpus. Unjudged papers count as nonrelevant for the benchmark; some may in fact contain useful evidence.

MIRACL was investigated first. Its English corpus contains 32.9 million passages, and a bounded corpus lookup timed out in this environment. SciFact lets us include the entire fixed corpus without an arbitrary convenience subset or a large download. This experiment is therefore **not a MIRACL result**. [MIRACL corpus and annotation description](https://github.com/project-miracl/miracl).

## Models and scoring

**BM25:** lowercase ASCII letter/number tokens from each full title and abstract, no stemming or stopword removal, `k1=1.2`, `b=0.75`. Its index is built once. Ties use original corpus order. This is a transparent lexical baseline, not a neural model.

**MiniLM:** [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/751bff37182d3f1213fa05d7196b954e230abad9), pinned revision `751bff37182d3f1213fa05d7196b954e230abad9`, 8-bit ONNX weights. It runs all **six transformer layers**. Attention-mask mean pooling of the final states, followed by L2 normalization, produces **384 numbers**; dot product compares normalized query and document vectors. There is no generated text and no early exit. [Original model and training description](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Both queries and document title-plus-abstract inputs are capped at 256 WordPiece tokens including special tokens for MiniLM. BM25 reads the full text. This is a practical method comparison, **not an input-length-controlled ablation**. The result artifact counts truncated documents and queries. MiniLM was pretrained on broad sentence-pair collections including scientific text; we cannot claim its pretraining excluded these abstracts.

**Combined ranking:** reciprocal-rank fusion sums `1/(60+rank)` from each complete ranking. The constant is fixed before evaluation. The combined path independently runs both component methods for its timed request; it does not reuse their timed query output for free.

## Results and timing

The measured values are in [result.json](../results/search-ranking/result.json), with [all per-query top-ten rankings and relevance metrics](../results/search-ranking/predictions.json) and [all timing repetitions](../results/search-ranking/timings.json). The [demo page](../search-demo.html) presents relevant-first counts and median query time. Keep weak methods and errors visible; no quality-preserving speedup is assumed.

Metrics are hit@1, recall@5, nDCG@10, and reciprocal rank@10, macro-averaged over the 100 queries. The fixed official relevance judgments determine gains. The full 5,183 candidates are used by every method.

Each method runs three timed passes per query; order rotates. Query time includes text processing, MiniLM query tokenization/encoding where applicable, complete corpus scoring, sorting, and fusion. It excludes evaluation against labels. Loading the model, constructing the lexical index, and encoding the document corpus happen separately and are reported as setup costs. This is the usual reusable-index scenario, not a claim that document encoding is free. A changing corpus requires fresh document work.

The local experiment uses ONNX Runtime 1.23.2, CPU execution, two intra-op threads, one inter-op thread, and sequential graph execution. Other numeric pools are capped at two. Exact hardware/runtime, date, source hashes, and setup times are in the artifacts. These local Python measurements are distinct from the browser's displayed timings.

The browser runs the same pinned q8 model through Transformers.js 3.8.1 and WebAssembly with one compute thread. It downloads stored document vectors and encodes each new user query. Its query timing includes scoring and encoding but excludes corpus/model download and setup, which are shown separately. The browser model may remain in cache; Stop terminates its worker. No query or result is uploaded by the application.

## Reproduce

The runner uses Python, NumPy, Transformers, PyArrow 21.0.0, and ONNX Runtime 1.23.2. Optional isolated dependencies can live under `experiments/.cache/search-ranking-tools`; the script also honors the repository's pinned Transformers runtime there. Run:

```text
python experiments/search_ranking.py --prepare
python experiments/search_ranking.py --run
```

Preparation saves the fixed protocol before scoring. Source downloads are pinned and checksummed. A completed result is not overwritten. Reproduce in a separate checkout or study directory instead of modifying an existing measured run. Dataset and weight downloads stay in the ignored cache; the browser corpus, document vectors, manifests, queries, rankings, and timings are published under `results/search-ranking`.

## Learned combination follow-up

A five-feature logistic ranker combines normalized BM25 score, MiniLM cosine, the two rank features `60/(60+rank)`, and the product of normalized BM25 and cosine. MiniLM stays frozen. This is a tiny learned scoring layer on top of two complete searches, not an early exit or a fine-tuned encoder. The live demo's optional combined-results section shows both the fixed and learned formulas, using the exported five weights and intercept locally.

The [pinned official training judgments](https://huggingface.co/datasets/BeIR/scifact-qrels/blob/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029/train.tsv) contain 809 claims. Seed 45110 splits them into **600 fit / 209 development** queries with no query-ID overlap with the 100 test queries. Fit examples are all judged positives plus the union of each method's top 20 candidates: 20,454 pairs, including 671 positives. Unjudged candidates in this training pool count as negatives and may contain useful but unannotated evidence. Gold positives are added only during supervised fitting; every development and test query still ranks the full 5,183-document corpus.

[Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) uses L2 regularization, balanced class weights, and the LBFGS solver. A predeclared grid of `C = 0.1, 1, 10` is selected by mean development nDCG@10, with ties favoring smaller C. Development selected **C=0.1**; no development refit or test-driven threshold selection follows. Its five weights and intercept are [published](../results/search-ranking/learned/model.json). They produce ranking scores, not calibrated relevance probabilities.

| Method | Relevant first | Recall@5 | nDCG@10 |
| --- | ---: | ---: | ---: |
| BM25 | 55/100 | 72.67% | 0.6652 |
| MiniLM | 47/100 | 69.50% | 0.6194 |
| Fixed combination | 54/100 | 76.00% | 0.6839 |
| Learned combination | 59/100 | 75.17% | 0.7001 |

The learned score corrects four BM25 first-result misses without losing any of BM25's first-result hits on these 100 queries, but slightly trails fixed fusion's recall@5. This was requested after seeing the initial test outcomes and **reuses the same test sample**: it is exploratory, not fresh confirmation or evidence of statistical significance. Official train/test claims may concern the same source paper; this is not a document-held-out evaluation. There is **no measured speed claim** for the follow-up because other jobs ran concurrently. Both underlying searches remain necessary.

[Frozen protocol and split IDs](../results/search-ranking/learned/protocol.json), [every development result](../results/search-ranking/learned/development.json), [every test ranking](../results/search-ranking/learned/predictions.json), and [summary](../results/search-ranking/learned/result.json) are separate from the sealed first pilot. Run `python experiments/search_reranker.py` with scikit-learn 1.8.0 and the baseline dependencies/cache to reproduce in a clean study directory. The runner refuses to replace completed results.

The real browser worker also passed a functional parity check: all three original methods returned the same top five as Python for one real test claim. [Browser record](../results/search-ranking/browser-check.json). This small check establishes functionality for that query, not whole-test model parity or browser performance.

The JavaScript scoring functions separately match Python's top five for **all four methods on two real queries**, using [saved query vectors](../results/search-ranking/browser-query-fixtures.json). This checks the learned feature normalization and exported weights without another model run. BM25's top ten also match all 100 frozen queries. Run `node scripts/search-demo-test.js`. Saved-vector arithmetic parity is distinct from browser encoder parity; only the single-query functional check above actually runs the browser model.

## Attribution and licenses

The [original SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md) identifies claims and evidence annotations as **CC BY 4.0**, and the abstract collection from Semantic Scholar S2ORC as **ODC-By 1.0**. The BEIR mirror's card separately lists CC BY-SA 4.0; retain both provenance and the original authors' more specific component terms. The demo preserves original titles and abstract text and identifies each corpus document. Tokenization, index weights, and embeddings are our derived representations.

Cite David Wadden and colleagues, *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020), the Semantic Scholar S2ORC authors, and Nandan Thakur and colleagues, *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* (2021). MiniLM's model and ONNX conversion are distributed under Apache 2.0. Our result artifacts do not change the upstream data or model licenses.

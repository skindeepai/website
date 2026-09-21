# Give search a second check

The previously frozen learned combination put a relevant paper first on **119/200 additional SciFact test IDs**, versus **105/200** for keywords or MiniLM alone. It corrected 15 keyword first-result misses and lost one previously correct first result. One claim duplicates training text under another ID; excluding it leaves 119/199 versus 105/199. It is a useful held-out-ID result on this collection, not proof of a general improvement across search domains.

A generic two-layer reranker reached 111/200. Calling it only when the keyword winner was close to the runner-up reached 113/200, while checking 42 queries. This conditional route is a different speed/quality tradeoff; it did not beat the learned combination.

## What actually runs

1. BM25 searches all 5,183 titles and abstracts.
2. The reranker receives the query alongside each of the top 20 documents. It returns one relevance score per pair, without generating text.
3. The optional gate retains the keyword ranking when its first result has enough of a score lead. Otherwise it runs the same second model.

The reranker is the pretrained [MS MARCO TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2), pinned to revision `81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc`. It has two BERT layers, 128 hidden dimensions and about 4.39 million parameters. We use its float32 ONNX graph, all 20 pairs in one batch, maximum 256 tokens per pair including special tokens. Both layers run for every pair. Skipping the second model avoids all 40 pair-layer executions for that query; it is not an exit inside the model. No SciFact reranker fine-tuning was done.

The original [retrieve-and-rerank documentation](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) describes this shortlist-and-second-check design. The model card's GPU throughput is not a measurement of our CPU experiment.

The learned combination is the existing five-feature logistic model, unchanged: BM25 score, MiniLM cosine similarity, their ranks and their interaction. It was trained on 600 official training claims; a separate 209 selected regularization. MiniLM is the existing six-layer q8 encoder with 384-dimensional stored document vectors. See the [first study](search-ranking.md).

## Data separation and frozen choices

[SciFact](https://github.com/allenai/scifact) contains real scientific abstracts, expert-authored claims and human evidence judgments. Relevant documents may support **or refute** the claim; relevance is not a truth verdict. The data is a research benchmark, not production search traffic.

The initial pilot consumed 100 of the 300 official test queries. This study uses their exact remaining 200-ID complement, with no ID overlap with the 600 fit or 209 development claims. An adversarial review found that IDs do not guarantee unique text: test ID 870 repeats fit ID 871, “Obesity decreases life quality.” A follow-up comparison against all fit, development and old-test query texts found only this overlap after case-folding and collapsing whitespace. The complete corpus is shared, and official splits can contain claims about the same paper. Unknown pretraining overlap remains a limitation.

The [protocol](../results/search-next/protocol.json) was written before development or new-test scoring. The model revision, hashes, candidate count, truncation, metrics, timing and gate selection rule were fixed. Development evaluates seven predeclared thresholds: 0, .05, .1, .2, .3, .5, 1.01. The gate signal is `(best BM25 score - second score) / best score`, with a safe denominator for a zero score. Rerank when the signal is below the threshold.

Selection minimizes reranked development queries while retaining at least always-rerank's first-hit rate and losing at most .01 nDCG@10. The selected .05 threshold reranked 49/209 development queries; that selection artifact was hashed in an [evaluation seal](../results/search-next/evaluation-seal.json) before scoring the new 200. No method changed after their outcomes were seen. This is an empirical development criterion, not a quality guarantee.

## Primary 200-ID results

| Method | Relevant first / 200 | Recall@5 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 keywords | 105 | .7231 | .6581 | .6256 |
| MiniLM meaning | 105 | .7318 | .6594 | .6185 |
| Frozen learned combination | 119 | .7609 | .7098 | .6804 |
| BM25 + TinyBERT rerank | 111 | .7301 | .6597 | .6380 |
| Conditional TinyBERT rerank | 113 | .7193 | .6701 | .6460 |

### Duplicate-text sensitivity check

We retained the original protocol and all 200 outcomes, then excluded ID 870 without retuning. All five methods missed that query, so correct-first totals stay 105, 105, 119, 111 and 113, now out of **199**. Learned-combination recall@5 becomes .7647 and nDCG@10 .7134; all method metrics and exclusions are in the [text-overlap audit](../results/search-next/text-overlap-audit.json), produced by a [separate script](../experiments/search_next_overlap.py). No other exact normalized query overlaps with prior fit, development or pilot IDs were found; paraphrases and shared papers remain possible.

The fusion-versus-keywords comparison has 15 improvements and one loss in both primary and sensitivity sets. A post-hoc exact paired sign test gives p=.000519. This was examined after several method results, without a multiple-comparison correction; it is exploratory evidence, not a predeclared population guarantee. The timing table below remains the original 200-query run.

Always reranking corrected 26 keyword first-result misses but lost 20 previously correct results. Conditional reranking corrected 10 and lost two versus keywords. Its total beats always reranking by two, but it corrected 18 of that method's misses and lost 16 of its successes: the outcomes are not interchangeable. We make no equivalence or zero-quality-loss claim.

The BM25 top-20 pool contained at least one judged relevant paper for **171/200 queries**, a first-hit ceiling of 85.5%. Mean candidate recall@20 was **.8338**. No judged relevant document was injected into any evaluation shortlist. Candidate lists are retained for every query. The second model cannot recover omitted papers.

**3,394 of 4,000** query-document pairs exceeded 256 tokens before truncation. BM25 sees full abstracts; MiniLM and the reranker have bounded inputs. This is a practical method comparison, not a controlled equal-context architecture comparison. Unjudged papers may still be useful; benchmark judgments are incomplete.

<!-- SEARCH_TIMING_START -->
## Separate timing run: first 50 queries

This timing-only run uses the first 50 sorted queries from the completed 200-ID study, after other launched model jobs finished. It does not introduce new quality data or change any method. Quality totals above remain out of 200; timing totals below cover only 50. Operating-system background load is uncontrolled.

| Method | Mean ms/query | Median ms/query | 95th percentile ms | Relevant first / 50 |
|---|---:|---:|---:|---:|
| BM25 | 5.97 | 6.20 | 9.40 | 24 |
| MiniLM | 5.83 | 5.51 | 8.63 | 25 |
| Learned combination | 12.39 | 12.19 | 18.01 | 26 |
| Always rerank | 99.62 | 101.91 | 112.24 | 25 |
| Conditional rerank | 23.38 | 6.72 | 109.25 | 25 |

The optional reranker ran on 9/50 queries here, versus 42/200 in the complete quality set. Three rotating paired repeats produced 750 timed calls, preserving every recorded top-10 ranking and gate choice. Mean times include the slow reranked requests; the median alone hides them.

Timers include raw-query input processing, required encoders, retrieval, gate and sorting. Model loading and document indexing are excluded. ONNX numerical pools are capped at two threads, inter-op one. This is warm local CPU timing, not browser or production latency.

[Timing protocol](../results/search-next/isolated/protocol.json), [summary](../results/search-next/isolated/result.json), [all 750 measured calls](../results/search-next/isolated/records.json), and [runner](../experiments/search_next_timing.py). The concurrent 200-query measurements below remain retained as the initial diagnostic run.
<!-- SEARCH_TIMING_END -->

## Timing: concurrent diagnostic run

| Method | Mean ms/query | Median ms/query | 95th percentile ms |
|---|---:|---:|---:|
| BM25 | 5.55 | 5.24 | 10.34 |
| MiniLM | 7.42 | 6.97 | 11.47 |
| Learned combination | 13.86 | 13.26 | 22.33 |
| Always rerank | 113.50 | 111.60 | 140.54 |
| Conditional rerank | 28.83 | 6.58 | 123.58 |

Three warm repeats for every query and method; method order rotates, yielding 3,000 timed calls. Timers start at raw input and include tokenization, required encoding, retrieval, sorting, gate decisions and pair inference. They exclude model loading (0.177 s), loading vectors and building the keyword index (1.009 s), and the earlier creation of document vectors. Each repeat preserved every first-10 ranking and gate choice exactly.

These were measured while other experiments used the host. They are diagnostic timings, **not an isolated benchmark or published controlled speedup**. Conditional latency is bimodal: most queries skip the model; the others are much slower. Its median alone would hide this cost, so the page reports means. Runtime: Windows 11, Python 3.13.2, ONNX Runtime 1.23.2, NumPy 2.2.4; numerical pools capped at two threads, ONNX inter-op one, tokenizer parallelism disabled.

A separate [timing-only runner](../experiments/search_next_timing.py) freezes the first 50 of these same 200 queries and is intended to run after other experiment jobs finish. It must preserve all saved rankings. That subset is not a new quality evaluation and must not be conflated with the 200-query quality table.

## Reproduction and checks

- [Runner](../experiments/search_next.py): `python experiments/search_next.py --prepare --develop --evaluate`. Completed phases refuse overwriting.
- [Development outcomes and selected gate](../results/search-next/development.json).
- [All 200 predictions, references and candidate pools](../results/search-next/predictions.json).
- [All 3,000 timings](../results/search-next/timings.json) and [summary](../results/search-next/result.json).
- [Independent record audit](../experiments/search_next_audit.py) recomputed all five ranking metrics, timing means/medians, candidate recall, split exclusions and paired first-result changes; [audit output](../results/search-next/audit.json). This is independent arithmetic on stored records, not an independent model rerun.

All source downloads and earlier artifacts are pinned by SHA-256. Model files remain in the ignored local cache; the public runner downloads their pinned versions. Model: Apache 2.0. SciFact claims and annotations: CC BY 4.0; underlying abstract database: ODC-By 1.0. The BEIR mirror separately lists CC BY-SA 4.0; retain original attributions and consult the [original license](https://github.com/allenai/scifact/blob/master/LICENSE.md).

The [original search demo](../search-demo.html) runs keywords, MiniLM and the fixed/learned combinations. The [reranking demo](../rerank-demo.html) now runs the pinned TinyBERT cross-encoder and optional gate in the browser; [runtime checks](method-demos.md) reproduce the first two sealed queries' top-ten rankings. Domain-specific hard negatives or a better candidate pool remain useful next experiments, selected on development data; these now-consumed 200 queries cannot certify a later change as a fresh improvement.

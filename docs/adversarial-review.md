# Adversarial review of early-exit claims

Reviewed 2026-09-19 at Steve's request by a separate agent tasked with challenging the evidence, without editing the implementation. Scope: the local working tree after checkpoint `3881aff`, not a verification of a hosted deployment. This is an independent agent review within the same workflow, not external peer review or independent reproduction of model inference.

## Verdict and ratings

**Credible research prototype, weak evidence of real-world reliability.** The reviewer found no numerical contradiction or code path that merely pretends to skip layers. Public wording is broader than the evidence supports.

Scale: 0 means unsupported or misleading; 10 means independently verified and adequately supported for the stated scope. Ratings are review judgments, not statistical probabilities.

| Area | Rating | Reason |
| --- | --- | --- |
| Numerical integrity | 9/10 | Independently recomputed reported accuracy, exit counts, harmful/corrected errors, timing medians and trace consistency; all matched. |
| Evidence of actual skipping | 8/10 | Hooks terminate execution at a selected block; saved traces and predictions agree. The reviewer did not rerun Qwen or observe the original execution. |
| Real-world validity | 3/10 | One fixed-intent benchmark, one backbone, no unfamiliar-input evaluation or changing-rule validation; every reliability guard failed. |
| Honesty of public claims | 7/10 | Failed guards are explicitly disclosed, but homepage wording and simplified labels overstate the evidence. |
| Reproducibility | 6/10 | Scripts, split IDs and hashes are available; the exact development-library revision and trained heads are not preserved in the public record. |

## Confirmed findings

1. **Readiness is overstated.** The homepage says models stop "once an answer is ready" and skip processing "when it is no longer needed." The implementation thresholds classifier scores; it does not detect completed reasoning or know that further processing is unnecessary. [Method notes](early-exit.md) correctly acknowledge this. This is a copy overstatement, not evidence of fabricated execution.

2. **Relative harm is not absolute error.** The [guard implementation](../experiments/banking77.py) counts full-head-correct / early-head-wrong events divided by all inputs. It does not bound the probability that an early answer is wrong. Independently recomputed:

   | Test | Early exits | Wrong early answers | New errors relative to full depth |
   | --- | --- | --- | --- |
   | BANKING77, seed 17 | 1,614 | 86 (5.33%) | 33 |
   | Conservative reserve test | 133 | 4 (3.01%) | 0 |

   The full-depth classifier also got the conservative test's four wrong early answers wrong. Neither zero harmful exits nor the proposed 1% guard implies 99% correct early answers.

3. **No candidate passed its acceptance rule.** All three initial seeds and the conservative follow-up failed calibration. A system respecting that rule would use full depth and obtain none of the candidate's early-exit savings. The site discloses this. See [initial results](banking77-results.md) and [conservative results](conservative-exits.md).

4. **The strongest test is fixed-intent classification.** Its prompt and 77 categories are fixed. The decision page's visible claim about reading "your instructions" is broader than this benchmark supports; the limitation is in collapsed details. The synthetic instruction pilot does not establish arbitrary changing-rule behavior.

5. **The timing table needs clearer labels.** The results table combines accuracy on 3,080 queries with latency measured on 96 queries, repeated twice per path. Its note should state both sample sizes. "Full model" should say "Full-depth trained classifier": this comparison is not against ordinary Qwen generation. The detailed report correctly documents the samples.

## Limits, not evidence of wrongdoing

- The original and fresh-reserve tests use the same public benchmark. Three head seeds are not three independent dataset replications. Disclosed exact duplicates barely affect the sensitivity results; paraphrase and pretraining overlap remain unknown.
- The 5.33% latency improvement is a warm, single-machine, batch-one result. Its bootstrap interval describes query variation in that recorded run, not fresh sessions, other machines or deployments.
- A lexical classifier already scored 80%, versus roughly 83% here. Practical superiority over a tuned small specialist or equally trained token-output model is unproven.
- Saved JSON and local provenance are inspectable evidence, not independent execution attestation or proof that a protocol existed before results. The documentation acknowledges this.
- The exact source revision of Transformers 4.50.0.dev0 is missing. Trained heads are excluded from git; scripts can train replacements. See [environment notes](../experiments/README.md).

## Checks actually performed by the reviewer

The separate agent matched saved labels against cached source CSVs, verified dataset-file hashes, recomputed all three original seeds and both reserve splits, reconstructed original exit decisions from per-head scores, checked every saved timed trace for contiguous executed layers and agreement with predictions, and recomputed timing means and medians. All passed. No heavy model inference was rerun.

The primary agent separately recomputed early-exit error counts and timing summaries, with matching results. Site copy was not changed as part of this review; the corrections above remain recommendations.

## Defensible claim

On one public banking-classification benchmark, trained intermediate classifiers sometimes retained similar aggregate accuracy while skipping real transformer computation. Their reliability gates did not pass.

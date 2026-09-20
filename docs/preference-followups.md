# Preference follow-ups

These are completed local synthetic checks of specific gaps in the earlier pilots. They do not complete human preference, real-generator, privacy or application validation. No production sampling policy was changed.

## Does the actual browser sampler learn more efficiently?

The experiment uses the actual browser training core and extracted sampling function: 16 settings, 380 candidates and a shuffled batch of five predicted likes, four uncertain choices and three random choices. The UI refreshes its queue every six ratings, so only half of each 12-item batch is rated; the experiment reproduces that behavior.

Random and uncertainty controls receive the same candidate pools and consume the same random-number budgets. All policies start from the same six examples. We retrain after each rating and record 12, 24, 48, 72 and 96 ratings across five fixed seeds. Evaluation uses 100 separate uniform examples per seed.

Mean balanced accuracy at 96 ratings (average of positive recall and negative recall):

| Synthetic preference | Random | Uncertain examples | Browser mixture |
| --- | --- | --- | --- |
| linear | 90.51% | 96.18% | 91.50% |
| noisy_linear | 83.60% | 88.52% | 78.66% |
| two_modes | 48.89% | 53.63% | 53.43% |

The mixture is not consistently better. It shows more truly liked examples under the linear rules, but that is different from learning accurately. With noisy ratings, it performs worse than random selection. The linear model also struggles with two separated preferred regions, whatever the sampler.

Noisy training flips 15% of ratings independently; evaluation asks whether the clean underlying preference was recovered. The second nonlinear rule likes examples where |z0| > 0.55 and |z1| < 0.65. These authored rules and five seeds are not people or confidence intervals.

### Learning curves: every policy and checkpoint


#### linear

| Ratings | Random | Uncertain examples | Browser mixture |
| --- | --- | --- | --- |
| 12 | 70.96% | 70.57% | 69.15% |
| 24 | 82.48% | 83.49% | 78.81% |
| 48 | 86.68% | 91.90% | 86.46% |
| 72 | 89.59% | 93.63% | 86.65% |
| 96 | 90.51% | 96.18% | 91.50% |

#### noisy_linear

| Ratings | Random | Uncertain examples | Browser mixture |
| --- | --- | --- | --- |
| 12 | 63.13% | 64.21% | 62.24% |
| 24 | 70.02% | 59.23% | 69.06% |
| 48 | 77.15% | 72.26% | 73.49% |
| 72 | 81.18% | 79.18% | 76.03% |
| 96 | 83.60% | 88.52% | 78.66% |

#### two_modes

| Ratings | Random | Uncertain examples | Browser mixture |
| --- | --- | --- | --- |
| 12 | 47.93% | 49.71% | 50.21% |
| 24 | 50.73% | 49.06% | 50.77% |
| 48 | 51.52% | 50.98% | 50.62% |
| 72 | 50.99% | 56.43% | 52.65% |
| 96 | 48.89% | 53.63% | 53.43% |

[Protocol](../results/preference-sampling/protocol.json), [all checkpoint predictions and weights](../results/preference-sampling/result.json), [every rated example](../results/preference-sampling/runs.json), [independent evaluation fixtures](../results/preference-sampling/fixtures.json), [runner](../experiments/preference_sampling.cjs). A recorded first checkpoint at 90% is not an exact label requirement or a guarantee of maintaining that score.

## Can selecting a high-scoring candidate help?

| Same 64-candidate pool | Mean true utility |
| --- | --- |
| Random candidate | -2.160 |
| Highest predicted preference | -1.391 |

Reranking improved 4 of 5 seeds and worsened 1. Its worst utility change was -0.828; the average improvement is not a guarantee.

Higher utility is better. Each method sees the same 64 new candidates after 64 training labels; the first independently uniform candidate is the random control. This checks selection quality at equal candidate count, not image-generation time or human taste. The true utility rewards proximity to a declared center. All five seed outcomes and candidates are saved; the earlier optimization failure remains in the [original synthetic results](../results/synthetic/result.json).

## What about recurring contexts?

The supplied context alternates A, B, A, B, with opposite preferences and 24 new ratings per phase. We compare one combined-history model, a reset at each phase, and separate retained models for the known context IDs. The context is given to the system; it does not infer a mood.

| Phase | all_history | phase_reset | known_context |
| --- | --- | --- | --- |
| 1 | 90.8 / 100 | 90.8 / 100 | 90.8 / 100 |
| 2 | 42.8 / 100 | 88.8 / 100 | 88.8 / 100 |
| 3 | 81.6 / 100 | 87.4 / 100 | 92.4 / 100 |
| 4 | 41.6 / 100 | 91.6 / 100 | 95.0 / 100 |

| Return to A after final B | Correct / 100 |
| --- | --- |
| all_history | 58.4 |
| phase_reset | 8.4 |
| known_context | 92.4 |

These are five-seed averages on the same 100 held-out points per seed. Context-specific models preserve earlier context data and use more model storage; this is an explicit design difference, not a matched-memory comparison. No real sessions, inferred context or human retention is measured.

## Do explicit constraints prevent a conflicting edit?

| Known requirement: z0 ≤ 0 | Violations / 5 |
| --- | --- |
| Unconstrained preference maximum | 5 |
| Explicit coordinate cap | 0 |

These cases deliberately train a preference that conflicts with the known requirement. The coordinate cap enforces the declared inequality by construction. It tests a mechanism, not a learned safety model, unseen constraints or real-world compliance. The earlier fixture had zero violations before either method and could not show this difference.

[Prospective protocol](../results/preference-followups/protocol.json), [all results](../results/preference-followups/result.json), [runner](../experiments/preference_followups.cjs). Both studies use one CPU compute thread and the existing browser training core. To reproduce, use a separate checkout; completed result files are protected from overwriting. Run each script with `--prepare` first, then without it.

## Review

A separate agent in this work session reconstructed all 22,500 sampler evaluation decisions, 225 checkpoints and 4,320 selected training examples, including the six-rating queue refresh. It also replayed all 85 candidate-selection, context and constraint records and verified the source/protocol hashes. The calculations matched. This is an internal code/artifact audit, not external replication or evidence about people.

# What would justify stronger early-exit claims?

The [adversarial review](adversarial-review.md) is a historical assessment. Its scores are judgments, not a target metric to optimize. Better documentation can earn confidence in a narrow claim; it cannot establish broad real-world reliability by itself. The [follow-up review](adversarial-followup.md) reassesses the completed evidence and retains a low broad-validity rating because the reliability tests failed.

## Acceptance checklist

| Area | Evidence required | Work in this iteration |
| --- | --- | --- |
| Numerical integrity | Predictions, source labels, split hashes, independently recomputed summaries; distinguish absolute error from added error | Separate reviewer recomputation, exact risk bounds and explicit denominators |
| Actual skipping | A separate execution implementation, real block traces and label parity, including every reachable exit path | Fresh-process replays with portable heads, including a targeted layer-18 path |
| Honest claims | Visible scope and failed gates; precise baseline and timing sample names; no claim of detecting completed reasoning | Corrected homepage, decisions text and results-table notes |
| Reproducibility | Obtainable dependencies, portable trained weights, model/data revisions and repeat commands | Published-release replay, numeric NPZ heads, source and wheel hashes |
| Broader validity | A second dataset, unknown requests, paired changing rules, matched controls and prospectively fixed acceptance criteria | CLINC subset with independent calibration plus a separate changing-rule stress test |

## Evidence that this machine cannot manufacture

Production reliability needs a specified use case, representative consented traffic, error costs, independently held-out sessions and continued monitoring under distribution shift. Neither an agent's review nor a public benchmark provides that. External reproduction and hardware generalization require other people or machines. Public-data pretraining contamination cannot be ruled out without knowledge of the model's training corpus.

We must not lower error limits after observing failures, choose a successful test slice retrospectively, or equate higher average accuracy with fewer harmful shortcuts. A rejected gate has no accepted early-exit speedup. A full-depth classifier that also makes an error does not make that early answer correct.

## Wording policy

Describe what was actually tested: a trained readout can sometimes return a category before all transformer blocks execute. Say whether a policy passed its stated calibration checks and on which data. Report a skipped-block percentage as computation depth, not memory, energy or wall-clock savings. A confidence score is not proof that a model has finished thinking.

Keep the approved site layout and styling. Detailed methods, portable weights, unsuccessful tests and review notes belong in linked references. Any later scores should come from another adversarial review of the resulting record; do not rewrite the original review to claim it was always stronger.

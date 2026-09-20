# Training and stopping on 500 unused messages

This study compares two training objectives across three matched random seeds, then tests a development-selected model and two stopping rules on 500 previously unused ToxicChat messages. The original shared model remains a frozen reference. Results are published on [the focused validation page](../fresh-message-results.html).

## Findings

The original model answered 440/500 correctly. Its frozen stopping rule preserved every full-depth decision while proposing 400 exits after layer 2: 40% fewer transformer-block executions. This is stronger evidence than the reused 100-message sample, but it preserves the original model's 3 toxic misses and 57 false blocks.

The development-selected candidate was `distill-9927`, epoch 2. It answered 451/500 correctly, reducing false blocks to 44 but increasing toxic misses to 5. Its confidence gate proposed 395 early exits; the learned risk gate proposed 386. Both preserved all of its full-depth decisions. The extra learned gate did not improve coverage, and the selected training change did not improve every quality measure. The browser therefore retains the original checkpoint.

| Training / seed | Correct / 500 | Toxic misses / 18 | False blocks / 482 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Human labels / 9927 | 440 | 3 | 57 | 85.8% |
| Human labels / 9928 | 448 | 3 | 49 | 86.6% |
| Human labels / 9929 | 434 | 2 | 64 | 87.8% |
| Teacher + labels / 9927, selected on development | 451 | 5 | 44 | 81.5% |
| Teacher + labels / 9928 | 453 | 1 | 46 | 92.5% |
| Teacher + labels / 9929 | 456 | 3 | 41 | 87.4% |

The apparently stronger seed 9928 was **not** selected by development. Promoting it because of these test outcomes would require another fresh confirmation. Three seeds and only 18 toxic cases leave substantial uncertainty about toxic recall.

Zero added errors in 500 gives a one-sided exact 95% upper bound of about 0.60% for that sample's overall added-error rate. For zero added toxic misses in only 18 toxic cases, the corresponding bound is about 15.3%. This result does not establish a tight rare-error guarantee or general lossless moderation. All routing counts above come from full-depth output simulation; actual layer traces and timings are recorded separately.

The frozen Qwen reference answered 435/500 correctly, with 5 toxic misses and 60 false blocks. Always returning SAFE reaches 482/500 while missing all 18 toxic cases. Training differs: the small BERT encoder and heads are task-trained, while Qwen's backbone stays frozen and only its classifier received task supervision. These outcomes compare those specific trained systems, not architecture or parameter count in isolation. [Reference protocol](../results/compact-next/reference-protocol.json), [summary](../results/compact-next/reference.json), [every reference output](../results/compact-next/reference-records.json).

## What was frozen

The original protocol fixes 1,398 training examples, 500 tuning examples and 796 calibration examples, inherited from the deduplicated earlier experiment. The new 500-message sample is drawn without label stratification from 730 eligible human-annotated official test messages. It excludes every ToxicChat-shaped ID found in existing result artifacts, all training text, conversations shared with earlier rows, and duplicate effective inputs after Qwen's 256-token bound and BERT tokenization. Selection is seeded and the exact source-data hashes are checked.

This is the **eligible annotated remainder**, not a sample of production prevalence. Earlier studies deliberately consumed many toxic examples. Do not transfer the previous balanced 100-message accuracy or this remainder's label mix to a live service. Public benchmark data may also occur in model pretraining.

The 500 messages are a sample of that 730-message remainder. An independent agent reproduced the complete exclusion procedure, exact seeded sample, all six model and gate hashes, and development-only winner selection. Its [split and selection audit](../results/compact-next/split-selection-audit.json) also checks all 3,500 stored source labels.

The reference Qwen classifier and an always-SAFE baseline have a separate supplemental protocol frozen before fresh evaluation. Their outcomes make the label imbalance and the teacher's actual quality visible; they do not choose the new model.

## Improve the small model

Both variants start from the same pinned `google/bert_uncased_L-4_H-256_A-4` checkpoint, with four layers, 256 hidden dimensions and two 256-to-2 classifier heads. Masked mean pooling reads layers 2 and 4. All encoder weights and both heads train.

- **Human labels:** equal cross-entropy losses at layers 2 and 4, with inverse-frequency class weights.
- **Teacher plus labels:** 75% of that loss plus 25% KL divergence to the frozen Qwen full-depth classifier's softened scores, at temperature 2 with the standard temperature-squared correction. The teacher sees training rows only.

Seeds 9927, 9928 and 9929 each run both variants for three epochs, with batch 16, AdamW learning rate 0.0001 and weight decay 0.01. Tuning balanced accuracy selects checkpoint and final-answer threshold; ties prefer fewer toxic misses, then more correct answers, then the first candidate. The same rule selects one candidate across all six before the new test is evaluated. Every seed's full-depth and layer-2 outcomes remains in the result, even when it loses selection.

This tests a fixed distillation recipe, not every possible teacher or objective. It does not add manually invented examples or silently replace human labels. The recipe follows the general softened-target approach described in [PyTorch's distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html).

## Learn whether to continue

The new gate predicts a specific event: **the early answer is wrong and the full-depth answer would be correct**. Its inputs are available after layer 2: 16 principal components of the pooled representation, the BLOCK probability, binary entropy, and input length. A standardized, class-balanced L2 logistic regression returns a risk score. It never sees layer 4 while deciding whether to stop.

The gate trains on the 500 tuning rows, which were not used for encoder gradient updates. However, those rows also selected the encoder checkpoint: this is not fully nested cross-validation. Separate calibration rows choose two class-dependent risk cutoffs, maximizing accepted early answers subject to no added errors on that development set. The frozen fresh test is the check against overfitting that rule.

The confidence baseline instead selects low/high probability acceptance regions on the same calibration rows, with the same zero-added-error condition. Accepted answers use the independently tuned layer-2 label threshold. A low acceptance region is therefore not universally a SAFE output: if it crosses the layer-2 label threshold, an accepted example can be BLOCK. The original frozen model uses its earlier explicit SAFE/BLOCK rule; its low cutoff is below its label threshold.

Neither rule promises calibrated probabilities or reliable absence of added errors. Predicting that more computation will not help also does not mean the early answer is correct: both depths can make the same mistake.

## What the numbers mean

Quality evaluation computes both depths, then simulates each frozen routing rule. One early exit uses two of four transformer blocks. Reported block savings exclude embeddings, tokenization, classifiers and gate overhead; they are not exact FLOPs or latency savings. The separate actual-execution runner checks contiguous layer traces and predictions on the first 50 new messages, with three counterbalanced passes, after other model jobs finish.

The report separates new errors from corrected errors, toxic misses from false blocks, and individual seeds from the selected candidate. A one-sided exact 95% upper bound describes the new-error rate on this sample; it is not a guarantee under distribution shift. Even matching all full-depth decisions would preserve that model's existing errors.

## Execution and evidence

The first serial training attempt was stopped during its first candidate to run the six independent candidates concurrently. Its partial checkpoint is retained locally. The scheduling wrapper does not change seeds, data, losses, batches, epochs or selection; each process caps numerical pools at two threads, and the overall workload stays at or below half the available logical processors. Its separate execution amendment is retained. A separate agent reviewed the wrapper for candidate equivalence and first-tie selection order.

The completed parallel CE/9927 run selected epoch 2 and reproduced the interrupted serial run's saved checkpoint **byte for byte**. [Checkpoint hashes](../results/compact-next/parallel-parity.json) check that scheduling change for this candidate; they do not establish cross-platform reproducibility.

Commands, in order: `compact_next.py prepare`, `compact_next_reference.py prepare`, the six `compact_next_parallel.py KIND-SEED` calls, `compact_next_parallel.py merge`, `compact_next_audit.py preflight`, `compact_next.py evaluate`, `compact_next_reference.py run`, `compact_next_audit.py audit`, and `compact_next_timing.py`. These files live under `experiments/`. Completed outputs are preserved.

Training requires the pinned model/tokenizer and ToxicChat caches from the parent studies, including the cached Qwen training representations. Selected PyTorch encoder checkpoints stay in the ignored local cache; their hashes and all predictions are retained publicly, and training scripts regenerate them. Direct dependency versions are in [the chat research requirements](../experiments/requirements-chat600.txt); the recorded environment below identifies what actually ran. This is not a one-command clean-checkout installer.

- [Original protocol, exact split IDs and historical exclusions](../results/compact-next/protocol.json)
- [Scheduling amendment](../results/compact-next/execution-amendment.json)
- [Frozen selection and every seed's development history](../results/compact-next/fit.json)
- [Fresh quality summary](../results/compact-next/result.json)
- [Every prediction and proposed exit depth](../results/compact-next/predictions.json)
- [Arithmetic, routing and confidence-bound audit](../results/compact-next/audit.json)
- [Library versions and numerical thread limits](../results/compact-next/environment.json)
- [Experiment source](../experiments/compact_next.py), [parallel wrapper](../experiments/compact_next_parallel.py), [independent arithmetic audit](../experiments/compact_next_audit.py)
- [Actual browser implementation of the original model](shared-browser.md)

No raw chat messages are copied into this report. The [ToxicChat dataset](https://huggingface.co/datasets/lmsys/toxic-chat) is CC BY-NC 4.0; the base BERT model is Apache 2.0. These task-trained artifacts are research models, not a ready-made moderation policy.

## Actual execution and corrected gate loading

The first 50 fresh messages were processed through all five paths in three rotating passes, with other launched model jobs stopped. Tokenization, bounding, classification, gate checks and actual model execution are included; model and gate loading are excluded. Each call records exactly which consecutive layers ran.

| Path | Mean milliseconds for 50 | Correct / 50 | Stops at layer 2 / 50 |
|---|---:|---:|---:|
| Original full depth | 330.1 | 45 | 0 |
| Original stopping rule | 215.6 | 45 | 40 |
| Selected new full depth | 333.2 | 47 | 0 |
| Selected confidence rule | 220.7 | 47 | 39 |
| Selected learned risk rule | 212.9 | 47 | 40 |

The original rule took **34.7% less time** than its own full-depth control, with all 50 decisions preserved. All 750 calls matched the quality evaluation's predictions and exit depths. The timing subset contains different proportions of accepted messages from the full 500: the learned risk rule exits 40 here versus confidence's 39, although it exits fewer across all 500. These short CPU measurements do not reliably rank small differences between the gates or establish a deployment latency guarantee.

The original timing runner kept the learned gate in a lazy NPZ archive, unnecessarily decompressing its six arrays inside each risk check. That run remains available: risk inference took 267.2 ms. The follow-up materializes the arrays once before timing, reducing measured risk time to 212.9 ms. It retimes all five methods and verifies every decision, exit depth and risk score against the original run. No model weights, threshold or quality outcome changed.

The first cached attempt used an unnecessary array copy, which changed the stored PCA matrix from column-major to row-major layout. An exact floating-point risk-score parity assertion failed before results were published. Its source, sealed protocol and failure are retained. Removing the redundant copy preserves the original layout; the completed follow-up passes exact parity across all 750 calls. An independent reviewer reproduced the layout-induced rounding difference and audited the final run.

- [Corrected timing protocol](../results/compact-next/timing-cached-protocol.json), [summary](../results/compact-next/timing-cached.json), [every timed call](../results/compact-next/timing-cached-records.json)
- [Original timing protocol](../results/compact-next/timing-protocol.json), [summary](../results/compact-next/timing.json), [records](../results/compact-next/timing-records.json)
- [Failed copy-attempt record](../results/compact-next/timing-cache-copy-attempt.json) and [sealed source](../results/compact-next/timing-cache-copy-attempt.py)
- [Follow-up timing source](../experiments/compact_next_timing_cached.py) and [independent timing audit](../results/compact-next/timing-cached-audit.json)

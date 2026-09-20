# Shared checkpoints, combined execution savings, and an int8 diagnostic

Combining a fixed instruction cache with batches of four reduced the measured 50-message workload from **31.49 to 17.99 seconds (42.86% less)** while preserving all decisions. Shared checkpoint readouts produced smaller savings. Simple readout and threshold repairs did not rescue the failing default int8 model.

These are three separate ideas: reuse a fixed prompt while batching messages, let one Qwen forward continue past an uncertain checkpoint, and repair a classifier after quantization changes its input representation. No experiment trains the Qwen transformer weights; the int8 diagnostic changes their numerical representation. The checkpoint method uses one shared backbone, not independently trained small and large language models.

The checkpoint gates were selected on historical development data before scoring the 100 messages already consumed by the refinement study. Those 100 outcomes are exploratory diagnostics, not another fresh holdout.

## Shared checkpoint readout

A single small readout is trained across layers 6, 12, 18 and 24. It has a shared 896-to-64 projection, a learned 64-dimensional bias for each checkpoint, GELU, training-only dropout, and a shared two-class output. Each depth uses normalization fitted only on training examples. Two predefined versions use the same initialization and minibatches:

- **Joint classification:** weighted cross-entropy against human toxicity labels at every checkpoint.
- **Joint classification plus distillation:** equal-weight classification and temperature-2 distillation from the original full-depth classifier, using training examples only.

Training uses 1,398 historical examples, 300 AdamW steps and batches of 128. Per-depth temperatures are selected on the old 500-example tuning split. Separate SAFE/BLOCK confidence thresholds, optional agreement with the previous checkpoint, and a minimum checkpoint are selected on the cleaned 796-example development split. Selection minimizes mean depth subject to zero newly introduced errors relative to the original full-depth classifier. This is an empirical development constraint, not a guaranteed error bound. The candidate grid includes a disabled gate.

If a gate accepts, inference returns its enum. Otherwise the same forward continues through the remaining blocks and uses the original full-depth classifier. It never starts Qwen again. The jointly trained layer-24 readout is reported as a diagnostic but is not the deployed fallback.

| Locked method on consumed 100 | Correct | Toxic messages missed | Benign messages blocked | Exit at 12 / 18 / 24 | Projected blocks skipped |
|---|---:|---:|---:|---:|---:|
| Original full-depth classifier | 86 | 9 | 5 | 0 / 0 / 100 | 0% |
| Joint classification gate | 86 | 9 | 5 | 9 / 1 / 90 | 4.75% |
| Joint distillation gate | 87 | 9 | 4 | 11 / 5 / 84 | 6.75% |

Neither selected policy exits at layer 6, because both require checkpoint agreement. The classification gate preserves all original decisions on these 100 examples. The distilled gate corrects one false block and introduces no new errors here. This one-message difference is not sufficient evidence of an accuracy improvement.

The classification gate uses SAFE threshold 1.01 (disabled), BLOCK threshold 0.95 and agreement. The distilled gate uses SAFE 0.99, BLOCK 0.975 and agreement. On the historical development split their projected block savings are only 0.69% and 3.20%. All selected and rejected gate candidates are retained. A percentage score is a development-fitted model score, not a certified probability of correctness.

Fixed-depth predictions also remain visible:

| Fixed readout | Layer 6 | Layer 12 | Layer 18 | Layer 24 |
|---|---:|---:|---:|---:|
| Joint classification, correct / 100 | 80 | 83 | 84 | 82 |
| Joint distillation, correct / 100 | 78 | 85 | 88 | 85 |

No depth was chosen after seeing that table. For example, the distilled layer-18 result is 88/100 but introduces six new errors relative to the original classifier, including five newly missed toxic messages. Net accuracy alone hides those changes.

## Execution measurements

The continuation test ran the first 50 consumed evaluation messages once per path with rotating full/classification/distillation order. It includes raw tokenization, gate computation and classifier work, and records every executed block. All 150 calls matched the previously saved predictions and stopping depths, with contiguous layer traces and no repeated lower blocks.

| Actual continuation path | Total time / 50 | Compared with paired full path | Correct / 50 | Early exits / 50 |
|---|---:|---:|---:|---:|
| Original full-depth | 31.175 s | reference | 45 | 0 |
| Joint classification gate | 28.330 s | 9.13% less | 45 | 6 |
| Joint distillation gate | 28.672 s | 8.03% less | 45 | 8 |

This is one paired pass, not a stable ranking of the two gates. Sequence lengths and timing variation matter, and these particular 50 examples differ from the 100-example quality table. Actual skipped blocks on the timed subset are 5.5% and 6.5%; block percentages are not wall-time percentages.

The combined prefix/batching test ran four paths in two counterbalanced corpus passes over the same 50 messages:

| Complete 50-message workload | Pass 1 | Pass 2 | Mean | Reduction vs full input, batch 1 |
|---|---:|---:|---:|---:|
| Full input, batch 1 | 31.419 s | 31.553 s | 31.486 s | reference |
| Full input, batch 4 | 28.118 s | 29.597 s | 28.857 s | 8.35% |
| Prefix reuse, batch 1 | 20.584 s | 20.384 s | 20.484 s | 34.94% |
| Prefix reuse, batch 4 | 17.711 s | 18.268 s | 17.990 s | 42.86% |

All eight workloads completed with equivalent outputs: 400/400 recorded labels match the sealed original float predictions, and all numerical comparisons pass the predeclared tolerance. The maximum absolute logit change is 0.00007915. Every path remains correct on 45/50 messages. This execution optimization preserves the classifier's errors as well as its correct answers.

Every path includes fresh tokenization and stable length sorting. Prefix construction, private key/value copies, padding, inference and readout are inside the corpus timer. The fixed prefix contains no user message; every batch receives its own cloned cache. Padding masks and rotary positions account for different suffix lengths. All 24 layers still execute. Batch size four has twelve full batches plus one two-message batch; every complete forward is recorded. Failed or slower attempts would be retained; none failed this run.

The fixed prefix is 53 tokens. Full-input batch 1 forwards 6,104 positions; batch 4 forwards 6,390 including padding. Prefix reuse forwards 3,507 or 3,793 respectively, including the once-per-workload prefix build. Each original prefix cache has the same hash before and after its workload. These counts are newly computed positions, not exact FLOPs.

The timer concerns queued workloads on one CPU, not live per-user latency. Newly processed positions count padding and the once-per-workload prefix build; they do not measure exact FLOPs, since suffix queries still attend to the cached prefix.

## Quantization diagnostic

The earlier unchanged float readout performed poorly after default dynamic int8 quantization. That failure does not identify whether the quantized backbone lost information, shifted its representation, changed the decision threshold, or combined those effects.

This bounded diagnostic uses a fixed balanced 128-example historical training subset and a separate balanced 64-example historical development subset. It compares the original float readout, the unchanged readout on quantized features, a matched 128-example float linear refit, a quantized-feature linear refit, and a train-only threshold adjustment to the quantized original readout. No new test set or timing is used for these comparisons. Quantized features are collected one message at a time; the historical float cache was produced with batches of eight, so numerical batching differences are a limitation of the comparison.

The first attempt failed during head training because PyTorch cannot use target tensors created in inference mode in the backward pass. No quantization quality outcomes were produced. The original source and failure record remain sealed. A separately sealed repair clones collected tensors outside inference mode before fitting, preserving the model, examples, optimizer, threshold rule and comparisons.

The completed diagnostic did **not** recover useful performance with either simple repair:

| Method on historical development 64 | Correct | Toxic messages missed | Benign messages blocked |
|---|---:|---:|---:|
| Original float model and original readout | 54 | 6 | 4 |
| Float model, matched 128-example linear refit | 53 | 6 | 5 |
| Int8 model, unchanged float readout | 33 | 0 | 31 |
| Int8 model, 128-example linear refit | 29 | 17 | 18 |
| Int8 model, train-only threshold correction | 33 | 14 | 17 |

The unchanged int8 path blocks 63/64 messages. Adjusting the threshold removes some false blocks but creates toxic misses, with no net recovery. Refitting the quantized representation performs worse here, while the matched float refit remains near the original float result. These observations argue against assuming that the earlier failure is merely an easily repaired output threshold or linear-readout mismatch. They do not establish that every int8 implementation, quantization configuration, nonlinear readout or quantization-aware training method must fail. There is no additional int8 speed claim in this diagnostic.

The mean cosine similarity between full-depth float and quantized representations is 0.447, with an aggregate RMS change of 9.90. These are descriptive feature shifts, not a causal explanation or a general information-loss measure.

## Evidence and reproduction

- [Source and stage commands](../experiments/chat_next_methods.py)
- [Sealed protocol, splits and dependency hashes](../results/chat-next-methods/protocol.json)
- [Training choices and every development gate candidate](../results/chat-next-methods/fit.json)
- [Historical development probabilities](../results/chat-next-methods/development.json)
- [Consumed-data quality summary](../results/chat-next-methods/quality.json)
- [Per-message checkpoint predictions and depths](../results/chat-next-methods/quality-predictions.json)
- [Actual continuation timings](../results/chat-next-methods/continuation-timing.json) and [all 150 layer traces](../results/chat-next-methods/continuation-records.json)
- [Combined prefix/batch timings](../results/chat-next-methods/combined-timing.json), [all 400 outputs and cache/layer traces](../results/chat-next-methods/combined-records.json), and [excluded warm-ups](../results/chat-next-methods/combined-warmup.json)
- [Quantization diagnostic](../results/chat-next-methods/quantization.json) and [all 64 development predictions](../results/chat-next-methods/quantization-records.json)
- [Preserved failed attempt](../results/chat-next-methods/quant-failure.json), [repair source](../experiments/chat_next_quant_repair.py) and [separately sealed repair protocol](../results/chat-next-methods/quant-repair-protocol.json)
- [Original fresh refinement and its limitations](chat-refinement.md)

The runner stages are `prepare`, `fit`, `collect`, `quant`, `continuation` and `combined`. It verifies the sealed dependencies and refuses to overwrite completed results. The quantization comparison uses the separately sealed repair runner described above. Training and inference use four CPU threads; timing stages require other launched model jobs to be paused. The execution and checkpoint experiments keep the original float transformer unchanged; only the quantization diagnostic converts it to int8.

A separate agent independently recomputed all 324 development gate candidates per method, verified selected gates and all 100 diagnostic predictions, and audited the 150 continuation calls, layer traces, order and timing sums. It also checked the quantization repair's source, hashes and 64-message metric arithmetic. For prefix/batching, it verified all eight timed workloads and four warm-ups, all 400 timed predictions, cache hashes, token accounting, padding, positions, 24-layer traces, logit tolerances and timing means. This internal artifact audit is not an external replication or independent quality holdout.

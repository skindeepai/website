# Adversarial review of the fresh-message refinement

The fresh sample supports the recorded counts, but does not establish a useful quality-preserving speedup for the new conservative cascade. It exactly matches full Qwen's decisions while avoiding Qwen on only two of 100 messages; mean recorded time falls only from 70.23 to 69.90 seconds. The int8 recipe is faster but fails badly: it blocks 47 of the 50 benign messages. These failures remain in the experiment.

This is a separate agent's source, artifact and dataset audit within the same project. It is not an external replication. I did not run new transformer inference for this review. I independently checked the input exclusions with both tokenizers, recomputed policy selection from the saved scores, and checked decisions against the original source labels and recorded execution traces.

## Split integrity and frozen choices

The new source was reviewed before its protocol was sealed or models were fitted. The final source pins the old specialist's policy as well as its weights, the Qwen readout, the historical protocol, the calibration feature cache and the specialist model files. The float model and dynamically quantized model are separate objects. Assertions require 168 float linear projections in the reference, 168 dynamic-int8 linear projections in the other model, and 24 retained transformer blocks in each.

The resulting partitions contain:

| Partition | Benign | Toxic | Role |
|---|---:|---:|---|
| Training | 1,210 | 188 | Fit the new specialist |
| Tuning | 428 | 72 | Select epoch and standalone decision threshold |
| Development selection | 693 | 103 | Select the cascade gate |
| Fresh evaluation | 50 | 50 | Evaluate frozen candidates |

Six effective duplicates were excluded: two training rows and four development-selection rows. I independently retokenized the selected inputs using the Qwen chat-template API and the BERT tokenizer. The fresh 100 share no ID, conversation, normalized full text, effective Qwen token sequence or effective BERT token sequence with all 3,300 historical study IDs. The retained new partitions also have no within-partition or cross-partition collisions under those checks. Fresh messages are additionally excluded when their normalized full text matches any human-annotated source training row. Original CSV and frozen dependency hashes match the manifest.

These checks establish freshness relative to the local experiments, not absence from model pretraining or semantic near-duplicates. The dataset is still the same archived ToxicChat collection. Exact token deduplication does not establish generalization to another chat service or policy.

The specialist trains on the retained 1,398 historical training messages for four epochs. I reproduced the ranking of all saved epoch/decision-threshold candidates: epoch 4 and standalone threshold 0.4 win. This verifies selection from the saved development summaries; it is not an independent rerun of training.

I recomputed all 144 cascade threshold candidates from the 796 saved development probabilities and original labels. The Qwen reference decisions match the historical calibration artifact exactly. The selected gate maximizes coverage among candidates introducing zero errors relative to the reference. It accepts SAFE when the specialist BLOCK score is at most 0.001 and BLOCK when the score is at least 1; otherwise it runs float Qwen. An all-fallback candidate was available.

The selected gate accepts 87 SAFE messages and no BLOCK messages in development, introduces no new errors and corrects three false blocks. Its fitted weights and policy were frozen before fresh predictions. The fresh sample does not select the epoch, label threshold or gate.

One limitation was flagged before fresh outcomes: the BLOCK threshold of 1 activates a branch with no accepted development examples. A softmax can round to exactly 1; that number is not certified certainty. A future protocol could prefer stricter thresholds among candidates with identical development decisions. The present frozen candidate was not changed. It accepts no BLOCK messages in the fresh sample either.

## Fresh outcomes independently recalculated

I checked all 600 recorded decisions across six paths against the original human-annotated source labels. The confusion counts, paired new harms, routing, exact one-sided binomial upper bounds, and recorded two-layer BERT or contiguous 24-layer Qwen traces agree with the saved results.

| Path | Correct / 100 | Missed toxic / 50 | False blocks / 50 | New errors versus float Qwen | New toxic misses | Qwen calls |
|---|---:|---:|---:|---:|---:|---:|
| Float Qwen | 86 | 9 | 5 | Reference | Reference | 100 |
| Dynamic-int8 Qwen | 53 | 0 | 47 | 43 | 0 | 100 |
| Original tiny specialist | 81 | 11 | 8 | 14 | 7 | 0 |
| Original cascade | 86 | 8 | 6 | 2 | 0 | 52 |
| Newly trained tiny specialist | 82 | 13 | 5 | 11 | 7 | 0 |
| New conservative cascade | 86 | 9 | 5 | 0 | 0 | 98 |

The original cascade's identical overall accuracy hides two new false blocks, offset by correcting one toxic miss and one false block. The new specialist gains one correct decision overall while missing two more toxic messages than the original specialist. That is a tradeoff, not a clear moderation-quality improvement.

The conservative cascade's two specialist-resolved messages are both SAFE, both correct, and both already correct under Qwen. Its remaining 98 messages use Qwen and include all 14 errors. The observed match in predictions is real, but the useful routing coverage drops from 10.93% in development to 2% here. Balanced evaluation and small sample variation matter; this is not a live-stream coverage estimate.

Zero observed new errors in 100 examples has a one-sided 95% binomial upper bound of 2.95%; zero new toxic misses in 50 toxic examples has an upper bound of 5.82%. This study cannot meet the earlier 1%/5% margins from those bounds, even with zero observed harm. They are individual descriptive bounds for this sample design, not a joint guarantee or a deployment-prevalence estimate. Absolute baseline quality remains limited: float Qwen misses nine of 50 toxic messages.

## Completed timing audit

I independently checked all 900 recorded calls in the main comparison and all 600 calls in the separate tiny-model comparison. Each run uses the same 100 evaluation IDs for three repetitions. Message/path ordering matches the prescribed rotations, every prediction and routing decision matches its frozen quality record, and the recorded two-layer or contiguous 24-layer traces match the path. All durations are positive and finite. I recalculated every per-pass total and checked the protocol, fit and evaluation hashes.

| Main timing path | Mean sum of request durations / 100 | Change versus paired float Qwen | Fresh quality |
|---|---:|---:|---|
| Float Qwen | 70.233 seconds | Reference | 86/100 correct |
| Dynamic-int8 Qwen | 43.726 seconds | 37.74% less time | 53/100; quality failure |
| New conservative cascade | 69.897 seconds | 0.48% less time | Identical 100 predictions |

This is actual execution timing, not a projection from fewer parameters or block counts. Dynamic int8 still executes all 24 layers; its timing improvement does not compensate for its severe false-block regression. The conservative cascade's observed gain is very small, and three warm repetitions in one process do not establish a meaningful deployment improvement. BERT runs for every message, including the 98 that then require Qwen.

The standalone old and new specialists average 0.2932 and 0.2941 seconds per 100 messages, respectively: approximately 2.93 and 2.94 milliseconds per request. Their timing run pairs them with each other, not with Qwen. It cannot supply a paired speedup estimate versus Qwen. Both are fast here, but their fresh errors remain those in the quality table: the new specialist misses 13 toxic messages versus nine for float Qwen.

The corpus totals are sums of individual measured request durations, not separate end-to-end queue clocks. Original text preparation, model computation, readout, trace hooks and all fallback calls are included; loading, downloads, training and warm-up are excluded. Timing repeats do not turn 100 accuracy examples into 300 independent examples. No cold-start, memory, live queue or hardware-generalization claim follows. The source and artifact audit verifies recorded consistency; it is not an external reproduction of the CPU execution.

The standalone timing wrapper does not itself check the new portable-weight hash against the fit record. I performed that check independently before its timing stage: the weight file matches the frozen fit SHA256. The main runner also verifies it. The separate tiny-model timing is not paired with the Qwen timing run, which matters when reporting ratios.

## What the failures suggest testing next

The quantized model emits BLOCK on 97 of 100 messages. The current artifacts do not identify whether activation/weight quantization drift, sensitivity of the float-trained standardized readout, or both cause the collapse. This result rejects this particular post-training dynamic-int8 recipe with its unchanged float-trained head; it does not reject all quantization methods.

A bounded follow-up should use historical development data only:

1. Record paired float/int8 hidden-state changes and classifier logit margins on a fixed development subset. Diagnose where the decision shifts before assigning a cause.
2. Freeze the quantized backbone and fit a readout with train-only normalization on its own representations. Compare with a float readout trained on exactly the same examples and recipe. Positive temperature scaling alone cannot change an argmax decision, so it cannot by itself fix the present false-block collapse.
3. Compare default and per-channel-weight dynamic quantization on the same development subset. Retain both failures and successes. Per-channel weights are a hypothesis to test, not a guaranteed fix for activation outliers.
4. If needed, compare quantizing only the MLP projections while retaining float attention projections.

The present fresh 100 are now consumed evaluation data. They must not select a repaired readout or quantization recipe and then serve as its new validation set. A new candidate requires a new locked holdout. The new specialist also changes training quantity, seed and gate-selection procedure together, so its outcome cannot be attributed to additional data alone.

Evidence: [source](../experiments/chat_refinement.py), [sealed protocol](../results/chat-refinement/protocol.json), [fit and every gate candidate](../results/chat-refinement/fit.json), [development scores](../results/chat-refinement/development-predictions.json), [fresh decisions](../results/chat-refinement/predictions.json), [fresh result](../results/chat-refinement/result.json), [900 main timing records](../results/chat-refinement/timings.json), [main timing totals](../results/chat-refinement/benchmark.json), [standalone timing source](../experiments/chat_refinement_tiny_timing.py), [standalone timing protocol](../results/chat-refinement/tiny-timing-protocol.json), [600 standalone timing records](../results/chat-refinement/tiny-timings.json), [standalone totals](../results/chat-refinement/tiny-benchmark.json).

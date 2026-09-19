# Short Qwen adapter experiments

These are exploratory reused-data results. Read the [combined comparison](chat-smoke-results.md) and [independent audit](chat-smoke-review.md) before interpreting the numbers.

Rank 4, scale 2 LoRA updates are attached to q/v projections in every retained block; original weights stay frozen. A task readout receives the last input position. Joint training averages human-label losses at 6/12/18/24. Distillation combines those losses with temperature 2 teacher KL loss. The fixed12 variant retains only the first 12 blocks and reads raw block 12 output, matching its initialization; full-24 readouts receive the final normalization.

Fresh forwards are used after weights change. The old hidden-state cache is used only to initialize the frozen starting heads. First-layer q/v adapter gradients are asserted nonzero; frozen base gradients stay absent. Every forward asserts contiguous execution through its retained blocks.

Initial heads use seed 73, 200 AdamW steps, train-only standardization and the common 384 labels. Adapter AdamW uses learning rate 0.0005, head rate 0.001, weight decay 0.01 and gradient clipping 1. The fixed 32-step budget covers 128 adapter message presentations from shuffled length buckets. Tune 32 selects step 16 or 32; development 64 and evaluation 100 do not select checkpoints.

## full

Selected step: 16. Retained decoder blocks: 24. Trainable adapter parameters: 270336.

| Readout layer | Correct /100 | Toxic missed /50 | Benign blocked /50 |
| --- | --- | --- | --- |
| 6 | 68 | 14 | 18 |
| 12 | 69 | 20 | 11 |
| 18 | 77 | 10 | 13 |
| 24 | 73 | 19 | 8 |

[Protocol](../results/chat-smoke-adaptation/full-protocol.json), [predictions and training history](../results/chat-smoke-adaptation/full.json), [portable weights](../results/chat-smoke-adaptation/full-weights.npz).

## joint

Selected step: 16. Retained decoder blocks: 24. Trainable adapter parameters: 270336.

| Readout layer | Correct /100 | Toxic missed /50 | Benign blocked /50 |
| --- | --- | --- | --- |
| 6 | 69 | 19 | 12 |
| 12 | 68 | 21 | 11 |
| 18 | 76 | 8 | 16 |
| 24 | 77 | 17 | 6 |

[Protocol](../results/chat-smoke-adaptation/joint-protocol.json), [predictions and training history](../results/chat-smoke-adaptation/joint.json), [portable weights](../results/chat-smoke-adaptation/joint-weights.npz).

## distill

Selected step: 16. Retained decoder blocks: 24. Trainable adapter parameters: 270336.

| Readout layer | Correct /100 | Toxic missed /50 | Benign blocked /50 |
| --- | --- | --- | --- |
| 6 | 67 | 24 | 9 |
| 12 | 61 | 26 | 13 |
| 18 | 78 | 13 | 9 |
| 24 | 80 | 14 | 6 |

[Protocol](../results/chat-smoke-adaptation/distill-protocol.json), [predictions and training history](../results/chat-smoke-adaptation/distill.json), [portable weights](../results/chat-smoke-adaptation/distill-weights.npz).

## fixed12

Selected step: 16. Retained decoder blocks: 12. Trainable adapter parameters: 135168.

| Readout layer | Correct /100 | Toxic missed /50 | Benign blocked /50 |
| --- | --- | --- | --- |
| 6 | 66 | 22 | 12 |
| 12 | 66 | 22 | 12 |

[Protocol](../results/chat-smoke-adaptation/fixed12-protocol.json), [predictions and training history](../results/chat-smoke-adaptation/fixed12.json), [portable weights](../results/chat-smoke-adaptation/fixed12-weights.npz).

The full-only adaptation loss does not train its intermediate heads after representation changes; their scores are diagnostic probes. Partial training, tiny development selection, one seed and a potentially weak teacher prevent conclusions about the limits of the methods. Concurrent training elapsed times are not speed measurements.

[Implementation used](../results/chat-smoke-adaptation/implementation.py), [budget amendment](../results/chat-smoke-adaptation/budget-amendment.json), [original interrupted protocol](../results/chat-smoke-adaptation/initial-full-protocol.json), [independently reconstructed initial baseline](../results/chat-smoke-adaptation/initial-frozen-baseline.json), [baseline reconstruction code](../experiments/chat_smoke_adaptation_baseline.py).

The root runner's optional timing path uses already tokenized input and computes all retained diagnostic readouts. It was not used for the comparison below. The separate runtime runner includes original text preparation and only the target head.

Prior methods: [LoRA](https://arxiv.org/abs/2106.09685), [FastBERT](https://arxiv.org/abs/2004.02178), [LayerSkip](https://arxiv.org/abs/2404.16710). These are small local adaptations of ideas, not reproductions of those papers' training budgets or published results.

## Actual target-head runtime replay

| Variant | Messages | Blocks | Summed request seconds | Prediction parity |
| --- | --- | --- | --- | --- |
| full | 50 | 24 | 33.77 | True |
| joint | 50 | 24 | 34.21 | True |
| distill | 50 | 24 | 33.69 | True |
| fixed12 | 50 | 12 | 16.73 | True |

Each variant executes a separate warm pass on the same 50 IDs, using four CPU threads. The boundary includes original text preparation, actual transformer execution and only the target readout. Models were timed sequentially with no concurrent model jobs. These non-interleaved single passes establish runnable paths and prediction parity, not a replicated speed ranking. Their totals cannot be compared directly to the 100-message head/cascade totals.

[Timing protocol](../results/chat-smoke-adaptation/runtime-protocol.json), [runtime implementation](../experiments/chat_smoke_adaptation_runtime.py). Each variant runtime JSON records every duration and contiguous executed block trace.

# More training, stricter fallback, and int8 Qwen

This is a new local diagnostic on 100 previously unused, human-labeled ToxicChat messages: 50 toxic and 50 benign. The examples were excluded from every earlier training/development/test partition and deduplicated against their effective model inputs. This does not establish that public messages were absent from model pretraining.

The model and fallback settings were frozen before scoring these messages. All outcomes are retained; the test is not used to choose a new threshold. Timing repeats reuse the same 100 messages and do not increase the accuracy sample.

## What changed

The same two-layer Google BERT miniature (4,369,666 parameters) now learns from 1398 historical training messages instead of 384. All encoder weights and the 128-to-2 linear classifier are trained. Four epochs use the original optimizer settings; 500 separate development messages select epoch 4 and the standalone threshold. Duplicate removal is recorded in the protocol.

A further 796 development messages select the fallback thresholds, allowing zero newly introduced errors relative to the frozen Qwen reference. The selected BLOCK-score limits are SAFE at or below 0.001, BLOCK at or above 1; all other cases run full Qwen. These development constraints are not a guarantee on new data.

The other candidate changes Qwen’s 168 attention/MLP linear projections to dynamic int8 execution. Embeddings, normalization and the trained classifier remain float32. It uses all 24 layers; this is weight/arithmetic compression, not early exit. The float reference is a separate, unchanged model instance.

## Accuracy on the same new messages

| Method | Correct / 100 | Toxic missed / 50 | Benign blocked / 50 | New errors vs Qwen | New toxic misses vs Qwen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original full Qwen | 86 | 9 | 5 | 0 | 0 |
| Qwen with int8 linear layers | 53 | 0 | 47 | 43 | 0 |
| Original tiny BERT | 81 | 11 | 8 | 14 | 7 |
| Original BERT + Qwen | 86 | 8 | 6 | 2 | 0 |
| More-trained tiny BERT | 82 | 13 | 5 | 11 | 7 |
| More-trained BERT + conservative fallback | 86 | 9 | 5 | 0 | 0 |

“New” counts a case the float Qwen reference answered correctly and the alternative answered incorrectly. Correcting a different mistake does not cancel this harm. These balanced-sample accuracies cannot be compared directly with natural chat prevalence.

## Actual request time

| Method | Pass 1 / 100 | Pass 2 / 100 | Pass 3 / 100 | Mean / message | Time reduction vs paired Qwen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original full Qwen | 69.99 s | 69.67 s | 71.04 s | 702.3 ms | 0.0% |
| Qwen with int8 linear layers | 43.71 s | 43.34 s | 44.13 s | 437.3 ms | 37.7% |
| More-trained BERT + conservative fallback | 69.75 s | 69.32 s | 70.63 s | 699.0 ms | 0.5% |

Three warm passes use four CPU threads, one request at a time, with rotating method order and no other launched model work. Request times include original text preparation, classifier readouts, trace hooks and every fallback call. Model loading, downloads, training and warm-up are excluded. Both float and quantized models reside in memory for this comparison; no device-memory saving or production throughput is measured.

Every timed prediction, routing choice and contiguous executed layer trace must match the earlier evaluation. The sums are accumulated request durations, not a separately timed service queue. Negative time reduction means a slowdown.

## Which model answered?

| Final answer from | Messages | Cascade correct | Qwen correct on these same messages |
| --- | ---: | ---: | ---: |
| BERT, accepted by the gate | 2 | 2 | 2 |
| Qwen, after BERT was uncertain | 98 | 84 | 84 |

BERT runs its two layers on all 100 messages. The new cascade runs all 24 Qwen layers on 98 messages and avoids Qwen on 2; it does not stop either model halfway. Both direct Qwen paths execute all 2,400 Qwen blocks across the 100 requests.

The routed groups differ in difficulty. Their accuracy fractions are not a model ranking. The last column provides the same-message comparison.

## What this can and cannot establish

The exact one-sided 95% upper bound on the rate of newly introduced errors in this balanced sampling setup is 2.95% for the new cascade and 51.72% for int8 Qwen. For additional toxic misses, the bounds are 5.82% and 5.82% respectively. These are separate bounds, not a joint guarantee.

Even zero observed new errors in 100 cases gives an upper bound of about 2.95%; zero new toxic misses in 50 gives about 5.82%. This smoke test cannot establish a 1% added-error tolerance or prove zero future quality loss. Broader untouched data, meaningful error budgets and a stronger reference are needed for an acceptance claim.

Training quantity, random seed and gate selection changed together. This comparison does not isolate the effect of more training data. The fallback remains the original Qwen classifier. Historical development examples have been repeatedly inspected. Only user-message text is provided, bounded to 256 Qwen tokens; dataset labels may depend on omitted material.

| Split | Qwen truncations | Additional BERT truncations |
| --- | ---: | ---: |
| train | 67 | 0 |
| tune | 27 | 0 |
| calibration | 37 | 0 |
| evaluation | 6 | 0 |

## Standalone tiny-model timing

These three paired passes time the old and new BERT alone. They are separate from the Qwen timing run, so no paired speedup against Qwen is claimed. All 600 predictions and two-layer traces matched the quality record.

| Model | Pass 1 / 100 | Pass 2 / 100 | Pass 3 / 100 | Mean / message |
| --- | ---: | ---: | ---: | ---: |
| Original tiny BERT | 0.306 s | 0.303 s | 0.271 s | 2.93 ms |
| More-trained tiny BERT | 0.309 s | 0.305 s | 0.268 s | 2.94 ms |

[Additional timing protocol](../results/chat-refinement/tiny-timing-protocol.json), [all calls](../results/chat-refinement/tiny-timings.json), [totals](../results/chat-refinement/tiny-benchmark.json).

## Sources and reproduction

- [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat), pinned ToxicChat0124 files and CC-BY-NC-4.0 license. Public artifacts contain IDs and predictions, not raw messages.
- [Protocol, source hashes, selected IDs and duplicate exclusions](../results/chat-refinement/protocol.json).
- [Every training epoch and gate candidate](../results/chat-refinement/fit.json), [gate-development predictions](../results/chat-refinement/development-predictions.json).
- [Every fresh prediction and executed block](../results/chat-refinement/predictions.json), [quality summary](../results/chat-refinement/result.json).
- [All 900 timed calls](../results/chat-refinement/timings.json), [timing totals and parity](../results/chat-refinement/benchmark.json).
- [Portable new BERT weights](../results/chat-refinement/specialist.npz), [runner](../experiments/chat_refinement.py).

- [Independent agent review](chat-refinement-review.md).

Run `python experiments/chat_refinement.py prepare`, then `fit`, `evaluate`, and `benchmark` in a separate checkout with the pinned local caches. Completed outputs and sealed source hashes are protected. Preserve this record; change the output directory in a new copy to reproduce.

# A small specialist with a Qwen fallback

The compact model is a useful speed candidate, but its standalone decisions are worse on this smoke sample. Sending uncertain messages to the old full-depth Qwen classifier recovers most of the difference. This is **exploration on 100 previously inspected messages**, not independent validation or a deployment recommendation.

| Path | Correct / 100 | Missed toxic / 50 | False blocks / 50 | Messages sent to Qwen | Sum of request times |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny BERT specialist | 72 | 20 | 8 | 0 | 0.31 s |
| Specialist then Qwen when uncertain | 79 | 14 | 7 | 49 | 32.46 s |
| Previous full Qwen classifier | 78 | 15 | 7 | 100 | 64.75 s |

The cascade makes two errors that the Qwen reference avoids and corrects three of its errors. One of those new errors misses a toxic message. A one-message gain in total accuracy does not establish safer moderation.

## What was trained

Google's [BERT miniature](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) has two encoder layers and 128 hidden dimensions. We use revision `30b0a37ccaaa32f332884b96992754e246e48c5f`, licensed Apache-2.0. The used encoder and new head contain 4,369,666 parameters. The model card links Google's original work on compact pretrained models. This experiment does not inherit the card's published benchmark results.

All encoder parameters and a new two-class linear head are trained on the common 384-message development sample, with inverse-frequency class weights, AdamW, learning rate 0.0001, batch size 16 and four epochs. The head reads the attention-mask-weighted mean of the final token representations; padded tokens never count toward that mean. The unused BERT pooler is removed.

Each input is the same user message bounded to the first 256 Qwen tokens used by the other smoke experiments. BERT retokenizes that text; a 512-WordPiece limit introduces **zero additional truncations** in these partitions. The specialist does not need Qwen's instruction template. No effective BERT input repeats across partitions.

The 128-message tuning split selects epoch 4 and a standalone BLOCK threshold of 0.3 by balanced accuracy. The same tuning split selects asymmetric cascade thresholds: return SAFE when the BLOCK softmax score is at most 0.05, BLOCK when at least 0.8, otherwise ask the full Qwen classifier. The score is not a calibrated probability of toxicity in a live stream. Selection maximizes specialist coverage subject to zero newly missed toxic messages and at most 1% new errors relative to Qwen on tuning. These are development criteria, not a guarantee beyond that sample.

Epoch, decision threshold and cascade thresholds all use the same tuning split, so its scores are optimistic. The shared split named `calibration` is also development data. We do not tune this specialist policy on it. On those 128 messages the cascade adds three errors, including two new toxic misses, while correcting nine Qwen errors. It therefore fails to preserve the tuning constraints there. We keep the selected policy and report the failure.

The old Qwen reference was trained on **1,400 messages**, more than this specialist's 384, and it is itself an unreliable moderator. It is a fallback reference, not a proven strong teacher. The evaluation sample deliberately contains 50 toxic and 50 benign messages; its accuracy and fallback rate do not estimate a natural chat stream's rates.

## Timing

The timed replay makes **300 actual calls**, one pass of 100 messages per path. Each message runs all three paths, rotating their order. The cascade executes two BERT blocks every time and all 24 Qwen blocks on its 49 fallback messages. Saved decisions match every timed prediction, and executed block traces are checked. The sum of measured request durations is 32.46 seconds for the cascade versus 64.75 seconds for its full-Qwen reference, approximately half the time in this single run. This is not a repeated-run speed estimate or a claim of preserved moderation quality.

The original three-pass timing plan was reduced to a single 100-message smoke pass before timing outcomes, following the user's request to speed exploration; the separate timing protocol records that amendment. The benchmark includes original text tokenization, the shared Qwen-token input bound, model calls, readout and duplicated fallback processing. Loading, training and one warmup call per path are excluded. Both models reside in memory, which increases the cascade's memory requirement. Downloads and cold start are not measured. Other launched model workloads had finished before this run.

PowerShell's redirected stderr reported a `NativeCommandError` for the pre-existing Qwen eager-attention warning and returned a nonzero shell status. The log has no traceback; all 100 messages reached completion, and both final timing artifacts were written and independently checked. We therefore rely on the completed trace/parity evidence rather than claiming an observed zero Python exit status for that shell invocation.

## Reproduce and inspect

- [Common data selection](../results/chat-smoke/protocol.json)
- [Specialist protocol](../results/chat-smoke-specialist/protocol.json)
- [Training history and results](../results/chat-smoke-specialist/result.json)
- [Every label, probability and fallback decision](../results/chat-smoke-specialist/predictions.json)
- [Portable trained weights](../results/chat-smoke-specialist/specialist.npz)
- [Timing amendment](../results/chat-smoke-specialist/timing-protocol.json)
- [Every timed request and executed block trace](../results/chat-smoke-specialist/timings.json)
- [Timing totals](../results/chat-smoke-specialist/benchmark.json)
- [Post-run runtime source and artifact hashes](../results/chat-smoke-specialist/runtime-provenance.json)
- [Implementation](../experiments/chat_smoke_specialist.py)

The specialist's fit source was not prospectively hash-sealed. The linked runtime snapshot and hashes were recorded **after** timing; they document the code used for the completed runtime replay, not a prospective attestation of the original training source. The pinned input/model hashes, training settings, history and portable weights remain available.

With the pinned local ToxicChat/Qwen caches and published Transformers 4.50.3 runtime used by the preceding studies, run `python experiments/chat_smoke_specialist.py --fit` into a new output directory (change `OUT` in a copy of the script; completed outputs are protected from overwriting). Download the pinned Google model's `config.json`, `vocab.txt` and `model.safetensors` into `experiments/.cache/bert-tiny` first. Run `--benchmark` only after other model work finishes. The benchmark can reload the portable NPZ; it does not require the ignored training checkpoint. ToxicChat remains CC-BY-NC-4.0, and no raw messages are published.

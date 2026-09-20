# Locate the quantization failure

The default dynamic INT8 recipe lost substantial quality in the earlier study. We kept the same Qwen2.5-0.5B-Instruct backbone, original trained classifier and input preparation, then converted different groups of linear projections. This is a development diagnostic, not a new quality holdout or timing benchmark.

| Conversion | Correct / 32 | New errors | Corrected errors |
|---|---:|---:|---:|
| Float32 reference | 28 | 0 | 0 |
| All linear projections | 14 | 17 | 3 |
| Attention projections only | 28 | 2 | 2 |
| Feed-forward projections only | 14 | 17 | 3 |
| First 12 layers | 14 | 17 | 3 |
| Last 12 layers | 21 | 9 | 2 |
| All projections, per-channel weights | 15 | 16 | 3 |

Attention-only conversion changes four decisions despite matching the reference's correct total. One new error is a missed toxic message. It is a candidate for further investigation, not a lossless optimization. All 24 transformer layers still execute; neither speed nor storage size was benchmarked here.

## Method

The first 32 IDs of the earlier 64-message quantization development split were fixed before execution. There are 14 toxic and 18 benign messages. No examples, labels or model parameters were selected using these new outcomes. The historical development set had already been inspected, so it cannot establish generalization.

Every path uses identical batch-one token IDs and attention masks, CPU float32 eager attention, two explicitly capped numerical threads, and the original full-depth classifier. PyTorch's x86 dynamic quantization converts selected linear modules; embeddings and normalization remain floating point. Attention-only converts 96 projections, feed-forward-only 72, either 12-layer half 84, and full conversion 168. The per-channel path changes the weight quantizer; it does not calibrate activations or retrain anything.

Forward hooks capture the **last input token only** after all 24 blocks, before the final RMS normalization. A 25th recorded state is the final normalized output, not another transformer layer. Cosine similarity and RMS differences compare those states with the float run. They are descriptive changes, not an explanation of all activations or proof of an exact causal failure point. The final-state mean cosine is approximately 0.976 for attention-only conversion and 0.434 for all-default conversion.

These results make selective quantization more promising than another full conversion followed by threshold tuning. A next candidate should retain sensitive feed-forward operations at higher precision, verify error counts on separate data, and then measure actual runtime. General guidance for matching float and quantized activations is available in the [official ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html); this particular experiment uses PyTorch rather than ONNX.

## Reproduction and review

Run `python experiments/quant_localize.py` in the pinned local research environment. The script refuses to overwrite the sealed protocol. The first invocation stopped before creating a protocol because the local tooling import path was missing; adding that path preceded all recorded inference.

- [Executable source](../experiments/quant_localize.py)
- [Sealed protocol, model revision and exact IDs](../results/quant-localize/protocol.json)
- [Summary and all layer means](../results/quant-localize/result.json)
- [All 192 quantized predictions, reference logits and layer measurements](../results/quant-localize/records.json)
- [Independent review record and artifact hashes](../results/quant-localize/review.json), with [reproducible arithmetic checks](../experiments/quant_localize_review.py)
- [Earlier failed conversion and readout repair](chat-next-methods.md)

A separate agent recomputed every confusion count and all 25 state means for all six methods from the retained records, checked source and inherited dependency hashes, and confirmed that matched aggregate accuracy does not imply preserved decisions. This is an internal artifact review, not external replication.

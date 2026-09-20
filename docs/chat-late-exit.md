# Classify the hidden state instead of generating a reply

[Website depth comparison](../depth-results.html#late-layers) · [Output-method comparison](../output-results.html)

All six methods in the original moderation comparison already return numerical classifier decisions. The `Full Qwen + classifier` baseline is not ordinary chat generation. It runs all 24 Qwen blocks, reads the final prompt-position state, and applies a trained task head. SAFE and BLOCK are names assigned to its two outputs.

| Existing method | Readout and transformer work |
|---|---|
| Full Qwen + classifier | Final normalization, then a trained 896 → 64 → 2 neural classifier; all 24 blocks. |
| Stop at layer 12 | The same head architecture trained for the raw layer-12 state; blocks 13–24 never run. |
| Learned stopping | Classifiers at 6, 12 and 18, plus learned error/benefit checks; otherwise the classifier at 24. |
| Tiny model only | Two-layer BERT, pooled features and a trained 128 → 2 linear classifier. |
| Tiny model → Qwen | BERT's classifier first; uncertain cases use all 24 Qwen blocks and an older trained linear classifier. |
| With distillation | Qwen with trained adapters and a linear classifier; the reported variant uses all 24 blocks. |

The [six-method replay](chat-label-suite.md) documents the exact implementations and different training budgets. Our separate written-reply controls use ordinary greedy generation and retain actual reply tokens. They should not be confused with these classifier rows.

The browser demos also distinguish these paths. The tiny-model, shared-model and method demos use trained numerical classifier heads. The Qwen moderation benchmark's direct-score option reads two entries from the ordinary final vocabulary output; it still computes the full vocabulary projection and all 24 layers. Its single-token and JSON controls generate text. That direct-score browser option is not the trained Qwen head in the six-method research table.

## What can actually save work?

A classifier can choose an integer, after which an application serializes it as `BLOCK` or JSON. That is not necessarily autoregressive text generation. For a decoder model, the first output token is available from the input forward pass; generating subsequent tokens requires additional decoding passes. A constrained single-token readout can be very close to a classifier.

There are three separate optimizations:

1. Replace vocabulary scoring with a small task-specific readout.
2. Avoid additional generated tokens and their transformer calls.
3. Read an earlier hidden state and physically skip later transformer blocks.

The [matched 77-category test](matched-output.md) used exactly the same trained scores for an enum and a single mapped token. It found identical decisions and no reliable latency advantage for the enum alone. Changing the representation is distinct from skipping model computation. A freshly trained classifier versus an untouched prompted model also changes supervision; its accuracy difference cannot be attributed solely to removing text output.

The [AMD CtrlVox article](https://www.amd.com/en/blogs/2026/amd-ryzen-ai-powers-on-device-voice-chat-moderation.html) does not disclose the internal readout or decoding loop needed to establish whether this optimization applies. Saying an API returns text does not settle that question. No CtrlVox model was tested here, and no cross-product performance comparison is made.

## New experiment: omit only the final one or two blocks

Earlier moderation probes covered blocks 6, 12, 18 and 24. This follow-up compares 22, 23 and 24 with equally trained linear classifiers, answering a narrower question than the existing early-exit tests.

Training uses the same 384 human-annotated ToxicChat messages for each head. Evaluation uses 50 previously inspected messages: the first 25 of each class in the fixed earlier evaluation order. Each head has 896 inputs and two outputs, fixed training settings and an argmax decision. There is no learned stopping gate, threshold search or selection of a preferred depth using these results.

At 22 and 23 the classifier reads the raw post-block state. At 24 it reads the final normalized state. The input prompt and selected token position are unchanged. All three heads are newly trained linear models; their scores are not the older MLP comparison.

Timing uses three rotating passes, batch size one, four CPU compute threads, one interop thread, float32 and eager attention. It includes text preparation, actual stopped execution and classifier output; loading and training are excluded. Hooks verify every executed layer, and the language vocabulary head must never run. Fixed-depth checks do not establish adaptive detection of when an answer is ready.

Source and exact example IDs were recorded before inference in the [protocol](../results/chat-late-exit/protocol.json). The [runner](../experiments/chat_late_exit.py) protects existing output from overwrite. A separate [NumPy audit](../experiments/check_chat_late_exit.py) recomputes predictions from saved features and verifies per-class errors and timing arithmetic.

## Results

| Classifier location | Correct / 50 | Toxic missed / 25 | Benign blocked / 25 | Blocks skipped | Mean ms/message |
|---|---:|---:|---:|---:|---:|
| After block 22 | 37 | 9 | 4 | 2/24 (8.3%) | 577.8 |
| After block 23 | 36 | 10 | 4 | 1/24 (4.2%) | 618.3 |
| After block 24 | 36 | 9 | 5 | 0 | 640.8 |

Layer 22 took 9.8% less time, but introduced two errors, both toxic misses, while correcting three other full-depth errors. Its equal total toxic-miss count does not mean it missed the same messages. Layer 23 took 3.5% less time and exchanged one corrected false block for one new toxic miss. Neither result establishes a quality-preserving shortcut.

All 450 timed calls executed exactly the claimed blocks and generated zero tokens. Independently calculated NumPy predictions matched every timing record; repeated passes gave identical decisions. Actual whole-pass times vary across the three repeats, so these are paired, single-machine exploratory timings rather than a hardware-independent speed claim.

[Results](../results/chat-late-exit/result.json) · [Every timed call](../results/chat-late-exit/records.json) · [Independent audit](../results/chat-late-exit/audit.json) · [Saved linear heads](../results/chat-late-exit/heads.npz)

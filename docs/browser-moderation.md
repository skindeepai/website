# Real-message moderation in the browser

The browser experiment runs **Qwen2.5-0.5B-Instruct locally** on up to 100 real ToxicChat messages. It compares reading the SAFE/BLOCK decision from final vocabulary logits with generating a short JSON reply, alongside a minimal one-token control and a fixed-instruction cache comparison. No result is pre-filled and no speed or accuracy improvement is assumed.

## Exact model and input

- Model: [onnx-community/Qwen2.5-0.5B-Instruct](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/tree/cc5cc01a65cc3ff17bdb73a7de33d879f62599b0), revision `cc5cc01a65cc3ff17bdb73a7de33d879f62599b0`, four-bit ONNX weights, WebAssembly, one compute thread. All four paths use every one of the 24 transformer layers.
- Runtime: Transformers.js **3.8.1**, loaded from jsDelivr after Run is pressed. Its [model API](https://huggingface.co/docs/transformers.js/v3.8.1/api/models) exposes a forward call and text generation; the [pinned implementation](https://github.com/huggingface/transformers.js/blob/3.8.1/src/models.js) supplies the decoder inputs and cache handling.
- Dataset: [LMSYS ToxicChat0124](https://huggingface.co/datasets/lmsys/toxic-chat), revision `29df8e4dba60e1f4af4b4075c0705c5b313548a8`, CC BY-NC 4.0. The downloaded test CSV must have SHA-256 `3c2e49889626f7738dca0a29bface0ba0a0595b2ffdd17f0e02f19df7c3c4c9b` or the run fails before inference.
- IDs: the exact 100 evaluation IDs from [the existing smoke protocol](../results/chat-smoke/protocol.json), 50 toxic and 50 benign. The selectable 10/50-message runs use fixed prefixes; they are not necessarily balanced. These messages were previously inspected during research. This is not fresh validation or deployment prevalence.
- Each message is bounded to its first 256 Qwen tokens, using identical bounded text for all four formats. Count every truncation. The policy is the same as the original local moderation study, except for the final output-format instruction.

## The four paths

**Direct decision:** `model(inputs)` executes one full forward pass. Read the final sequence position's logits for the single vocabulary tokens SAFE and BLOCK; select the higher value, breaking exact ties toward the lower vocabulary token ID to match greedy generation. Tokenization is checked at runtime. The application returns an enum without decoding an answer. Both raw scores are included in the download; they are not calibrated probabilities.

**Written JSON:** `model.generate()` uses greedy decoding and at most 16 new tokens. Decode the continuation and accept only a JSON object containing exactly one `label` key with SAFE or BLOCK. Empty, truncated, malformed, fenced, or extra-field responses remain visible and count as incorrect. There is no retry or hidden output repair.

**One-token control:** the same exact prompt as direct mode, with greedy generation capped at one token and a custom `LogitsProcessorList` masking every vocabulary token except SAFE and BLOCK. This prevents an unnecessarily verbose baseline from being the only comparison. Repetition penalty is explicitly 1. The [pinned processor API](https://github.com/huggingface/transformers.js/blob/3.8.1/src/generation/logits_process.js) applies the label restriction to final logits. The generated token is decoded and checked.

The direct mode **still computes the normal vocabulary projection**. It is not a trained internal head, a vocabulary-free model, or early stopping. A difference versus JSON includes both avoiding repeated decoding steps and the different output instructions. Compare direct mode with the one-token control before attributing a gain to output representation. This browser's quantized model and zero-shot readout are also different from the trained CPU float32 classifiers elsewhere on the site.

**Cached instructions:** the same direct prompt and final-score readout, but save the fixed system-prefix state once and copy it privately for each message. Verify the token prefix before reuse. No message content is cached. All 24 layers still process each new message. The timed total includes cache construction on the first request, after discarding the warm-up cache. Numerical changes may change decisions; report disagreements with uncached direct mode.

See also the separate [Qwen 3.5 / SSN prompt and cache experiments](moderation-transfer.md).

## Timing and quality

Each path receives one warm-up on a separate authored benign message. The timed workload uses one pass per format per message, rotating and reversing the four execution paths. Per-format totals sum measured input preparation, inference, and output parsing. **They exclude the subsequent tensor disposal and UI updates.** Download, integrity checking, runtime/model loading, and warm-up are reported separately. Overall benchmark wall time includes the between-request cleanup and is also stored; it can be longer than the sum of decision times. The completion summary compares these measured decision-time totals and shows the corresponding correct counts alongside each speed ratio. No ratio is shown for a partial run.

Report correct/count, toxic recall, toxic misses, benign messages incorrectly blocked, invalid outputs, truncation, generated tokens, and the full per-example predictions. Invalid output on a toxic message is a missed toxic message; invalid benign output is not mislabeled as a false block. The interface shows disagreement count as well as each path's accuracy. Equal totals can hide different mistakes.

Stop terminates the worker and retains completed rows as an explicitly partial download. A partial run may have unequal path counts and must not be presented as a complete comparison. Browser hardware, background work, caching, and memory limits affect measurements. Repeated runs and a larger untouched dataset are needed before any quality-preserving speed claim.

## Data handling and reproducibility

The browser downloads public data and model files; inference and scoring happen on the device. The application does not upload messages or benchmark results. The JSON report omits source message text and includes pinned IDs, hashes, revisions, scores, outputs, and timings. External download hosts still receive ordinary file requests. The model may remain in the browser cache.

Source: [worker](../scripts/moderation-benchmark-worker.js), [dataset/parser/metrics](../scripts/moderation-benchmark-core.js), [interface](../scripts/moderation-benchmark.js), [page generator](../scripts/refresh_moderation_benchmark.py).

The [10-message cached-path check](../results/moderation-transfer/browser-qwen25.json) completed all four paths. Cached instructions took 25.55 seconds versus 38.29 seconds for uncached direct readout, including prefix creation, with identical decisions. Both scored only 4/10: this is an execution check, not adequate moderation quality. JSON scored 6/10 but missed all four toxic messages. One input was truncated. The interface passed width checks at 320, 375, 390, 768 and 1440 pixels.

## Implementation checks

Run `node scripts/moderation-benchmark-test.js` for CSV edge cases, pinned IDs, exact-tie readout, malformed replies, and per-class error accounting. When the original CSV exists in the local research cache, the same test verifies its hash and every selected label.

A [Chromium/WebAssembly functional check](../results/ui/moderation-functional.json) completed all three warm-ups and all three paths on the first **two** pinned real messages. Direct and one-token predictions matched each other; JSON differed on one. This checked execution and reporting only. Two examples cannot validate moderation quality, and these timings are not presented as benchmark results. On a single CPU thread, the full 100-message run can take tens of minutes or longer.

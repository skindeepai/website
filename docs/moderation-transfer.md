# Moderation prompts and browser caching

The useful change was **keeping the original moderation rules and removing unrelated co-host instructions from single-message requests**. Shortening the rules themselves made results worse. SSN now reads the final OK/BLOCK scores directly on its default Qwen 3.5 0.8B moderation path.

## Prompt comparison

Actual SSApp/Electron, packaged Transformers.js 4.2.0, WebGPU, q4. These are 50 previously inspected, human-annotated [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) development messages: 24 toxic and 26 benign. Every method received the full same message, without history or truncation. These labels measure ToxicChat toxicity, which is not identical to every application's profanity policy.

| Method | Correct / 50 | Mean ms/message | Toxic missed | Benign blocked |
|---|---:|---:|---:|---:|
| Existing prompt plus co-host instructions, reply capped at 8 tokens | 38 | 804.5 | 7 | 5 |
| Original rules as system instructions, same reply cap | 41 | 416.7 | 5 | 4 |
| Same focused prompt, first generated token only | 41 | 371.4 | 5 | 4 |
| Same focused prompt, direct OK/BLOCK scores | 41 | 363.8 | 5 | 4 |
| Shorter rules, direct scores | 32 | 305.4 | 3 | 15 |
| Original rules kept in the user message, no co-host instructions | 37 | 363.5 | 13 | 0 |

The last row is a subsequent placement experiment on the same development IDs. Its own paired baseline was 819.3 ms and 38/50. Full outputs, prompts and every tested variant are retained in the records below; the table does not pool timings across runs.

## Follow-up on 50 different messages

The fixed first 50 IDs from the existing browser benchmark: 25 toxic, 25 benign, disjoint from the development IDs. They were already used in earlier research, so this is **not fresh validation**. The focused prompt was chosen before this run.

| Method | Correct / 50 | Mean ms/message | Toxic missed | Benign blocked |
|---|---:|---:|---:|---:|
| Existing SSN framing and written reply | 31 | 853.9 | 13 | 6 |
| Focused prompt, one token | 38 | 408.9 | 7 | 5 |
| Focused prompt, direct scores | 38 | 395.7 | 7 | 5 |
| Direct scores with fixed-prefix caching | 37 | 260.4 | 8 | 5 |
| Cached, recompute close decisions without cache | 38 | 462.8 | 7 | 5 |

Direct scores and one-token replies made identical allow/block decisions on both 50-message sets after the existing parser. On the follow-up, the one-token path produced one invalid reply, which the parser blocked; the baseline produced four invalid replies. Correct counts here score the final allow/block behavior, including that fail-closed policy, not just valid label generation. Direct readout always returned a label. The focused direct path took **53.7% less time** than the existing framing in the follow-up. This is a combined prompt-and-readout change, not a 53.7% saving attributable to avoiding text alone. A separate execution of the patched production worker reproduced all 50 decisions, averaging 421.0 ms; that repeat was not interleaved with the baseline.

All paths use **all 24 layers**. There is no trained task classifier here. The direct path compares two entries in the normal final vocabulary output. In this tokenizer OK=3793, SAFE=80898 and BLOCK=38637, each one token. One token is not one letter. A normal valid reply also generated its end-of-reply token; direct readout needs neither text decoding nor that extra forward pass.

## What happened with caching?

SSN already caches downloaded model files. That avoids downloading weights again; it does not reuse inference work. We separately tested saving the computation for 95 fixed instruction tokens, then processing only each new message and its history. Every message still visits all layers.

Qwen 3.5 needs **48 state tensors**, including recurrent and convolution state as well as attention keys and values. The experiment retains only an immutable CPU copy of the policy state, makes a private copy for each request, checks the exact token prefix, and never caches chat content. In the follow-up, warm cache reuse changed three of 50 decisions versus uncached direct inference. Net accuracy lost one correct answer; it also changed which messages were missed.

Recomputing when the two cached scores differed by less than 1 preserved all 50 reference decisions, but triggered 27 full recomputations and became slower than the uncached path. That margin is a numerical guard selected during exploration, not calibrated confidence or a guarantee of safety. **Prefix caching is therefore not enabled in SSN's production moderation path.**

Cache timing above is warm, after prefix construction during warm-up. The initial nine-method development run used a single cache slot and switched policies repeatedly; its cached timings include those rebuilds and are not steady-state throughput. A separate cached-generation attempt failed in the packaged multimodal runtime with missing image inputs. The successful cache experiments use direct forward calls.

The [SkinDeep browser benchmark](../moderation-benchmark.html) now offers a fourth, cached-instructions path using its separate Qwen 2.5 0.5B model. Its total charges for building the prefix on the first timed request and compares changed decisions explicitly. This is an experiment users can run, not a claim that caching preserves quality.

## Production history and scope

History cannot be assumed equivalent to isolated-message testing. Moving the policy into the system message incorrectly blocked an authored supportive split phrase. The final patch therefore uses the focused prompt **only when both recent history and compact candidates are empty**. With history, it retains the previous prompt framing and directly reads the label scores. All authors, recent messages and compact candidates remain present. Four authored context checks match the previous decisions; the model still misses the split F/U/C/K example in isolation. SSN's existing compact-profanity rule handles that separately and was retained. These are regression checks, not a real-world conversation benchmark.

Neither moderation path imports remembered co-host facts or conversation memory. The fast path is gated to stateless, text-only `localqwen` requests using the default OPT 0.8B model. Co-host chat, vision, Gemma, Qwen 2B and other providers retain their existing paths. Actual `ai.js → client → worker` testing also exposed and fixed a moderation-specific handoff that lost the initialized model host/provider before generation. No model weights or global app settings changed.

## Reproduction and remaining limits

- Model: [onnx-community/Qwen3.5-0.8B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT), revision `fafab72d87a9e6be3925b38caf48286d2838f2d0`. [Asset hashes and original source snapshots](../results/moderation-transfer/assets.json). Public weights were served locally; the production-hosted weight bytes were not accessed or verified.
- Dataset: ToxicChat0124 revision `29df8e4dba60e1f4af4b4075c0705c5b313548a8`, CC BY-NC 4.0. IDs and full inputs: [development](../results/moderation-transfer/tasks.json), [follow-up](../results/moderation-transfer/validation-tasks.json).
- Raw decisions: [development](../results/moderation-transfer/development.json), [follow-up](../results/moderation-transfer/validation.json), [user-role placement](../results/moderation-transfer/user-placement.json), [production repeat](../results/moderation-transfer/production.json), [context checks](../results/moderation-transfer/production-context.json), [actual integration](../results/moderation-transfer/production-integration.json).
- [Recomputed summary](../results/moderation-transfer/summary.json), [independent record checker](../experiments/check_moderation_transfer.py), [experiment harness](../experiments/moderation_transfer.cjs), [prompt/cache implementations](../experiments/moderation_transfer_worker.js), [runtime and hardware](../results/moderation-transfer/context-final-runtime.json).

Timing is one pass per method and message with alternating order; download, initialization and warm-up are excluded. Decision times include prompt preparation and readout, but exclude subsequent cleanup. SSApp used an isolated profile, the real packaged runtime and local-only network access. ONNX CPU work was capped at one thread. These are device-specific exploratory results, not deployment-grade moderation accuracy.

Next useful work is a labeled multi-message chat benchmark. Another implementation opportunity is that this packaged runtime drops `num_logits_to_keep` on the multimodal forward path, so even direct mode computes vocabulary scores at every input position. We observed the output shapes and retained the existing packaged runtime; no claimed speedup assumes that issue has been fixed.

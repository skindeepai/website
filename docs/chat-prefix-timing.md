# Reusing the fixed instruction prefix

Reusing the fixed instruction prefix reduced the original float Qwen classifier's measured 100-message workload from **60.77 to 37.73 seconds**, a **37.92% reduction**, while preserving all 100 decisions in each of three passes. The classifier remains correct on 86/100 messages; this optimization does not fix its moderation errors.

The cache contains only the fixed system prompt and user-message opening, not any user's message or answer. Every remaining token still passes through all 24 layers. It is not an early-exit method.

## Measured outcomes

| Path | Pass 1 / 100 | Pass 2 / 100 | Pass 3 / 100 | Mean workload time |
|---|---:|---:|---:|---:|
| Full input every time | 60.942 s | 61.170 s | 60.213 s | 60.775 s |
| Reuse fixed prefix | 38.965 s | 37.472 s | 36.753 s | 37.730 s |

All six workloads completed. All 600 recorded labels match the sealed float-Qwen quality evaluation, and every logit comparison meets the predeclared tolerance. The largest absolute logit change is 0.00002313. This is measured numerical agreement on these inputs, not a guarantee of identical floating-point output for every possible message.

A separate agent independently checked the saved 600 decisions, logit deltas, workload order, timing arithmetic, layer traces, hashes and token accounting. It reconstructed the fixed prefix and all 100 complete input lengths with the pinned tokenizer and source CSV, confirming every exact prefix match. This was an internal artifact and tokenizer audit, not an external replication of model execution.

The reusable prefix is 53 tokens and its 24-layer key/value cache occupies 1,302,528 bytes. A workload normally forwards 11,755 input tokens. Reuse forwards the 53-token prefix once plus 6,455 suffix tokens, totaling 6,508 newly processed positions. The fixed cache is cloned for each request, and all original-cache hashes remain unchanged before and after each workload. Those copies and the once-per-workload cache build are included in the measured time.

Use the paired full-input reference in this table when interpreting the reduction. Other studies used different execution schedules; their absolute times are not interchangeable. Three warm passes on one CPU do not establish the same gain on other hardware or a live service.

## Fixed protocol

The study uses the same 100 messages already evaluated in the refinement study, with the original frozen Qwen model and classifier. There is no training, gate, changed label threshold or selection using these outcomes. This is an execution-equivalence test on consumed data, not new moderation validation.

We render the original template with a fixed marker in place of user content, take the text before that marker, and unconditionally remove trailing carriage returns/newlines before tokenization. This boundary rule was chosen before execution. The omitted newline remains in each request's suffix. Every complete input must begin with exactly the resulting prefix token sequence; a mismatch fails the attempt instead of changing the prefix based on a message.

The two paths are:

- **Full input:** tokenize and process the complete original prompt for each message.
- **Prefix reuse:** build the fixed prefix cache once per 100-message workload, then process each request's suffix against a fresh private copy of that cache.

Three whole-corpus passes alternate path order: full/reuse, reuse/full, full/reuse. Message order stays fixed. The first measured full-input pass supplies same-runtime reference logits; all labels also have to match the sealed float-Qwen evaluation. Absolute/relative logit tolerances of 0.001/0.0001 were fixed before execution. Slower or failed outcomes are retained.

The corpus wall timer includes raw text tokenization, truncation and templating, cache construction, copying, input preparation, all model forwards, classifier readout, trace hooks and result conversion. The prefix cache is rebuilt inside every reuse-path timer. Loading, excluded warm-up calls, subsequent parity comparisons and file writes are outside the timer. Four CPU threads are used, with other launched model jobs paused.

## Cache isolation and model positions

The published Transformers 4.50.3 source was inspected before implementation. Its dynamic cache appends keys and values when attention receives a cache object; merely setting a return-cache flag false would not protect a supplied cache. Converting it to a legacy tuple also returns tensor references, not copies.

The runner therefore preserves a tuple of the fixed prefix tensors and never passes those tensors directly into a user-message forward. Every request clones all 24 key/value pairs and constructs a new dynamic cache. Storage addresses must differ from the fixed tensors. Each private cache must start with the prefix length and finish with that request's full length at every layer. The fixed tuple is hashed before and after the entire workload to detect mutation.

The attention mask covers the complete prefix-plus-suffix context. Suffix positions and cache positions start at the prefix length. The runner requires default RoPE scaling, because sequence-length-dependent rotary frequencies would require additional validation. Both prefix prefill and every suffix forward must execute the contiguous 24-block trace.

Processed-token counts report tokens newly forwarded through the network, including the once-per-workload prefix build. Suffix queries still attend to cached prefix keys and values; token-count reduction is not an exact FLOP or latency reduction. No model layer is removed or skipped.

## Limits and reproduction

This optimization applies only while the instruction template, model, tokenization, positions and numerical configuration remain compatible. It does not reuse user-message KV across requests. It does not improve moderation accuracy, establish universal numerical equivalence or measure a live chat queue. There is one CPU and one sequence of three warm passes.

Run `python experiments/chat_prefix_timing.py --prepare`, then review the sealed protocol before running `python experiments/chat_prefix_timing.py` in an isolated timing window. Existing attempts cannot be silently overwritten.

- [Source](../experiments/chat_prefix_timing.py)
- [Sealed protocol and dependency hashes](../results/chat-prefix-timing/protocol.json)
- [All six workload records, cache checks and decisions](../results/chat-prefix-timing/records.json)
- [Timing and equivalence summary](../results/chat-prefix-timing/result.json)
- [Excluded warm-up records](../results/chat-prefix-timing/warmup.json)
- [Parent quality study](chat-refinement.md)

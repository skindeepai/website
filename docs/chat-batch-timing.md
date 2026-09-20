# Process an available queue together

Processing four messages together reduced the 100-message queue from **61.20 to 55.28 seconds**, a **9.67% reduction in time**. Every decision stayed identical to the existing full-depth classifier: **86/100 correct**, including the same errors. Groups of eight helped less on average and were slower than one-at-a-time processing in the third pass.

This experiment keeps the existing float32 Qwen model and classifier and changes how messages are grouped for execution. A queue of 100 messages is processed one at a time, four at a time, or eight at a time. Every message still executes all 24 transformer layers. This measures queued-work throughput, not the waiting time of a newly arriving live-chat message.

The method was selected after inspecting the [refinement experiment](../results/chat-refinement/result.json). It reuses that experiment's 100 evaluation messages and frozen full-depth predictions. It is an execution-equivalence check, not another fresh test set or a new quality claim.

## Results

All figures use this experiment's own one-message baseline. Mean durations include fresh preparation and sorting in each of three complete corpus passes.

| Messages per batch | Mean time / 100 | Messages / second | Less queue time | Correct / 100 | Changed decisions | Layers skipped |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 61.20 s | 1.63 | 0% | 86 | 0 | 0% |
| 4 | 55.28 s | 1.81 | 9.67% | 86 | 0 | 0% |
| 8 | 57.70 s | 1.73 | 5.72% | 86 | 0 | 0% |

Four-message batching increased measured throughput by 10.70%. That is the reciprocal of the duration change, not an additional saving. The modest improvement is specific to this queue and CPU; larger batches were not consistently better.

| Messages per batch | Pass 1 | Pass 2 | Pass 3 | Calls per 100 | Real token slots | Padded token slots |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 62.31 s | 62.49 s | 58.81 s | 100 | 11,755 | 11,755 |
| 4 | 56.59 s | 56.47 s | 52.79 s | 25 | 11,755 | 12,024 |
| 8 | 55.74 s | 56.75 s | 60.61 s | 13 | 11,755 | 12,484 |

Eight-message processing uses 12 full batches plus a final batch of four. Padding adds 269 token slots for groups of four and 729 for groups of eight. All batch calls execute blocks 1 through 24; batching shares execution across messages without skipping any message's computation.

All 900 recorded decisions across the nine corpus passes match the frozen reference. The largest raw-logit difference from batch one is 0.00005925 for batch four and 0.00003588 for batch eight, both inside the sealed tolerance. All 414 batch traces are contiguous and contain 24 blocks. These repeats still contain only 100 unique quality examples.

A separate internal review agent independently recomputed the source-label comparisons, timing summaries, numeric differences, batch sizes, padding counts, traces and source hashes from the saved files. This is an internal artifact audit, not an external replication.

## What is held fixed

The model is Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`. After all 24 blocks and final normalization, the original standardized linear 896-to-2 classifier returns SAFE or BLOCK. There is no text generation, quantization, retraining or early exit.

The real messages are human-annotated ToxicChat0124 examples. The 100 IDs and data checksums come from the [refinement protocol](../results/chat-refinement/protocol.json); 50 are benign and 50 toxic. The existing float model got 86/100 correct. Matching those predictions preserves its existing errors, rather than making it a reliable moderation service. [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) is licensed CC-BY-NC-4.0.

Each timed corpus call starts again from raw messages. The same tokenizer limits user text to 256 Qwen tokens, renders the same chat instruction and tokenizes it. All batch sizes use the same stable sort by actual input length. Consecutive similarly sized messages are grouped together to limit wasted padding; the last short group is retained. This assumes that the queue is already available and its requests can be processed out of order.

Inputs are explicitly left-padded, with a zero attention mask on padding. Real-token position IDs start at zero independently for each message. Every final sequence position is a real token, so the classifier reads the same final-token representation as the one-message reference. The timing includes padding, masks and position preparation.

## Measurement and safeguards

The machine is an AMD Ryzen 9 3950X using four compute threads and one inter-op thread. The runtime is PyTorch 2.6.0+cpu and Transformers 4.50.3, float32, eager attention, with no KV cache. No other agent-launched model workload runs during the recorded window.

After one excluded warm-up per batch size, three full-corpus passes rotate the size order: 1/4/8, 4/8/1, then 8/1/4. Each corpus timer includes input preparation, length sorting, padding, actual forward passes, classifier output, output-record conversion and hook removal. It excludes model and dataset loading, warm-up, post-corpus validation, reporting and file writes.

Every output label is checked against the saved float-Qwen refinement prediction. The first timed batch-one pass also supplies reference logits. The protocol requires logits to agree within absolute tolerance 0.001 plus relative tolerance 0.0001; observed maximum differences are reported. Every batch records its actual size and the contiguous block trace 1 through 24. Reducing the number of forward calls does not reduce the layers executed by any message.

All requested batch sizes and passes are retained, including slower outcomes or failures. The runner never substitutes one-message outputs or silently falls back to a smaller batch. An incomplete pass has no complete-corpus throughput figure. A pass with changed labels or logits outside tolerance is explicitly marked as an equivalence failure.

## Evidence

- [Sealed protocol](../results/chat-batch-timing/protocol.json)
- [Runner](../experiments/chat_batch_timing.py)
- [Timing, padding and parity summary](../results/chat-batch-timing/result.json)
- [All nine corpus attempts and 900 outputs](../results/chat-batch-timing/records.json)
- [Excluded warm-up batches](../results/chat-batch-timing/warmup.json)
- [Original full-depth reference predictions](../results/chat-refinement/predictions.json)

These results concern an already available 100-message queue on one CPU. They do not measure a 600-message workload, peak production memory, another device, or the delay needed to wait for a live batch to fill. Identical predictions on this sample cannot guarantee universal numerical equivalence.

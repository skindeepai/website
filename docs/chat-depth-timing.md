# Timing the exact fixed-depth classifiers

This follow-up closes the missing timing measurement for the existing MLP classifiers at blocks 6, 12, 18 and 24. It uses the same saved heads and the same 100 ToxicChat messages as the [head comparison](chat-smoke-heads.md), with no retraining. These are new execution measurements on reused examples, not a fresh test of accuracy.

The exact block-12 classifier takes **28.10 seconds per 100 messages**, compared with **56.26 seconds** at full depth: 50.06% less measured request time while skipping 12 of 24 blocks. Its accuracy remains worse: **71/100 versus 77/100**. The timing gap is closed; the quality gap is not.

## Results

Durations are means of three paired passes over the same 100 messages. They sum the individual timed request durations, rather than timing a whole-service workload.

| Stop after block | Blocks skipped | Correct / 100 | Mean time / 100 | Less request time vs full |
|---|---:|---:|---:|---:|
| 6 of 24 | 18 (75%) | 75 | 13.85 s | 75.39% |
| 12 of 24 | 12 (50%) | 71 | 28.10 s | 50.06% |
| 18 of 24 | 6 (25%) | 78 | 41.71 s | 25.86% |
| 24 of 24 | 0 (0%) | 77 | 56.26 s | 0% |

Block 18 has one more correct answer overall than full depth, but it introduces seven errors that full depth avoided while correcting eight other errors. Three of the newly introduced errors are missed toxic messages. It therefore does not demonstrate no quality loss.

| Stop after block | Missed toxic / 50 | False blocks / 50 | New errors vs full | Corrected errors vs full | New toxic misses vs full |
|---|---:|---:|---:|---:|---:|
| 6 | 17 | 8 | 12 | 10 | 6 |
| 12 | 20 | 9 | 14 | 8 | 8 |
| 18 | 16 | 6 | 7 | 8 | 3 |
| 24 | 18 | 5 | 0 | 0 | 0 |

All three measured passes are retained:

| Stop after block | Pass 1 / 100 | Pass 2 / 100 | Pass 3 / 100 | Median request | 95th-percentile request |
|---|---:|---:|---:|---:|---:|
| 6 | 13.80 s | 13.72 s | 14.02 s | 106.37 ms | 345.54 ms |
| 12 | 27.61 s | 28.32 s | 28.36 s | 214.67 ms | 731.04 ms |
| 18 | 41.37 s | 41.37 s | 42.39 s | 320.83 ms | 1,055.40 ms |
| 24 | 56.82 s | 56.52 s | 55.43 s | 419.25 ms | 1,517.30 ms |

All 1,200 timed labels match the saved classifier labels. Every trace confirms exactly the selected contiguous blocks ran. The largest probability difference from the cached scores is 0.000007034, below the predeclared 0.0005 tolerance. This supports execution parity, not independent quality validation.

## What runs

Qwen2.5-0.5B-Instruct has 24 transformer blocks. After the selected block, a small classifier converts the final input token's 896-number hidden representation into SAFE or BLOCK scores. The classifier has 64 hidden units with GELU and two output units. Each depth has its own previously trained weights; this is not one classifier moved between layers.

At blocks 6, 12 and 18, a hook calculates the classifier output and immediately interrupts the forward pass. No later block executes. The full-depth path completes block 24 and the final normalization, then reads its classifier. This matches how the original features were collected. Neither path generates text or runs a vocabulary decoder. These fixed-depth paths have no confidence gate: every message stops at the selected block.

Stopping at block 12 skips 12 of 24 blocks: 50% of the transformer blocks. That is not a claim to remove 50% of the model's stored parameters or halve every part of request processing. The full checkpoint remains loaded; embedding, tokenization and the classifier still run.

The heads were trained on 384 messages; temperature scaling used a separate 128. Evaluation reuses the already inspected 100-message balanced subset, containing 50 benign and 50 toxic messages. Accuracy on this artificial class balance is not expected live-service accuracy. See the [original training protocol](../results/chat-smoke-heads/protocol.json) and [shared split manifest](../results/chat-smoke/protocol.json).

## Timing method

The runner uses one AMD Ryzen 9 3950X CPU, four compute threads, one inter-op thread, float32, eager attention, batch size one and Transformers 4.50.3. Model loading and training are excluded. Each timed call starts with the raw message and includes truncation to 256 Qwen tokens, chat templating, tokenization, hook registration, the actual executed blocks, classifier readout and hook removal. Validation assertions and result writing occur outside the timer.

One execution per depth warms the model before recording. Three passes process the same 100 messages through all four depths, giving 1,200 timed forwards. Within each message the depth order rotates; each depth occupies each position exactly 25 times per pass. Other agent-launched model jobs are paused during the run. This balances local execution order, but it is not a benchmark of a loaded production queue.

Each execution must match its cached classifier label and probability scores within an absolute tolerance of 0.0005. Every recorded block trace must contain exactly the contiguous blocks from 1 through the selected depth. Predictions must remain the same across all three repeats. Timing repeats do not increase the quality sample from 100 to 300.

## Evidence and reproduction

The protocol is sealed before execution, including hashes of the runner, training protocol, split manifest, head weights and reference predictions. Completed results cannot be overwritten by the runner. To reproduce, preserve the original artifacts and run in a separate copy with a new output directory and protocol; do not replace these measurements.

- [Sealed protocol](../results/chat-depth-timing/protocol.json)
- [Runner](../experiments/chat_depth_timing.py)
- [Timing and quality summary](../results/chat-depth-timing/result.json)
- [All 1,200 timed executions](../results/chat-depth-timing/records.json)
- [Four excluded warm-up executions](../results/chat-depth-timing/warmup.json)
- [Original classifier scores](../results/chat-smoke-heads/predictions.json)
- [ToxicChat dataset](https://huggingface.co/datasets/lmsys/toxic-chat), human-annotated rows from the pinned ToxicChat0124 files in the [parent protocol](../results/chat600/protocol.json). The dataset license is CC-BY-NC-4.0.

This measures four fixed stopping points, not a new adaptive policy. It does not establish preserved quality, hardware portability or production throughput. The original accuracy outcomes remain unchanged; faster execution cannot make those outcomes independent validation.

Use this run's full-depth timing as the reference for these fixed-depth savings. The earlier adaptive-gate and BERT cascade measurements were separate runs; comparing their durations with this run is not a paired speed comparison.

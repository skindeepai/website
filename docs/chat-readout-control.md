# Where the output-path speedup comes from

Scoring only SAFE and BLOCK reduced total request time by **6.4%** against a full-vocabulary readout with the same cache setting. All four paths returned identical decisions on the same 50 messages, in both timing passes. This confirms a useful execution saving without changing the model's answers.

| Path | KV cache | Mean ms/message | Pass 1 / pass 2 |
|---|---|---:|---:|
| Two vocabulary rows | Off | 650.9 | 666.2 / 635.5 |
| Full vocabulary, then select SAFE/BLOCK | Off | 695.5 | 722.1 / 669.0 |
| Two vocabulary rows | On | 656.2 | 676.2 / 636.2 |
| Generate one forced SAFE/BLOCK token | On | 700.0 | 727.2 / 672.9 |

The readout stage alone averaged **0.055 ms** for two rows versus **41.106 ms** for the full vocabulary. Both read the same last-position representation after all 24 Qwen blocks. The paired whole-request difference was 44.7 ms; the small remaining difference includes variation in the otherwise identical backbone calls. No layers were skipped.

The original [6.6% comparison](chat-output-steps.md) changed vocabulary scoring, cache use and the generation API together. This follow-up directly isolates full versus two-row vocabulary scoring. The cached two-row control was 5.3 ms slower on average than its uncached counterpart, with a smaller difference in the second pass. The generation row still combines full projection and generation machinery; it does not independently measure API overhead. Manual cached-state cleanup occurs after the timer.

## What was measured

Pinned Qwen2.5-0.5B-Instruct, CPU float32, eager attention, four compute threads and one interop thread, batch one. These are the same reused 25 toxic and 25 benign ToxicChat examples as the preceding output study. No model or threshold was trained or selected. Every path was correct on 31/50, missed 3/25 toxic messages and falsely blocked 16/25 benign messages. Faster execution preserves those mistakes too.

Two passes rotate the four methods per message and reverse their order on the second pass. Timers include truncation, input preparation, hooks, inference and readout; generation also includes detokenization. Model loading, warm-up, file writes and audit calculations are excluded. Separate stage timers record input preparation, backbone and readout. Warm-ups use a separate harmless sentence. Timing repeats are not additional quality examples.

The independent checker recalculates all 400 call records, source labels, execution order, layer traces, selected scores and summary arithmetic. It also verifies every decision against both passes of the original output study. This is a CPU experiment; the browser benchmark still uses its existing full-vocabulary implementation.

## Evidence

- [Protocol and frozen IDs](../results/chat-readout-control/protocol.json)
- [Every measured call and stage](../results/chat-readout-control/records.json)
- [Summary](../results/chat-readout-control/result.json) and [independent audit](../results/chat-readout-control/audit.json)
- [Inference runner](../experiments/chat_readout_control.py) and [checker](../experiments/check_chat_readout_control.py)

Run `python experiments/check_chat_readout_control.py` to check retained evidence. The inference runner refuses to overwrite a completed directory; repeat experiments must preserve the original evidence.

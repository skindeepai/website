# Trying eight approaches with small experiments

**Exploration, not validation.** These are 100 previously inspected ToxicChat messages, balanced to 50 toxic and 50 benign labels. Every candidate is reported. The sample helps choose follow-ups; it does not estimate natural chat prevalence or establish deployment reliability.

## Readouts, stopping rules and a specialist

| Method | Correct /100 | Toxic missed /50 | Benign blocked /50 | Projected Qwen blocks skipped |
| --- | --- | --- | --- | --- |
| Linear readout | 77 | 16 | 7 | 0.00% |
| Small neural readout | 77 | 18 | 5 | 0.00% |
| Separate SAFE/BLOCK thresholds | 78 | 17 | 5 | 17.25% |
| Checkpoint agreement | 79 | 15 | 6 | 16.50% |
| Learned gate, linear readout | 75 | 18 | 7 | 9.00% |
| Learned gate, neural readout | 79 | 15 | 6 | 33.75% |
| Tiny specialist | 72 | 20 | 8 | Different architecture |
| Specialist then Qwen | 79 | 14 | 7 | Different architecture |
| Older Qwen fallback reference (1,400 training labels) | 78 | 15 | 7 | 0.00% |

The two new full-depth readouts use 384 training messages. The specialist is a 4.37-million-parameter BERT trained on those same messages; its Qwen fallback is the older 1,400-message-trained classifier. That is an unequal-supervision fallback reference, not a controlled model-architecture ranking.

The neural learned gate adds one error and corrects three relative to its full-depth readout. It newly misses no toxic messages in these 50 positive examples, but the sample is too small for the earlier reliability tolerance. The specialist cascade adds two errors and corrects three, including one new toxic miss. Net accuracy cannot cancel those harms.

REVIEW is also tested. The neural readout accepts 64 messages and gets 56 right (87.5%), leaving 36 unresolved. It fails its development target of 95% accepted-answer accuracy. The linear rule sends all 100 to REVIEW because no eligible development threshold exists. Neither outcome counts unresolved requests as correct.

[All head/gate comparisons and completed timing](chat-smoke-heads.md), [specialist and cascade](chat-smoke-specialist.md). Feature-cache projections alone are not speedups.

## Actual early-stop execution

The full neural readout took 63.6 seconds for 100 messages; the learned gate took 42.1 seconds, a 33.8% reduction in summed request time. This is one warm paired pass, alternating path order, on one CPU with four threads and no concurrent model jobs. Original text preparation, readouts, gates and hooks are included; loading and training are excluded.

The gate stopped 32 messages at block 6, 11 at block 12, 17 at block 18, and carried 40 through all 24 blocks. Every executed block trace and prediction matched the recorded policy. The gate uses checkpoint scores and their changes to predict an error or whether more layers would help; it does not inspect an unavailable final-layer answer.

[Timing summary](../results/chat-smoke-heads/runtime.json), [all 200 timed calls](../results/chat-smoke-heads/runtime-records.json). This is measured computation saving on reused examples, not a verified equal-quality deployment speedup.

## Training the model, rather than only its readout

All four adapter recipes fit their initial linear heads on 384 messages, then take only 32 batches of four adapter updates: 128 message presentations. Checkpoints at steps 16 and 32 are selected on a 32-message development subset. This is a short-budget experiment, not a comparison of converged training methods. The original slower attempt and the prospective budget amendment are preserved.

| Recipe | Correct /100 | Toxic missed /50 | Benign blocked /50 | Selected step |
| --- | --- | --- | --- | --- |
| Frozen starting readout, layer 24 | 77 | 16 | 7 | Initial |
| Task adaptation, full 24 | 73 | 19 | 8 | 16 |
| Intermediate training losses, full 24 | 77 | 17 | 6 | 16 |
| Intermediate losses + distillation, full 24 | 80 | 14 | 6 | 16 |
| Frozen starting readout, layer 12 | 71 | 17 | 12 | Initial |
| Physically truncated 12 + adaptation | 66 | 22 | 12 | 16 |

| Recipe versus its initial readout | New errors | Corrected errors | New toxic misses |
| --- | --- | --- | --- |
| full | 5 | 1 | 4 |
| joint | 1 | 1 | 1 |
| distill | 0 | 3 | 0 |
| fixed12 | 6 | 1 | 5 |

Intermediate-loss and distillation rows above still use all 24 layers for the primary output. Their separate layer scores are diagnostic; they do not by themselves establish an adaptive stopping policy. The fixed-12 model actually contains twelve decoder blocks.

The distillation teacher is the selected task-adapted full model trained on the same labels. Its training-set logits are in-sample soft targets, not independent labels or proof that the teacher is stronger. Initial heads can already nearly fit this small training set, leaving little classification loss to drive useful adapter changes. Checkpoint selection excludes step 0; regressions against that initial readout remain visible.

[Adapter methods, every layer and artifacts](chat-smoke-adaptation.md).

## Maze and coordinate follow-ups

- A structured maze policy trained on the original 256 states still reaches 0/10 goals; forbidding wall collisions raises it to 2/10. Training on 4,992 states from the same training layouts raises this to 5/10, or 7/10 with the explicit legal-action mask. BFS reaches 10/10. More supervision and legal constraints help, but legal loops remain. [Maze details](maze-smoke.md).
- A confidence-only coordinate rejection rule rejects both absent targets across two screenshot folds, with no rejected present targets. It still accepts one wrong coordinate. These are eight previously inspected examples of the same interface and repeated instructions, not a learned visual presence head or general absent-target validation. [Coordinate diagnostic](coordinate-abstention-smoke.md).

## What these tests can establish

A development split chooses each policy before that policy is evaluated; however, all examples were already inspected in the earlier project. Different optimization budgets and selected variants make this exploration, not a definitive ranking. The old 600-message study remains intact.

Even zero added errors in 100 independent examples would give an individual one-sided 95% upper error bound of about 2.95%, above the earlier 1% requirement. With 50 toxic examples, zero added toxic misses would give an upper bound of about 5.82%, above 5%. These reused samples offer still less basis for a confirmatory claim.

Next, carry promising development-selected recipes to genuinely unused messages with frozen thresholds, matched error limits and repeated isolated timing. Do not keep choosing variants against this 100-message set.

[Separate adversarial review](chat-smoke-review.md), [shared sample/protocol](../results/chat-smoke/protocol.json), [selection code](../experiments/chat_smoke_common.py). No raw chat text is published.

## Specialist timing

| Path | Messages | Summed request seconds |
| --- | --- | --- |
| specialist | 100 | 0.31 |
| cascade | 100 | 32.46 |
| full_qwen_reference | 100 | 64.75 |

One warm interleaved pass with rotating path order, four CPU threads and actual BERT plus Qwen execution for fallback requests. Includes original text preparation and readouts; excludes loading and training. These timings are not repeated-run estimates. The cascade newly misses one toxic message that its full-depth fallback gets right. [Every timed call](../results/chat-smoke-specialist/timings.json), [timing protocol](../results/chat-smoke-specialist/timing-protocol.json).

## Separate shorter-warm-up repair

The adversarial review found that the original starting heads already fit almost every training label. A separately recorded repair reduces head warm-up from 200 to five optimizer steps, then repeats the joint adapter recipe. This was proposed after observing earlier failures; it is additional exploration, not a replacement for them.

| Layer | Initial correct /100 | After training /100 | Initial toxic misses /50 | After training /50 |
| --- | --- | --- | --- | --- |
| 6 | 66 | 70 | 12 | 26 |
| 12 | 71 | 65 | 11 | 27 |
| 18 | 79 | 71 | 7 | 22 |
| 24 | 78 | 69 | 8 | 26 |

[Recorded rationale](../results/chat-smoke-reset/rationale.json), [protocol](../results/chat-smoke-reset/joint-protocol.json), [result](../results/chat-smoke-reset/joint.json), [matched initialization](../results/chat-smoke-reset/initial-frozen-baseline.json).

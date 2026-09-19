# Frozen-feature heads and stopping rules: exploratory smoke test

The small neural-network head did not improve full-depth accuracy over a new linear head: both got 77 of 100 decisions right. Its learned stopping gate reached 79/100 while executing 33.75% fewer transformer blocks in a follow-up parity run. The measured request durations totaled 42.06 seconds versus 63.56 seconds at full depth, a 33.84% reduction. These are reused examples and one CPU timing pass, not independent quality validation.

The same learned gate recipe made the linear model worse. A REVIEW policy that met 95% covered accuracy in development fell to 87.5% on the evaluation subset. Both failures are retained below.

## What was trained

This uses the [shared smoke protocol](../results/chat-smoke/protocol.json): 384 training examples (256 benign, 128 toxic), 128 temperature/gate-fit examples, 128 development selection examples, and 100 evaluation examples (50 benign, 50 toxic). The evaluation examples come from the previously inspected 600-message ToxicChat study. Balanced evaluation is useful for seeing both error types; its overall accuracy does not estimate live moderation accuracy and should not be compared directly with the old 600-message accuracy.

Head and gate fitting use inputs and cached hidden states from the old frozen Qwen2.5-0.5B-Instruct run. No raw messages are published. Three previously identified repeated effective inputs were excluded by the shared manifest. No transformer weights changed here. The selected MLP gate was subsequently checked with actual transformer execution, described below.

At each of blocks 6, 12, 18 and 24, we compared:

- A linear 896-to-2 head, trained for 300 AdamW steps, learning rate 0.01.
- An 896-to-64-to-2 head with GELU and 25% dropout, trained for 200 AdamW steps, learning rate 0.002. Dropout is disabled during evaluation.

Both use train-only input standardization, weight decay 0.1, inverse-frequency class weighting, and seed 109. Temperature scaling is fitted on the 128 gate-fit examples. These weighted and temperature-scaled scores are not guaranteed probabilities of correctness.

Every gate is fitted or selected before the evaluation outcomes are used:

- **Asymmetric confidence:** separate thresholds for SAFE and BLOCK at each available checkpoint. The selected values are shared across checkpoints.
- **Agreement:** the same approach, additionally requiring the current and previous checkpoint predictions to match. It cannot exit at block 6.
- **Learned gate:** two logistic outputs trained on held-out gate-fit examples. One predicts whether the current decision is wrong; the other predicts whether it is wrong and full depth would correct it. Inputs are current BLOCK probability, confidence, entropy, previous confidence, prediction agreement and change in BLOCK probability. Full-depth information is used only to construct the training target, never as an input to the stopping decision.
- **REVIEW:** use the full-depth classifier only above a selected confidence threshold; leave all other messages unresolved. This is an abstention test, not an evaluated human or model fallback service.

Early-exit selection minimizes mean depth subject to at most 1% newly introduced errors and **zero additional missed toxic messages** on the 128 development selection examples. It does not permit one corrected error to cancel a new error. There is no minimum early-exit coverage requirement, so a policy that continues every message is allowed. These development constraints are not statistical acceptance guarantees.

Temperature and gate fitting reuse the same 128 development examples. Threshold selection uses a separate 128, but all data and earlier outcomes have informed this research. The experiment is explicitly exploratory.

## Every evaluated method

All rows use the same 100 messages, including 50 toxic messages. “Added” counts mistakes that the corresponding full-depth head got right. “Skipped” is an arithmetic projection from the chosen exit depths, not measured runtime savings.

| Head | Policy | Correct / 100 | Missed toxic / 50 | False blocks / 50 | Added errors | Projected blocks skipped |
|---|---|---:|---:|---:|---:|---:|
| Linear | Fixed block 6 | 69 | 17 | 14 | 16 | 75% |
| Linear | Fixed block 12 | 71 | 17 | 12 | 14 | 50% |
| Linear | Fixed block 18 | 75 | 15 | 10 | 10 | 25% |
| Linear | Full block 24 | 77 | 16 | 7 | 0 | 0% |
| Linear | Asymmetric confidence | 78 | 16 | 6 | 0 | 13% |
| Linear | Agreement | 77 | 16 | 7 | 0 | 8% |
| Linear | Learned gate | 75 | 18 | 7 | 2 | 9% |
| MLP | Fixed block 6 | 75 | 17 | 8 | 12 | 75% |
| MLP | Fixed block 12 | 71 | 20 | 9 | 14 | 50% |
| MLP | Fixed block 18 | 78 | 16 | 6 | 7 | 25% |
| MLP | Full block 24 | 77 | 18 | 5 | 0 | 0% |
| MLP | Asymmetric confidence | 78 | 17 | 5 | 0 | 17.25% |
| MLP | Agreement | 79 | 15 | 6 | 1 | 16.5% |
| MLP | Learned gate | 79 | 15 | 6 | 1 | 33.75% |

The MLP learned gate exits 32 messages at block 6, 11 at block 12, 17 at block 18, and continues 40 to block 24. It introduces one false block while correcting three missed toxic messages. The linear learned gate introduces two new missed toxic messages. There is no consistently superior gate across both heads.

For context only, the original classifier trained on 1,400 messages scores 76, 75, 81 and 78 at blocks 6, 12, 18 and 24 on these same 100 messages. Its different training budget and previously selected temperatures prevent a clean attribution of those differences to architecture.

## REVIEW does not solve the problem by itself

| Head | Resolved | Correct among resolved | Missed toxic | False blocks | Unresolved | Toxic among unresolved |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0 | Not defined | 0 | 0 | 100 | 50 |
| MLP | 64 | 56/64 (87.5%) | 7 | 1 | 36 | 22 |

The development target was at least 95% covered accuracy and at least 20% coverage. No linear confidence band qualified, so its fallback policy reviews everything. The MLP chose threshold 0.8, then missed that accuracy target on evaluation. Unresolved messages are not counted as correct. Neither row demonstrates a functioning fallback or safe moderation service.

## Selected policies and actual execution

Linear asymmetric and agreement policies use SAFE 0.90 / BLOCK 0.95. MLP asymmetric uses SAFE 0.95 / BLOCK 0.90; agreement uses SAFE 0.95 / BLOCK 0.80. The learned gate uses error/benefit limits 0.10/0.05 for linear and 0.15/0.10 for MLP. Gates evaluate checkpoints in order and retain full depth if no earlier gate accepts.

After inspecting the exploratory table, we selected the MLP learned gate for a separate actual-forward check. The runner processes all 100 evaluation messages once through each path, alternating which path runs first. One unrecorded warm-up call per path precedes the recorded 200 calls. Other launched model workloads were paused until this run finished. Four CPU threads were used.

| Actual path | Correct / 100 | Sum of request durations | Mean request duration | 95th-percentile request duration |
|---|---:|---:|---:|---:|
| Matched MLP full depth | 77 | 63.56 seconds | 635.64 ms | 1,682.73 ms |
| MLP learned stopping gate | 79 | 42.06 seconds | 420.57 ms | 1,221.27 ms |

All 200 predictions and exit depths match the cached evaluation, and every execution trace contains exactly the contiguous blocks from 1 through the recorded stopping depth. A hook actually terminates the transformer forward at an accepted checkpoint. The gate uses only the current and previous checkpoint outputs. Both paths include text tokenization, model computation, classifier output and trace instrumentation; the early path also includes its gate work. Loading and training are excluded. These totals sum measured request durations; they are not a separately timed whole-corpus service run.

The 33.84% duration reduction supports a local execution claim for this selected method. There is only one paired pass, with no repeated-run confidence interval, hardware replication, batching or live queue benchmark. Choosing this method after seeing the table remains exploratory method selection; measuring it does not make its quality independently validated. It still introduces one new false block. A fresh, larger held-out set is needed before any claim of preserved moderation quality. A stronger full-depth baseline remains necessary: 77/100 on this balanced sample is weak.

## Reproduction and evidence

Run `python experiments/chat_smoke_heads.py` after preparing the shared manifest and original cached features. The script uses two CPU threads. It overwrites this smoke directory on rerun; preserve the directory before changing its configuration. No parent study files are changed.

- [Implementation](../experiments/chat_smoke_heads.py)
- [Executed protocol and script hash](../results/chat-smoke-heads/protocol.json)
- [All results](../results/chat-smoke-heads/result.json)
- [Every selection candidate](../results/chat-smoke-heads/selection-sweeps.json)
- [Per-message scores and decisions](../results/chat-smoke-heads/predictions.json)
- [Linear heads](../results/chat-smoke-heads/linear-heads.npz), [MLP heads](../results/chat-smoke-heads/mlp-heads.npz), [linear gates](../results/chat-smoke-heads/linear-gates.npz), [MLP gates](../results/chat-smoke-heads/mlp-gates.npz)
- [Actual execution runner](../experiments/chat_smoke_heads_runtime.py), [timing summary](../results/chat-smoke-heads/runtime.json), [all 200 execution traces](../results/chat-smoke-heads/runtime-records.json)
- [Adversarial audit](chat-smoke-review.md)

Cached-state depth projections for the other policies still omit head/gate overhead, tokenization, memory traffic and scheduling. They must not be presented as measured latency reductions. Run `python experiments/chat_smoke_heads_runtime.py` for the selected MLP execution check only when other model jobs have finished.

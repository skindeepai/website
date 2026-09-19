# Six hundred real chat moderation decisions

This benchmark follows Steve's proposed practical test: complete a fixed set of SAFE/BLOCK decisions faster without materially worse decisions. The [machine-readable protocol](../results/chat600/protocol.json) was recorded before model inference. It is local prospective documentation, not external preregistration.

## Data and labels

Source: [ToxicChat0124 by Lin and colleagues](https://huggingface.co/datasets/lmsys/toxic-chat), [paper](https://arxiv.org/abs/2310.17389), pinned revision `29df8e4dba60e1f4af4b4075c0705c5b313548a8`, CC BY-NC 4.0. These are real archived messages sent to the Vicuna online demo, not generated messages or live-stream channel logs. This is a noncommercial research evaluation; no live moderation actions are taken.

Use only `human_annotation=True` rows. The source also contains automatically screened benign rows; those are excluded. Remove exact normalized-text duplicates and prohibit conversation-ID overlap across partitions. Fixed seeded selection gives:

| Partition | Messages | Toxic | Benign |
| --- | --- | --- | --- |
| Head training | 1,400 | 189 | 1,211 |
| Temperature and gate selection | 500 | 72 | 428 |
| Independent calibration | 800 | 106 | 694 |
| Held-out workload | 600 | 81 | 519 |

This human-annotated subset has a different toxicity prevalence from the entire source dataset. It is not a random sample of current Twitch chat or any user's channel. Public records contain IDs, labels and predictions, not the raw messages or model responses.

An [effective-input overlap check](../results/chat600/effective-input-overlap.json) subsequently found three calibration messages whose clipped, rendered inputs matched development inputs, despite distinct original texts. None of the 600 test inputs matched any development/calibration input. Before fitting outcomes were available, an [additional acceptance requirement](../results/chat600/calibration-overlap-amendment.json) was recorded: the original guard must pass **and** the same frozen policy must pass the same limits on calibration with those three IDs removed. No training, gate selection or test messages are changed. The original protocol remains intact; this extra check cannot turn an original failure into an acceptance.

BLOCK maps to the dataset's toxicity label, SAFE to its non-toxic label. Those annotations are an operational reference, not universal moderation truth. The classifier's policy wording, missing conversation context and changing community rules may disagree with that reference.

## Identical bounded inputs

Each path receives the same moderation instruction and user message. For compute bounds, messages longer than 256 Qwen tokens are clipped to their first 256 tokens before the chat template is constructed; the instruction and assistant-start marker remain intact. Record every clipped message and report its results separately. Original full-message labels remain unchanged, which can make clipped examples ambiguous. Do not hide them or call this a full-context benchmark.

## Models and comparisons

Frozen Qwen2.5-0.5B-Instruct, pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`. Published Transformers 4.50.3, float32 CPU, eight threads. Linear readouts after blocks 6, 12, 18 and 24 receive identical head architecture and training budget; training standardization uses training data only. Inverse-frequency loss weights address class imbalance. Head temperatures are chosen on tuning data.

Compare four actual execution paths:

1. **Trained full-depth enum:** all 24 blocks, then the task-trained readout returns a numeric category.
2. **Trained full-depth SAFE/BLOCK token:** identical task-trained readout rows, emitting one of two actual Qwen token IDs and decoding it. Equality with the enum path follows by construction. This is the fair supervised output-format control, not an independently trained model.
3. **Early candidate:** the same input, task-trained intermediate readouts and a selected gate; execution actually terminates at an accepted checkpoint.
4. **Untouched LM SAFE/BLOCK token:** all 24 blocks, then Qwen's original vocabulary rows for SAFE/BLOCK. This is a minimal constrained zero-shot output; it does not compute every vocabulary row or generate a long reply. It gets no task training, so accuracy differences cannot be attributed solely to the output architecture.

Also report always-SAFE and a TF-IDF logistic classifier trained on identical examples. A high overall accuracy score is insufficient when most messages are benign.

## Quality criteria fixed before results

On tuning examples, choose the candidate with lowest mean depth among those losing at most one percentage point overall accuracy and five points toxic recall relative to the full-depth trained readout, with at least 5% early coverage. The confidence/agreement/earliest-layer grid is in the JSON protocol. No qualifying candidate disables early exit.

Freeze that choice. Independent calibration must meet all these checks:

- Exact one-sided 95% upper bound on **additional errors / all messages** no greater than 1%.
- Exact one-sided 95% upper bound on **additional missed toxic messages / all toxic messages** no greater than 5%.
- At least 5% early coverage.
- The full-depth reference itself must reach at least 90% observed accuracy and 80% toxic recall on calibration.

These are individual bounds, not a simultaneous 95% guarantee, and additional errors are relative to the full-depth classifier, not absolute error. Report both confusion matrices and every new mistake. Failed gates remain visible as candidates but are not accepted quality-preserving speedups. A policy enforcing the guard falls back to full depth.

## Actual completion time

After other launched model jobs finish, run all 600 test messages sequentially through every path. Repeat each entire workload three times, cycling path order between repetitions. Count actual executed block indices and assert prediction/depth parity against saved readouts for every message.

Report total time to finish each 600-message pass, mean and p95 message latency, exit distribution, correct decisions, missed toxic messages and false blocks. Timing includes tokenization, transformer execution, gate/readout work and token decoding where used. Downloads, model loading and training are excluded. Warm CPU batch-one results do not establish concurrent queue latency, accelerator throughput or a fresh-session result.

Three cyclic orders over four paths are not fully balanced for position. Paired-query bootstrap intervals do not capture thermal drift, corpus-order effects or between-session variation; retain that limitation when interpreting small differences.

Implementation: [chat600.py](../experiments/chat600.py). The initial frozen script hash, source-file hashes and split IDs are in the protocol. Do not change its thresholds after viewing the 600 test outcomes.

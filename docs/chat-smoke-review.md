# Adversarial review of the development smoke tests

The completed smoke tests identify candidates worth checking, but do not establish preserved moderation quality at lower latency. Their numerical records are internally consistent. The most promising gate now has an actual stopped-forward timing check: it saves 33.84% of summed request time in one warmed pass, improves total accuracy by two messages and introduces a new error. The specialist cascade improves accuracy by one while introducing a new toxic miss. These are deliberately small, reused evaluation samples, not fresh validation.

This is a separate agent's code and artifact audit within the same project and session. It is not an external replication or independent attestation that every historical model execution occurred. The checks below use saved probabilities, portable weights, original cached representations, source labels and recorded trajectories. No new transformer inference was launched for this review.

## Scope and completed checks

- Inspected the common manifest and training, gating, abstention, specialist, adapter, maze and coordinate diagnostic implementations.
- Recomputed every head and specialist split summary against saved decisions and original ToxicChat source labels. All recorded IDs followed the shared manifest; saved head labels were also checked for human annotation. This check, including the head policy sweeps, passed 12,396 assertions.
- Independently recalculated all 408 asymmetric/agreement/learned gate candidates, eligibility and selected policies, plus the REVIEW grids. Reconstructed all 740 specialist decisions and its 100 cascade threshold candidates, including the selected winner.
- Reconstructed 4,984 two-output head/gate probability vectors from the portable NPZ weights and original cached hidden states using one CPU thread. They exactly matched the saved vectors. This verifies those readouts, not fresh transformer execution or the historical origin of the cache.
- Audited all 200 new MLP runtime records: alternating paired order, IDs, labels, predictions, exit depths, contiguous actual block traces, elapsed-time sums, means, percentiles and the reported reduction.
- Audited all 300 specialist/cascade/reference runtime records: rotating path order, IDs, predictions, fallback parity, actual BERT/Qwen block traces and request-time totals.
- Audited all 200 adapter runtime records across four variants: the same sealed 50 IDs, source/protocol/weights/result hashes, labels and prediction parity, actual retained-block traces and summed durations.
- Replayed 1,678 maze state/action records using separately written boundary/wall transitions and breadth-first distances; checked episode termination and aggregate outcomes.
- Recalculated both coordinate leave-one-screenshot-out thresholds and all eight saved decisions.

All five adapter quality results and the completed MLP, specialist/cascade and adapter timing artifacts are checked below. No speed claim is accepted merely from a depth projection or parameter count. The earlier 600-message benchmark has its own [completed timing audit](chat600-review.md).

## Moderation outcomes and adverse findings

All rows below use the same 100 previously inspected messages, deliberately balanced to 50 toxic and 50 benign. A new error means the corresponding full-depth reference was correct and the candidate was wrong. Correcting another message does not erase that harm.

| Candidate | Correct | Its full-depth reference | New errors | New toxic misses | What is established |
| --- | ---: | ---: | ---: | ---: | --- |
| Linear asymmetric gate | 78 | 77 | 0 | 0 | 13% projected blocks skipped |
| Linear learned gate | 75 | 77 | 2 | 2 | Failed to preserve toxic decisions |
| MLP asymmetric gate | 78 | 77 | 0 | 0 | 17.25% projected blocks skipped |
| MLP learned gate | 79 | 77 | 1 | 0 | Actual 33.75% blocks skipped; one new false block |
| Tiny BERT alone | 72 | 78 | 15 | 9 | Lower quality than the old Qwen reference |
| Tiny BERT then Qwen | 79 | 78 | 2 | 1 | 49 messages require Qwen; one new toxic miss |

The two full-depth readouts trained on the common 384 messages both score 77/100, but make different errors: the linear head misses 16 toxic messages and the MLP misses 18. The specialist's Qwen reference instead uses the old head trained on 1,400 examples, scoring 78/100. It must not be presented as a matched training-budget comparison with the new heads. The cascade also inherits that larger supervised fallback.

The MLP REVIEW policy resolves 64 messages, getting 56 correct (87.5%) and leaving 36 unresolved, including 22 toxic messages. Its 95% covered-accuracy development target did not carry over. The linear policy resolves nothing. Neither is a tested human review service or successful automatic fallback.

Zero additional toxic misses on 50 toxic examples is weak evidence. For scale only, even an independently drawn sample with zero errors in 50 trials has a one-sided 95% binomial upper bound of about 5.82%. The present reused, selected sample does not earn that prospective guarantee. A lack of observed added errors must not become a claim of equal or safe quality.

The [MLP runtime records](../results/chat-smoke-heads/runtime-records.json) strengthen the mechanism evidence. All 200 actual-forward decisions agree with the cached predictions and depths. The learned policy stops 32 messages after block 6, 11 after block 12, 17 after block 18, and carries 40 through block 24. Unlike the earlier unconditional block-12 shortcut, this policy uses different depths for different inputs. The inspected hook raises immediately at the accepting block, before later blocks execute.

The full path sums to 63.5638 seconds for 100 requests; the learned path sums to 42.0566 seconds, a 33.8356% reduction. These sums include original input tokenization, readout, learned gate and hook overhead. They exclude model loading and training. The two paths alternate first/second position across messages, with one warmup for each. The orchestrating agent reports that other launched model jobs were paused; the records cannot independently attest machine-wide isolation. One paired pass on one CPU supplies no repeated-run uncertainty or deployment guarantee. Runtime success does not turn the reused quality sample into independent validation or erase its new false block.

The [specialist timing records](../results/chat-smoke-specialist/timings.json) also verify actual work rather than cached fallback answers. For 100 messages, the specialist alone sums to 0.3098 seconds, the cascade to 32.4581 seconds and the old full-depth Qwen reference to 64.7527 seconds. The cascade's 49.8738% reduction includes 100 actual two-block BERT calls and 49 actual 24-block Qwen calls. Every decision and fallback matches the saved quality result. All three paths rotate position across messages, with one warmup each, in one timed pass. These request-time sums exclude loading/training and have no repeated-run uncertainty. The standalone specialist's large speed difference accompanies lower quality; the cascade still introduces a toxic miss. Its outer redirected PowerShell session likewise did not report zero, so these conclusions rest on completed code/trace/artifact checks rather than an invented successful shell status.

The specialist's [source snapshot and artifact hashes](../results/chat-smoke-specialist/runtime-provenance.json) all match. They were recorded after the run, following the audit's provenance concern. They are explicitly not a prospective seal or the exact original training implementation; that reproducibility limitation remains visible.

## Methodological limits that remain

1. **Evaluation has already influenced the research.** The 100 messages come from the old 600-message study. The common split named `calibration` is development data used to select gates, not a new acceptance set. Choosing a winner from this table is further development selection. No new validation sample was consumed.
2. **The prevalence is artificial.** Balanced accuracy here is useful for comparing error types. Overall accuracy, fallback coverage and expected latency on these 50/50 samples do not estimate a natural chat stream. The old 600-message accuracy cannot be compared directly with this table.
3. **The reference moderator is weak.** Preserving a reference with 15–18 missed toxic messages out of 50 would still be inadequate evidence of practical moderation quality. Relative non-inferiority and absolute quality are separate requirements.
4. **Scores are not certified confidence.** Heads use class weighting; temperatures and learned gates reuse the same 128 development examples. A gate's logistic output estimates an error target but has not demonstrated reliable calibration under new domains or rules. The benefit gate estimates whether full depth would correct the current error, not whether a generic answer is ready.
5. **Most current gate savings are arithmetic.** Only the MLP learned gate now has an actual stopped-forward parity and timing run. Other gates still use cached depth projections; their block savings cannot be called measured latency improvements. The MLP run establishes a mechanism and single-pass timing result, not repeated hardware performance or accepted moderation quality.
6. **The specialist also misses its selection constraints elsewhere.** It selects epoch, label threshold and cascade thresholds on the same tune set. On the separate development split it adds three errors, including two new toxic misses. Its selected policy is retained rather than silently repaired on evaluation. Both models must reside in memory for the proposed warm cascade; cold start and memory cost remain relevant.
7. **This is one small training recipe, not a method ceiling.** No repeated-seed uncertainty or broad hyperparameter study supports an architecture ranking. More methods and more threshold candidates increase the chance that a small winning result is noise.

## Adapter methods: prospective code review

The reviewed implementation freezes the original Qwen weights and trains rank-4 adapters on q/v projections plus readout heads. It uses fresh forwards after changing weights, rather than stale original representations. The selected full-model teacher's training logits remain aligned to training IDs; evaluation labels are not teacher inputs. Intermediate-loss and distillation variants supervise multiple readouts. The fixed-12 variant physically retains 12 transformer modules and deliberately reads raw block-12 output to match its initial head's normalization convention.

The saved [budget amendment](../results/chat-smoke-adaptation/budget-amendment.json) preserves the abandoned 96-step implementation/protocol and records that it was stopped after eight steps, before candidate evaluation. The replacement recipe uses 32 steps of four messages, checkpoint selection on 32 development examples at steps 16/32, a 64-message development report and the unchanged 100-message evaluation. Initial heads still fit all 384 training examples; adapter updates see only 128 message presentations. This distinction must remain visible.

No blocking gradient or label-leakage error was found in this inspection. Step-two assertions check nonzero A/B gradients for the first q and v adapters and no gradients in frozen base parameters. They do not independently prove every adapter received a useful update. The full-only variant does not retune intermediate heads, so its intermediate scores are diagnostic probes on altered representations. Joint-head accuracy alone is not evidence of an operational early-exit policy.

The initial heads nearly fit the small training set before adapter updates begin: the independent reconstruction gets 384/384 training labels correct at blocks 6, 12 and 24, and 383/384 at block 18. This may leave little supervised loss to improve the backbone; it does not establish the cause of later degradation. Checkpoint selection excludes step zero. A negative outcome would therefore criticize this short-budget warm-start recipe, not establish that LoRA, joint losses or distillation fail. A future controlled comparison could use a newly initialized or more strongly regularized readout, more training data and a separately specified optimization budget. That should be a new protocol, not a post-result repair of this one.

To provide the missing matched initialization comparison, I independently reconstructed the exact seed-73, 200-step initial readouts from cached frozen features at one CPU thread, without changing the candidate models or selection. The [saved baseline](../results/chat-smoke-adaptation/initial-frozen-baseline.json) records source/cache/code hashes, all decisions and [portable weights](../results/chat-smoke-adaptation/initial-frozen-heads.npz); its [script](../experiments/chat_smoke_adaptation_baseline.py) reproduces the calculation. Initial evaluation accuracy at blocks 6/12/18/24 is 69/71/77/77 out of 100. This is distinct from the other head experiment's seed-109, 300-step linear control.

All four original variants' initial tune metrics exactly match this reconstruction. Their split IDs, source labels, per-split metrics, checkpoint selection, reported gradient checks and retained module counts are consistent. Current source and the archived implementation match all four protocol hashes; the initial abandoned source also matches its preserved protocol hash. These checks concern artifacts and code. The earlier outer PowerShell sessions reported a nonzero status in the presence of redirected warnings; I do not replace those statuses with a claim of observed zero exit codes. The orchestrating agent reports explicit Python exit zero for the later distillation and short-warmup repair launches.

| Adaptation recipe | Selected step | Correct / 100 | Matched initial head | Toxic misses / 50 | New errors vs initial | Corrected errors | New toxic misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-depth task loss | 16 | 73 | 77 at block 24 | 19 | 5 | 1 | 4 |
| Joint intermediate losses | 16 | 77 | 77 at block 24 | 17 | 1 | 1 | 1 |
| Joint losses plus teacher distillation | 16 | 80 | 77 at block 24 | 14 | 0 | 3 | 0 |
| Physically retained 12 blocks | 16 | 66 | 71 at block 12 | 22 | 6 | 1 | 5 |

Equal overall accuracy for joint training hides a newly missed toxic message and a different corrected error. Full-depth and fixed-12 training reduce accuracy in this recipe. The selected joint model's intermediate heads reach 69/68/76 correct at blocks 6/12/18, versus matched initial scores 69/71/77.

Distillation is the strongest full-depth adapter result, correcting three initial errors without introducing another on these 100 messages. That is a candidate finding, not a validated improvement. Its intermediate results are mixed: block 6 gets 67 correct, block 12 only 61 (nine newly missed toxic messages), and block 18 gets 78 (one newly missed toxic message). Full-depth improvement is not evidence that its early layers are ready to stop. The student also uses a separately trained teacher and thus additional training compute.

The original embedded adapter timing path excludes original text tokenization and computes all active readouts. It is not directly comparable to a specialist timing path that includes text tokenization and only its needed output head. The separately reviewed [runtime runner](../experiments/chat_smoke_adaptation_runtime.py) corrects that boundary: it includes original text processing, runs one target head and checks actual block traces and stored-prediction parity. It seals the first 50 evaluation IDs before timing. Its single, separately executed passes remain diagnostics without a replicated speed estimate. Distillation also uses an additional trained teacher, so equal student update counts do not mean equal total training compute.

The completed replay runs all four original adapter variants on exactly those 50 IDs (24 toxic and 26 benign). I verified every record's label, prediction, contiguous block trace and duration, plus the linked protocol, source, weight and result hashes.

| Adapter path | Actually retained/executed blocks | Correct / 50 | Sum of request seconds |
| --- | ---: | ---: | ---: |
| Full task adaptation | 24 | 36 | 33.7697 |
| Joint intermediate losses | 24 | 37 | 34.2112 |
| Joint losses plus distillation | 24 | 39 | 33.6895 |
| Fixed-12 adaptation | 12 | 31 | 16.7318 |

This confirms the fixed-12 mechanism and a substantial observed time difference, accompanied by lower decision quality. The small differences among the three 24-block timings have no reliable ranking interpretation from these sequential single passes. Do not compare these 50-message totals with the other experiments' 100-message totals or claim that distillation itself accelerates inference; its evaluated path still executes all 24 blocks.

A subsequent [recorded repair](../results/chat-smoke-reset/rationale.json) reduces head initialization from 200 to five optimizer steps before joint adapter training. It was proposed after seeing the original failures and near-perfect training fit, so it is explicitly post-observation exploration. I reconstructed its five-step initial heads separately, preserving the original baseline and its source. The [new baseline](../results/chat-smoke-reset/initial-frozen-baseline.json) scores 66/71/79/78 at blocks 6/12/18/24, with 12/11/7/8 toxic misses.

The repair fails against that matched baseline. Its selected step-16 full-depth result drops from 78 to 69 correct, introduces 21 errors while correcting 12, and introduces 20 toxic misses. Total toxic misses increase from 8 to 26. Block 6 improves total correct from 66 to 70 while introducing 14 newly missed toxic messages, illustrating why accuracy alone is insufficient. I checked the repair's source/archive hashes, initial tune parity, all source labels and split metrics, and its prescribed checkpoint selection. The improvement attempt is retained as a failure; it does not establish that the initial head's low loss caused the earlier failures.

## Maze and coordinate follow-ups

The maze follow-up's counts are real within its synthetic setup: original-state training reaches 0/10 goals without a legality mask and 2/10 with one; expanded training reaches 5/10 and 7/10 respectively; breadth-first search reaches 10/10. Independently replayed actions and state labels agree with those results. The expanded unmasked policy makes only 48 legal moves out of 126 actions. The mask makes all 99 executed actions legal, but three episodes still fail through legal loops.

The learned maze policy reads structured wall/agent/goal arrays. Its optional mask uses visible walls and boundaries, not shortest-path answers. Breadth-first search supplies training labels and the explicit oracle control, not hidden inference inputs. Expanded training enumerates 4,992 agent/goal pairs from the original 32 training layouts. Wall layouts remain disjoint, but evaluation episodes were previously inspected. This is useful evidence for task-specific representation, supervision and constraints; it is not a controlled proof that an LLM learned navigation or that early exit improved. The new training quantity and representation differ from the historical ASCII-Qwen experiment.

The coordinate diagnostic selects each confidence threshold from the other screenshot, as documented. It rejects both absent targets and keeps all six present targets, but accepts one wrong coordinate. There are only two screenshots of the same authored UI, and the same requests recur across them. Leaving out a screenshot does not leave out the task. Peak patch confidence is neither calibrated target-presence probability nor coordinate correctness. The 30 saved ScreenSpot cases all have present targets, so they cannot validate absent-target rejection.

## What would change this verdict

Freeze a small number of candidates and their input/timing boundaries before a fresh evaluation. Measure actual stopped execution for the gate, all fallback work for the cascade, and complete unresolved handling for REVIEW. Evaluate absolute toxic recall and false-block rates as well as paired new harms, with enough new positive examples to support the desired margin. Include a stronger reference and meaningful distribution shifts. Preserve failures and report uncertainty rather than promoting whichever exploratory row wins by one or two messages.

Evidence: [common manifest](../results/chat-smoke/protocol.json), [heads implementation](../experiments/chat_smoke_heads.py), [head results](../results/chat-smoke-heads/result.json), [head selection sweep](../results/chat-smoke-heads/selection-sweeps.json), [specialist implementation](../experiments/chat_smoke_specialist.py), [specialist results](../results/chat-smoke-specialist/result.json), [adapter implementation](../experiments/chat_smoke_adaptation.py), [maze results](../results/maze-smoke/result.json), [coordinate diagnostic](../results/coordinate-abstention-smoke/result.json).

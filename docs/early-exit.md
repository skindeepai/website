# What is actually stopping inside Qwen?

The first pilot uses the frozen Qwen2.5-0.5B-Instruct checkpoint `7ae557604adf67be50417f59c2c2f167def9a775`. Its configuration has 24 transformer blocks and a hidden width of 896.

The complete input message and rule are tokenized first. Every executed block processes the input sequence. At checkpoints 6, 12 and 18, a hook reads the 896 numbers at the final prompt position. These numbers are the model's internal representation, not a sentence, generated answer, or explanation of its thoughts. A separate readout after the full model uses the final RMS-normalized representation at layer 24.

## The classifier

Each checkpoint has a separately trained multinomial logistic regression: standardize with training-only mean and standard deviation, multiply by a weight matrix, add a bias, and apply softmax. There is no hidden layer inside this classifier. The backbone weights are frozen.

In the original three-label task, each classifier has 896 × 3 + 3 = **2,691 trained parameters**. The outputs represent ALLOW, BLOCK and IGNORE. A temperature rescales the scores, and the largest score is compared with a threshold. The classifier returns a numeric label; an application can map it to an enum. No text token is decoded.

This predicts a task answer. It does **not** directly observe whether Qwen has “finished thinking.” Confidence is an estimate from a learned readout, and can be wrong. An uncertainty gate answers “should we keep computing?”; it is separate from task categories such as ALLOW or IGNORE. “Too hard” is not established merely because an early head is uncertain.

Avoiding text means avoiding repeated model steps that emit tokens. Converting a token ID into a character is not the expensive operation. A fair text baseline can emit a single constrained enum token; that is much closer to a classifier than a long JSON response. The custom head also avoids projecting the internal vector into Qwen's entire vocabulary.

## The first stopping rule was too easy to pass

The training set had 80 generated examples; 32 separate generated examples were used for both temperature and threshold selection. The rule grammar was fixed, with held-out topic words. The 48 test examples were variations of that same grammar, including a short previous-message reference. This is a controlled integration test, not representative customer language or arbitrary policy understanding.

For each checkpoint the script selected the lowest tested threshold accepting at least eight calibration examples with at least 95% observed accuracy. All selected thresholds were **0.5**, with temperature **0.5**. Every test query passed the first gate after block 6.

That means 6 blocks executed and 18 did not: **75% of transformer blocks skipped**. It does not mean 75% of parameters, memory, energy, input processing, or all system work disappeared. Token embeddings and the first six blocks still run; the full checkpoint stays in memory.

The timed execution installs block hooks and raises a local stop signal as soon as an accepted head returns. A counter asserts that exactly the selected number of blocks executed. This is actual early termination. Extracting all intermediate representations after a full pass is used to train the heads, but is not counted as early-exit inference.

On this CPU the original warm model-only medians were 309 ms at full depth and 80 ms with early exit. Accuracy fell from 47/48 to 44/48. These timings excluded tokenization and loading. They demonstrate a working shortcut with a quality cost, not a quality-preserving speedup.

## Harder validation

The next experiment uses public BANKING77 queries with 77 intents and all 3,080 official test examples. Each linear head now has **69,069 parameters**. Training, policy selection, and independent calibration are separate. It compares confidence-only exits with agreement between successive heads and records accuracy, harmful exits, layer counts, and actual runtime.

The calibration guard can reject the selected shortcut; the guarded policy then uses full depth. A public banking benchmark still does not establish dynamic rule following, absent-category rejection, or contamination-free evaluation. These are separate follow-ups.

## What the browser comparison does

The optional browser benchmark loads pinned four-bit Qwen weights and runs actual inference in a worker. It compares ordinary one-letter generation with JSON generation, retaining every output and failure. It does not yet attach our trained heads or skip transformer layers. The existing browser ONNX graph includes the full decoder; implementing true early exits requires exported intermediate-output graphs and parity checks for their quantized heads.

The first browser check used three authored queries twice. One-letter output was correct on two of the three distinct queries; JSON on one. Its CPU timings overlapped other local experiments. The raw record is useful for checking that the browser path works, not for an equal-quality speed claim. [Recorded run](../results/ui/browser-qwen.json).

## Next experiments

1. **Changing rules:** real held-out customer queries paired with independently varied instruction-to-label mappings. Keep the same query and change the rule; check that the answer changes accordingly. Hold out both queries and rule templates. This tests the original dynamic-decision idea rather than a fixed intent specialist.
2. **Better exits:** compare a classifier trained to predict early-head error with confidence-only and agreement gates. Train the error predictor on a separate development split, calibrate once, and retain a fresh evaluation set. Report risk versus coverage, including unfamiliar and absent-category inputs.
3. **Cheaper alternatives:** compare the selected fixed-depth head, a small dedicated text encoder, and the lexical control. A complex gate is only useful when it improves the cost/quality trade-off over those simpler options.
4. **Browser parity:** export fixed-depth graphs first, check labels against float32 on a frozen test slice, and measure download size, compilation, peak memory and warm inference separately. Only then connect checkpoints into a conditional runtime; running every graph from the beginning is not activation reuse.
5. **Coordinates:** validate absence rejection and grounding before adding intermediate pointer heads. Image encoding may dominate: in the original coordinate run, the vision encoder accounted for a median 77% of the direct path's elapsed time. Skipping language layers cannot remove that cost. Resolution and small-target accuracy need paired tests.

## Reproduce and inspect

- [Original implementation](../experiments/qwen_decisions.py), [exact generated fixtures](../results/qwen-decisions/fixtures.json), [per-query predictions](../results/qwen-decisions/predictions.json), [run settings and timings](../results/qwen-decisions/result.json).
- [BANKING77 protocol](banking77-protocol.md) and [implementation](../experiments/banking77.py).
- [Pinned Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json).

Training and inference are CPU float32 in the recorded environment. Neither pilot establishes a fair advantage over an equally trained text-output baseline. The zero-shot three-code baseline in the first pilot received no task training while the heads did.

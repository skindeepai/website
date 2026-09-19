# Changing the rule for the same request

This stress test pairs each held-out public utterance with two opposite A/B rules. Ignoring the rule cannot get both answers right. It uses 20 known intents across ten CLINC domains, 100 held-out utterances and 200 test prompts. Training used different utterances and an authored rule template; testing changes that template.

**This is public language paired with authored rules, not production traffic or arbitrary instruction following.** Intent labels and pairs are shared across splits.

| Readout layer | Correct prompts | Both opposite rules correct | Output changed | 90% paired criterion met |
| --- | --- | --- | --- | --- |
| 6 | 100/200 | 0/100 | 0/100 | False |
| 12 | 117/200 | 20/100 | 23/100 | False |
| 18 | 100/200 | 0/100 | 0/100 | False |
| 24 | 107/200 | 14/100 | 21/100 | False |

A changed output can still be wrong in both cases, so the both-correct count is the useful measure. The 90% criterion was recorded before inference. No stopping gate was trained or accepted in this test, and no latency claim is made. Tuning features were extracted but not used to select models or policies.

The task labels are derived from the dataset categories and authored rules; the category-to-natural-language descriptions have not received independent human adjudication. This limits the interpretation of failures and successes.

- [Prospective protocol and every prompt](../results/changing-rules/protocol.json).
- [Results](../results/changing-rules/result.json), [predictions](../results/changing-rules/predictions.json), [weights](../results/changing-rules/heads.npz), [code](../experiments/changing_rules.py).
- Source utterances: [CLINC / Larson and colleagues](https://github.com/clinc/oos-eval), CC BY 3.0. Prompts add authored rules; the original data is unchanged.

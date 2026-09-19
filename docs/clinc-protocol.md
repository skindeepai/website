# Second-dataset stopping test

Recorded locally before inference on 2026-09-19. The machine-readable protocol is [sealed here](../results/clinc-validation/protocol.json); the script refuses to silently overwrite a changed protocol. This is not external preregistration.

## Question

Does the early-exit mechanism transfer to a second classification benchmark, and does confidence avoid early answers on unsupported requests?

We use [CLINC150 / OOS-eval](https://github.com/clinc/oos-eval), from Larson and colleagues' [2019 paper](https://aclanthology.org/D19-1131/). Its utterances were crowdsourced; they are not production traffic. The pinned dataset revision is `828f8093932c8fe6ca7936c3d2e52903b1c523de`, licensed CC BY 3.0. Original data stays in the ignored cache; public manifests contain source IDs, hashes and attribution.

## Predetermined scope

- First three alphabetical intents from each of ten domains: 30 supported intents. This is explicitly a subset, not a full CLINC150 benchmark.
- Per intent: 32 official training examples for the head and ten different training examples for tuning. All 20 official validation examples are reserved for independent calibration; all 30 official test examples for evaluation.
- Official out-of-scope training/validation/test splits provide 100/100/1,000 examples for tuning/calibration/testing.
- An additional stress test uses one predetermined official test example from each of the 120 unsupported intents.
- Exact normalized-text duplicates are removed in partition order before inference. Retained IDs and removals are recorded. This does not establish paraphrase or pretraining independence.

## Model and decisions

Frozen Qwen2.5-0.5B-Instruct, the same pinned revision as BANKING77. New 30-way linear classifiers after blocks 6, 12, 18 and 24; 200 fixed training steps and one initialization seed. Train-only standardization; temperature selected on tuning data. This tests whether the method can be retrained on another dataset, not zero-shot transfer of banking weights.

On tuning data, choose the cheapest candidate with no more than 0.5 percentage points net accuracy loss, 5% wrong answers among early exits, 5% early acceptance of out-of-scope examples, and at least 5% in-scope early coverage. Candidate choices are fixed in the JSON protocol. No candidate means full-depth fallback.

Before looking at calibration/test outcomes, freeze the chosen rule. Independently require:

1. A one-sided exact 95% Clopper-Pearson upper bound of at most 1% on new errors relative to full depth, divided by all in-scope queries.
2. An upper bound of at most 5% on wrong answers divided by early exits.
3. An upper bound of at most 5% on early acceptance of out-of-scope inputs.
4. At least 5% observed early coverage on in-scope inputs.

These are individual bounds, not a simultaneous 95% guarantee, and describe this sampled distribution. A failed check rejects the gate. The 5% absolute-error criterion is a new, explicit research criterion; it must not be confused with the previous 1% additional-error criterion.

Unknown requests that continue to layer 24 are **not counted as successfully rejected**. These heads have no unknown class, so the final result remains unresolved. This experiment measures whether an early shortcut makes premature decisions; it does not implement complete unfamiliar-input handling.

## Controls and reporting

Compare all fixed-depth classifiers, the full-depth classifier and TF-IDF logistic regression trained on the identical examples. Publish all evaluated predictions, exit counts, wrong early answers, added errors, unknown acceptance, guard outcomes and portable classifier weights. Full feature extraction is not a timing benchmark; no new runtime saving is inferred from cached features.

Code: [clinc_validation.py](../experiments/clinc_validation.py). Run `--prepare` to record the protocol, then run without that option. No results may silently revise this protocol or replace failed candidates with test-selected alternatives.

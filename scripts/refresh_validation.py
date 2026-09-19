"""Render the bounded follow-up evidence without changing the approved UI."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def save(name,lines):(ROOT/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    if (ROOT/'results/clinc-validation/result.json').exists():
        r=read('results/clinc-validation/result.json');p=read('results/clinc-validation/protocol.json');test=r['splits']['test'];cal=r['splits']['calibration']
        lines=['# Second-dataset validation: CLINC subset','',
               '**Scope: 30 supported intents across ten domains, not the complete 150-intent benchmark.** New heads were trained on this dataset; the BANKING77 heads were not transferred. Public utterances are crowdsourced, not production traffic.','',
               f'Independent calibration accepted the selected gate: **{r["guard_passed"]}**. A rejected gate falls back to full depth; its candidate savings are not accepted deployment savings.','',
               '## Known-intent results','',
               '| Split | Full-depth correct | Candidate correct | Early exits | Wrong early answers | Added errors | Projected blocks skipped |',
               '| --- | --- | --- | --- | --- | --- | --- |']
        for name in ['calibration','test']:
            s=r['splits'][name];lines.append(f'| {name} | {s["full_correct"]}/{s["n"]} | {s["correct"]}/{s["n"]} | {s["early_count"]} | {s["early_wrong"]} | {s["harmful"]} | {100*s["blocks_skipped"]:.2f}% |')
        lines+=['','The projected depths above come from full-depth feature extraction and applying the stopping policy to intermediate readouts. They are not new wall-clock or actual stopped-execution measurements. The separate BANKING77 replay verifies that execution mechanism on the first task.','',
                '## Calibration checks','',
                f'- Added-error upper bound, divided by all known-intent queries: **{100*cal["harm_upper95"]:.3f}%**, limit 1%.',
                f'- Wrong-answer upper bound among early exits: **{100*cal["early_error_upper95"]:.3f}%**, limit 5%.',
                f'- Early acceptance of unfamiliar inputs, upper bound: **{100*r["splits"]["oos_calibration"]["acceptance_upper95"]:.3f}%**, limit 5%.',
                f'- Observed known-intent early coverage: **{100*cal["early_coverage"]:.2f}%**, minimum 5%.','',
                'Exact one-sided 95% Clopper-Pearson bounds are computed separately. They are not a simultaneous 95% guarantee or a guarantee under distribution shift.','',
                '## Unknown requests','',
                '| Set | Queries | Premature early answers | Continued to full depth |','| --- | --- | --- | --- |']
        for name in ['oos_calibration','oos_test','unsupported_test']:
            s=r['splits'][name];lines.append(f'| {name} | {s["n"]} | {s["early_accepted"]} | {s["continued"]} |')
        lines+=['','Continuing does **not** mean a request was correctly rejected. This classifier has no unknown class. These inputs remain unresolved at full depth; the table measures premature early decisions only.','',
                '## Controls','', '| Depth | Correct test answers |','| --- | --- |']
        for d,s in r['fixed_heads'].items():lines.append(f'| {d} | {s["test_correct"]}/{test["n"]} |')
        lines += [f'| TF-IDF logistic regression | {r["lexical"]["correct"]}/{r["lexical"]["n"]} |','',
                  'The lexical control receives identical training examples. No matched latency comparison was run here. One head seed, alphabetically selected classes, unknown pretraining overlap and a limited input distribution constrain generalization.','',
                  '## Inspect and reproduce','',
                  '- [Protocol written before inference](clinc-protocol.md), [sealed IDs and settings](../results/clinc-validation/protocol.json).',
                  '- [Result JSON](../results/clinc-validation/result.json), [every evaluated prediction](../results/clinc-validation/predictions.json), [portable heads](../results/clinc-validation/heads.npz).',
                  '- [Implementation](../experiments/clinc_validation.py), [frozen selected policy](../results/clinc-validation/selected-policy.json).',
                  '- [CLINC dataset and attribution](https://github.com/clinc/oos-eval), [original paper](https://aclanthology.org/D19-1131/).']
        save('docs/clinc-results.md',lines)
    if (ROOT/'results/changing-rules/result.json').exists():
        r=read('results/changing-rules/result.json')
        lines=['# Changing the rule for the same request','',
               'This stress test pairs each held-out public utterance with two opposite A/B rules. Ignoring the rule cannot get both answers right. It uses 20 known intents across ten CLINC domains, 100 held-out utterances and 200 test prompts. Training used different utterances and an authored rule template; testing changes that template.','',
               '**This is public language paired with authored rules, not production traffic or arbitrary instruction following.** Intent labels and pairs are shared across splits.','',
               '| Readout layer | Correct prompts | Both opposite rules correct | Output changed | 90% paired criterion met |','| --- | --- | --- | --- | --- |']
        for d,s in r['depths'].items():lines.append(f'| {d} | {s["correct"]}/{s["n"]} | {s["both_correct"]}/{s["pairs"]} | {s["changed_prediction"]}/{s["pairs"]} | {s["criterion_passed"]} |')
        lines+=['','A changed output can still be wrong in both cases, so the both-correct count is the useful measure. The 90% criterion was recorded before inference. No stopping gate was trained or accepted in this test, and no latency claim is made. Tuning features were extracted but not used to select models or policies.','',
                'The task labels are derived from the dataset categories and authored rules; the category-to-natural-language descriptions have not received independent human adjudication. This limits the interpretation of failures and successes.','',
                '- [Prospective protocol and every prompt](../results/changing-rules/protocol.json).',
                '- [Results](../results/changing-rules/result.json), [predictions](../results/changing-rules/predictions.json), [weights](../results/changing-rules/heads.npz), [code](../experiments/changing_rules.py).',
                '- Source utterances: [CLINC / Larson and colleagues](https://github.com/clinc/oos-eval), CC BY 3.0. Prompts add authored rules; the original data is unchanged.']
        save('docs/changing-rules.md',lines)
    if (ROOT/'results/clinc-unknown/result.json').exists():
        r=read('results/clinc-unknown/result.json')
        lines=['# Adding an unsupported-request output','',
               'This prospective variant adds a 31st output, UNKNOWN, to the 30 supported CLINC intents. It trains on 80 out-of-scope training queries and uses the remaining 20 for tuning, alongside the original in-scope partitions. It reuses frozen Qwen features; no backbone weights change.','',
               f'Independent calibration accepted this gate: **{r["guard_passed"]}**. Failure means full-depth fallback, not an accepted early-exit saving.','',
               '| Set | Full-depth correct | Candidate correct | Early exits | Wrong early answers | Early UNKNOWN | Early known category |','| --- | --- | --- | --- | --- | --- | --- |']
        for name,s in r['splits'].items():lines.append(f'| {name} | {s["full_correct"]}/{s["n"]} | {s["candidate_correct"]}/{s["n"]} | {s["early_count"]} | {s["early_wrong"]} | {s["early_unknown"]} | {s["early_known"]} |')
        lines+=['','For out-of-scope and unsupported-intent inputs, UNKNOWN is the correct dataset-derived label. For supported inputs, UNKNOWN is an error. It means outside this classifier\'s supported categories, not "too hard for the full model."','',
                '## Independent calibration bounds','', '| Check | Exact one-sided 95% upper bound | Passed |','| --- | --- | --- |']
        for name,value in r['upper95'].items():lines.append(f'| {name} | {100*value:.3f}% | {r["guard_checks"][name]} |')
        lines+=['','Limits are 1% added errors per all calibration requests, 5% wrong answers per early exit, 5% unknown inputs prematurely routed to known categories, and 5% known inputs prematurely rejected. Require at least 5% early coverage. These are individual bounds, not a joint 95% guarantee. The combined calibration counts reflect this experiment\'s 600:100 known/unknown mixture; deployment prevalence may differ.','',
                'The variant protocol was saved before the 30-output experiment wrote its results. It changes the output categories, adds OOS supervision and uses another seed, so differences cannot be attributed solely to one extra output. It shares benchmark partitions with the first variant, not an independent dataset replication. Calibration has 600 known and 100 OOS queries; testing has 900 known and 1,000 OOS queries. Pooled rates cannot be transferred across those prevalences. The guard concerns early decisions, not full-depth rejection reliability. Small OOS training/tuning sets and unknown pretraining exposure remain limits. Features were collected at full depth; projected exits are not measured runtime savings here.','',
                '- [Prospective protocol](../results/clinc-unknown/protocol.json), [frozen policy](../results/clinc-unknown/selected-policy.json), [results](../results/clinc-unknown/result.json).',
                '- [Every prediction](../results/clinc-unknown/predictions.json), [portable weights](../results/clinc-unknown/heads.npz), [implementation](../experiments/clinc_unknown.py).',
                '- [Dataset and attribution](https://github.com/clinc/oos-eval), CC BY 3.0.']
        save('docs/clinc-unknown.md',lines)
    if (ROOT/'results/matched-output/result.json').exists():
        r=read('results/matched-output/result.json');a=r['paired_token_minus_enum_ms']
        lines=['# A matched one-token output control','',
               'The numeric enum and constrained text paths share the same frozen Qwen representations and the same supervised output rows. Each of the 77 labels maps to a single Qwen token. The text path emits that token and converts it to text. Neither path stops early.','',
               f'All **{r["queries"]}** saved examples produced identical decisions, with **{r["correct"]}** correct. This equality follows from the shared classifier and bijective token mapping; it is an invariant check, not new evidence of equal independently trained model quality.','',
               'The constrained readout computes only its 77 supported token scores. It does not run an untouched full-vocabulary language decoder or the ordinary generation API. This is an efficient custom one-token baseline, not an independently fine-tuned conversational model.','',
               '## Actual warm CPU timing','',
               '| Output | Mean input-to-result | Median |','| --- | --- | --- |']
        for name,s in r['timing'].items():lines.append(f'| {name} | {s["mean"]:.2f} ms | {s["p50"]:.2f} ms |')
        lines+=['',f'Timing uses {r["timing_queries"]} predetermined queries, two alternating repeats per path, eight CPU threads and an otherwise idle launched-model workload. Includes tokenization, all 24 blocks, trained readout and token-to-text conversion where applicable. Loading is excluded.',
                f'Text minus enum mean time: **{a["mean"]:.2f} ms**; paired-query bootstrap 95% interval **{a["query_bootstrap_ci95"][0]:.2f} to {a["query_bootstrap_ci95"][1]:.2f} ms**. This describes one run, not fresh sessions or different hardware.','',
                'A single trained output token can implement the same decision. Avoiding text alone does not establish a useful speedup. Skipping transformer blocks is a separate optimization.','',
                '- [Implementation](../experiments/matched_output.py), [timing protocol](../results/matched-output/protocol.json), [results](../results/matched-output/result.json).',
                '- [Every equivalent output](../results/matched-output/predictions.json), [all actual timings](../results/matched-output/timings.json).']
        save('docs/matched-output.md',lines)
    print('Refreshed completed validation references.')

if __name__=='__main__':main()

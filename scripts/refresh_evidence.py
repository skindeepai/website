"""Write readable research references from completed, independently recorded runs."""
import csv, json, statistics, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def percent(value):return f'{100*value:.2f}%'
def paired_timing(directory):
    rows=read(directory+'/timings.json');groups={}
    for r in rows:groups.setdefault(r['id'],{}).setdefault(r['path'],[]).append(r['end_to_end_ms'])
    pairs=[(statistics.mean(v['full_head']),statistics.mean(v['candidate_early_exit'])) for v in groups.values()]
    def saving(values):return 1-sum(v[1] for v in values)/sum(v[0] for v in values)
    rng=random.Random(19);bootstrap=sorted(saving(rng.choices(pairs,k=len(pairs))) for _ in range(2000))
    result={'queries':len(pairs),'mean_latency_reduction':saving(pairs),'paired_bootstrap_ci95':[bootstrap[49],bootstrap[1949]],
            'scope':'Resamples query pairs, averaging each query’s two repeats first. Describes this warm CPU run, not other machines or production traffic.'}
    (ROOT/directory/'timing-analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

def main():
    q=read('results/banking77/result.json');m=read('results/banking77/data-manifest.json');p=read('results/banking77/predictions.json')
    lexical=read('results/banking77/lexical-baseline.json')
    rows={}
    for split in ['train','test']:
        with (ROOT/f'experiments/.cache/banking77/{split}.csv').open(encoding='utf-8',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):rows[f'{split}:{i}']=row['text'].strip().casefold()
    exposed={rows[i] for s in ['train','tune','calibration'] for i in m['splits'][s]}
    duplicate_ids={i for i in m['splits']['test'] if rows[i] in exposed}
    sensitivity=[]
    for run in q['runs']:
        clean=[r for r in p if r['seed']==run['seed'] and r['id'] not in duplicate_ids]
        sensitivity.append({'seed':run['seed'],'n':len(clean),'full_correct':sum(r['full_prediction']==r['label'] for r in clean),
                            'candidate_correct':sum(r['candidate_prediction']==r['label'] for r in clean)})
    (ROOT/'results/banking77/duplicate-sensitivity.json').write_text(json.dumps({'excluded_ids':sorted(duplicate_ids),'runs':sensitivity},indent=2)+'\n')
    lines=['# Qwen early exits on real banking queries','',
           'BANKING77, 77 intents, all 3,080 official test queries. The backbone is frozen; only the small linear classifiers are trained. This is a fixed-intent task, not arbitrary instruction following.','',
           '## Data and methods','',
           f'Training: {q["split_sizes"]["train"]} queries. Tuning: {q["split_sizes"]["tune"]}. Independent calibration: {q["split_sizes"]["calibration"]}. Test: {q["split_sizes"]["test"]}. The fixed per-class slices are shorter for a few small classes. Three initialization seeds share these same examples.',
           '', 'A head reads 896 internal numbers after layer 6, 12, 18 or 24. It has 69,069 trained parameters and emits 77 scores without generating text. Policy selection uses tuning data. An independent calibration guard rejects a shortcut if the 95% upper bound on harmful exits exceeds 1%. A rejected gate deploys the full model.', '',
           '## Fixed-depth classifiers','', '| Seed | Layer 6 | Layer 12 | Layer 18 | Layer 24 | Fixed layer chosen on tuning |', '| --- | --- | --- | --- | --- | --- |']
    for r in q['runs']:lines.append('| '+str(r['seed'])+' | '+' | '.join(percent(h['test_accuracy']) for h in r['heads'])+' | '+str(r.get('fixed_depth_selected_on_tune','not recorded'))+' |')
    lines+=['',f'The simple TF-IDF word/phrase classifier scored **{lexical["correct"]}/3,080 ({percent(lexical["accuracy"])})** with the same training IDs. Its configuration was fixed before the Qwen test results. This matters: an early-exit transformer still needs to justify its cost against a small specialist.','',
            '## Selected early-exit policies','', '| Seed | Confidence threshold | Require agreement? | Guard passed? | Candidate correct | Mean layers | Harmful / corrected errors |', '| --- | --- | --- | --- | --- | --- | --- |']
    for r in q['runs']:
        c=r['candidate_test'];g=r['selected_policy']
        lines.append(f'| {r["seed"]} | {g["threshold"]} | {g["agreement"]} | {r["calibration_guard_passed"]} | {c["correct"]}/{c["count"]} ({percent(c["accuracy"])}) | {c["mean_depth"]:.2f} | {c["harmful_exits"]} / {c["corrected_full_errors"]} |')
    lines+=['','“Harmful” means the full head was correct and the early head was wrong. “Corrected” means the reverse. Candidate rows remain visible even if calibration rejected them. A passed finite-sample guard is not a production guarantee.','', '### Where requests stopped','', '| Seed | Layer 6 | Layer 12 | Layer 18 | Layer 24 | Transformer blocks skipped |','| --- | --- | --- | --- | --- | --- |']
    for r in q['runs']:
        c=r['candidate_test'];counts=c['exit_counts']
        lines.append('| '+str(r['seed'])+' | '+' | '.join(str(counts.get(str(d),0)) for d in [6,12,18,24])+' | '+percent(c['blocks_skipped_fraction'])+' |')
    lines+=['','These are skipped transformer blocks, not skipped model parameters or a measured energy reduction. The checkpoint remains in memory.','',
            '## Actual CPU execution','', '| Path | Model p50 | Including tokenization p50 | Including tokenization p95 |','| --- | --- | --- | --- |']
    for path,t in q['timing'].items():lines.append(f'| {path} | {t["model_ms"]["p50"]:.1f} ms | {t["end_to_end_ms"]["p50"]:.1f} ms | {t["end_to_end_ms"]["p95"]:.1f} ms |')
    lines+=['',f'Seed 17, {q["threads"]} CPU threads, {q["timing_rows"]} timed passes over 96 fixed queries and two alternating repeats per path. Warm model; loading excluded. The final recorded timing run occurs after the other launched model experiments finish. Each pass counts every executed block and asserts agreement with its cached prediction. The candidate is timed whether or not the calibration guard accepts it.',
            '', '## Exact-duplicate sensitivity','',f'The published dataset has {len(duplicate_ids)} test queries whose normalized text also appears in the selected development data. Their IDs are recorded; the official score retains them. Excluding them gives:', '', '| Seed | Remaining queries | Full model correct | Candidate correct |','| --- | --- | --- | --- |']
    for r in sensitivity:lines.append(f'| {r["seed"]} | {r["n"]} | {r["full_correct"]} ({percent(r["full_correct"]/r["n"])}) | {r["candidate_correct"]} ({percent(r["candidate_correct"]/r["n"])}) |')
    lines+=['','This sensitivity check does not detect paraphrase overlap or pretraining contamination. Classifier seeds are not independent dataset replications. There is no equally trained text-decoder baseline, GPU/browser early-exit measurement, or unfamiliar-category rejection test.','',
            '## Inspect and reproduce','',
            '- [Pre-run protocol](banking77-protocol.md) and [implementation](../experiments/banking77.py).',
            '- [Settings and results](../results/banking77/result.json), [all predictions](../results/banking77/predictions.json), [executed layers and timing samples](../results/banking77/timings.json).',
            '- [Data revision, hashes, categories and split IDs](../results/banking77/data-manifest.json), [duplicate check](../results/banking77/duplicate-sensitivity.json), [lexical baseline](../results/banking77/lexical-baseline.json).',
            '- [BANKING77 authors and CC-BY-4.0 source](https://github.com/PolyAI-LDN/task-specific-datasets).',
            '- Run `python experiments/banking77.py --threads 8`; feature extraction is cached locally. Run `python experiments/banking77_lexical.py` for the lexical control. See [environment details](../experiments/README.md).','']
    (ROOT/'docs/banking77-results.md').write_text('\n'.join(lines),encoding='utf-8')
    paired_timing('results/banking77')
    c=read('results/screenspot/result.json');cp=read('results/screenspot/predictions.json')
    lines=['# Direct points on public interface screenshots','',f'Tested {c["samples"]} predetermined ScreenSpot examples across Windows, macOS, iOS, Android and web interfaces. Three examples per platform/type stratum. This is a small stratified sample, not the full 1,272-example benchmark.', '',
           f'- Highest-probability patch center: **{c["hits"]["max_patch"]}/{c["samples"]} hits**.',
           f'- Connected-region weighted center: **{c["hits"]["connected_region"]}/{c["samples"]} hits**.', '',
           'The two readouts use exactly the same probability map from one model pass. The region method follows the published GUI-Actor rule; it is not a newly trained model. No text tokens or JSON are generated. All 28 language-model blocks and the vision encoder run.', '',
           '## Every result','', '| Source row | Platform | Type | Single patch hit | Region hit |','| --- | --- | --- | --- | --- |']
    for r in cp:lines.append(f'| {r["row_index"]} | {r["data_source"]} | {r["data_type"]} | {r["hits"]["max_patch"]} | {r["hits"]["connected_region"]} |')
    lines+=['','This run enforces at most 576 visual patches, checked after preprocessing. Smaller elements may disappear at that resolution. It does not test missing targets or prove a speed advantage over JSON. Recorded elapsed times are diagnostic because other CPU work overlapped. Unknown training overlap with this public benchmark remains possible.','',
            '## Evidence','', '- [Protocol](screenspot-protocol.md), [implementation](../experiments/screenspot.py), [resolution-cap correction](coordinate-correction.md).',
            '- [Pinned sample IDs, boxes and image hashes](../results/screenspot/data-manifest.json), [all points and probabilities](../results/screenspot/predictions.json), [summary](../results/screenspot/result.json).',
            '- [ScreenSpot dataset](https://huggingface.co/datasets/bevaya/ScreenSpot), [SeeClick authors](https://github.com/njucckevin/SeeClick), [Microsoft GUI-Actor](https://github.com/microsoft/GUI-Actor).',
            '- Reproduce with `python experiments/screenspot.py --threads 4`. Public screenshots download to the ignored local cache; no live clicks are executed.','']
    (ROOT/'docs/screenspot-results.md').write_text('\n'.join(lines),encoding='utf-8')
    conservative=read('results/banking77-conservative/result.json');protocol=read('results/banking77-conservative/protocol.json')
    lines=['# A more conservative stopping rule','',
           'This follow-up was designed after the first real-data candidate failed its harmful-exit guard. It reuses the seed-17 classifiers; only policy selection changes. It is explicitly an exploratory revision, checked on a fresh reserve.', '',
           f'The chosen rule starts at layer {conservative["policy"]["minimum"]} and requires a {percent(conservative["policy"]["threshold"])} score. Agreement with the preceding checkpoint: {conservative["policy"]["agreement"]}. It was selected on the original tuning queries by requiring the one-sided 95% upper bound on harmful exits to be at most 0.5%. The chosen rule was saved before any reserve inference.', '',
           f'Fresh calibration: {protocol["sizes"]["calibration"]} queries. Fresh test: {protocol["sizes"]["test"]}. Both cover {protocol["classes"]["test"]} of 77 intents; two small classes had no unused examples. All prior query IDs and exact normalized-text duplicates were excluded, including duplicates within this reserve.', '',
           '## Results','', '| Split | Full head | Early candidate | Mean layers | Blocks skipped | Harmful / corrected |','| --- | --- | --- | --- | --- | --- |']
    for split,values in conservative['results'].items():
        f=values['full'];c=values['candidate']
        lines.append(f'| {split} | {f["correct"]}/{f["count"]} ({percent(f["accuracy"])}) | {c["correct"]}/{c["count"]} ({percent(c["accuracy"])}) | {c["mean_depth"]:.2f} | {percent(c["blocks_skipped_fraction"])} | {c["harmful_exits"]} / {c["corrected_full_errors"]} |')
    cal=conservative['results']['calibration']['candidate']
    lines+=['',f'Independent calibration guard passed: **{conservative["guard_passed"]}**. Its one-sided 95% Wilson upper bound on harmful exits was **{100*cal["harm_rate_upper95"]:.3f}%**, against a 1% limit. If this guard fails, the guarded policy uses full depth. This approximate finite-sample check is not a production safety guarantee.', '',
            '### Test exit counts','', '| Layer | Requests |','| --- | --- |']
    for d in [6,12,18,24]:lines.append(f'| {d} | {conservative["results"]["test"]["candidate"]["exit_counts"].get(str(d),0)} |')
    if 'timing' in conservative:
        lines+=['','## Actual runtime','', '| Path | Mean including tokenization | Median | p95 |','| --- | --- | --- | --- |']
        for path,values in conservative['timing'].items():
            t=values['end_to_end_ms'];lines.append(f'| {path} | {t["mean"]:.1f} ms | {t["p50"]:.1f} ms | {t["p95"]:.1f} ms |')
        lines+=['','Warm CPU, first 96 fixed reserve queries, two alternating repeats per path. Other launched model experiments were stopped for this timing run. Every executed block is counted and runtime predictions must match cached ones. Loading is excluded. The mean is shown because a policy that rarely exits early may leave the median unchanged.']
        paired=paired_timing('results/banking77-conservative')
        lines+=['',f'Mean latency reduction in this paired run: **{percent(paired["mean_latency_reduction"])}**. Query-paired bootstrap 95% interval: {percent(paired["paired_bootstrap_ci95"][0])} to {percent(paired["paired_bootstrap_ci95"][1])}. This does not override a failed calibration guard. [Calculation record](../results/banking77-conservative/timing-analysis.json).']
    lines+=['','## Limits and evidence','',
            'One head seed, the same underlying public benchmark, and 75-intent reserve coverage. A fresh reserve avoids reusing the first test examples for this confirmation, but does not establish cross-domain or pretraining independence. No absent-category detection or changing-rule generalization was tested.', '',
            '- [Frozen policy and reserve IDs](../results/banking77-conservative/protocol.json), [results](../results/banking77-conservative/result.json), [every reserve prediction](../results/banking77-conservative/predictions.json).',
            '- [Actual layer traces and timings](../results/banking77-conservative/timings.json), [implementation](../experiments/banking77_conservative.py).',
            '- Run `python experiments/banking77_conservative.py --time-runtime` after the first BANKING77 run. Use an otherwise idle model-testing environment for timings.', '']
    (ROOT/'docs/conservative-exits.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote measured research references and duplicate sensitivity results.')

if __name__=='__main__':main()

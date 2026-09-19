"""Make the real-message workload results readable without changing shared styles."""
import json,random,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/chat600'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def main():
    if not (OUT/'result.json').exists():print('Chat fitting has not completed.');return
    result=read('result.json');protocol=read('protocol.json');rows=[r for r in read('predictions.json') if r['split']=='test'];s=result['splits']['test']
    integrity=read('calibration-integrity.json') if (OUT/'calibration-integrity.json').exists() else None
    accepted=result['guard_passed'] and bool(integrity and integrity['accepted'])
    pairs=[int(r['candidate']==r['label'])-int(r['full']==r['label']) for r in rows]
    rng=random.Random(23);boot=sorted(statistics.mean(rng.choices(pairs,k=len(pairs))) for _ in range(4000))
    quality={'queries':len(rows),'accuracy_delta':statistics.mean(pairs),'paired_query_bootstrap95':[boot[99],boot[3899]],
             'scope':'Describes this fixed human-annotated sample and its labels; does not include annotation uncertainty or user/session clustering.'}
    (OUT/'quality-analysis.json').write_text(json.dumps(quality,indent=2)+'\n',encoding='utf-8')
    lines=['# Six hundred real chat decisions','',
           'This is the requested practical workload: process 600 held-out real messages and return SAFE or BLOCK. Messages come from the human-annotated portion of ToxicChat0124, collected from an online chatbot. They are archived real user messages, not generated text or current Twitch chat.','',
           f'**Calibration and the added input-overlap check accepted the early-exit rule: {accepted}.** A failed gate means a guarded application uses the full-depth classifier. Candidate timing remains visible for research, not as an accepted quality-preserving speedup.','',
           '## Decision quality on the same 600 messages','',
           'There are 81 toxic and 519 benign reference labels. Always returning SAFE would score 86.5% accuracy while missing every toxic message.','',
           '| Path | Correct | Missed toxic (of 81) | Wrongly blocked benign (of 519) | Toxic recall |','| --- | --- | --- | --- | --- |']
    names=[('Untouched Qwen: constrained SAFE/BLOCK','zero_shot_lm'),('Trained full-depth readout','full'),('Trained early-exit candidate','candidate'),('Always SAFE','always_safe')]
    for title,key in names:
        m=s[key];lines.append(f'| {title} | {m["correct"]}/600 | {m["missed_toxic"]} | {m["false_block"]} | {100*m["toxic_recall"]:.1f}% |')
    m=result['lexical'];lines.append(f'| TF-IDF logistic regression | {m["correct"]}/600 | {m["missed_toxic"]} | {m["false_block"]} | {100*m["toxic_recall"]:.1f}% |')
    a=s['candidate'];lines+=['',f'The candidate adds **{a["added_errors"]}** mistakes that the full-depth classifier did not make and fixes **{a["corrected_errors"]}** full-depth mistakes. It adds **{a["additional_missed_toxic"]}** missed toxic messages. These changes must not be hidden by net accuracy.',
            f'Candidate minus full-depth accuracy: **{100*quality["accuracy_delta"]:.2f} percentage points**, paired-query bootstrap 95% interval **{100*quality["paired_query_bootstrap95"][0]:.2f} to {100*quality["paired_query_bootstrap95"][1]:.2f}**. This exploratory test-set interval does not override failed independent calibration.','',
            'The trained enum and trained SAFE/BLOCK-token paths share exactly the same output rows and predictions. The untouched Qwen control uses its original two vocabulary rows and gets no task training. It is not an equally supervised architecture comparison. No path generates a long explanation or JSON.','',
            '## Where the candidate stops','', '| Layer | Messages |','| --- | --- |']
    for d,n in a['exit_counts'].items():lines.append(f'| {d} | {n} |')
    if a['exit_counts'].get('12')==600:
        lines+=['','The selected binary confidence threshold is 0.5 with earliest exit at layer 12. A binary maximum score is always at least 0.5, so this rule reduces to a **fixed layer-12 shortcut**. It does not demonstrate input-dependent readiness or dynamic effort. That distinction is retained even if its workload time improves.']
    confident=[r for r in rows if r['heads']['12']['confidence']>=.9]
    confident_wrong=sum(r['heads']['12']['prediction']!=r['label'] for r in confident)
    lines+=['',f'At layer 12, **{len(confident)}** test messages received a confidence score of at least 0.9; **{confident_wrong}** of those predictions were wrong. A score is evidence for a gate to evaluate, not proof that the answer is ready. This is a post-run diagnostic, not a replacement threshold selected on the test set.']
    lines+=['',f'Mean depth: **{a["mean_depth"]:.2f} of 24 blocks**; **{100*a["blocks_skipped"]:.2f}%** of blocks skipped by the candidate policy. Wrong answers among early exits: **{a["early_wrong"]}/{a["early_count"]}**. This is not a memory or energy measurement.','',
            '## Independent calibration','']
    c=result['splits']['calibration'];a=c['candidate'];f=c['full']
    lines += [f'- Added-error upper bound per all calibration messages: **{100*a["added_upper95"]:.3f}%**, limit 1%.',
              f'- Additional missed-toxic upper bound per toxic calibration message: **{100*a["added_miss_upper95"]:.3f}%**, limit 5%.',
              f'- Early coverage: **{100*a["early_count"]/a["n"]:.1f}%**, minimum 5%.',
              f'- Full-depth calibration accuracy: **{100*f["accuracy"]:.1f}%**, required minimum 90%.',
              f'- Full-depth toxic recall: **{100*f["toxic_recall"]:.1f}%**, required minimum 80%.','',
              'Bounds are exact one-sided 95% Clopper-Pearson bounds, individually. They are not a simultaneous 95% guarantee, and relative added errors are not the absolute error rate.','',
              'Clipping created three calibration inputs identical to development inputs. An extra acceptance requirement was recorded before fitting outcomes: exclude those three IDs and require the original guard and the clean-calibration guard to pass. None of the 600 test inputs overlap development/calibration after clipping. [Overlap evidence](../results/chat600/effective-input-overlap.json), [prospective amendment](../results/chat600/calibration-overlap-amendment.json), [clean-calibration result](../results/chat600/calibration-integrity.json).','',
              '## Clipped inputs and jailbreak subset','',
              'All paths use the same first 256 user-message tokens, followed by the intact assistant-start marker. Longer originals keep their dataset labels. Clipping can remove information needed to judge the original message.','',
              '| Test subset | Messages | Full-depth correct | Candidate correct |','| --- | --- | --- | --- |']
    for title,predicate in [('Clipped',lambda r:r['truncated']),('Not clipped',lambda r:not r['truncated']),('Dataset jailbreaking label',lambda r:r['jailbreak']==1)]:
        group=[r for r in rows if predicate(r)];lines.append(f'| {title} | {len(group)} | {sum(r["full"]==r["label"] for r in group)} | {sum(r["candidate"]==r["label"] for r in group)} |')
    if (OUT/'benchmark.json').exists():
        b=read('benchmark.json');lines+=['','## Actual time to finish all 600 messages','',
             '| Path | Pass 1 | Pass 2 | Pass 3 | Median complete workload | p95 per message |','| --- | --- | --- | --- | --- | --- |']
        for p,m in b['paths'].items():lines.append('| '+p+' | '+' | '.join(f'{x:.1f} s' for x in m['total_seconds_each_pass'])+f' | {m["median_total_seconds"]:.1f} s | {m["p95_message_ms"]:.1f} ms |')
        lines+=['',f'Candidate mean per-message saving versus the equally supervised full-depth SAFE/BLOCK-token path: **{100*b["relative_mean_saving_vs_trained_token"]:.2f}%**, paired-query bootstrap 95% interval **{100*b["paired_query_bootstrap95"][0]:.2f}% to {100*b["paired_query_bootstrap95"][1]:.2f}%**.',
                f'All **{b["actual_forward_passes"]} actual forward passes** counted executed blocks and checked prediction/depth parity. Each workload includes all 600 messages. These are actual calls, not cached feature timings. Other launched model work finished before this benchmark.',
                'Warm batch-one CPU, eight threads, float32, one process. Includes tokenization, readout/gate work and token decoding. Excludes loading, downloads and training. Three cyclic orders over four paths are not completely balanced; query bootstrap intervals do not include thermal drift, ordering effects or independent-session uncertainty. This is not live streaming queue latency or GPU throughput.','',
                '- [Every timed message and executed layer trace](../results/chat600/timings.json), [all complete-workload passes](../results/chat600/passes.json), [timing summary](../results/chat600/benchmark.json).']
    else:lines+=['','Actual all-600-message runtime measurement is not complete; no timing claim is made yet.']
    lines+=['','## Scope and reproduction','',
            'The task is agreement with dataset toxicity annotations, not a universal SAFE/BLOCK judgment. Only the human-annotated subset is used; it has different prevalence from the full corpus. Exact text duplicates and conversation-ID overlap are excluded, but shared users, paraphrases, annotator uncertainty and pretraining exposure remain unknown. A single-message classifier does not enforce a changing channel policy or use full conversation context.','',
            '- [Prospective method and acceptance criteria](chat600-protocol.md), [sealed source hashes and IDs](../results/chat600/protocol.json).',
            '- [Separate adversarial review](chat600-review.md), [next experiments](../PLAN.md#practical-workload-findings-and-the-next-iteration).',
            '- [Reproduction commands and dependencies](chat600-reproduce.md).',
            '- [Results](../results/chat600/result.json), [every held-out prediction](../results/chat600/predictions.json), [paired accuracy calculation](../results/chat600/quality-analysis.json), [portable heads](../results/chat600/heads.npz).',
            '- [Implementation](../experiments/chat600.py), [source dataset](https://huggingface.co/datasets/lmsys/toxic-chat), [original paper](https://arxiv.org/abs/2310.17389). Source messages are CC BY-NC 4.0 and remain in the local research cache.']
    (ROOT/'docs/chat600-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote real-message workload report.')

if __name__=='__main__':main()

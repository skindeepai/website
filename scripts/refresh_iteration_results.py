"""Publish fresh moderation and quantization findings from retained artifacts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def table(headers,rows):
    return '<div class="table-scroll" role="region" tabindex="0" aria-label="Measured comparison"><table><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(('<th scope="row">'+str(v)+'</th>') if i==0 else '<td>'+str(v)+'</td>' for i,v in enumerate(row))+'</tr>' for row in rows)+'</tbody></table></div>'
def append(pages,name,identifier,body):
    if name in pages:pages[name]['body']=pages[name]['body'].split('<section id="'+identifier+'">')[0]+'<section id="'+identifier+'">'+body+'</section>'
def update(pages):
    append(pages,'shared-model-results.html','live-shared','<p><a href="@/shared-decision-demo.html">Run both paths on your device</a></p>')
    if (ROOT/'results/quant-localize/result.json').exists():
        r=read('results/quant-localize/result.json');methods=r['methods'];float_m=methods['float']
        rows=[['Original float32',f'{float_m["correct"]} / 32',0]]
        for key,label in [('all_default','All linear projections'),('attention_only','Attention projections only'),('mlp_only','Feed-forward projections only'),('first12_only','First 12 layers'),('last12_only','Last 12 layers'),('all_per_channel','All projections, per-channel weights')]:
            m=methods[key]['metrics'];rows.append([label,f'{m["correct"]} / 32',m['added_errors']])
        body='<p><a href="@/shared-qwen-results.html">Back to Qwen experiments</a></p><p>Smaller numbers can make a model cheaper to store and run. Our first INT8 conversion broke its answers. This test checks which parts are sensitive to that conversion.</p>'
        body+=table(['What uses INT8?','Correct','New errors versus float32'],rows)
        attention=methods['attention_only']['metrics']
        body+=f'<p><strong>Attention-only conversion was less damaging.</strong> It introduced {attention["added_errors"]} errors and corrected {attention["corrected_errors"]} others. Matching the total number correct does not mean preserving the answers.</p>'
        body+='<p>This is a 32-message development diagnostic. All 24 layers still execute. We have not measured a speed benefit or selected this conversion for the demos.</p><details><summary>What was inspected?</summary><p>We compared the final input token\u2019s internal representation after every layer against the same float32 model. Feed-forward conversion caused much larger changes than attention-only conversion on this sample. This narrows down the investigation; it does not prove a single cause or show that all INT8 methods fail.</p><p>The classifier, inputs and thresholds stayed fixed. Per-channel conversion was tested too. No message text is copied into these reports.</p></details><p><a href="@/docs/quant-localize.md">Method, layer measurements and every outcome</a></p>'
        pages['quantization-results.html']=dict(title='Which parts tolerate smaller numbers?',description='Locate the accuracy loss before attempting another compressed model.',status='Development diagnostic',styles=['results.css'],body=body)
        append(pages,'shared-qwen-results.html','quant-localize','<p><a href="@/quantization-results.html">Why the INT8 conversion failed: a closer test</a></p>')
    if not (ROOT/'results/compact-next/result.json').exists():return
    r=read('results/compact-next/result.json');selected=r['selected'];m=r['methods'];n=r['n'];original=m['original'];candidate=m[selected]
    body='<p><a href="@/shared-model-results.html">Back to the shared model</a></p>'
    body+=f'<p>The original stopping rule preserved all full-depth decisions on <strong>500 unused messages</strong>. It would finish {original["confidence"]["early_count"]} at layer 2, skipping {100*original["confidence"]["blocks_skipped"]:.0f}% of transformer blocks. The model still got {n-original["full"]["correct"]} messages wrong.</p>'
    rows=[]
    for label,v in [('Previous model, all 4 layers',original['full']),('Previous stopping rule',original['confidence']),('Selected new model, all 4 layers',candidate['full']),('New model, confidence check',candidate['confidence']),('New model, learned risk check',candidate['risk'])]:
        rows.append([label,f'{v["correct"]} / {n}',f'{100*v["blocks_skipped"]:.1f}%'])
    body+=table(['Method','Correct','Projected blocks skipped'],rows)
    body+='<p class="small">These 500-message routing counts are simulated from full-depth outputs. Actual stop/continue execution is checked separately on the first 50 messages.</p>'
    body+='<p>The learned check sees only the state after layer 2. It estimates whether stopping would create an error that the remaining layers would correct. Both stopping rules were chosen on separate development examples.</p>'
    risk=candidate['risk'];confidence=candidate['confidence']
    body+=f'<p>The risk rule would stop {risk["early_count"]} messages after layer 2. Compared with its own full-depth model, it added <strong>{risk["added_errors"]} errors</strong> and corrected {risk["corrected_errors"]}. The confidence check added {confidence["added_errors"]} and corrected {confidence["corrected_errors"]}. Equal totals can conceal different mistakes.</p>'
    body+='<p><strong>The more complex risk check did not improve coverage here.</strong> The selected teacher-trained model reduced false blocks from 57 to 44, but toxic misses rose from 3 to 5. We have kept the original model in the browser demo.</p>'
    body+=f'<p>There were {r["toxic"]} toxic messages in this sample. The selected full model missed {candidate["full"]["missed_toxic"]} of them and falsely blocked {candidate["full"]["false_block"]} benign messages. This samples the unused human-annotated remainder of ToxicChat; it is not a deployment safety estimate.</p>'
    if (ROOT/'results/compact-next/reference.json').exists():
        reference=read('results/compact-next/reference.json')
        body+=table(['Same messages, reference methods','Correct','Toxic messages missed','False blocks'],[[label,v['correct'],v['missed_toxic'],v['false_block']] for label,v in [('Always say SAFE',reference['always_safe']),('Frozen Qwen classifier, 24 layers',reference['qwen']),('Selected small model, 4 layers',candidate['full'])]])
        body+='<p class="small">Always saying SAFE shows how label imbalance can inflate accuracy. Development selection used balanced accuracy. Training differs: the BERT encoder was task-trained; Qwen\u2019s backbone stayed frozen and its classifier was trained.</p>'
    timing_name='results/compact-next/timing-cached.json' if (ROOT/'results/compact-next/timing-cached.json').exists() else 'results/compact-next/timing.json'
    if (ROOT/timing_name).exists():
        timing=read(timing_name)['methods']
        body+=table(['Actual execution, first 50 messages','Mean time, 3 passes'],[[label,f'{1000*timing[key]["mean_seconds"]:.0f} ms'] for key,label in [('original_full','Original model, all 4 layers'),('original_confidence','Original model, stop early'),('full','New model, all 4 layers'),('confidence','New model, confidence check'),('risk','New model, learned risk check')]])
        body+='<p class="small">Paired CPU timing includes input preparation, the checks and actual skipped layers. Model and gate weights are loaded before timing. The 50-message timing subset is smaller than the quality test. The detailed report retains an earlier run that unnecessarily reloaded gate arrays inside each check.</p>'
    body+='<details><summary>What changed in training?</summary><p>We trained all four BERT layers and both answer heads. Three matched seeds compared human labels alone with human labels plus a frozen Qwen teacher\u2019s output scores. Development data selected the checkpoint, answer thresholds and one candidate before these 500 labels were evaluated. Every seed remains in the report.</p><p>Each accepted early answer skips 2 of 4 transformer blocks. Embeddings, tokenization and classifier work are still required, so block savings are not a percentage of total computation or time.</p></details><p><a href="@/docs/compact-next.md">Training, all seeds, errors and confidence bounds</a> &middot; <a href="@/shared-decision-demo.html">Try the original shared model in your browser</a></p>'
    pages['fresh-message-results.html']=dict(title='Does stopping hold up on new messages?',description='Stronger training and a learned check tested on 500 unused messages.',status='Fresh validation',styles=['results.css'],body=body)
    append(pages,'shared-model-results.html','fresh-validation','<p><a href="@/fresh-message-results.html">New: what happened on 500 unused messages?</a></p>')
    append(pages,'shared-decision-demo.html','fresh-validation','<p><a href="@/fresh-message-results.html">Fresh validation of this model and newer training candidates</a></p>')

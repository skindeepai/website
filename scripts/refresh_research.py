"""Publish the experiment register and scoped results from local JSON artifacts."""
import json, re, statistics
from html import escape as e
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))
def table(headers, rows):
    return '<div class="table-scroll" tabindex="0" role="region" aria-label="Results table"><table><thead><tr>'+''.join('<th scope="col">'+e(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def paragraph(text): return '<p>'+e(text)+'</p>'
def pct(x): return f'{100*x:.1f}%'

def main():
    pages=read('content/pages.json');protocol=(ROOT/'EXPERIMENTS.md').read_text(encoding='utf-8')
    evidence=read('content/research-evidence.json');groups={'P':'preferences','D':'decisions','C':'coordinates','X':'combined'}
    entries=[]
    for match in re.finditer(r'^### ([PDCX]\d\d) [^\w\n]+ ([^\n]+)\n(.*?)(?=^### |^## |\Z)',protocol,re.M|re.S):
        identifier,title,body=match.groups();title=re.sub(r' \[.*?\]','',title)
        fields=dict(re.findall(r'\*\*(.+?):\*\* (.*?)(?=\n\n|\Z)',body,re.S))
        record=evidence[identifier]
        entries.append({'id':identifier,'title':title,'track':groups[identifier[0]],'fields':fields,**record})
    assert len(entries)==29,len(entries)
    (ROOT/'content/experiments.json').write_text(json.dumps(entries,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    # Public answers are rendered by refresh_faq through organize_results.
    body='<div class="note warning"><p><strong>Exploratory results, not deployment claims.</strong> These runs establish working mechanisms and expose failure cases. They do not validate arbitrary policies, human preferences or general GUI control.</p></div>'
    synthetic=read('results/synthetic/result.json');records=synthetic['records'];math=synthetic['math']
    body+='<section class="paper" id="preferences"><h2>Preference learning: mechanics pass, assumptions matter</h2>'+paragraph(f'The browser solver passes {math["browser_fixtures"]} fixtures. On {math["random_cases"]} random bounded edits, its largest squared-distance difference from independent SciPy optimization was {math["max_squared_distance_gap_vs_scipy"]:.2g} (tolerance 1e-7). This verifies latent geometry, not perceptual minimality.')
    rows=[]
    def average(exp,metric,**filters):
        vals=[r[metric] for r in records if r['experiment']==exp and all(r.get(k)==v for k,v in filters.items())]
        return statistics.mean(vals),min(vals),max(vals)
    for n in [20,50,100]:
        a,lo,hi=average('P02','accuracy',n=n);rows.append([f'Linear utility, {n} ratings',pct(a),f'{pct(lo)}–{pct(hi)}'])
    for head in ['linear','quadratic']:
        a,lo,hi=average('P04','balanced_accuracy',head=head);rows.append([f'Disconnected utility: {head} head (balanced accuracy)',pct(a),f'{pct(lo)}–{pct(hi)}'])
    body+=table(['Synthetic test','Mean, five seeds','Observed seed range'],rows)
    body+='<p>Seed ranges are descriptive, not confidence intervals. The disconnected-utility test uses a different label function from the linear test.</p>'
    rows=[]
    for policy in ['random','uncertainty','mixed']:
        a,lo,hi=average('P03','accuracy',policy=policy);rows.append([policy,pct(a),f'{pct(lo)}–{pct(hi)}'])
    body+=table(['Sampling at 80 labels','Held-out accuracy','Observed seed range'],rows)+'<p>This pilot uses a 2/2/2 mixed batch with 240 candidates. It does not evaluate the browser’s 5/4/3 policy or prove label savings.</p>'
    rows=[]
    for method in ['box-optimum','truncated','reranked','random']:
        a,_,_=average('P05','oracle_utility',method=method);p,_,_=average('P05','predicted',method=method);rows.append([method,pct(p),f'{a:.3f}'])
    body+=table(['Misspecified-model choice','Predicted score','True synthetic utility (higher is better)'],rows)+'<p>The maximum learned score is not the maximum true utility. These methods also use different candidate/generation budgets, so this is a failure illustration, not a fair efficiency benchmark.</p><p><a href="@/results/synthetic/result.json">All synthetic results</a> · <a href="@/results/synthetic/math-fixtures.json">Numerical fixtures</a></p></section>'
    q=read('results/qwen-decisions/result.json');times=q['timing_ms'];full=next(h['test_accuracy'] for h in q['heads'] if h['depth']==24)
    body+='<section class="paper" id="decisions"><h2>Qwen: early exit works, the quality gate fails</h2>'+paragraph(f'Frozen Qwen2.5-0.5B-Instruct, 80 training / 32 calibration / 48 test examples. Rules vary by topic and ALLOW/BLOCK action; grammar is shared. Heads were trained at layers 6, 12, 18 and 24. This is a small synthetic policy task.')
    body+=table(['Output path','Test accuracy','Warm CPU p50','Warm CPU p95'],[['Constrained A/B/C token (zero-shot)',pct(q['token_accuracy']['test']),f'{times["minimal_token"]["p50"]:.1f} ms',f'{times["minimal_token"]["p95"]:.1f} ms'],['Trained full-depth head',pct(full),f'{times["direct_head"]["p50"]:.1f} ms',f'{times["direct_head"]["p95"]:.1f} ms'],['Trained adaptive head',pct(q['adaptive_accuracy']),f'{times["adaptive_head"]["p50"]:.1f} ms',f'{times["adaptive_head"]["p95"]:.1f} ms']])
    body+=paragraph(f'All 48 adaptive requests exited after layer 6; hooks verified the remaining 18 layers did not execute. Accuracy fell by {100*(full-q["adaptive_accuracy"]):.2f} percentage points against the trained final head, exceeding the proposed one-point tolerance. The small calibration set did not protect test quality.')
    body+='<p>The token baseline is zero-shot while the heads receive labels; its low accuracy is not evidence that heads inherently outperform tokens. This pilot did not include an equally trained token baseline; the later matched-output and chat studies provide that control. Layer 12 and 18 probes reached 48/48 on this fixture set, but choosing either after seeing test results requires a new held-out test.</p><p>Times include transformer/head execution and adaptive hooks, but exclude tokenization, loading and service overhead. CPU float32, eight threads, one seed. No equal-quality speedup, JSON speedup or rare-error guarantee is established.</p><p><a href="@/results/qwen-decisions/result.json">Run configuration and metrics</a> · <a href="@/results/qwen-decisions/predictions.json">Every test prediction</a> · <a href="@/results/qwen-decisions/fixtures.json">Train/calibration/test fixtures</a></p></section>'
    if (ROOT/'results/coordinates/result.json').exists():
        c=read('results/coordinates/result.json')
        body+='<section class="paper" id="coordinates"><h2>Coordinates without text decoding</h2>'+paragraph(f'Reproduced the pretrained GUI-Actor 2B pointer path on our own interface at desktop and mobile sizes. It hit {c["hits"]} of {c["target_count"]} labeled target boxes. Two absent-target prompts still received points: this head has no no-target class.')
        body+=table(['Path','Warm CPU p50','Samples'],[[k,f'{v["p50"]:.1f} ms',str(v['samples'])] for k,v in c['timing_ms'].items()])
        body+='<p>Both paths use the same pointer head and produced matching probabilities. “With vocabulary” adds one vocabulary projection; it is not JSON decoding or a trained coordinate-token baseline. This isolates a small piece of output overhead. All vision and transformer layers still run.</p><p>Eight prompts from two screenshots are a smoke test, not a general grounding benchmark. This uses published Microsoft weights, a shortened prompt, 936 desktop / 392 mobile visual patches, CPU float32 and eight threads. See the <a href="@/docs/coordinate-correction.md">resolution-cap correction</a>. Patch confidence is uncalibrated. No live clicks were executed.</p><p><a href="https://github.com/microsoft/GUI-Actor">Upstream GUI-Actor</a> · <a href="@/results/coordinates/result.json">Run artifact</a> · <a href="@/results/coordinates/predictions.json">Predictions and boxes</a> · <a href="@/coordinate-lab.html">Annotate a target</a></p></section>'
    body+='<section class="paper"><h2>What happens next</h2><p>Broaden the data before broadening the claim. The next priority is a reliable decision exit gate and a coordinate head that can reject missing targets, followed by fair token baselines and full input-to-result timings.</p><p><a href="@/research.html">Open experiments and prerequisites</a> · <a href="@/docs/claims.md">Claim ledger</a> · <a href="@/getting-started.html">Reproduce the pilots</a></p></section>'
    full_details=body
    if (ROOT/'results/banking77/result.json').exists():
        full_details+='<section class="paper"><h2>Public-data follow-ups</h2><p><a href="@/docs/banking77-results.md">BANKING77: every layer, seed, stopping rule and timed path</a> · <a href="@/docs/conservative-exits.md">Conservative gate on a fresh reserve</a> · <a href="@/docs/screenspot-results.md">ScreenSpot: every predicted point and target box</a>.</p><p><a href="@/docs/early-exit.md">How the classifiers work</a> · <a href="@/docs/coordinate-correction.md">Correction to the old image-resolution cap</a> · <a href="@/browser-benchmark.html">Run Qwen in your browser</a>.</p></section>'
    if (ROOT/'results/clinc-validation/result.json').exists():
        full_details+='<section class="paper"><h2>Adversarial follow-ups</h2><p>Separate implementation replays matched the original outputs and actual layer stopping. A second dataset and an explicit unknown-request output still failed their calibration checks. The changing-rule test also failed.</p><p><a href="@/docs/reproduction.md">Replay and portable weights</a> · <a href="@/docs/clinc-results.md">Second dataset</a> · <a href="@/docs/clinc-unknown.md">Unknown-request classifier</a> · <a href="@/docs/changing-rules.md">Changing rules</a> · <a href="@/docs/matched-output.md">Matched one-token control</a>.</p></section>'
    count=q['split_sizes']['test'];full_correct=round(full*count);early_correct=round(q['adaptive_accuracy']*count)
    body='<section class="result-summary"><h2>Preference learning</h2>'+paragraph(f'{math["browser_fixtures"]} calculation checks passed. The model can learn simple made-up preferences, but we have not yet shown that people prefer its suggestions.')+'</section>'
    body+='<section class="result-summary"><h2>Stopping early</h2><p>Faster, but more mistakes in this first test.</p>'+table(['Method','Correct answers','Typical time'],[['Use the full model',f'{full_correct} of {count}',f'{times["direct_head"]["p50"]:.0f} ms'],['Stop early',f'{early_correct} of {count}',f'{times["adaptive_head"]["p50"]:.0f} ms']])+'<p class="small">One small, made-up message task on this CPU. These times do not include loading the model or preparing the input.</p></section>'
    if (ROOT/'results/banking77/result.json').exists():
        banking=read('results/banking77/result.json');first=banking['runs'][0];candidate=first['candidate_test'];n=candidate['count']
        correct=round(first['heads'][-1]['test_accuracy']*n)
        summary='<section class="result-summary"><h2>Stopping early</h2><p>Tested on 3,080 public banking queries across 77 topics. Similar overall accuracy, but the stopping rule failed the stricter reliability check.</p>'+table(['Method','Correct answers','Typical time'],[['Full-depth trained classifier',f'{correct} of {n}',f'{banking["timing"]["full_head"]["end_to_end_ms"]["p50"]:.0f} ms'],['Early-stop candidate',f'{candidate["correct"]} of {n}',f'{banking["timing"]["candidate_early_exit"]["end_to_end_ms"]["p50"]:.0f} ms']])+'<p class="small">Accuracy: 3,080 queries. Timing: 96 queries, two repeats per method; seed 17, warm CPU medians including tokenization. The candidate skips about 24% of transformer blocks on average. A guarded system would keep using full depth.</p><p><a href="@/docs/banking77-results.md">Data, methods and all predictions</a> · <a href="@/docs/conservative-exits.md">Stricter follow-up on fresh queries</a></p></section>'
        start=body.index('<section class="result-summary"><h2>Stopping early</h2>')
        body=body[:start]+summary
    if (ROOT/'results/chat600/result.json').exists():
        chat=read('results/chat600/result.json')['splits']['test']
        timing=read('results/chat600/benchmark.json')['paths'] if (ROOT/'results/chat600/benchmark.json').exists() else None
        headers=['600 messages','All 24 layers','Stop at 12']
        rows=[['Correct answers',str(chat['full']['correct']),str(chat['candidate']['correct'])],['Toxic messages missed',str(chat['full']['missed_toxic'])+' of 81',str(chat['candidate']['missed_toxic'])+' of 81']]
        if timing:
            rows.append(['Total time']+[f'{timing[key]["median_total_seconds"]:.1f} s' for key in ['trained_full_token','early_candidate']])
        summary='<section class="result-summary"><h2>Stopping early</h2><p>600 real messages from an online chatbot. Stopping halfway gave three fewer correct answers overall, but missed seven more toxic messages.</p>'+table(headers,rows)
        summary+='<p class="small">The chosen rule always stops at layer 12; it does not detect readiness per message. It failed our reliability checks. Both methods use trained classifiers. The messages are archived ToxicChat data, not a live moderation trial.</p>'
        if timing:summary+='<p class="small">Time: median of three complete 600-message passes per method on this CPU, including input preparation. The full-depth comparison returns one trained SAFE/BLOCK token. Model loading is excluded.</p>'
        summary+='<p><a href="@/docs/chat600-results.md">Results, timings and every prediction</a> &middot; <a href="@/docs/chat600-review.md">Independent review</a></p></section>'
        start=body.index('<section class="result-summary"><h2>Stopping early</h2>')
        body=body[:start]+summary
        full_details+='<section class="paper"><h2>Practical workloads</h2><p><a href="@/docs/chat600-protocol.md">600-message protocol</a> &middot; <a href="@/docs/chat600-results.md">Chat results</a> &middot; <a href="@/docs/chat600-review.md">Claim audit</a>.</p><p>On ten new synthetic mazes, both learned policies reached zero goals; the shortest-path reference reached all ten. The failed moves are retained in the <a href="@/maze-benchmark.html">recorded maze replay</a>. <a href="@/docs/maze-actions.md">Full maze method and evidence</a>.</p></section>'
    if (ROOT/'results/coordinates/result.json').exists():
        marks=''.join('<span class="'+('hit' if i<c['hits'] else 'miss')+'" aria-hidden="true">'+('✓' if i<c['hits'] else '×')+'</span>' for i in range(c['target_count']))
        body+='<section class="result-summary"><h2>Finding where to click</h2><div class="hit-row" role="img" aria-label="'+str(c['hits'])+' of '+str(c['target_count'])+' targets found">'+marks+'</div>'+paragraph(f'{c["hits"]} of {c["target_count"]} visible targets found on two example screens. But when a target was missing, the model still guessed a point.')+'<p><a href="@/coordinate-lab.html">See the recorded clicks</a></p></section>'
        if (ROOT/'results/screenspot/result.json').exists():
            public=read('results/screenspot/result.json')
            body+='<p>Public screenshot follow-up: '+str(public['hits']['connected_region'])+' of '+str(public['samples'])+' targets found using connected regions, compared with '+str(public['hits']['max_patch'])+' using a single patch. <a href="@/docs/screenspot-results.md">See every result</a>.</p>'
    if (ROOT/'docs/chat-smoke-results.md').exists():
        full_details+='<section class="paper"><h2>Small follow-up experiments</h2><p><a href="@/docs/chat-smoke-results.md">Eight approaches on 100 previously inspected messages</a>, including a gate that chooses different stopping layers per message. These exploratory results do not replace the 600-message study above. <a href="@/docs/chat-smoke-review.md">Separate adversarial review</a>.</p></section>'
    body+='<details id="full-results"><summary>Test setup, measurements and limitations</summary>'+full_details+'</details><p><a href="@/research.html">Questions still to test</a> · <a href="@/getting-started.html">Run the experiments</a></p>'
    pages['results.html']['body']=body;pages['results.html']['status']='Early results'
    from organize_results import update
    update(pages, legacy=pages['results.html']['body'])
    (ROOT/'content/pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Refreshed 29 experiments and measured result summaries.')

if __name__=='__main__':main()

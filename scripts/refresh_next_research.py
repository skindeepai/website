"""Focused practical pages and concise links; preserve the approved site layout."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def table(headers,rows):
    return '<div class="table-scroll" role="region" tabindex="0" aria-label="Measured comparison"><table><thead><tr>'+''.join('<th scope="col">'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(('<th scope="row">'+str(x)+'</th>') if i==0 else '<td>'+str(x)+'</td>' for i,x in enumerate(row))+'</tr>' for row in rows)+'</tbody></table></div>'
def page(pages,name,title,lead,body):
    pages[name]=dict(title=title,description=lead,status='Exploratory results',styles=['results.css'],body='<p><a href="@/decisions.html">Back to decisions without text</a></p>'+body)
def update(pages):
    from refresh_tiny_decision import update as tiny
    from refresh_search_demo import update as search
    from refresh_action_demo import update as action
    tiny(pages);search(pages);action(pages)
    from refresh_shared_demo import update as shared_demo
    shared_demo(pages)
    from refresh_search_next import update as search_next
    search_next(pages)
    privacy=read('results/practical-privacy/result.json');receipt=read('results/practical-receipts/result.json');routing=read('results/practical-routing/result.json')
    body='<p>Before sharing a document, identify the names that may need masking. This first test covers names in court documents; it does not establish complete anonymization.</p>'
    body+=table(['Method','Name tokens found','Other tokens incorrectly marked'],[['Title rule',f'{privacy["title_rule"]["tp"]} / 1,229',privacy['title_rule']['fp']],['Trained token classifier',f'{privacy["learned"]["tp"]} / 1,229',privacy['learned']['fp']]])
    body+='<p>The trained classifier found more names, but also marked more unrelated words. It missed 141 name tokens.</p><details><summary>What was tested?</summary><p>50 real TAB court documents, limited to the first 1,200 tokens each. Human annotations supply the references. A separate 100 documents train the classifier and 25 select its threshold. No privacy or speed guarantee is established.</p><p><a href="@/docs/practical-baselines.md#find-names-before-redacting-text">Data, method and every outcome</a></p></details>'
    page(pages,'redaction-results.html','Find names to redact','Return marked text spans instead of rewriting a document.',body)
    body='<p>Pick the total from a receipt\u2019s text and positions. This test starts with the supplied transcription; it does not run OCR on the image.</p>'
    body+=table(['Method','Correct total'],[['Choose largest amount',f'{receipt["largest_correct"]} / {receipt["scorable_receipts"]}'],['Trained candidate selector',f'{receipt["learned_correct"]} / {receipt["scorable_receipts"]}']])
    body+='<p>A little context helps distinguish the total from cash paid, change and individual prices. The trained selector still got 17 totals wrong.</p><details><summary>What was tested?</summary><p>156 receipts trained the selector. Fifty held-out receipts were grouped separately by merchant name; one had no usable reference total, leaving 49 scorable cases. This uses a contestant-corrected SROIE annotation mirror, not its official test split. OCR time and noisy scans remain untested.</p><p><a href="@/docs/practical-baselines.md#read-a-receipts-total">Source, split and every outcome</a></p></details>'
    page(pages,'receipt-results.html','Choose the receipt total','Select a value that already appears on the receipt.',body)
    body='<p>A request router needs to recognize when a message does not fit any supported action. Adding a rejection rule catches more unsupported requests, but can also reject valid ones.</p>'
    body+=table(['Rule','Supported intent correct / 900','Unsupported rejected / 1,000'],[['Highest category score',routing['raw']['known_correct'],routing['raw']['unsupported_rejected']],['Also reject low scores',routing['with_rejection']['known_correct'],routing['with_rejection']['unsupported_rejected']]])
    body+='<p>The cautious rule rejected 152 supported requests, versus 16 without the cutoff. This is a tradeoff, not a universal improvement.</p><details><summary>What was tested?</summary><p>Word/phrase TF-IDF and logistic regression, 30 supported CLINC intents plus UNKNOWN. Development data chooses the cutoff; the 1,900 test requests are reused from the earlier study. CLINC is human-crowdsourced scenario data, not live support traffic.</p><p><a href="@/docs/practical-baselines.md#route-a-request-or-decline-to-route-it">Data, selection and every outcome</a></p></details>'
    page(pages,'routing-results.html','Route or ask for help','Choose a supported action, or return UNKNOWN.',body)
    if (ROOT/'results/compact-specialist/result.json').exists():
        compact=read('results/compact-specialist/result.json')['variants'];joint=compact['joint']['splits']['evaluation'];single=compact['full']['splits']['evaluation']
        body='<p>The first two layers do the shared work. A small classifier checks the result. Easy cases can finish there; other cases continue through layers three and four, using the same internal state.</p>'
        body+='''<figure class="small-example"><svg viewBox="0 0 360 125" role="img" aria-label="Input runs through layers one and two. A classifier can answer immediately or continue through layers three and four." style="width:100%;max-width:360px;height:auto"><g fill="none" stroke="#395775" stroke-width="2"><rect x="2" y="43" width="46" height="36" rx="5"/><rect x="68" y="43" width="100" height="36" rx="5"/><path d="M48 61h20 M168 61h20V25h20 M188 61v41h20"/><rect x="208" y="7" width="150" height="36" rx="5"/><rect x="208" y="84" width="150" height="36" rx="5"/></g><g fill="#152c45" font-family="system-ui,sans-serif" font-size="18" text-anchor="middle"><text x="25" y="67">Input</text><text x="118" y="67">Layers 1–2</text><text x="283" y="31">Answer now</text><text x="283" y="108">Layers 3–4</text></g></svg><figcaption>One model. The first two layers are never repeated.</figcaption></figure>'''
        body+=table(['Method','Correct / 100','Average layers'],[['Four layers, final answer only',single['full']['correct'],'4'],['Train answers at layers 2 and 4; use all four',joint['full']['correct'],'4'],['Same jointly trained model; allow early answers',joint['cascade']['correct'],f'{joint["mean_depth"]:.2f}']])
        body+=f'<p><strong>{joint["early_count"]} of 100 messages stopped after layer 2.</strong> The adaptive route preserved all 100 decisions of its own four-layer model. That skips 36% of its transformer-block executions on this sample. It does not establish equal quality to Qwen, which scored 86/100 on these messages.</p>'
        if (ROOT/'results/compact-specialist/runtime.json').exists():
            timing=read('results/compact-specialist/runtime.json');full=sum(timing['paths']['full']['seconds'])/3;adaptive=sum(timing['paths']['adaptive']['seconds'])/3
            body+=f'<p>Actual execution on the first 50 messages took {full*1000:.0f} ms at full depth and {adaptive*1000:.0f} ms with the check, averaged over three passes. '+(f'That was {100*(1-adaptive/full):.1f}% less time.' if adaptive<full else 'The extra checking did not save time in this run.')+'</p>'
        body+='<details><summary>Training and limits</summary><p>Google BERT small has four layers and about 11.1 million parameters here. Both classifier heads and the shared encoder were trained on 1,398 labeled messages. Separate development data selected the epoch, thresholds and stopping rule. The 100 evaluation messages had already been inspected in earlier experiments.</p><p>This is one jointly trained backbone with multiple exits, not two unrelated pretrained models fused together. The comparison changes the training objective as well as the exit rule. One seed and one small consumed test cannot establish a general quality improvement.</p><p><a href="@/docs/compact-specialist.md">Training, layer traces, timing and every outcome</a></p></details>'
        page(pages,'shared-model-results.html','One model, two chances to answer','Compute the beginning once; continue only when needed.',body)
    if (ROOT/'results/chat-next-methods/combined-timing.json').exists():
        combined=read('results/chat-next-methods/combined-timing.json');runs=read('results/chat-next-methods/combined-records.json');paths=combined['paths']
        assert all(p['all_equivalent'] for p in paths.values()),'Do not publish successful-comparison copy for failed equivalence.'
        correct=sum(r['prediction']==r['label'] for r in runs[0]['records'])
        body='<p>Compute the fixed instructions once, then process several messages together. This tests the two optimizations together on the same queued workload.</p>'
        body+=table(['Execution','Correct / 50','Time for 50'],[[name,correct,f'{paths[key]["mean_seconds"]:.2f} s'] for key,name in [
            ('full_b1','One message at a time'),('full_b4','Groups of four'),('prefix_b1','Reuse instructions'),('prefix_b4','Reuse instructions + groups of four')]])
        saving=100*(1-paths['prefix_b4']['mean_seconds']/paths['full_b1']['mean_seconds'])
        body+=f'<p><strong>{saving:.1f}% less time with identical decisions on these 50 messages.</strong> Every message still uses all 24 layers. This combines fewer repeated instruction calculations with fewer separate model calls.</p>'
        body+='<details><summary>What was measured?</summary><p>Two counterbalanced CPU passes, four threads, on 50 previously inspected ToxicChat messages. Input preparation, sorting, padding, cache construction and copying are included. All eight runs matched labels and the specified logit tolerance. The cache contains only the fixed instructions and is copied privately for each batch.</p><p>The workload is already queued. This is not a live-response latency estimate, a new accuracy test, or a guarantee of identical outputs on other data.</p><p><a href="@/docs/chat-next-methods.md#execution-measurements">All passes, layer traces and cache checks</a> · <a href="@/results/chat-next-methods/combined-timing.json">Timing summary</a></p></details>'
        page(pages,'combined-results.html','Reuse instructions and group messages','Test two execution improvements together.',body)
    if (ROOT/'results/chat-next-methods/continuation-timing.json').exists():
        body='<p>Qwen processes the input once. At layers 6, 12 and 18, a small trained classifier checks whether to answer. If it continues, the lower layers are not repeated.</p>'
        body+=table(['Locked stopping rule','Correct / 100','Layer blocks skipped'],[['Original full classifier',86,'0%'],['Jointly trained checkpoint classifier',86,'4.75%'],['Checkpoint classifier with distillation',87,'6.75%']])
        body+='<p>The distilled rule answered early on 16 messages and corrected one false block. It introduced no new errors on these 100 previously inspected messages. A one-message improvement is not enough to establish better accuracy.</p>'
        t=read('results/chat-next-methods/continuation-timing.json')['totals_seconds']
        body+=f'<p>A separate actual-execution pass over the first 50 messages took {t["full"]:.2f} s at full depth, {t["joint_ce"]:.2f} s with joint classification and {t["joint_distill"]:.2f} s with distillation. One paired pass is noisy; it does not reliably rank the two gates.</p>'
        body+='<details><summary>What learns, and when does it stop?</summary><p>The Qwen backbone stays frozen. One shared 896-to-64-to-2 readout learns across checkpoints, with depth-specific normalization and biases. The distilled version also learns from the original full-depth classifier on training examples. The final fallback remains the original classifier.</p><p>Historical development data selects confidence thresholds and agreement checks. Both selected gates require consecutive checkpoint agreement, so neither stops at layer 6. These empirical rules have not passed fresh reliability acceptance.</p><p><a href="@/docs/chat-next-methods.md">Training, all fixed-depth comparisons, timing and failed int8 repair</a> · <a href="@/results/chat-next-methods/quality.json">Full quality record</a></p></details>'
        page(pages,'shared-qwen-results.html','Check Qwen before continuing','Several chances to answer within one model pass.',body)
    from refresh_iteration_results import update as validation
    validation(pages)
    from refresh_visual_next import update as visual_next
    marker='<section id="new-practical-demos">'
    links='<ul class="page-links">'+''.join('<li><a href="@/'+name+'">'+label+'</a></li>' for name,label in [
        ('tiny-decision-demo.html','Try the small trained classifier'),('shared-decision-demo.html','Compare early stopping on your device'),('search-demo.html','Search real papers locally'),
        ('screenshot-demo.html','See real screenshot-to-click results'),('image-action-demo.html','Try a live image-to-action model'),
        ('redaction-results.html','Find names to redact'),('receipt-results.html','Choose a receipt total'),('routing-results.html','Route or ask for help'),
        ('shared-model-results.html','One model with a small early exit')])+'</ul>'
    for name in ['decisions.html','sitemap.html']:
        pages[name]['body']=pages[name]['body'].split(marker)[0]+marker+'<h2>Try a specific task</h2>'+links+'</section>'
    marker='<section id="new-action-demos">'
    pages['coordinates.html']['body']=pages['coordinates.html']['body'].split(marker)[0]+marker+'<h2>See it work</h2><p><a href="@/screenshot-demo.html">Explore 30 real screenshots</a> · <a href="@/image-action-demo.html">Run the small image-to-action model</a></p></section>'
    marker='<section id="shared-model-followup">'
    pages['adaptive.html']['body']=pages['adaptive.html']['body'].split(marker)[0]+marker+'<h2>One shared model, two exits</h2><p>A jointly trained four-layer model can answer after two layers, then continue from the same state when needed. <a href="@/shared-model-results.html">See the layer counts, accuracy and actual timing</a>.</p></section>'
    marker='<section id="next-execution-results">'
    new_links=''.join('<li><a href="@/'+n+'">'+pages[n]['title']+'</a></li>' for n in ['shared-model-results.html','shared-qwen-results.html','combined-results.html'] if n in pages)
    for name in ['decision-results.html','sitemap.html']:
        pages[name]['body']=pages[name]['body'].split(marker)[0]+marker+'<h2>Further measured tests</h2><ul class="page-links">'+new_links+'</ul></section>'
    visual_next(pages)

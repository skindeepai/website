"""Direct, honestly labelled live demos on topic and method pages."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def update(pages):
    demos=[
        ('preference-method-demo','Learn more than one favorite style','Compare example selection and simple preference models.',['preference-method-demo']),
        ('method-demo','Compare ways to decide sooner','Run real classifiers on your text or a public message sample.',['method-demo']),
        ('practical-demo','Try a small task classifier','Edit the input and get a direct result from trained weights.',['practical-demo-core','practical-demo']),
        ('execution-demo','Reuse work inside Qwen','Measure instruction caching and batching on your device.',['execution-demo']),
        ('maze-live','Let a trained model choose the moves','Run the policy, change the walls, and see whether it reaches the goal.',['maze-live-core','maze-live']),
        ('point-demo','Find visible text in a screenshot','Upload an image and get a point back, computed locally.',['point-demo']),
        ('rerank-demo','Give your search a second check','A two-layer model scores the first 20 keyword matches.',['rerank-demo'])]
    for name,title,lead,scripts in demos:
        pages[name+'.html']=dict(title=title,description=lead,status='Live browser demo',styles=['results.css','live-demos.css'],scripts=['scripts/'+s+'.js' for s in scripts],template='content/'+name+'.html')
    mapping={}
    def links(names,target,label,note=''):
        for n in names:mapping[n]=dict(target=target,label=label,note=note)
    links(['preferences.html','how-it-works.html','preference-results.html'],'demo.html','Teach the model your preferences')
    links(['active-learning-strategies.html'],'preference-method-demo.html','Compare how examples are chosen','Live random versus least-certain sampling from your ratings.')
    links(['multi-modal-preferences-deep-dive.html'],'preference-method-demo.html','Teach two different styles','Live linear versus nonlinear preference learning; this illustrates separated preferences without a generator mixture.')
    links(['examples/art.html','examples/music.html','examples/social-media.html','examples/dating.html','examples/science.html','examples/index.html'],'demo.html','Try the underlying preference-learning loop','The working example learns from drawings. The application described here is a proposal, not a completed product demo.')
    links(['decisions.html','chat-results.html','decision-checks.html'],'moderation-benchmark.html','Live demo: compare labels and written replies','Run Qwen 0.5B on real messages and compare accuracy and time. All 24 layers run; this demo does not use the trained banking classifier.')
    links(['adaptive.html','shared-model-results.html','fresh-message-results.html'],'shared-decision-demo.html','Live demo: compare early stopping and full depth','Compare accuracy and time on real messages. This demo uses a four-layer BERT model that can stop after layer 2; the Qwen tests below are separate.')
    links(['decision-results.html'],'method-demo.html','Compare the methods live','Browser BERT comparisons; the recorded Qwen results remain separate.')
    links(['output-results.html'],'moderation-benchmark.html','Compare output formats in the browser','Related q4 Qwen demo: direct vocabulary scores, one token or JSON. Its direct path still computes the full vocabulary; it does not implement the two-row CPU optimization or the trained classifier below.')
    for mode,names,label,note in [
        ('fixed',['depth-results.html'],'Try stopping at a fixed layer','Live layer-2 versus layer-4 BERT; not a replication of the Qwen depths below.'),
        ('adaptive',['early-exit-results.html','shared-qwen-results.html'],'Try the stop-or-continue check','Live BERT checks layer 2 of 4. The Qwen multi-checkpoint experiment below is separate.'),
        ('cascade',['cascade-results.html','cascade-test-results.html'],'Run a two-model fallback','This uses tiny BERT → larger BERT. It is a new pairing, not the measured tiny → Qwen cascade.'),
        ('training',['training-results.html'],'Compare the trained models','Runs earlier and updated BERT weights. Training happened offline; this is not browser distillation.'),
        ('quantization',['quantization-results.html'],'Compare float32 and INT8','Live conversion of BERT. The Qwen conversion investigated below is a separate experiment.')]:
        links(names,'method-demo.html?method='+mode,label,note)
    for mode,names in [('prefix',['prefix-results.html']),('batch',['batch-results.html']),('combined',['combined-results.html'])]:
        links(names,'execution-demo.html?method='+mode,'Run this comparison with Qwen','Live q4 Qwen with vocabulary scores and a 64-token message limit; it differs from the trained Python benchmark below.')
    for task,name,label in [('routing','routing-results.html','Route your own request'),('privacy','redaction-results.html','Try the name detector'),('receipts','receipt-results.html','Try a receipt total')]:
        links([name],'practical-demo.html?task='+task,label)
    links(['maze-results.html','maze-benchmark.html'],'maze-live.html','Run the trained maze policy','Live small neural policy; earlier Qwen recordings remain labelled separately.')
    links(['coordinates.html','coordinate-results.html','visual-refusal-results.html','coordinate-lab.html','screenshot-demo.html'],'point-demo.html','Find a point in your own screenshot','Live OCR and text matching; a simpler alternative to the GUI-Actor model used in the recordings.')
    links(['search-next-results.html'],'rerank-demo.html','Try the actual reranker')
    links(['getting-started.html','results.html','future-explorations.html','roadmap.html'],'demo-directory.html','Choose a live demo')
    for name,item in mapping.items():
        body=re.sub(r'<section id="live-demo-link">.*?</section>','',pages[name].get('body',''),flags=re.S)
        pages[name]['body']='<section id="live-demo-link"><div class="actions"><a class="button" href="@/'+item['target']+'">'+item['label']+'</a></div>'+('<p class="small">'+item['note']+'</p>' if item['note'] else '')+'</section>'+body
    directory=[('demo.html','Learn your preferences','Rate drawings; train a model here.'),('method-demo.html','Compare decision methods','Fixed depth, adaptive stopping, two-model fallback, training and INT8.'),('shared-decision-demo.html','Share the first two layers','Run the saved adaptive model and full-depth control.'),('execution-demo.html','Reuse instructions and batch work','Actual Qwen processing; larger download.'),('practical-demo.html','Routing, names and receipt totals','Edit the input; run trained task classifiers.'),('search-demo.html','Keyword and meaning search','Search real scientific abstracts.'),('rerank-demo.html','Check search results again','Run the same TinyBERT reranker as the study.'),('point-demo.html','Screenshot to coordinate','Local OCR locates visible text in your image.'),('image-action-demo.html','Image to direction','Small learned model recognizes synthetic arrows.'),('maze-live.html','Maze decisions','Run the trained policy and edit the walls.'),('moderation-benchmark.html','Direct output versus a reply','Qwen on real messages: scores, one token or JSON.')]
    body='<p>These run real inference or learning on your device. Each demo explains where its model differs from the research report.</p><ul class="page-links">'+''.join('<li><a href="@/'+n+'">'+title+'</a><p>'+desc+'</p></li>' for n,title,desc in directory)+'</ul><p><a href="@/screenshot-demo.html">Recorded GUI-Actor screenshots</a> and <a href="@/maze-benchmark.html">earlier maze replays</a> remain available as recordings.</p>'
    pages['demo-directory.html']=dict(title='Run a demo',description='Choose a task or an approach, then try it yourself.',body=body)
    pages['demo-directory.html']['body']+='<p><a href="@/preference-method-demo.html">Compare preference models and example selection</a></p>'
    body=re.sub(r'<section id="demo-directory-link">.*?</section>','',pages['sitemap.html']['body'],flags=re.S)
    pages['sitemap.html']['body']='<section id="demo-directory-link"><p><a href="@/demo-directory.html">Browse all live demos</a></p></section>'+body
    (ROOT/'content/demo-coverage.json').write_text(json.dumps(mapping,indent=2)+'\n',encoding='utf-8',newline='\n')

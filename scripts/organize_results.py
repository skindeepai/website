"""Build focused, plain-language research pages from existing measured artifacts."""
import json
from html import escape as e
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOXIC = 'https://huggingface.co/datasets/lmsys/toxic-chat'
SCREEN = 'https://huggingface.co/datasets/bevaya/ScreenSpot'


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def link(href, label):
    return f'<a href="{href}">{e(label)}</a>'


def table(headers, rows, row_links=None):
    row_links = row_links or {}
    return '<div class="table-scroll"><table><thead><tr>' + ''.join('<th scope="col">'+e(v)+'</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<th scope="row">'+(link('@/'+row_links[v],str(v)) if v in row_links else e(str(v)))+'</th>' if i == 0 else '<td>'+e(str(v))+'</td>' for i, v in enumerate(row))+'</tr>' for row in rows)+'</tbody></table></div>'


def refs(items):
    return '<p class="small">'+ ' · '.join(link('@/'+p, label) for p, label in items) + '</p>'


def flow(items, caption):
    return '<figure class="method-figure"><ol class="method-flow">'+''.join('<li>'+e(v)+'</li>' for v in items)+'</ol><figcaption>'+e(caption)+'</figcaption></figure>'


def choices(items):
    return '<div class="topic-links">'+''.join('<a href="@/'+path+'"><span><strong>'+e(title)+'</strong>'+e(description)+'</span></a>' for path, title, description in items)+'</div>'


def blocks(depth):
    return '<span class="processing-steps" aria-hidden="true">'+''.join('<i'+(' class="used"' if i < depth else '')+'></i>' for i in range(24))+'</span>'


def layers(rows, caption):
    return '<figure class="method-figure layer-figure">'+''.join(f'<div class="layer-row"><span>{e(label)}</span>{blocks(d)}<strong>{e(note)}</strong></div>' for label, d, note in rows)+'<figcaption>'+e(caption)+'</figcaption></figure>'


def sample():
    return '<p class="study-sample">'+link(TOXIC, 'Dataset: ToxicChat0124')+' · 100 previously inspected messages: 50 toxic, 50 benign. A small exploratory test, not live moderation.</p>'


def maze_picture():
    episode=read('results/maze-smoke/episodes.json')['all_train_goal_pairs_masked'][0]
    layout=read('results/maze-actions/protocol.json')['mazes'][episode['maze']]
    move=episode['steps'][0]
    boards=[]
    for position in [move['position'],move['next_position']]:
        cells=[]
        for i in range(16):
            kind='wall' if i in layout['walls'] else 'player' if i==position else 'goal' if i==episode['goal'] else 'empty'
            cells.append(f'<span class="maze-cell {kind}">'+({'wall':'#','player':'P','goal':'G'}.get(kind,''))+'</span>')
        boards.append('<div class="mini-maze">'+''.join(cells)+'</div>')
    direction=['UP','RIGHT','DOWN','LEFT'][move['prediction']]
    return '<figure class="method-figure"><div class="maze-example" role="img" aria-label="Recorded first move: the player moves down from the top-right cell. P is player, G is goal, and hash marks are walls.">'+boards[0]+'<strong>'+direction+'<br>→</strong>'+boards[1]+'</div><figcaption>First recorded maze, before and after one move. P = player; G = goal; # = wall. This policy reached the goal in '+str(episode['actions'])+' moves on this example.</figcaption></figure>'


def page(pages, name, title, description, body, parent='decision-results.html', back=None):
    breadcrumb = refs([back]) if back else refs([('results.html', 'Test results')]+([(parent, 'Decision methods')] if parent else []))
    pages[name] = dict(title=title, description=description, status='Research results', styles=['results.css'], body=breadcrumb+body)


def update(pages, legacy=None):
    if legacy is not None or 'results-record.html' not in pages:
        pages['results-record.html'] = dict(title='Earlier combined results', description='The previous report, preserved as a reference. Start with the topic pages for shorter explanations.', status='Reference', body=refs([('results.html', 'Results by topic')])+(legacy or pages['results.html']['body']))
    heads=read('results/chat-smoke-heads/result.json')
    mlp=heads['mlp']
    gate=mlp['learned']['splits']['evaluation']
    runtime=read('results/chat-smoke-heads/runtime.json')
    specialist=read('results/chat-smoke-specialist/result.json')['splits']['evaluation']
    timing=read('results/chat-smoke-specialist/benchmark.json')['total_seconds']
    initial=read('results/chat-smoke-adaptation/initial-frozen-baseline.json')
    adapted={v:read(f'results/chat-smoke-adaptation/{v}.json') for v in ['full','joint','distill','fixed12']}

    pages['results.html'] = dict(title='Test results', description='Choose a question. Each comparison explains the method, the data and what happened.', status='Research', body=''.join('<section id="'+anchor+'"><h2>'+title+'</h2>'+choices(items)+'</section>' for anchor,title,items in [
        ('preferences','Learning what you like',[('preference-results.html','Can a model learn your taste?','Tests of learning from ratings and changing a design.')]),
        ('decisions','Decisions and computation',[('decision-results.html','How much processing does a decision need?','Stopping at different layers, checking confidence, trying a smaller model, and training changes.')]),
        ('coordinates','Finding a place to click',[('coordinate-results.html','Can a model return a point directly?','Screenshot targets, neighbouring image patches and missing buttons.')]),
        ('mazes','Choosing an action',[('maze-results.html','Can a small model finish a maze?','More training examples, legal moves and the failures that remain.')])])+refs([('results-record.html','Earlier combined report'),('docs/claims.md','Evidence and limitations'),('research.html','Open questions')]))

    distill_time=read('results/chat-smoke-adaptation/distill-runtime.json')
    comparison=sample()+'<h2>Speed and accuracy</h2>'+table(['Method','Correct / 100','Time / message'],[
        ['Full Qwen',mlp['fixed_24']['splits']['evaluation']['correct'],f'{runtime["paths"]["full"]["mean_ms"]:.0f} ms'],
        ['Stop at layer 12',mlp['fixed_12']['splits']['evaluation']['correct'],'Not timed'],
        ['Learned stopping',gate['correct'],f'{runtime["paths"]["learned"]["mean_ms"]:.0f} ms'],
        ['Tiny model only',specialist['specialist']['correct'],f'{1000*timing["specialist"]/100:.1f} ms'],
        ['Tiny model → Qwen',specialist['cascade']['correct'],f'{1000*timing["cascade"]/100:.0f} ms'],
        ['With distillation',adapted['distill']['metrics']['evaluation']['24']['correct'],f'{1000*distill_time["seconds"]/distill_time["n"]:.0f} ms'],
    ],{'Full Qwen':'depth-results.html','Stop at layer 12':'depth-results.html','Learned stopping':'early-exit-results.html','Tiny model only':'cascade-test-results.html','Tiny model → Qwen':'cascade-results.html','With distillation':'training-results.html'})
    comparison+='<p>The learned stop reduced time by about 34% against its full-model reference. The tiny-model cascade reduced it by about 50% against its own Qwen reference, but introduced one new toxic-message miss.</p><p class="small">Same 100 messages for accuracy; training differs between methods. Warm CPU time includes input preparation: 50 timed messages for distillation, 100 for the other timed rows. The cascade’s own Qwen reference scores 78/100 at 648 ms.</p><h2>How each approach works</h2>'
    page(pages,'decision-results.html','Ways to make a decision sooner','Compare the measured speed and accuracy, then open a method for its test results.',comparison+choices([
        ('depth-results.html','1. Stop at a fixed point','Use the first quarter, half or three quarters of Qwen. Compare the mistakes at each depth.'),
        ('early-exit-results.html','2. Check before continuing','A small checker decides whether to return a label or run more layers. Different messages can stop at different points.'),
        ('cascade-results.html','3. Try a tiny model first','Accept its confident answers. Send the other messages to a larger model.'),
        ('training-results.html','4. Teach the model differently','Train intermediate answers, learn from another model, or remove later layers.')])+refs([('decision-checks.html','Other datasets and stress tests'),('chat-results.html','Original 600-message comparison'),('browser-benchmark.html','Try output formats in your browser')]),back=('adaptive.html','Back to stopping early'))

    body=sample()+layers([(f'Layer {d} of 24',d,f'{100*(24-d)//24}% skipped') for d in [6,12,18,24]],'Each square is one transformer block. Filled squares run; outlined squares are skipped.')+'<p>A small classifier reads Qwen’s internal numbers at the chosen layer and returns SAFE or BLOCK. It does not write a reply.</p><h2>Earlier is cheaper. Accuracy is uneven.</h2>'
    body+=table(['Stop after','Correct / 100','Toxic missed / 50'],[[f'{label}: layer {d}',mlp[f'fixed_{d}']['splits']['evaluation']['correct'],mlp[f'fixed_{d}']['splits']['evaluation']['missed_toxic']] for d,label in [(6,'Early'),(12,'Halfway'),(18,'Late'),(24,'Full model')]])
    body+='<p>These use the same small neural classifier design at each checkpoint. Layer 18 did slightly better than layer 24 here; a one-message difference is not a reliable ranking.</p><p class="small">Skipped means transformer blocks, not the same percentage of model parameters or elapsed time. These fixed-depth rows come from stored internal states; each depth was not separately timed.</p><details><summary>Linear classifiers and the larger test</summary>'+table(['Layer','Linear correct / 100','Toxic missed / 50'],[[d,heads['linear'][f'fixed_{d}']['splits']['evaluation']['correct'],heads['linear'][f'fixed_{d}']['splits']['evaluation']['missed_toxic']] for d in [6,12,18,24]])+'<p>The nonlinear classifier has 896 inputs, 64 hidden units and two outputs. The linear classifier maps the same 896 numbers directly to two outputs. Both learn from 384 messages; Qwen stays frozen.</p>'+refs([('chat-results.html','All four depths on 600 messages'),('docs/chat-smoke-heads.md','Training and every comparison'),('results/chat-smoke-heads/result.json','Recorded results')])+'</details>'+refs([('early-exit-results.html','Let each message stop at a different point')])
    page(pages,'depth-results.html','Where should processing stop?','Qwen has 24 transformer layers. Here is what happens when we stop earlier.',body)

    body=sample()+flow(['Read through a checkpoint','Predict SAFE or BLOCK','Stop, or continue to the next checkpoint'],'The stopping checker uses information available so far. It cannot see the final answer in advance.')+'<h2>Where the learned checker stopped</h2>'+layers([(f'{gate["exit_counts"][str(d)]} messages: layer {d}',d,f'{100*(24-d)//24}% skipped') for d in [6,12,18,24]],'100 messages in total. The chosen depth varies by message.')
    body+=f'<p>On average, it used <strong>{gate["mean_depth"]:.1f} of 24 layers</strong>, skipping <strong>{100*gate["projected_blocks_skipped"]:.2f}% of transformer blocks</strong>. We verified that later blocks really did not execute.</p>'+table(['Same 100 messages','All 24 layers','Learned stopping'],[['Correct',77,79],['Toxic missed / 50',18,15],['Total request time',f'{runtime["paths"]["full"]["sum_seconds"]:.1f} s',f'{runtime["paths"]["learned"]["sum_seconds"]:.1f} s']])+'<p>It corrected three mistakes but introduced one new false block. This is promising exploration; it has not passed a fresh reliability test.</p><h2>Three ways to decide when to stop</h2>'
    body+=table(['Checker','Correct / 100','Blocks skipped'],[[label,mlp[k]['splits']['evaluation']['correct'],f'{100*mlp[k]["splits"]["evaluation"]["projected_blocks_skipped"]:.2f}%'] for k,label in [('asymmetric','Separate SAFE / BLOCK confidence limits'),('agreement','Confidence + agreement with the previous checkpoint'),('learned','Learn when an answer may be wrong')]])
    body+='<p>Only the learned-checker row above has this new measured runtime comparison. The other savings are block-count estimates.</p><details><summary>What the checker learns, and what failed</summary><p>Two logistic classifiers look at confidence, uncertainty and changes between checkpoints. They predict whether the current answer is wrong, and whether continuing would fix it. Full-model answers supply training targets only.</p><p>The same gate recipe reduced accuracy with a linear answer classifier: 77 to 75 correct. A separate REVIEW option answered only 64 messages, with 56 correct; the remaining 36 were unresolved.</p><p>Timing: one warmed, alternating paired pass on a CPU with four threads. Includes input preparation, classifier and checker; excludes model loading. Every exit trace matched the recorded policy.</p>'+refs([('docs/chat-smoke-heads.md','All variants and training'),('results/chat-smoke/protocol.json','Exact sample IDs'),('results/chat-smoke-heads/runtime.json','Runtime record'),('docs/chat-smoke-review.md','Adversarial review')])+'</details>'
    page(pages,'early-exit-results.html','Check whether more layers are needed','Keep processing uncertain messages. Stop earlier on others.',body)

    body=sample()+flow(['Tiny BERT reads the message','Confident: return its label','Uncertain: Qwen reads the message'],'The fallback starts from the original message. It does not continue BERT’s internal processing.')+f'<p><strong>51 messages avoided Qwen entirely.</strong> The other {specialist["fallback_count"]} ran both the tiny model and all 24 Qwen layers. There is no early exit inside either model.</p>'+table(['Method','Correct / 100','Time for 100'],[[label,specialist[k]['correct'],f'{timing[k]:.2f} s'] for k,label in [('specialist','Tiny model only'),('cascade','Tiny model, then Qwen if needed'),('full_qwen_reference','Qwen for every message')]])+'<p>The cascade roughly halved time. It missed 14 toxic messages versus Qwen’s 15, but one of those misses was a message Qwen got right.</p><details><summary>Models, confidence and timing</summary><p>The specialist is a trained 4.37-million-parameter, two-layer BERT. The fallback is Qwen2.5-0.5B with a trained label classifier. BERT and Qwen layers are different sizes; their layer counts are not directly comparable.</p><p>Development examples select separate confidence limits for SAFE and BLOCK. BERT learned from 384 messages; the older Qwen fallback classifier learned from 1,400. This is not an equal-training comparison of architectures.</p><p>One warm CPU pass, 100 calls per path, rotating order. Timing includes input preparation and all fallback work. Loading is excluded and both models are already in memory.</p>'+refs([('docs/chat-smoke-specialist.md','Training and evidence'),('results/chat-smoke-specialist/timings.json','Every timed request'),('docs/chat-smoke-review.md','Review')])+'</details>'
    body=body.replace('<details><summary>Models, confidence and timing</summary>','<p class="next-links">'+link('@/cascade-test-results.html','See the cascade test results →')+'</p><details><summary>Models, confidence and timing</summary>')
    page(pages,'cascade-results.html','Try a tiny model first','Use the larger model only when the small one is uncertain.',body,back=('decision-results.html','Back to the decision comparison'))

    predictions=[r for r in read('results/chat-smoke-specialist/predictions.json') if r['split']=='evaluation']
    timed={r['id']:r for r in read('results/chat-smoke-specialist/timings.json') if r['path']=='cascade'}
    body=sample()+'<h2>The complete comparison</h2>'+table(['Method','Correct / 100','Toxic missed / 50','Benign blocked / 50','Time for 100'],[[label,specialist[k]['correct'],specialist[k]['missed_toxic'],specialist[k]['false_block'],f'{timing[k]:.2f} s'] for k,label in [('specialist','Tiny model only'),('cascade','Tiny model → Qwen'),('full_qwen_reference','Qwen for every message')]])
    body+='<p>The cascade corrected three Qwen mistakes and introduced two new ones, including one toxic-message miss. Its higher total accuracy does not remove that trade-off.</p><h2>Which model answered?</h2>'+table(['Route','Messages','Correct'],[[label,len(part),sum(r['cascade']==r['label'] for r in part)] for fallback,label in [(False,'Tiny model alone'),(True,'Tiny model, then Qwen')] for part in [[r for r in predictions if r['fallback']==fallback]]])
    body+='<p>Every message ran BERT’s two layers. Only the 49 fallback messages ran Qwen’s 24 layers; 51 avoided Qwen entirely.</p><details><summary>Inspect all 100 decisions</summary><p>SAFE and BLOCK follow the dataset’s non-toxic and toxic labels. IDs identify source rows; raw chat text is not republished. Time includes both models when Qwen was needed.</p>'+table(['Message ID','Dataset label','Cascade answer','Answered by','Request time'],[[r['id'],'BLOCK' if r['label'] else 'SAFE','BLOCK' if r['cascade'] else 'SAFE','Qwen' if r['fallback'] else 'Tiny model',f'{1000*timed[r["id"]]["seconds"]:.1f} ms'] for r in predictions])+'</details><details><summary>How this test was run</summary><p>The tiny model trained on 384 messages; the older Qwen classifier trained on 1,400. Separate development examples selected the confidence thresholds. These 100 evaluation messages had already been inspected in earlier work.</p><p>Timing uses one warm CPU pass per method, rotating their order for each message. Input preparation and actual fallback work are included; loading is excluded. This is exploratory evidence, not validated moderation performance.</p></details><h2>Source records</h2>'+refs([('results/chat-smoke/protocol.json','Exact sample IDs'),('results/chat-smoke-specialist/predictions.json','Download all predictions'),('results/chat-smoke-specialist/timings.json','Download timings'),('docs/chat-smoke-specialist.md','Training and reproduction'),('docs/chat-smoke-review.md','Adversarial review')])
    page(pages,'cascade-test-results.html','Cascade: test results','The recorded decisions, mistakes and processing time for the tiny-model fallback experiment.',body,back=('cascade-results.html','Back to the cascade explanation'))

    rows=[]
    for v,label,d in [('full','Train the final answer',24),('joint','Train answers at four depths',24),('distill','Also learn from a teacher model',24),('fixed12','Remove the last 12 layers, then train',12)]:
        m=adapted[v]['metrics']['evaluation'][str(d)]
        rows.append([label,m['correct'],m['missed_toxic'],f'{100*(24-d)//24}%'])
    body=sample()+flow(['Examples with known labels','Update small parts of Qwen','Read an answer at chosen layers'],'Training changes the model. A separate stopping rule is still needed to save work on individual messages.')+'<h2>Four short training experiments</h2>'+table(['Training recipe','Correct / 100','Toxic missed / 50','Blocks skipped'],rows)+'<p>All recipes use 24 layers except the truncated model, which uses 12. Before training, the final-layer classifier got 77/100 right and missed 16 toxic messages; the layer-12 classifier got 71/100 right and missed 17.</p><h2>Learning from a teacher: distillation</h2><p>Alongside the known labels, a second trained model supplies scores to learn from. This raised the final answer to 80/100, but did not make the earliest answers better.</p>'+table(['Layer','Before training','With distillation'],[[d,initial['metrics']['evaluation'][str(d)]['correct'],adapted['distill']['metrics']['evaluation'][str(d)]['correct']] for d in [6,12,18,24]])+'<p class="small">Correct answers out of the same 100. The main distillation result uses all 24 layers: 0% skipped.</p><details><summary>Each recipe, limitations and failed follow-up</summary><p>Task adaptation trains the final answer. Joint training also rewards correct answers at layers 6, 12 and 18. Distillation adds the teacher’s scores. Truncation physically keeps only the first 12 blocks.</p><p>These use small LoRA weight updates, not full-model retraining: 32 batches of four messages after fitting the initial classifiers. A 32-message development subset selects the checkpoint. The teacher uses the same training labels and is not an independent oracle.</p><p>A separately recorded attempt with less initial classifier training also failed: final accuracy fell from 78 to 69, while toxic misses rose from 8 to 26. These small budgets do not establish the limits of the methods.</p>'+refs([('docs/chat-smoke-adaptation.md','All layers, timings and exact training'),('docs/chat-smoke-results.md','Every exploratory outcome'),('docs/chat-smoke-review.md','Review')])+'</details>'
    page(pages,'training-results.html','Teach useful answers to appear earlier','Changing the readout is one option. Changing what the model learns is another.',body)

    chat=read('results/chat600/result.json');bench=read('results/chat600/benchmark.json')['paths']
    body='<p class="study-sample">'+link(TOXIC,'Dataset: ToxicChat0124')+' · 600 archived, human-annotated messages: 81 toxic and 519 benign. No live moderation.</p>'+layers([('Chosen shortcut: layer 12',12,'50% of blocks skipped')],'All 600 messages stopped at the same point. This was not an adaptive readiness detector.')+table(['Depth','Correct / 600','Toxic missed / 81'],[[f'{d} of 24 layers',chat['fixed_heads'][str(d)]['correct'],chat['fixed_heads'][str(d)]['missed_toxic']] for d in [6,12,18,24]])+f'<p>The selected layer-12 path took {bench["early_candidate"]["median_total_seconds"]:.1f} seconds versus {bench["trained_full_token"]["median_total_seconds"]:.1f} seconds at full depth. It missed seven more toxic messages and failed the reliability check.</p><p class="small">Median of three complete passes per method, including input preparation. Full depth returns one trained label token. Only the selected shortcut and full-depth paths have this timing comparison; the table does not imply measured times at every depth.</p>'+refs([('docs/chat600-results.md','Full results'),('docs/chat600-protocol.md','Sampling and training'),('docs/chat600-review.md','Independent audit'),('early-exit-results.html','Later adaptive follow-up')])
    page(pages,'chat-results.html','The original 600-message test','A real-message workload showed both the savings and the extra mistakes.',body)

    screen=read('results/screenspot/result.json')
    body='<p class="study-sample">'+link(SCREEN,'Dataset: ScreenSpot')+' · 30 public screenshots, with a requested target and its marked location.</p>'+flow(['Screenshot + request','Score the image patches','Return a point (x, y)'],'The published GUI-Actor model returns a location without generating coordinate text.')+table(['How the point is chosen','Targets hit / 30'],[['Highest-scoring single patch',screen['hits']['max_patch']],['Combine neighbouring high-scoring patches',screen['hits']['connected_region']]])+'<p>Combining nearby patches helped on this small sample. The model still ran <strong>all 28 transformer layers: 0% skipped</strong>. It generated no text tokens; that does not make image understanding free.</p><h2>What if the button is missing?</h2><p>On two local screenshots, a confidence cutoff rejected both missing targets and kept all six present targets. It also kept one wrong point. Eight reused examples are too few to establish reliable rejection.</p>'+refs([('coordinate-lab.html','See the six local recorded targets')])+'<details><summary>Model, subsets and limitations</summary><p>This reproduces Microsoft’s pretrained GUI-Actor 2B pointer head. We did not train a new vision model. Both patch-selection methods share the same model pass, so this is a location comparison, not a measured speed advantage over JSON.</p><p>The missing-target check uses the other screenshot to choose each cutoff. It is a confidence diagnostic on a repeated interface, not a learned presence classifier. The 30 ScreenSpot cases all have visible targets.</p>'+refs([('docs/screenspot-results.md','Every public screenshot result'),('docs/screenspot-protocol.md','Sample selection'),('docs/coordinate-abstention-smoke.md','Missing-target diagnostic'),('results/coordinates/result.json','Original local run')])+'</details>'
    page(pages,'coordinate-results.html','Finding the right point','Return a location directly, then check whether it actually hits the target.',body,parent=None)

    maze=read('results/maze-smoke/result.json')
    rows=[]
    for k,label in [('original_states_unmasked','256 training states'),('original_states_masked','256 states + forbid wall moves'),('all_train_goal_pairs_unmasked','4,992 training states'),('all_train_goal_pairs_masked','4,992 states + forbid wall moves'),('bfs','Shortest-path search')]:
        m=maze[k]['episodes'];rows.append([label,f'{m["goals_reached"]} / {m["n"]}'])
    body='<p class="study-sample">Generated task: ten 4 × 4 mazes, reused for exploration. '+link('@/results/maze-actions/protocol.json','Layouts and protocol')+' · '+link('@/experiments/maze_actions.py','Maze generator')+'.</p>'+flow(['Walls, player and goal','Four scores: ↑ → ↓ ←','Choose a move; repeat'],'The small network reads the grid directly. It does not describe a move in text.')+table(['Method','Goals reached'],rows)+'<p>More examples helped. Forbidding moves through walls helped further, reaching 7/10 goals. The remaining failures were legal moves repeated in loops.</p><p>This is a small, separate network, not a Qwen early-exit result. No Qwen layer-skipping or timing saving is claimed here.</p><details><summary>Training and the original failed run</summary><p>The network maps 48 grid values through two 128-unit hidden layers to four action scores. A search algorithm supplies training labels. More training states come from the same 32 training layouts; the ten evaluation layouts are different but previously inspected.</p><p>The optional legal-move rule sees walls and boundaries, not the correct path. Search remains the stronger baseline on these simple fully visible mazes.</p><p>The original Qwen policies both reached 0/10 goals. The existing browser replay shows those original failures, not the later 7/10 policy.</p>'+refs([('maze-benchmark.html','Replay the original Qwen failures'),('docs/maze-smoke.md','Later training results and trajectories'),('results/maze-smoke/episodes.json','Follow-up episode records')])+'</details>'
    body=body.replace(flow(['Walls, player and goal','Four scores: ↑ → ↓ ←','Choose a move; repeat'],'The small network reads the grid directly. It does not describe a move in text.'),maze_picture()+'<p>The small network reads walls, player and goal, then returns one of four moves. It does not write directions in text.</p>')
    page(pages,'maze-results.html','From a decision to a finished task','A correct-looking move is not enough. The agent needs to reach the goal.',body,parent=None)

    syn=read('results/synthetic/result.json')
    body=flow(['Rate examples','Learn which settings matter','Score or adjust another example'],'The browser demo learns from settings behind drawings, rather than from image pixels.')+f'<p>{syn["math"]["browser_fixtures"]} browser calculation checks passed. In {syn["math"]["random_cases"]} generated bounded-edit cases, the solution agreed with an independent numerical solver.</p><p>On made-up preferences, the model learns simple patterns. It can fail when someone likes two separated styles or when a high predicted score does not match their real taste.</p><p><strong>We have not yet shown that people prefer its suggestions.</strong> The calculation tests verify the mechanics, not that human outcome.</p>'+refs([('demo.html','Try your own ratings'),('how-it-works.html','See how learning works')])+'<details><summary>Data and detailed tests</summary><p>These are generated numerical fixtures and synthetic preference rules, not a human-rating dataset.</p>'+refs([('results/synthetic/result.json','Five-seed learning results'),('results/synthetic/math-fixtures.json','Numerical fixtures'),('results-record.html#preferences','Original tables and comparisons'),('docs/method.md','Mathematics')])+'</details>'
    page(pages,'preference-results.html','Does preference learning work?','The calculations work. Whether the suggestions are useful still needs a human test.',body,parent=None)

    body='<h2>Public banking questions</h2><p>On all 3,080 BANKING77 test queries, one trained Qwen classifier identified 2,551 topics correctly. Early stopping had similar total accuracy but failed the stricter error check.</p><p>'+link('https://github.com/PolyAI-LDN/task-specific-datasets','BANKING77 dataset')+'</p>'+refs([('docs/banking77-results.md','All depths, seeds and timing'),('docs/conservative-exits.md','Stricter follow-up')])+'<h2>Unfamiliar requests and changing rules</h2><p>A second dataset also failed the early-stop check. Adding an UNKNOWN answer did not reliably handle unfamiliar requests. In a separate test with opposite rules, the full model answered both correctly on only 14 of 100 pairs.</p><p>'+link('https://github.com/clinc/oos-eval','CLINC dataset')+'</p>'+refs([('docs/clinc-results.md','Second dataset'),('docs/clinc-unknown.md','UNKNOWN output'),('docs/changing-rules.md','Changing-rule test')])+'<h2>A label versus a written reply</h2><p>A number and a single trained output token can use the same classifier. Our matched test did not establish a reliable speed advantage for the number alone; avoiding transformer work is a different question.</p>'+refs([('docs/matched-output.md','Matched output comparison'),('browser-benchmark.html','Try one-letter and JSON output'),('docs/reproduction.md','Separate implementation replay')])
    page(pages,'decision-checks.html','Other decision tests','Public data and harder cases show where the shortcuts break.',body)

    # Keep existing technical explanation while replacing the vague opening.
    old=pages['how-it-works.html']['body'];detail=old[old.index('<details'):]
    mug='<svg viewBox="0 0 64 64" width="64" height="64" aria-hidden="true"><path d="M12 14h32v29q0 10-12 10h-8q-12 0-12-10z" fill="COLOR" stroke="#263b57" stroke-width="2"/><path d="M44 21h5c15 0 13 21-5 21" fill="none" stroke="#263b57" stroke-width="4"/></svg>'
    example='<figure class="method-figure"><div class="rating-example"><div>'+mug.replace('COLOR','#709cd1')+'<strong>Like</strong><span>Blue, wide</span></div><div>'+mug.replace('COLOR','#c98676')+'<strong>Pass</strong><span>Red, wide</span></div></div><figcaption>Illustration: the shape stays the same, so these ratings suggest colour matters. Real preferences need more examples.</figcaption></figure>'
    body='<h2>1. Start with the settings behind a drawing</h2><p>A drawing can be described by a few numbers: colour, width, roundness. A renderer turns those settings into the picture you see.</p>'+flow(['Settings: blue, wide, round','Renderer draws the mug','You choose Like or Pass'],'For a learned image generator, similar internal numbers are called a latent vector; its controls are usually less tidy.')+'<h2>2. Learn which settings go with your ratings</h2>'+example+'<p>The small preference model learns a weight for each setting. A setting associated with likes raises its score; one associated with passes lowers it. Each rating updates those weights.</p><h2>3. Use that score to make a choice</h2>'+flow(['New settings','Preference model scores them','Rank, suggest, or make a small edit'],'To suggest something, search the allowed settings for a high score, then render those settings as a picture.')+'<p>The generator makes the image. The preference model predicts whether you will like it. Your next rating is the check on that prediction.</p>'+refs([('demo.html','Try the drawing demo'),('preference-results.html','What the tests show')])+detail
    pages['how-it-works.html'].update(body=body,styles=['results.css'])

    replacements={'@/results.html#coordinates':'@/coordinate-results.html','@/results.html#decisions':'@/decision-results.html','@/results.html#preferences':'@/preference-results.html'}
    for name,p in pages.items():
        if name=='results-record.html':continue
        for old,new in replacements.items():p['body']=p.get('body','').replace(old,new)
    pages['decisions.html']['body']=pages['decisions.html']['body'].replace('First test results','Compare the methods')
    pages['adaptive.html']['body']=pages['adaptive.html']['body'].replace('href="@/results.html">See the comparison','href="@/decision-results.html">Compare the approaches')
    # Make the new work discoverable without expanding the topic introduction.
    marker='<details class="plain-details">'
    intro='<p>Later small tests tried different stopping depths, learned confidence checks and a tiny model with Qwen as fallback. '+link('@/decision-results.html','See how the methods compare')+'.</p>'
    pages['adaptive.html']['body']=pages['adaptive.html']['body'].replace(intro,'').replace(marker,intro+marker,1)
    section='<section id="focused-results"><h2>Results by method</h2><ul class="page-links">'+''.join('<li>'+link('@/'+n,pages[n]['title'])+'</li>' for n in ['preference-results.html','decision-results.html','depth-results.html','early-exit-results.html','cascade-results.html','cascade-test-results.html','training-results.html','chat-results.html','decision-checks.html','coordinate-results.html','maze-results.html'])+'</ul></section>'
    site=pages['sitemap.html']['body'];start=site.find('<section id="focused-results">')
    if start>=0:site=site[:start]
    pages['sitemap.html']['body']=site+section


if __name__ == '__main__':
    pages=read('content/pages.json');update(pages)
    (ROOT/'content/pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Organized topic and method result pages.')

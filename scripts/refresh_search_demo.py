"""Add the live, local search demo and its measured pilot summary."""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update(pages):
    result = json.loads((ROOT / 'results/search-ranking/result.json').read_text(encoding='utf-8'))
    names = {'bm25': 'Keywords (BM25)', 'minilm': 'Meaning (MiniLM)', 'hybrid': 'Combined ranking'}
    rows = ''.join('<tr><th scope="row">' + names[key] + '</th><td>' + str(values['correct_first']) + ' / 100</td><td>' + f'{values["latency_ms"]["median"]:.1f} ms' + '</td></tr>' for key, values in result['metrics'].items())
    learned_path = ROOT / 'results/search-ranking/learned/result.json'
    learned_note = ''
    if learned_path.exists():
        learned = json.loads(learned_path.read_text(encoding='utf-8'))
        rows += '<tr><th scope="row">Learned combination</th><td>' + str(learned['metrics']['correct_first']) + ' / 100</td><td>Not timed</td></tr>'
        learned_note = '<p>A small learned combination reached ' + str(learned['metrics']['correct_first']) + '/100 after training on 600 separate claims and selecting its settings on another 209. This is an exploratory repeat of the same test, not fresh confirmation. It improved first results but found slightly fewer relevant papers in the top five than the fixed combination. <a href="@/docs/search-ranking.md#learned-combination-follow-up">Method and tradeoffs</a>.</p>'
    pages['search-demo.html'] = {
        'title': 'Search real research',
        'description': 'Find papers with keywords or a small model that compares meaning.',
        'status': 'Live local demo',
        'scripts': ['scripts/search-demo.js'],
        'body': '''<p>Search <strong>5,183 real scientific abstracts</strong>. Type a question or topic. Results are documents, not a generated answer.</p>
<form id="search-form"><label for="search-query">Your search</label><input id="search-query" type="search" maxlength="1000" value="exercise and inflammation" style="display:block;width:100%;max-width:100%;padding:10px;margin:8px 0 12px" required><div class="actions"><button class="button" id="search-run" type="submit">Search keywords</button><button class="button secondary" id="search-compare" type="button">Compare meaning</button><button class="button secondary" id="search-stop" type="button" disabled>Stop</button></div></form>
<p class="small">Keyword search loads an 8 MB collection. Comparing meaning also loads a 23 MB MiniLM model and 8 MB of document vectors. Your query stays on this device.</p>
<p id="search-status" role="status">Ready. Nothing is downloaded until you search.</p><p id="search-timing" class="small"></p>
<div class="split"><section><h2>Keywords</h2><p class="small">BM25: matches words in titles and abstracts.</p><ol id="search-keywords" style="padding-left:20px"></ol></section><section><h2>Meaning</h2><p class="small">MiniLM: compares the query's numerical representation with each document.</p><ol id="search-meaning" style="padding-left:20px"></ol></section></div>
<details class="plain-details"><summary>Combine the two rankings</summary><h3>Fixed combination</h3><p>Reciprocal-rank fusion combines both lists with a fixed formula.</p><ol id="search-combined" style="padding-left:20px"></ol><h3>Learned combination</h3><p>Five learned weights combine word scores, meaning scores and ranks. Trained on separate SciFact claims; it does not learn from your query.</p><ol id="search-learned" style="padding-left:20px"></ol></details>
<h2>What happened on 100 test queries?</h2><p>Each method searched the same complete 5,183-document collection. “Relevant first” means the first paper was judged to contain evidence about the claim; it may support or refute it.</p>
<div class="table-scroll" role="region" tabindex="0" aria-label="Search pilot results"><table><thead><tr><th scope="col">Method</th><th scope="col">Relevant first</th><th scope="col">Query time</th></tr></thead><tbody>''' + rows + '''</tbody></table></div>
<p class="small">Exploratory sample of 100 official SciFact test claims. Warm local CPU medians, three repeats, including query encoding and ranking; document indexing and model loading are separate. These Python timings are not browser timings. No early exit or quality-preserving speedup is claimed.</p>''' + learned_note + '''
<details class="plain-details"><summary>Data, model and limitations</summary><p><a href="https://github.com/allenai/scifact">SciFact</a> contains real paper abstracts and expert-authored research claims with human evidence judgments. These are not production search logs. The demo and benchmark use the same fixed corpus, not all scientific literature or the web.</p><p><a href="https://huggingface.co/Xenova/all-MiniLM-L6-v2">MiniLM-L6</a> uses six transformer layers and returns a 384-number vector. All six layers run. The model is pretrained; we did not fit or select it on these 100 queries. BM25 sees full abstracts; MiniLM uses at most 256 tokens, which can hide relevant text.</p><p>A close meaning score is not a probability or a factual verdict. Scientific text may have appeared in model pretraining. The test uses benchmark relevance labels, which may not cover every useful paper. The browser uses the same pinned q8 ONNX model and stored document vectors, but its runtime and query timings differ.</p><p>Abstract collection: SciFact / Semantic Scholar S2ORC, <a href="https://opendatacommons.org/licenses/by/1-0/">ODC-By 1.0</a>. Claims and annotations: <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. See the <a href="https://github.com/allenai/scifact/blob/master/LICENSE.md">original license</a>; the BEIR mirror separately lists CC BY-SA 4.0.</p></details>
<p class="small"><a href="@/docs/search-ranking.md">Method, all metrics and setup costs</a> · <a href="@/results/search-ranking/predictions.json">Every test ranking</a> · <a href="@/results/search-ranking/protocol.json">Frozen protocol and sources</a> · <a href="@/experiments/search_ranking.py">Experiment code</a> · <a href="@/scripts/search-demo-worker.js">Browser code</a></p>'''
    }

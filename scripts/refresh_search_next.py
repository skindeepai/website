"""Focused fresh search results; preserve the existing demo and site styling."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def refresh_timing_notes(timing):
    path = ROOT / 'docs/search-next.md'
    text = path.read_text(encoding='utf-8')
    start, end = '<!-- SEARCH_TIMING_START -->', '<!-- SEARCH_TIMING_END -->'
    if start not in text or end not in text:
        return
    names = {'bm25': 'BM25', 'minilm': 'MiniLM', 'fusion': 'Learned combination',
             'rerank': 'Always rerank', 'gated': 'Conditional rerank'}
    lines = ['## Separate timing run: first 50 queries', '',
             'This timing-only run uses the first 50 sorted queries from the completed 200-ID study, after other launched model jobs finished. It does not introduce new quality data or change any method. Quality totals above remain out of 200; timing totals below cover only 50. Operating-system background load is uncontrolled.', '',
             '| Method | Mean ms/query | Median ms/query | 95th percentile ms | Relevant first / 50 |',
             '|---|---:|---:|---:|---:|']
    for key, value in timing['metrics'].items():
        lines.append(f'| {names[key]} | {value["mean_ms"]:.2f} | {value["median_ms"]:.2f} | {value["p95_ms"]:.2f} | {value["correct_first"]} |')
    lines += ['', f'The optional reranker ran on {timing["metrics"]["gated"]["reranked_queries"]}/50 queries here, versus 42/200 in the complete quality set. Three rotating paired repeats produced 750 timed calls, preserving every recorded top-10 ranking and gate choice. Mean times include the slow reranked requests; the median alone hides them.', '',
              'Timers include raw-query input processing, required encoders, retrieval, gate and sorting. Model loading and document indexing are excluded. ONNX numerical pools are capped at two threads, inter-op one. This is warm local CPU timing, not browser or production latency.', '',
              '[Timing protocol](../results/search-next/isolated/protocol.json), [summary](../results/search-next/isolated/result.json), [all 750 measured calls](../results/search-next/isolated/records.json), and [runner](../experiments/search_next_timing.py). The concurrent 200-query measurements below remain retained as the initial diagnostic run.']
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(before + start + '\n' + '\n'.join(lines) + '\n' + end + after, encoding='utf-8', newline='\n')


def update(pages):
    path = ROOT / 'results/search-next/result.json'
    if not path.exists():
        return
    result = json.loads(path.read_text(encoding='utf-8'))
    metrics = result['metrics']
    timing_path = ROOT / 'results/search-next/isolated/result.json'
    timing = json.loads(timing_path.read_text(encoding='utf-8')) if timing_path.exists() else None
    if timing:
        assert timing['queries'] == 50 and timing['all_top10_rankings_match']
        assert all(value['timed_calls'] == 150 for value in timing['metrics'].values())
        refresh_timing_notes(timing)
    time_heading = 'Mean time (50-query subset)' if timing else 'Mean query time'
    names = {'bm25': 'Keywords', 'minilm': 'Meaning', 'fusion': 'Learned combination',
             'rerank': 'Keywords + small second check', 'gated': 'Second check only when uncertain'}
    rows = ''.join('<tr><th scope="row">' + names[key] + '</th><td>' + str(value['correct_first']) + ' / 200</td><td>' + f'{(timing["metrics"][key]["mean_ms"] if timing else value["latency_ms"]["mean"]):.1f} ms' + '</td></tr>' for key, value in metrics.items())
    body = '<p><a href="@/search-demo.html">Try the search demo</a></p><p>A small model can read a search query beside a likely match and give it a relevance score. Here it checks only the first 20 keyword matches.</p>'
    body += '<figure class="explainer"><svg viewBox="0 0 360 118" role="img" aria-label="Search all 5,183 papers with keywords, then a two-layer model checks the best 20 matches." style="max-width:440px;width:100%;height:auto"><g fill="none" stroke="#395775" stroke-width="2"><rect x="2" y="10" width="148" height="76" rx="6"/><path d="M150 48h54m-8-6 8 6-8 6"/><rect x="206" y="10" width="152" height="76" rx="6"/></g><g fill="#152c45" font-family="system-ui,sans-serif" font-size="18" text-anchor="middle"><text x="76" y="39">5,183 papers</text><text x="76" y="65">Keyword search</text><text x="282" y="39">20 matches</text><text x="282" y="65">Small model</text></g></svg><figcaption>The model returns scores, without writing a reply.</figcaption></figure>'
    body += '<h2>200 additional test IDs</h2><p>These are the remaining official SciFact test IDs, excluded from our earlier 100-query experiments. One claim duplicates training text under another ID; removing it leaves 199 queries and the same correct-first totals below. Each search starts with the same collection of real scientific abstracts.</p>'
    body += '<div class="table-scroll" role="region" tabindex="0" aria-label="Fresh search comparison"><table><thead><tr><th scope="col">Method</th><th scope="col">Relevant first</th><th scope="col">' + time_heading + '</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
    if timing:
        body += '<p class="small">Quality totals cover all 200 test IDs. Times cover only the first 50 queries, measured separately after other model experiments finished: warm CPU means, three repeats per query. Input processing and ranking are included; model loading and building the document index are excluded. Background operating-system load is uncontrolled.</p>'
    else:
        body += '<p class="small">Warm CPU mean times, three repeats per query. Input processing and ranking are included; model loading and building the document index are excluded. Other experiments were running on the computer, so these timings are diagnostic, not an isolated speed benchmark.</p>'
    body += '<p><strong>The frozen learned combination performed best here: 119/200.</strong> Compared with keywords, it corrected 15 first results and lost one. It still missed 81 first results.</p>'
    body += f'<p>The optional check ran for <strong>{metrics["gated"]["reranked_queries"]} of 200 queries</strong>. Its rule was selected on 209 separate development claims before these results were measured.</p>'
    if timing:
        gate = timing['metrics']['gated']
        body += f'<p class="small">In the separate timing subset, {gate["reranked_queries"]}/50 queries used the check. Its mean includes those slower queries; the 95th-percentile time was {gate["p95_ms"]:.1f} ms. <a href="@/results/search-next/isolated/result.json">Timing results</a>.</p>'
    else:
        body += f'<p class="small">The mean includes the slower checked queries; the 95th-percentile time was {metrics["gated"]["latency_ms"]["p95"]:.1f} ms in the concurrent diagnostic run.</p>'
    difference = metrics['rerank']['correct_first'] - metrics['bm25']['correct_first']
    if difference > 0:
        body += f'<p>Checking every shortlist put a relevant paper first on {difference} more queries overall than keywords alone. It also adds processing; it is not a free speed improvement.</p>'
    elif difference < 0:
        body += f'<p>Checking every shortlist put a relevant paper first on {-difference} fewer queries overall than keywords alone. This generic small reranker did not transfer well enough to improve this benchmark.</p>'
    else:
        body += '<p>Checking every shortlist matched the keyword first-result total. Equal totals do not mean the same queries succeeded.</p>'
    body += f'<p>Only {result["queries_with_relevant_candidate"]}/200 shortlists contained a judged relevant paper. A second check cannot recover papers that keyword search left out.</p>'
    body += '<details class="plain-details"><summary>What model runs, and what gets skipped?</summary><p>The second check uses pretrained MS MARCO TinyBERT: two transformer layers, 128 hidden dimensions and about 4.39 million parameters. Both layers run for each of the 20 query-document pairs. When the keyword winner is clearly ahead, the optional route skips the entire second model. This is conditional reranking, not stopping inside the model.</p><p>The learned combination reuses our frozen five-weight model. Meaning search uses six-layer MiniLM and stored document vectors. Neither was retrained on these 200 claims. The live search demo supports keywords, meaning and their combinations; this new second check is an offline experiment.</p><p>Relevant means human-judged evidence about the claim, which may support or refute it. This is not a claim-truth classifier, production search traffic, or proof of a general improvement. Model pretraining overlap is unknown.</p></details>'
    body += '<p class="small"><a href="@/docs/search-next.md">Method, all metrics and limitations</a> &middot; <a href="@/results/search-next/predictions.json">Every query and result</a> &middot; <a href="@/results/search-next/protocol.json">Frozen protocol</a> &middot; <a href="@/experiments/search_next.py">Experiment code</a></p>'
    pages['search-next-results.html'] = {'title': 'Give search a second check', 'description': 'A small model checks a shortlist; compare quality and processing on fresh queries.', 'status': 'Additional held-out IDs', 'styles': ['results.css'], 'body': body}
    if 'search-demo.html' in pages:
        link = '<p><a href="@/search-next-results.html">New: compare the methods on 200 additional test IDs</a></p>'
        if link not in pages['search-demo.html']['body']:
            pages['search-demo.html']['body'] += link

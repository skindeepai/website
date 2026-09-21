"""Connect topics, approaches, demos, exact studies and their evidence."""
import json
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(href, label):
    return '<a href="@/' + escape(href, quote=True) + '">' + escape(label) + '</a>'


def route(href, label):
    return dict(href=href, label=label)


def section(pages, name, ident, title, links):
    body = re.sub(r'<section id="' + ident + r'">.*?</section>', '', pages[name].get('body', ''), flags=re.S)
    pages[name]['body'] = body + '<section id="' + ident + '"><h2>' + title + '</h2><p class="next-links">' + ' · '.join(link(*item) for item in links) + '</p></section>'


def update(pages):
    def journey(name, topic, approach=None, results=None, evidence=None, note=''):
        item = {'topic': route(topic, pages[topic]['title'])}
        for key, value in [('approach', approach), ('results', results), ('evidence', evidence)]:
            if value:
                item[key] = route(*value)
        if note:
            item['note'] = note
        pages[name]['journey'] = item
        # Replace old "Back to" strips with the canonical topic/approach trail.
        if 'body' in pages[name]:
            pages[name]['body'] = re.sub(r'<p class="small"><a href="@/[^"]+">Back to [^<]+</a></p>', '', pages[name]['body'])
        return item

    # Saved browser runs deserve their own destination, not another model's numbers.
    labels = {'full': 'All 4 layers', 'fixed2': 'Stop after layer 2', 'adaptive': 'Check after layer 2',
              'cascade': 'Tiny BERT, then larger BERT', 'trained': 'Updated training', 'batch4': 'Groups of four', 'int8': 'INT8 weights'}
    methods = {'fixed': ('Fixed depth', 'depth-results.html'), 'adaptive': ('Check before continuing', 'early-exit-results.html'),
               'cascade': ('Small model, then larger model', 'cascade-results.html'), 'training': ('Compare trained weights', 'training-results.html'),
               'batch': ('Process a queue together', 'batch-results.html'), 'quantization': ('Float32 versus INT8', 'quantization-results.html')}
    body = '<p>These are saved runs of the BERT browser demo: 100 previously inspected ToxicChat messages, two reversed-order passes per comparison. Time includes input preparation and inference, but excludes loading and warmup. Each row belongs to its own paired run.</p><p>These are exploratory measurements on one computer, not fresh validation or a promise of the same speed on your device.</p>'
    for mode, (title, study) in methods.items():
        record = f'results/browser-demos/final/method-{mode}.json'
        run = json.loads((ROOT / record).read_text(encoding='utf-8'))
        body += '<details id="' + mode + '"><summary>' + title + '</summary><div class="table-scroll"><table><thead><tr><th>Path</th><th>Correct / 100</th><th>Time / 100</th><th>Mean blocks run</th><th>Changed answers</th></tr></thead><tbody>'
        for row in run['summary']:
            body += '<tr><th scope="row">' + labels[row['path']] + '</th><td>' + str(row['correct']) + '</td><td>' + f'{row["ms"] / 1000:.2f} s' + '</td><td>' + f'{row["meanBlocks"]:.2f}' + '</td><td>' + str(row['changed']) + '</td></tr>'
        body += '</tbody></table></div>'
        if mode == 'cascade':
            body += '<p>The fallback runs two independent BERT models. Their blocks have different widths; the sum is not a compute-saving percentage. The 0.05/0.95 gate was not validated for this pairing.</p>'
        if mode == 'quantization':
            body += '<p>The browser INT8 run matched these float32 decisions. Python INT8 differed on one message; this is not proof of lossless conversion across runtimes.</p>'
        body += '<p>' + link('method-demo.html?method=' + mode, 'Run this comparison') + ' · ' + link(study, 'Related Qwen study') + '</p><details><summary>Detailed evidence</summary><p>' + link(record, 'Every decision, timing and model setting') + ' · ' + link('results/browser-demos/final/protocol.json', 'Run protocol and source hashes') + ' · ' + link('docs/method-demos.md', 'Implementation and limitations') + '</p></details></details>'
    tiny = json.loads((ROOT / 'results/decision-export/result.json').read_text())
    body += '<details id="tiny"><summary>Standalone tiny classifier: separate export check</summary><p>The two-layer classifier in the small-model demo got ' + str(tiny['metrics']['correct']) + '/100 correct, missed ' + str(tiny['metrics']['missed_toxic']) + ' toxic messages and falsely blocked ' + str(tiny['metrics']['false_block']) + ' benign messages. The browser preserved all Python decisions. This is a separate single-pass functional check, not one of the paired comparisons above.</p><p>Both layers run. This later specialist uses a 0.4 BLOCK threshold; it is not the earlier 0.3-threshold model in the original Qwen cascade study.</p><p>' + link('tiny-decision-demo.html', 'Run this tiny classifier') + '</p><details><summary>Detailed evidence</summary><p>' + link('docs/decision-export.md', 'Model, training and export checks') + ' · ' + link('results/decision-export/browser.json', 'Every browser decision and timing') + '</p></details></details>'
    pages['browser-method-results.html'] = dict(title='Recorded browser comparisons', description='Saved BERT comparisons and the standalone tiny-classifier check.', status='Browser test results', styles=['results.css'], body=body)
    run = json.loads((ROOT / 'results/browser-demos/final/qwen-execution.json').read_text())
    labels = {'full': 'Full input each time', 'prefix': 'Reuse instructions', 'batch': 'Groups of four', 'prefix-batch': 'Reuse + groups of four'}
    body = '<p>This saved browser check used four real messages, with two passes per path. Every path got only <strong>1 of 4 correct</strong>. It checks that the execution paths run; it does not establish useful moderation accuracy.</p><p>All 24 Qwen layers ran. No layers were skipped. Loading and warmup are excluded; instruction-cache creation and copying are included in cached-path time.</p><div class="table-scroll"><table><thead><tr><th>Path</th><th>Correct / 4</th><th>Time / 4</th><th>Changed answers</th></tr></thead><tbody>'
    for row in run['summary']:
        body += '<tr><th scope="row">' + labels[row['path']] + '</th><td>' + str(row['correct']) + '</td><td>' + f'{row["ms"] / 1000:.2f} s' + '</td><td>' + str(row['changed']) + '</td></tr>'
    body += '</tbody></table></div><p>Qwen 0.5B q4, one WASM thread, 64 message tokens maximum. This differs from the trained float32 Python classifiers. Four messages are too few for a general speed or quality claim.</p>'
    for mode, title in [('prefix', 'Reuse instructions'), ('batch', 'Process a queue'), ('combined', 'Combine both')]:
        body += '<section id="' + mode + '"><h2>' + title + '</h2><p>' + link('execution-demo.html?method=' + mode, 'Run this browser comparison') + ' · ' + link(mode + '-results.html', 'Related Python study') + '</p></section>'
    body += '<details><summary>Detailed evidence</summary><p>' + link('results/browser-demos/final/qwen-execution.json', 'Every request and timing') + ' · ' + link('results/browser-demos/final/protocol.json', 'Run protocol') + ' · ' + link('docs/method-demos.md', 'Implementation and checks') + '</p></details>'
    pages['browser-execution-results.html'] = dict(title='Qwen browser execution check', description='The saved four-message run for caching and batching.', status='Functional check', styles=['results.css'], body=body)

    # One stable topic and approach per page; no referrer-dependent "back" behavior.
    for name in ['how-it-works.html', 'active-learning-strategies.html', 'multi-modal-preferences-deep-dive.html', 'preference-results.html']:
        journey(name, 'preferences.html')
    for name in ['depth-results.html', 'early-exit-results.html', 'training-results.html', 'cascade-results.html', 'chat-results.html', 'shared-model-results.html', 'shared-qwen-results.html', 'fresh-message-results.html', 'quantization-results.html']:
        journey(name, 'adaptive.html', ('decision-results.html', 'Compare approaches'))
    journey('cascade-results.html', 'adaptive.html', ('decision-results.html', 'Compare approaches'), ('cascade-test-results.html', 'Recorded cascade results'))
    journey('cascade-test-results.html', 'adaptive.html', ('cascade-results.html', 'Small model, then larger model'))
    journey('decision-results.html', 'adaptive.html')
    journey('output-results.html', 'decisions.html', evidence=('docs/chat-output-steps.md', 'Test design, chart and detailed evidence'))
    for name in ['batch-results.html', 'prefix-results.html', 'combined-results.html', 'decision-checks.html', 'redaction-results.html', 'receipt-results.html', 'routing-results.html', 'maze-results.html', 'search-next-results.html']:
        journey(name, 'decisions.html')
    for name in ['coordinate-results.html', 'visual-refusal-results.html']:
        journey(name, 'coordinates.html')
    journey('demo.html', 'preferences.html', ('how-it-works.html', 'How learning works'), ('preference-results.html', 'Preference results'), ('docs/method.md', 'Detailed evidence'))
    journey('shared-decision-demo.html', 'adaptive.html', ('shared-model-results.html', 'One model, two exits'), ('fresh-message-results.html', 'Fresh-message validation'), ('docs/shared-browser.md', 'Browser results and evidence'))
    journey('maze-live.html', 'decisions.html', ('maze-results.html', 'Learn a movement policy'), ('maze-results.html', 'Recorded policy results'), ('docs/maze-smoke.md', 'Detailed evidence'))
    journey('maze-benchmark.html', 'decisions.html', ('maze-results.html', 'Maze experiments'), ('maze-results.html', 'Later trained-policy results'), ('docs/maze-actions.md', 'Original Qwen evidence'))
    journey('search-demo.html', 'decisions.html', results=('search-next-results.html', 'Search follow-up results'), evidence=('docs/search-ranking.md', 'Original search evidence'))
    journey('rerank-demo.html', 'decisions.html', ('search-next-results.html', 'Rerank search results'), ('search-next-results.html', 'Recorded reranker results'), ('docs/search-next.md', 'Detailed evidence'))
    journey('screenshot-demo.html', 'coordinates.html', results=('coordinate-results.html', 'Results for these 30 screenshots'), evidence=('docs/screenspot-results.md', 'Detailed evidence'))
    journey('coordinate-lab.html', 'coordinates.html', results=('results-record.html#coordinates', 'Original local-test results'), evidence=('docs/coordinate-abstention-smoke.md', 'Local-target evidence'))
    journey('point-demo.html', 'coordinates.html', results=('screenshot-demo.html', 'Related vision-model recordings'), evidence=('docs/method-demos.md', 'OCR implementation and checks'), note='This live OCR demo is not the GUI-Actor model in the recordings. No accuracy benchmark has been published for this OCR example.')
    journey('image-action-demo.html', 'coordinates.html', evidence=('docs/action-demo.md', 'Arrow-model tests and evidence'))
    journey('preference-method-demo.html', 'preferences.html', ('active-learning-strategies.html', 'Choose the next example'), results=('multi-modal-preferences-deep-dive.html', 'Learning more than one style'), evidence=('docs/method-demos.md', 'Demo implementation and checks'), note='This small interactive model has no human-preference benchmark. The optional example ratings are synthetic.')
    journey('moderation-benchmark.html', 'decisions.html', results=('output-results.html', 'Related CPU output comparison'), evidence=('docs/browser-moderation.md', 'Browser method, dataset and limits'), note='Results are measured when you run this demo. The related CPU study uses different readouts and precision; this browser computes the full vocabulary and does not reproduce its trained classifier or two-row optimization.')
    journey('tiny-decision-demo.html', 'decisions.html', results=('browser-method-results.html#tiny', 'Results for this tiny classifier'), evidence=('docs/decision-export.md', 'Model export and browser checks'))
    journey('browser-benchmark.html', 'decisions.html', results=('moderation-benchmark.html', 'Run the real-message benchmark'), evidence=('scripts/browser-benchmark.js', 'Earlier example source'), note='Earlier timing example: three authored banking requests, not a real-message accuracy benchmark.')

    item = journey('method-demo.html', 'adaptive.html')
    item.update(selector='method-mode', parameter='method', modes={})
    for mode, (title, study) in methods.items():
        item['modes'][mode] = dict(approach=route(study, title), results=route('browser-method-results.html#' + mode, 'Recorded browser results'), evidence=route('docs/method-demos.md', 'Implementation and evidence'))
    item = journey('execution-demo.html', 'decisions.html')
    item.update(selector='execution-mode', parameter='method', modes={})
    for mode, title in [('prefix', 'Reuse instructions'), ('batch', 'Process a queue'), ('combined', 'Combine both')]:
        item['modes'][mode] = dict(approach=route(mode + '-results.html', title), results=route('browser-execution-results.html#' + mode, 'Recorded browser check'), evidence=route('docs/method-demos.md', 'Implementation and evidence'))
    item = journey('practical-demo.html', 'decisions.html')
    item.update(selector='practical-task', parameter='task', modes={})
    for task, study, title in [('routing', 'routing-results.html', 'Route a request'), ('privacy', 'redaction-results.html', 'Find names'), ('receipts', 'receipt-results.html', 'Choose a receipt total')]:
        evidence='docs/privacy-full-document.md' if task=='privacy' else 'docs/practical-baselines.md'
        item['modes'][task] = dict(approach=route(study, title), results=route(study, 'Recorded task results'), evidence=route(evidence, 'Dataset, method and evidence'))
    journey('browser-method-results.html', 'adaptive.html', ('decision-results.html', 'Compare approaches'))
    journey('browser-execution-results.html', 'decisions.html')

    # Fix missing topic paths, sample mismatches, and newer-validation discovery.
    section(pages, 'preferences.html', 'preference-approaches', 'Explore the approaches', [('how-it-works.html', 'Learn from ratings'), ('active-learning-strategies.html', 'Choose the next example'), ('multi-modal-preferences-deep-dive.html', 'Learn more than one style'), ('preference-results.html', 'Recorded results')])
    section(pages, 'how-it-works.html', 'preference-next', 'Two useful variations', [('active-learning-strategies.html', 'Choose the next example'), ('multi-modal-preferences-deep-dive.html', 'Learn more than one style')])
    section(pages, 'preference-results.html', 'preference-next', 'Related approaches', [('active-learning-strategies.html', 'Example selection'), ('multi-modal-preferences-deep-dive.html', 'Multiple styles')])
    section(pages, 'coordinate-results.html', 'matching-screenshots', 'Inspect this test and its follow-up', [('screenshot-demo.html', 'These 30 screenshot predictions'), ('visual-refusal-results.html', 'Harder screens and missing targets')])
    pages['coordinate-lab.html']['body'] = pages['coordinate-lab.html'].get('body', '').replace('href="@/results.html#coordinates"', 'href="@/results-record.html#coordinates"')
    # coordinate-lab is template-backed; the builder also applies this exact legacy link correction.
    pages['coordinate-lab.html']['link_replacements'] = {'results.html#coordinates': 'results-record.html#coordinates'}
    section(pages, 'adaptive.html', 'latest-validation', 'Later validation', [('fresh-message-results.html', '500 fresh messages'), ('browser-method-results.html#adaptive', 'Recorded browser run')])
    section(pages, 'decision-results.html', 'latest-validation', 'Latest checks', [('fresh-message-results.html', 'Fresh-message validation'), ('quantization-results.html', 'Smaller-number tests'), ('browser-method-results.html', 'Browser comparisons')])
    section(pages, 'results.html', 'practical-results', 'Results for practical tasks', [('search-next-results.html', 'Search ranking'), ('routing-results.html', 'Request routing'), ('redaction-results.html', 'Finding names'), ('receipt-results.html', 'Receipt totals')])
    section(pages, 'results.html', 'browser-results', 'Saved browser runs', [('browser-method-results.html', 'BERT method comparisons'), ('browser-execution-results.html', 'Qwen execution check')])
    section(pages, 'decisions.html', 'movement-policy', 'A decision on every step', [('maze-results.html', 'Learn to move through a maze'), ('maze-live.html', 'Run the policy')])

    # Trim clearly redundant adjacent calls to action; preserve contextual evidence links.
    for name, old, new in [
        ('adaptive.html', '<p>Later small tests tried different stopping depths, learned confidence checks and a tiny model with Qwen as fallback. <a href="@/decision-results.html">See how the methods compare</a>.</p>', ''),
        ('preferences.html', '<a href="@/demo.html">Try it with drawings</a> · ', ''),
        ('how-it-works.html', '<a href="@/demo.html">Try the corrected demo</a> · ', ''),
        ('how-it-works.html', '<a href="@/demo.html">Try the drawing demo</a> · ', ''),
        ('coordinates.html', '<a href="@/coordinate-lab.html">Open coordinate workbench</a>', '<a href="@/screenshot-demo.html">Inspect the public screenshot test</a>')]:
        pages[name]['body'] = pages[name].get('body', '').replace(old, new)

    groups = [
        ('Learn preferences', [('demo.html', 'Rate drawings'), ('preference-method-demo.html', 'Compare example selection and styles')]),
        ('Return decisions', [('moderation-benchmark.html', 'Qwen: labels versus replies'), ('tiny-decision-demo.html', 'Small trained message classifier'), ('practical-demo.html', 'Routing, names and receipt totals'), ('maze-live.html', 'Maze policy')]),
        ('Reduce computation', [('shared-decision-demo.html', 'Shared model with an early exit'), ('method-demo.html', 'Compare decision methods'), ('execution-demo.html', 'Cache instructions and batch messages')]),
        ('Search documents', [('search-demo.html', 'Keyword and meaning search'), ('rerank-demo.html', 'Rerank the first matches')]),
        ('Find an action in an image', [('point-demo.html', 'Live OCR: visible text to a point'), ('image-action-demo.html', 'Live synthetic-arrow model')]),
    ]
    pages['demo-directory.html']['body'] = '<p>Choose a task. Each demo links to its approach and the evidence for the model it runs.</p>' + ''.join('<section><h2>' + title + '</h2><ul class="page-links">' + ''.join('<li>' + link(*x) + '</li>' for x in links) + '</ul></section>' for title, links in groups) + '<details><summary>Recorded examples</summary><p>' + link('screenshot-demo.html', 'GUI-Actor screenshot recordings') + ' · ' + link('maze-benchmark.html', 'Earlier Qwen maze replay') + '</p></details>'
    pages['getting-started.html']['title'] = 'Run the code locally'
    pages['getting-started.html']['description'] = 'Run the website and reproduce the research experiments.'

    # The HTML sitemap and XML sitemap now share one authoritative page registry.
    groups = {'The lab and reference pages': []}
    for name in pages:
        if name == 'sitemap.html':
            continue
        topic = pages[name].get('journey', {}).get('topic', {}).get('href')
        title = pages[topic]['title'] if topic else 'The lab and reference pages'
        groups.setdefault(title, []).append(name)
    pages['sitemap.html']['body'] = '<p>Every current page, including demos and detailed results. Older versions remain under ' + link('history.html', 'History and archive') + '.</p>' + ''.join('<section><h2>' + escape(title) + '</h2><ul class="page-links">' + ''.join('<li>' + link(n, pages[n]['title']) + '</li>' for n in sorted(names, key=lambda n: pages[n]['title'].casefold())) + '</ul></section>' for title, names in groups.items())


if __name__ == '__main__':
    path = ROOT / 'content/pages.json'
    pages = json.loads(path.read_text(encoding='utf-8'))
    update(pages)
    path.write_text(json.dumps(pages, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')

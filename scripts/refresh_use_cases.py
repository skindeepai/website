"""Render application pages and short topic teasers from reviewed use-case copy."""
import json
import re
from collections import OrderedDict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(href, title):
    target = href if href.startswith('https://') else '@/' + href
    return '<a href="' + escape(target, quote=True) + '">' + escape(title) + '</a>'


def links(items):
    return '<p class="next-links">' + ' · '.join(link(*item) for item in items) + '</p>'


def teasers(cases):
    return '<ul class="use-case-list">' + ''.join(
        '<li>' + link('examples/' + item['id'] + '.html', item['title']) +
        '<span>' + escape(item['teaser']) + '</span></li>' for item in cases) + '</ul>'


def groups(cases):
    grouped = OrderedDict()
    for item in cases:
        grouped.setdefault(item['group'], []).append(item)
    return ''.join('<div class="use-case-group"><h3>' + escape(name) + '</h3>' + teasers(items) + '</div>' for name, items in grouped.items())


def update(pages):
    catalog = json.loads((ROOT / 'content/use-cases.json').read_text(encoding='utf-8'))
    cases, topics = catalog['cases'], catalog['topics']
    assert len({item['id'] for item in cases}) == len(cases)
    coverage_path = ROOT / 'content/demo-coverage.json'
    coverage = json.loads(coverage_path.read_text(encoding='utf-8'))
    coverage = {name: item for name, item in coverage.items() if not name.startswith('examples/')}
    for item in cases:
        topic = topics[item['topics'][0]]
        pref = item['topics'][0] == 'preferences'
        body = '<figure class="method-figure use-case-flow"><ol class="method-flow">' + ''.join(
            '<li>' + escape(step) + '</li>' for step in item['steps']) + '</ol><figcaption>Illustration of the proposed workflow.</figcaption></figure>'
        body += '<p>' + escape(item['scenario']) + '</p><h2>Where the small model fits</h2><p>' + escape(item['mechanism']) + '</p>'
        body += '<h2>What would need to work</h2><p>' + escape(item['check']) + '</p>'
        if item.get('medical'):
            body += '<p class="small">Research proposal only. SkinDeep has no clinical model, patient-use demo or validated medical performance for this task.</p>'
        if item.get('dataset'):
            body += '<h2>A dataset to start with</h2><p>' + escape(item['dataset']) + ' ' + ' · '.join(link(*x) for x in item.get('sources', [])) + '.</p>'
        elif item.get('sources'):
            body += links(item['sources'])
        demo = item.get('demo')
        demo_note = item.get('demo_note', '')
        if pref:
            demo = ['demo.html', 'Try the drawing-based learning loop']
            demo_note = 'This demo learns from simple drawings. It does not run the application described above.'
        if demo:
            body += '<section id="live-demo-link"><h2>Try the related work</h2>' + links([demo])
            if demo_note:
                body += '<p class="small">' + escape(demo_note) + '</p>'
            body += '</section>'
            coverage['examples/' + item['id'] + '.html'] = dict(target=demo[0], label=demo[1], note=demo_note,
                relationship='Related demonstration; not necessarily this application',
                kind='recorded' if demo[0] == 'screenshot-demo.html' else 'interactive')
        evidence = item.get('evidence', [])
        if pref:
            evidence = [['how-it-works.html', 'How preference learning works'], ['preference-results.html', 'What the current preference tests show']]
        if evidence:
            body += '<details><summary>Related experiments and evidence</summary>' + links(evidence) + '</details>'
        if item.get('legacy'):
            body += '<details><summary>Earlier proposal and example code</summary><p>The earlier page is preserved as a historical proposal; its claims are not new validation.</p>' + links([[item['legacy'], 'Read the archived example']]) + '</details>'
        if item.get('related'):
            body += links(item['related'])
        # Cross-topic cases have one page, not a separate near-duplicate for each method.
        body += '<nav aria-label="Related topics">' + links([[topics[t]['page'] + '#use-cases', topics[t]['title']] for t in item['topics']]) + '</nav>'
        body += links([['examples/index.html', 'All use cases']])
        pages['examples/' + item['id'] + '.html'] = dict(
            title=item['title'], description=item['teaser'], status='Use case · ' + item.get('status', 'Idea to test'),
            styles=['results.css', 'use-cases.css'], body=body,
            journey=dict(topic=dict(href=topic['page'], label=topic['title']),
                         approach=dict(href='examples/index.html#'+item['topics'][0], label='Use cases')))

    for key, topic in topics.items():
        selected = [item for item in cases if key in item['topics']]
        section = '<section id="use-cases" aria-labelledby="use-cases-title"><h2 id="use-cases-title">Where this could be useful</h2><p>' + escape(topic['intro']) + '</p>'
        if key == 'preferences':
            section += '<p>With a compatible generator, the small model can score or adjust its internal inputs <strong>before rendering</strong>. It can also rank things that already exist. These are the original generate, score and edit uses of the same learned preferences.</p>'
        # Keep the preference introduction compact while retaining every historical idea.
        if key in ['preferences', 'adaptive', 'decisions']:
            featured_ids = {
                'preferences': {'music', 'art', 'dating', 'social-media', 'beauty', 'architecture'},
                'adaptive': {'model-routing', 'faster-llm', 'image-safety', 'image-preferences'},
                'decisions': {'moderation', 'support', 'receipts', 'ecg'}
            }[key]
            section += teasers([item for item in selected if item['id'] in featured_ids])
            more = {
                'preferences': 'More use cases: everyday choices, creative work, science and engineering',
                'adaptive': 'More use cases: messages, devices and medical research',
                'decisions': 'More use cases: search, actions, signals and image checks'
            }[key]
            section += '<details class="more-use-cases"><summary>' + more + '</summary>' + groups([item for item in selected if item['id'] not in featured_ids]) + '</details>'
        else:
            section += groups(selected)
        section += links([['examples/index.html#' + key, 'Browse these use cases']]) + '</section>'
        page = pages[topic['page']]
        body = re.sub(r'<section id="use-cases".*?</section>', '', page.get('body', ''), flags=re.S)
        # Put applications after the short explanation, before the deeper evidence.
        position = body.find('<details')
        page['body'] = body[:position] + section + body[position:] if position >= 0 else body + section
        page['styles'] = list(dict.fromkeys(page.get('styles', []) + ['use-cases.css']))

    body = '<p>Choose a practical application. Each page explains the role of the small model and links to relevant work. An idea is not a completed product; related demos and tests are labelled separately.</p>'
    body += '<nav aria-label="Use-case topics">' + links([['examples/index.html#'+key, value['title']] for key, value in topics.items()]) + '</nav>'
    for key, topic in topics.items():
        body += '<section id="' + key + '"><h2>' + escape(topic['title']) + '</h2>'
        body += groups([item for item in cases if key in item['topics']]) + '</section>'
    body += '<details><summary>Original applications and source history</summary><p>The catalog restores the older SkinDeep application list and adds newer decision, spatial-output and computation examples. Historical proposals are not evidence that all applications were built.</p>' + links([['docs/use-case-history.md', 'Historical versions and coverage'], ['history.html', 'Original work and archive']]) + '</details>'
    pages['examples/index.html'] = dict(title='Use cases', description='Applications for preference models, direct decisions, spatial outputs and less computation.',
        status='', styles=['use-cases.css'], body=body)
    coverage_path.write_text(json.dumps(coverage, indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    path = ROOT / 'content/pages.json'
    pages = json.loads(path.read_text(encoding='utf-8'))
    update(pages)
    path.write_text(json.dumps(pages, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')

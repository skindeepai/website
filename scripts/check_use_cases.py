"""Check historical coverage and rendered topic-to-case routes without inference."""
import hashlib
import json
import re
import subprocess
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links, self.ids = [], set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])


def main():
    catalog = json.loads((ROOT/'content/use-cases.json').read_text(encoding='utf-8'))
    pages = json.loads((ROOT/'content/pages.json').read_text(encoding='utf-8'))
    cases = catalog['cases']
    assert len({r['id'] for r in cases}) == len(cases)
    old_revision = 'caadd362247dcaf821ab20e8b4d9fe91d21a9766'
    old = subprocess.check_output(['git', 'show', old_revision+':index.html'], cwd=ROOT)
    cards = re.findall(r'<div class="app-card"[^>]*>\s*<div[^>]*>.*?</div>\s*<h3>(.*?)</h3>.*?<a href="([^"]+)" class="app-link"', old.decode(), re.S)
    assert len(cards) == 15
    destinations = {'examples/'+x['id']+'.html' for x in cases}
    for title, target in cards:
        if target == 'getting-started.html':
            continue
        assert target in destinations, ('Missing historical application', title, target)
        item = next(x for x in cases if 'examples/'+x['id']+'.html' == target)
        assert 'preferences' in item['topics'], ('Original preference case lost its topic', title)
    parsed = {name: Links((ROOT/name).read_text(encoding='utf-8')) for name in pages}
    routes = 0
    for item in cases:
        name = 'examples/'+item['id']+'.html'
        assert name in pages
        for topic_key in item['topics']:
            topic = catalog['topics'][topic_key]['page']
            assert name in parsed[topic].links, (topic, name)
            assert '../'+topic+'#use-cases' in parsed[name].links, (name, topic)
            routes += 1
        assert '../'+name in parsed['examples/index.html'].links
        for href in parsed[name].links:
            target = urlsplit(href)
            if target.scheme or target.netloc:
                continue
            absolute = (ROOT/name).parent / unquote(target.path) if target.path else ROOT/name
            absolute = absolute.resolve()
            assert absolute.is_relative_to(ROOT), href
            assert absolute.is_file(), (name, href)
            if target.fragment and absolute.suffix == '.html':
                assert unquote(target.fragment) in Links(absolute.read_text(encoding='utf-8')).ids, (name, href)
        if item.get('medical'):
            body = pages[name]['body']
            assert 'Research proposal only' in body and 'patient-use demo' in body
            assert 'demo' not in item, 'Do not imply a clinical demo exists.'
    for name in pages:
        text = (ROOT/name).read_text(encoding='utf-8')
        assert 'label-results.html' not in text and 'chat-label-suite' not in text, 'Parser experiment belongs in research notes only.'
    for topic in catalog['topics'].values():
        text = (ROOT/topic['page']).read_text(encoding='utf-8')
        assert text.count('id="use-cases"') == 1
    coverage = json.loads((ROOT/'content/demo-coverage.json').read_text(encoding='utf-8'))
    for name, item in coverage.items():
        text = (ROOT/name).read_text(encoding='utf-8')
        section = re.search(r'<section id="live-demo-link">(.*?)</section>', text, re.S)
        assert section, ('Missing mapped demo entry point', name)
        targets = Links(section[1]).links
        assert len(targets) == 1, (name, targets)
        actual = ((ROOT/name).parent / urlsplit(targets[0]).path).resolve()
        expected = (ROOT/urlsplit(item['target']).path).resolve()
        assert actual == expected and urlsplit(targets[0]).query == urlsplit(item['target']).query
    output = ROOT/'results/use-cases'
    output.mkdir(exist_ok=True, parents=True)
    report = dict(status='passed', use_cases=len(cases), topic_case_round_trips=routes,
        original_application_cards=14, historical_source_commit=old_revision,
        historical_source_sha256=hashlib.sha256(old).hexdigest(),
        historical_coverage=[dict(title=unescape(title), page=target) for title, target in cards],
        medical_proposals=sum(bool(x.get('medical')) for x in cases),
        mapped_demo_links=len(coverage),
        checks=['Every original application has a dedicated preference page', 'Every topic teaser has a matching case and return route',
                'All case links and local fragments resolve', 'Cross-topic cases use a single canonical page',
                'Medical proposals do not advertise a clinical demo', 'Parser experiment remains in research notes'],
        limitations='Static content and route audit; no application efficacy or medical performance validation.')
    (output/'coverage.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k!='historical_coverage'}, indent=2))


if __name__ == '__main__':
    main()

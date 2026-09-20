"""Map generated-page navigation without changing the public site.

Run: python scripts/audit_navigation.py
Outputs a complete link inventory and an offline page-connection explorer.
"""
import hashlib
import json
import posixpath
import subprocess
from collections import Counter, deque
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.links = []
        self.anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag not in VOID:
            self.stack.append((tag, attrs))
        if tag == 'a' and 'href' in attrs:
            tags = [t for t, _ in self.stack]
            zone = ('sidebar' if 'aside' in tags else 'footer' if 'footer' in tags
                    else 'content' if 'main' in tags else 'header' if 'header' in tags else 'utility')
            self.anchor = {
                'href': attrs['href'], 'label': '', 'zone': zone,
                'collapsed': any(t == 'details' and 'open' not in a for t, a in self.stack),
                'hidden': any('hidden' in a for _, a in self.stack),
            }
        if tag == 'img' and self.anchor:
            self.anchor['label'] += ' ' + attrs.get('alt', '')

    def handle_data(self, data):
        if self.anchor is not None:
            self.anchor['label'] += data + ' '

    def handle_endtag(self, tag):
        if tag == 'a' and self.anchor is not None:
            self.anchor['label'] = ' '.join(self.anchor['label'].split())
            self.links.append(self.anchor)
            self.anchor = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def resolve(source, href, names):
    url = urlsplit(href)
    same_site = url.netloc.lower() in {'skindeep.ai', 'www.skindeep.ai'}
    if (url.scheme or url.netloc) and not same_site:
        return dict(kind='external', target=href, query=url.query, fragment=url.fragment)
    if url.path:
        target = (url.path.lstrip('/') if same_site or url.path.startswith('/')
                  else posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(url.path))))
        if not target or target.endswith('/'):
            target += 'index.html'
    else:
        target = 'index.html' if same_site else source
    kind = ('page' if target in names else 'archive' if target.startswith('archive/')
            else 'evidence' if target.startswith(('docs/', 'results/', 'experiments/', 'models/', 'scripts/'))
            else 'resource')
    return dict(kind=kind, target=target, query=url.query, fragment=url.fragment)


def adjacency(pages, predicate):
    return {p: sorted({e['target'] for e in data['links'] + data.get('mode_links', [])
                       if e['kind'] == 'page' and e['target'] != p and predicate(e)})
            for p, data in pages.items()}


def paths_from_home(graph, excluded=()):
    paths = {'index.html': ['index.html']}
    queue = deque(['index.html'])
    while queue:
        source = queue.popleft()
        for target in graph[source]:
            if target not in paths and target not in excluded:
                paths[target] = paths[source] + [target]
                queue.append(target)
    return paths


def main():
    config = json.loads((ROOT / 'content/pages.json').read_text(encoding='utf-8'))
    pages = {}
    for name, item in config.items():
        data = (ROOT / name).read_bytes()
        parser = PageLinks()
        parser.feed(data.decode('utf-8'))
        for link in parser.links:
            link.update(resolve(name, link['href'], config))
        mode_links = []
        for mode, routes in item.get('journey', {}).get('modes', {}).items():
            for key in ['approach', 'results', 'evidence']:
                if key in routes:
                    entry = dict(href=routes[key]['href'], label=routes[key]['label'], mode=mode,
                                 zone='content', collapsed=False, hidden=False)
                    entry.update(resolve(name, entry['href'], config))
                    mode_links.append(entry)
        pages[name] = dict(title=item['title'], description=item['description'],
                           sha256=hashlib.sha256(data).hexdigest(), links=parser.links, mode_links=mode_links)
    content = adjacency(pages, lambda e: e['zone'] == 'content')
    visible = adjacency(pages, lambda e: e['zone'] == 'content' and not e['collapsed'] and not e['hidden'])
    all_links = adjacency(pages, lambda e: True)
    for name, data in pages.items():
        data['incoming'] = [p for p, links in content.items() if name in links]
        data['outgoing'] = content[name]
        data['duplicate_content_links'] = []
        counts = Counter((e['target'], e['query'], e['fragment']) for e in data['links']
                         if e['zone'] == 'content' and e['kind'] == 'page' and e['target'] != name)
        for (target, query, fragment), count in counts.items():
            if count > 1:
                labels = [e['label'] for e in data['links'] if e['zone'] == 'content'
                          and (e['target'], e['query'], e['fragment']) == (target, query, fragment)]
                data['duplicate_content_links'].append(dict(target=target, query=query, fragment=fragment,
                                                            count=count, labels=labels))
    body_paths = paths_from_home(content)
    no_sitemap_paths = paths_from_home(content, ['sitemap.html'])
    visible_paths = paths_from_home(visible, ['sitemap.html'])
    global_paths = paths_from_home(all_links)
    for name, data in pages.items():
        data['paths'] = {'content': body_paths.get(name), 'without_sitemap': no_sitemap_paths.get(name),
                         'visible_content': visible_paths.get(name), 'all_links': global_paths.get(name)}
    coverage = json.loads((ROOT / 'content/demo-coverage.json').read_text(encoding='utf-8'))
    missing_returns = []
    for source, item in coverage.items():
        target = urlsplit(item['target']).path
        if source not in content[target]:
            missing_returns.append(dict(source=source, demo=target, query=urlsplit(item['target']).query))
    summary = {
        'pages': len(pages),
        'anchors': sum(len(d['links']) for d in pages.values()),
        'content_page_link_occurrences': sum(e['zone'] == 'content' and e['kind'] == 'page'
                                            for d in pages.values() for e in d['links']),
        'unique_content_connections': sum(len(v) for v in content.values()),
        'unique_all_connections': sum(len(v) for v in all_links.values()),
        'missing_from_sitemap': [n for n in pages if n != 'sitemap.html' and n not in content['sitemap.html']],
        'link_occurrences_by_zone': dict(Counter(e['zone'] for d in pages.values() for e in d['links'])),
        'pages_with_repeated_content_destinations': [n for n, d in pages.items() if d['duplicate_content_links']],
        'no_content_incoming': [n for n, d in pages.items() if not d['incoming']],
        'only_sitemap_incoming': [n for n, d in pages.items() if d['incoming'] == ['sitemap.html']],
        'no_current_page_content_exits': [n for n, d in pages.items() if not d['outgoing']],
        'unreachable_content': [n for n in pages if n not in body_paths],
        'requires_sitemap_for_content_route': [n for n in pages if n in body_paths and n not in no_sitemap_paths and n != 'sitemap.html'],
        'unreachable_visible_content_without_sitemap': [n for n in pages if n not in visible_paths and n != 'sitemap.html'],
        'unreachable_all_links': [n for n in pages if n not in global_paths],
        'demo_links_without_direct_return': missing_returns,
    }
    result = {
        'scope': 'Current local generated site, including uncommitted navigation edits; not a crawl of production.',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'definitions': {
            'content': 'Links inside main, excluding footer; includes closed details and declared mode-specific navigation routes unless marked otherwise. Static anchor counts exclude these additional mode variants.',
            'connection': 'Distinct source-to-current-page pair. Self links excluded. Modes/fragments retained per anchor but collapsed for graph reachability.',
            'boundary': 'Archive, documents, raw results, source, images and external destinations are listed but not recursively crawled.',
            'limitations': 'Static anchors only. No form actions, model downloads or runtime-generated export links. Collapsed means inside a closed details element, not a pixel visibility measurement. Missing return links are candidates for editorial review, not automatically defects.',
        },
        'summary': summary, 'pages': pages,
    }
    output = ROOT / 'docs/navigation-map.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    inline = json.dumps(result, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    template = (ROOT / 'scripts/navigation-map-template.html').read_text(encoding='utf-8')
    (ROOT / 'docs/navigation-map.html').write_text(template.replace('__MAP_DATA__', inline), encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Verify published metadata, favicon and sharing assets for every current page."""
import json
import struct
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class Head(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta, self.links, self.schemas = {}, {}, []
        self.title, self.in_title, self.schema = '', False, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            key = attrs.get('name', attrs.get('property'))
            if key:
                assert key not in self.meta, f'Duplicate metadata: {key}'
                self.meta[key] = attrs.get('content', '')
        if tag == 'link' and attrs.get('rel') in ['canonical', 'icon']:
            key = attrs['rel']
            assert key not in self.links, f'Duplicate link: {key}'
            self.links[key] = attrs
        if tag == 'title':
            self.in_title = True
        if tag == 'script' and attrs.get('type') == 'application/ld+json':
            self.schema = ''

    def handle_data(self, text):
        if self.in_title:
            self.title += text
        if self.schema is not None:
            self.schema += text

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        if tag == 'script' and self.schema is not None:
            self.schemas.append(json.loads(self.schema))
            self.schema = None


pages = json.loads((ROOT / 'content/pages.json').read_text(encoding='utf-8'))
canonicals = set()
for name, page in pages.items():
    head = Head()
    head.feed((ROOT / name).read_text(encoding='utf-8').split('</head>')[0])
    expected = 'https://skindeep.ai/' + ('' if name == 'index.html' else name)
    assert head.links['canonical']['href'] == expected, name
    assert expected not in canonicals, name
    canonicals.add(expected)
    assert head.meta['og:url'] == expected, name
    assert head.meta['og:type'] == 'website' and head.meta['og:site_name'] == 'SkinDeep', name
    assert head.title == head.meta['og:title'] == head.meta['twitter:title'], name
    assert head.meta['description'] == head.meta['og:description'] == head.meta['twitter:description'], name
    assert head.meta['description'].strip() and ' ? ' not in head.title, name
    assert head.meta['twitter:card'] == 'summary_large_image', name
    image_url = head.meta['og:image']
    assert image_url == head.meta['twitter:image'], name
    assert image_url == 'https://skindeep.ai/images/skindeep-research-card.png', name
    image = (ROOT / urlparse(image_url).path.lstrip('/')).read_bytes()
    assert image[:8] == b'\x89PNG\r\n\x1a\n', name
    assert struct.unpack('>II', image[16:24]) == (1200, 630), name
    assert head.meta['og:image:width'] == '1200' and head.meta['og:image:height'] == '630', name
    assert head.meta['og:image:alt'] == head.meta['twitter:image:alt'] and head.meta['og:image:alt'], name
    icon = (ROOT / name).parent / head.links['icon']['href']
    assert icon.resolve() == (ROOT / 'favicon.png').resolve(), name
    assert head.links['icon']['type'] == 'image/png', name
    if name == 'index.html':
        assert len(head.schemas) == 1 and head.schemas[0]['@type'] == 'WebSite'
        assert head.schemas[0]['url'] == expected
        assert head.schemas[0]['creator']['name'] == 'Steve Seguin'

assert (ROOT / 'favicon.png').read_bytes() == subprocess.check_output(['git', 'show', '705fa7e:favicon.ico'], cwd=ROOT)
urls = {node.text for node in ET.parse(ROOT / 'sitemap.xml').iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')}
assert urls == canonicals
assert 'Sitemap: https://skindeep.ai/sitemap.xml' in (ROOT / 'robots.txt').read_text()
print(f'Passed metadata, sitemap, share-card and original-favicon checks for {len(pages)} pages.')

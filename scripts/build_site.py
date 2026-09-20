"""Build the static lab pages from reviewed content; Python standard library only."""
from pathlib import Path
from html import escape
import json
import re

ROOT = Path(__file__).resolve().parents[1]
NAV = [
    ('The lab', [('index.html', 'Overview'), ('research.html', 'Research FAQ'), ('results.html', 'Test results')]),
    ('Topics', [('preferences.html', 'Learning what you like'), ('decisions.html', 'Decisions without text'), ('coordinates.html', 'Finding where to click'), ('adaptive.html', 'Stopping early')]),
    ('Explore', [('demo-directory.html', 'Live demos'), ('history.html', 'History & archive'), ('about.html', 'About')])
]

def journey_markup(config, prefix):
    def relative(value):
        if isinstance(value, dict):
            return {k: prefix + v if k == 'href' else relative(v) for k, v in value.items()}
        return value
    config = relative(config)
    selected = {**config, **next(iter(config.get('modes', {}).values()), {})}
    def anchor(key):
        item = selected.get(key)
        return ('<a data-journey="' + key + '" href="' + escape(item['href'], quote=True) + '">' + escape(item['label']) + '</a>') if item else ''
    trail = '<nav id="journey-trail" class="small" aria-label="Topic and approach">' + ' · '.join(filter(None, [anchor('topic'), anchor('approach')])) + '</nav>'
    links = ' · '.join(filter(None, [anchor('results'), anchor('evidence')]))
    context = ('<nav id="journey-links" class="next-links small" aria-label="Results and evidence">' + links + '</nav>') if links else ''
    if selected.get('note'):
        context += '<p id="journey-note" class="small">' + escape(selected['note']) + '</p>'
    if config.get('modes'):
        context += '<script id="journey-config" type="application/json">' + json.dumps(config, ensure_ascii=False).replace('<', '\\u003c') + '</script><script src="' + prefix + 'scripts/journey.js" defer></script>'
    return trail, context

def build():
    pages = json.loads((ROOT / 'content/pages.json').read_text(encoding='utf-8'))
    for name, p in pages.items():
        depth = len(Path(name).parts) - 1
        prefix = '../' * depth
        def url(link): return prefix + link
        nav = ''.join('<div class="nav-group"><p>' + title + '</p>' + ''.join(
            '<a ' + ('aria-current="page" ' if name == href else '') + 'href="' + url(href) + '">' + label + '</a>' for href, label in links) + '</div>' for title, links in NAV)
        body = p.get('body', '')
        if 'template' in p: body = (ROOT / p['template']).read_text(encoding='utf-8')
        for old, new in p.get('link_replacements', {}).items():
            body = body.replace('href="@/' + old + '"', 'href="@/' + new + '"')
        body = body.replace('href="@/', 'href="' + prefix).replace('src="@/', 'src="' + prefix)
        trail, context = journey_markup(p['journey'], prefix) if p.get('journey') else ('', '')
        extra_css = ''.join('<link rel="stylesheet" href="' + url(s) + '">' for s in p.get('styles', []))
        scripts = ''.join('<script src="' + url(s) + '" defer></script>' for s in p.get('scripts', []))
        topline = '<p class="page-status">' + escape(p['status']) + '</p>' if p.get('status') else ''
        header_note = 'Steve Seguin'
        canonical = 'https://skindeep.ai/' + ('' if name == 'index.html' else name)
        meta_title = p.get('meta_title', p['title'] + ' — SkinDeep Research')
        meta_description = p.get('meta_description', p['description'])
        share_image = 'https://skindeep.ai/images/skindeep-research-card.png'
        share_alt = 'SkinDeep research by Steve Seguin: learning preferences, returning decisions, finding click locations and stopping early.'
        structured = ''
        if name == 'index.html':
            structured = '<script type="application/ld+json">' + json.dumps({
                '@context': 'https://schema.org', '@type': 'WebSite', 'name': 'SkinDeep',
                'url': canonical, 'description': meta_description,
                'creator': {'@type': 'Person', 'name': 'Steve Seguin'}
            }, ensure_ascii=False).replace('<', '\u003c') + '</script>'
        html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(meta_title)}</title><meta name="description" content="{escape(meta_description, quote=True)}">
<link rel="canonical" href="{canonical}">
<link rel="icon" type="image/png" sizes="16x16" href="{url('favicon.png')}">
<meta property="og:type" content="website"><meta property="og:site_name" content="SkinDeep"><meta property="og:url" content="{canonical}">
<meta property="og:title" content="{escape(meta_title, quote=True)}"><meta property="og:description" content="{escape(meta_description, quote=True)}">
<meta property="og:image" content="{share_image}"><meta property="og:image:type" content="image/png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta property="og:image:alt" content="{share_alt}">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="{escape(meta_title, quote=True)}"><meta name="twitter:description" content="{escape(meta_description, quote=True)}"><meta name="twitter:image" content="{share_image}"><meta name="twitter:image:alt" content="{share_alt}">
{structured}
{extra_css}<link rel="stylesheet" href="{url('lab.css')}"><script src="{url('scripts/lab.js')}" defer></script>{scripts}
</head><body class="lab-page {'demo-page' if name == 'demo.html' else 'home-page' if name == 'index.html' else ''}">
<a class="skip-link" href="#main">Skip to content</a>
<header class="lab-header"><a class="wordmark" href="{url('index.html')}">SkinDeep<span>RESEARCH</span></a><button class="lab-menu" type="button" aria-expanded="false" aria-controls="lab-nav">Research menu <span aria-hidden="true">☰</span></button><span class="header-note">{header_note}</span></header>
<aside class="lab-sidebar" id="lab-nav"><nav aria-label="Research navigation">{nav}</nav><div class="rail-note"><a href="{url('sitemap.html')}">All pages</a><a href="https://github.com/skindeepai">Source on GitHub ↗</a></div></aside>
<main class="lab-main" id="main" tabindex="-1">{trail}{topline}
<h1>{escape(p['title'])}</h1><p class="page-lead">{escape(p['description'])}</p>{context}{body}
<footer class="lab-footer"><span>SkinDeep.ai · Steve Seguin</span><a href="{url('history.html')}">History</a><a href="{url('research.html')}">Research FAQ</a><a href="mailto:contact@skindeep.ai">Contact</a></footer></main></body></html>
'''
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding='utf-8', newline='\n')
    urls = ['https://skindeep.ai/' + ('' if n == 'index.html' else n) for n in pages]
    (ROOT / 'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + ''.join('<url><loc>' + u + '</loc></url>\n' for u in urls) + '</urlset>\n', encoding='utf-8', newline='\n')
    print(f'Built {len(pages)} static pages.')

if __name__ == '__main__': build()

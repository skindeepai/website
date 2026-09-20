"""Build the static lab pages from reviewed content; Python standard library only."""
from pathlib import Path
from html import escape
import json
import re

ROOT = Path(__file__).resolve().parents[1]
NAV = [
    ('The lab', [('index.html', 'Overview'), ('research.html', 'Research FAQ'), ('results.html', 'Test results')]),
    ('Topics', [('preferences.html', 'Learning what you like'), ('decisions.html', 'Decisions without text'), ('coordinates.html', 'Finding where to click'), ('adaptive.html', 'Stopping early')]),
    ('Explore', [('demo.html', 'Preference demo'), ('coordinate-lab.html', 'Recorded clicks'), ('getting-started.html', 'Try it yourself'), ('history.html', 'History & archive'), ('about.html', 'About')])
]

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
        body = body.replace('href="@/', 'href="' + prefix).replace('src="@/', 'src="' + prefix)
        extra_css = ''.join('<link rel="stylesheet" href="' + url(s) + '">' for s in p.get('styles', []))
        scripts = ''.join('<script src="' + url(s) + '" defer></script>' for s in p.get('scripts', []))
        topline = '<p class="page-status">' + escape(p['status']) + '</p>' if p.get('status') else ''
        header_note = 'Steve Seguin'
        html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(p['title'])} — SkinDeep Research</title><meta name="description" content="{escape(p['description'], quote=True)}">
<link rel="canonical" href="https://skindeep.ai/{'' if name == 'index.html' else name}">
<link rel="icon" type="image/svg+xml" href="{url('favicon-simple.svg')}">
<meta property="og:title" content="{escape(p['title'], quote=True)} — SkinDeep Research"><meta property="og:description" content="{escape(p['description'], quote=True)}"><meta property="og:image" content="https://skindeep.ai/images/og-image.png">
{extra_css}<link rel="stylesheet" href="{url('lab.css')}"><script src="{url('scripts/lab.js')}" defer></script>{scripts}
</head><body class="lab-page {'demo-page' if name == 'demo.html' else 'home-page' if name == 'index.html' else ''}">
<a class="skip-link" href="#main">Skip to content</a>
<header class="lab-header"><a class="wordmark" href="{url('index.html')}">SkinDeep<span>RESEARCH</span></a><button class="lab-menu" type="button" aria-expanded="false" aria-controls="lab-nav">Research menu <span aria-hidden="true">☰</span></button><span class="header-note">{header_note}</span></header>
<aside class="lab-sidebar" id="lab-nav"><nav aria-label="Research navigation">{nav}</nav><div class="rail-note"><a href="{url('sitemap.html')}">All pages</a><a href="https://github.com/skindeepai">Source on GitHub ↗</a></div></aside>
<main class="lab-main" id="main" tabindex="-1">{topline}
<h1>{escape(p['title'])}</h1><p class="page-lead">{escape(p['description'])}</p>{body}
<footer class="lab-footer"><span>SkinDeep.ai · Steve Seguin</span><a href="{url('history.html')}">History</a><a href="{url('research.html')}">Research FAQ</a><a href="mailto:contact@skindeep.ai">Contact</a></footer></main></body></html>
'''
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding='utf-8', newline='\n')
    urls = ['https://skindeep.ai/' + ('' if n == 'index.html' else n) for n in pages]
    (ROOT / 'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + ''.join('<url><loc>' + u + '</loc></url>\n' for u in urls) + '</urlset>\n', encoding='utf-8', newline='\n')
    print(f'Built {len(pages)} static pages.')

if __name__ == '__main__': build()

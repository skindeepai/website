"""Check generated pages and their local links without network access."""
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.append(attrs['id'])
        for key in ['href','src']:
            if key in attrs:self.links.append(attrs[key])

def main():
    names=list(json.loads((ROOT/'content/pages.json').read_text(encoding='utf-8')))
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'archive/2026-09').rglob('*.html')]
    errors=[];links=0
    for name in names:
        path=ROOT/name;parser=Links();parser.feed(path.read_text(encoding='utf-8'))
        if len(parser.ids)!=len(set(parser.ids)):errors.append(f'{name}: duplicate ID')
        for link in parser.links:
            target=urlsplit(link)
            if target.scheme or target.netloc or not target.path:continue
            links+=1;local=(path.parent/unquote(target.path)).resolve()
            if local.is_dir():local=local/'index.html'
            if not local.is_file():errors.append(f'{name}: missing {link}')
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Checked {len(names)} current/archive pages and {links} local references.')
if __name__=='__main__':main()

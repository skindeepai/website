"""Check generated pages and their local links without network access."""
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[];self.anchors=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.append(attrs['id']);self.anchors.add(attrs['id'])
        if tag=='a' and 'name' in attrs:self.anchors.add(attrs['name'])
        for key in ['href','src']:
            if key in attrs:self.links.append(attrs[key])

def main():
    names=list(json.loads((ROOT/'content/pages.json').read_text(encoding='utf-8')))
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'archive/2026-09').rglob('*.html')]
    errors=[];links=0;fragments=0;parsed={}
    def parse(path):
        if path not in parsed:
            parser=Links();parser.feed(path.read_text(encoding='utf-8'));parsed[path]=parser
        return parsed[path]
    for name in names:
        path=(ROOT/name).resolve();parser=parse(path)
        if len(parser.ids)!=len(set(parser.ids)):errors.append(f'{name}: duplicate ID')
        for link in parser.links:
            target=urlsplit(link)
            if target.scheme or target.netloc:continue
            links+=1;local=(path.parent/unquote(target.path)).resolve() if target.path else path
            if local.is_dir():local=local/'index.html'
            if not local.is_file():errors.append(f'{name}: missing {link}');continue
            if target.fragment and local.suffix.lower() in ['.html','.htm']:
                # Text-fragment directives are interpreted by the browser, not element IDs.
                anchor=unquote(target.fragment.split(':~:text=',1)[0])
                if anchor:
                    fragments+=1
                    if anchor not in parse(local).anchors:errors.append(f'{name}: missing anchor {link}')
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Checked {len(names)} current/archive pages, {links} local references and {fragments} anchors.')
if __name__=='__main__':main()

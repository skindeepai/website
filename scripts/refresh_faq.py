"""Render direct answers, retaining the old research IDs for incoming links."""
import json
import re
from html import escape
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def update(pages):
    rows=json.loads((ROOT/'content/research-faq.json').read_text(encoding='utf-8'))
    identifiers=[identifier for row in rows for identifier in row['ids']]
    assert len(identifiers)==29 and len(set(identifiers))==29
    body='';group=None
    for index,row in enumerate(rows):
        if row['group']!=group:
            if group is not None:body+='</section>'
            group=row['group'];body+='<section><h2>'+escape(group)+'</h2>'
        body+='<details class="faq" id="'+row['ids'][0]+'"'+(' open' if index==0 else '')+'><summary>'+escape(row['question'])+'</summary>'
        body+=''.join('<span id="'+identifier+'"></span>' for identifier in row['ids'][1:])
        body+='<p>'+escape(row['answer'])+'</p><p class="next-links">'+' · '.join('<a href="@/'+href+'">'+escape(label)+'</a>' for href,label in row['links'])+'</p></details>'
    body+='</section><p class="small"><a href="@/docs/research-status.md">Evidence behind these answers</a> · <a href="@/EXPERIMENTS.md">Technical study protocols</a></p>'
    pages['research.html'].update(title='Research FAQ',description='What the experiments show, how the methods work, and what is still untested.',status='',body=body)
    # Make incoming labels agree with the page. Detailed protocol links stay technical.
    for page in pages.values():
        def replace(match):
            label=match.group(1)
            if 'protocol' in label.lower() or re.search(r'[PDCX]\d\d',label):
                return '<a href="@/EXPERIMENTS.md">Technical study protocols</a>'
            return '<a href="@/research.html">Research FAQ</a>'
        page['body']=re.sub(r'<a href="@/research.html">([^<]+)</a>',replace,page.get('body',''))

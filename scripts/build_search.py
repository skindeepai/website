"""Build source-linked answer passages and, optionally, their local search vectors."""
import hashlib
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'search'
MODEL = 'all-MiniLM-L6-v2'


def update(pages):
    pages['search.html'] = dict(title='Search the research',
        description='Ask in your own words. Find a short answer, then follow the evidence.',
        template='content/search.html', styles=['search.css'], scripts=['scripts/research-search-core.js', 'scripts/research-search.js'])


def anchors(body):
    """Stable section targets shared by the published page and answer index."""
    count = 0
    used = set(re.findall(r'\bid=[\"\x27]([^\"\x27]+)', body))
    def add(match):
        nonlocal count
        if re.search(r'\bid\s*=', match[0]):
            return match[0]
        count += 1
        name = 'answer-' + str(count)
        while name in used:
            count += 1
            name = 'answer-' + str(count)
        used.add(name)
        return match[0][:-1] + ' id="' + name + '">'
    return re.sub(r'<(?:h2|h3|summary)\b[^>]*>', add, body)


class Passages(HTMLParser):
    def __init__(self, title, description):
        super().__init__(convert_charrefs=True)
        self.sections = []
        self.current = dict(title=title, anchor='', paragraphs=[description])
        self.stack = []
        self.heading = None
        self.block = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('br', 'hr', 'img', 'input', 'meta', 'link', 'source', 'wbr'):
            return
        skip = (self.stack and self.stack[-1][1]) or tag in ('script', 'style', 'svg', 'figure', 'table', 'pre', 'nav', 'form', 'button') or any(c in attrs.get('class', '').split() for c in ('next-links', 'page-links', 'ctx-jumps', 'actions', 'source-list'))
        self.stack.append((tag, bool(skip), attrs.get('id', '')))
        if skip:
            return
        if tag in ('h2', 'h3', 'summary'):
            anchor = attrs.get('id', '')
            if len(self.stack) > 1 and anchor.startswith('answer-'):
                parent = self.stack[-2]
                if parent[0] in ('section', 'details') and parent[2]:
                    anchor = parent[2]
            self.heading = dict(title='', anchor=anchor, paragraphs=[])
        elif tag in ('p', 'li', 'dd') and self.block is None:
            self.block = [tag, '']

    def handle_data(self, text):
        if self.stack and self.stack[-1][1]:
            return
        if self.heading is not None:
            self.heading['title'] += text
        elif self.block is not None:
            self.block[1] += text

    def handle_endtag(self, tag):
        if not self.stack:
            return
        skip = self.stack[-1][1]
        if not skip and self.block is not None and tag in ('strong', 'span', 'a'):
            self.block[1] += ' '
        if not skip and self.heading is not None and tag in ('h2', 'h3', 'summary'):
            self.sections.append(self.current)
            self.current = self.heading
            self.heading = None
        if not skip and self.block is not None and tag == self.block[0]:
            text = re.sub(r'\s+', ' ', self.block[1]).strip()
            if len(text) >= 55:
                self.current['paragraphs'].append(text)
            self.block = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def finish(self):
        return self.sections + [self.current]


def build(pages, bodies):
    records = json.loads((ROOT / 'content/search-answers.json').read_text(encoding='utf-8'))
    for row in records:
        row['curated'] = True
    seen = {r['answer'].casefold() for r in records}
    excluded = {'search.html', 'sitemap.html', 'history.html', 'results-record.html', 'whitepaper.html', 'vc-analysis.html', 'investment-response.html', 'roadmap.html', 'future-explorations.html'}
    for name, p in pages.items():
        if '/' in name or name in excluded:
            continue
        parser = Passages(p['title'], p['description'])
        parser.feed(bodies[name])
        topic = 'Longer context' if name.startswith('context') else p.get('journey', {}).get('topic', {}).get('label', 'Research')
        for section in parser.finish():
            answer = '\n\n'.join(section['paragraphs'])
            if len(answer) < 110 or answer.casefold() in seen:
                continue
            # Preserve whole paragraphs and their qualifications; the UI folds long excerpts.
            seen.add(answer.casefold())
            records.append(dict(title=unescape(section['title']).strip(), answer=answer,
                url=name + ('#' + section['anchor'] if section['anchor'] else ''),
                page=p['title'], topic=topic, questions=[]))
    questions = json.loads((ROOT / 'content/search-questions.json').read_text(encoding='utf-8'))
    missing = questions.keys() - {r['url'] for r in records}
    if missing:
        raise ValueError('Question aliases have no answer: ' + ', '.join(sorted(missing)))
    for index, row in enumerate(records):
        row['id'] = index
        row['questions'].extend(questions.get(row['url'], []))
        target = urlsplit(row['url'])
        if target.scheme or target.netloc or target.path not in bodies:
            raise ValueError('Answer needs a current local source: ' + row['url'])
        if target.fragment and not re.search(r'\bid=[\"\x27]' + re.escape(target.fragment) + r'[\"\x27]', bodies[target.path]):
            raise ValueError('Answer source section missing: ' + row['url'])
    raw = (json.dumps(records, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    OUT.mkdir(exist_ok=True)
    (OUT / 'answers.json').write_bytes(raw)
    print(f'Indexed {len(records)} source-linked answers across {len(set(r["url"].split("#")[0] for r in records))} pages.')


def embedding_text(row):
    return row['page'] + '. ' + row['title'] + '\n' + row['answer']


def vectors():
    # Optional publishing step; the normal website build uses only the standard library.
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    model = ROOT / 'models' / MODEL
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model / 'onnx/model_quantized.onnx'), options, providers=['CPUExecutionProvider'])
    raw = (OUT / 'answers.json').read_bytes()
    rows = json.loads(raw)
    result = []
    record_ids = []
    passages = []
    for row in rows:
        # Separate vectors preserve the meaning of each alternate question.
        texts = [embedding_text(row)] + row['questions']
        if '?' in row['title']:
            texts.append(row['title'])
        passages.extend((row['id'], text) for text in dict.fromkeys(texts))
    for record_id, text in passages:
        encoded = tokenizer.encode(text)
        feed = {'input_ids': np.array([encoded.ids], dtype=np.int64),
                'attention_mask': np.array([encoded.attention_mask], dtype=np.int64),
                'token_type_ids': np.array([encoded.type_ids], dtype=np.int64)}
        hidden = session.run(None, {i.name: feed[i.name] for i in session.get_inputs()})[0][0]
        # The tokenizer file may contain a saved padding policy. Match the browser's
        # attention-mask pooling even if padding is enabled again for batched export.
        mask = np.asarray(encoded.attention_mask, dtype=np.float32)[:, None]
        vector = (hidden * mask).sum(axis=0) / max(float(mask.sum()), 1)
        result.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
        record_ids.append(record_id)
    data = np.asarray(result, dtype='<f4').tobytes()
    (OUT / 'vectors.f32').write_bytes(data)
    manifest = dict(model=MODEL, revision='751bff37182d3f1213fa05d7196b954e230abad9', dimensions=384,
        count=len(rows), vector_count=len(result), record_ids=record_ids,
        answers_sha256=hashlib.sha256(raw).hexdigest(), vectors_sha256=hashlib.sha256(data).hexdigest())
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Prepared {len(result)} answer/question vectors ({len(data):,} bytes).')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--vectors']:
        vectors()
    else:
        raise SystemExit('Run build_site.py first; use build_search.py --vectors to prepare semantic search (numpy, tokenizers, onnxruntime).')

"""Bounded real-data search pilot: SciFact, BM25, MiniLM and fixed rank fusion.

Use --prepare, then --run. No labels are used to train or choose the methods.
CPU pools are capped at two threads. Downloads and model files stay in .cache.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'experiments/.cache/search-ranking'
OUT = ROOT / 'results/search-ranking'
sys.path.insert(0, str(ROOT / 'experiments/.cache/search-ranking-tools'))
sys.path.insert(0, str(ROOT / 'experiments/.cache/replay-runtime'))
MODEL = 'Xenova/all-MiniLM-L6-v2'
MODEL_REV = '751bff37182d3f1213fa05d7196b954e230abad9'
DATA_REV = 'b3b5335604bf5ee3c4447671af975ea25143d4f5'
QRELS_REV = '2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029'
SEED = 45109


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            length = int(response.headers.get('Content-Length', 0))
            if length > 100_000_000:
                raise ValueError('Unexpectedly large download: ' + url)
            data = response.read(100_000_001)
            if len(data) > 100_000_000:
                raise ValueError('Download exceeded its size budget.')
        path.write_bytes(data)
    return {'url': url, 'sha256': digest(path), 'bytes': path.stat().st_size}


def prepare():
    import pyarrow.parquet as parquet
    sources = {}
    files = {
        'corpus.parquet': f'https://huggingface.co/datasets/BeIR/scifact/resolve/{DATA_REV}/corpus/corpus-00000-of-00001.parquet',
        'queries.parquet': f'https://huggingface.co/datasets/BeIR/scifact/resolve/{DATA_REV}/queries/queries-00000-of-00001.parquet',
        'test.tsv': f'https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/{QRELS_REV}/test.tsv',
    }
    for name, url in files.items():
        sources[name] = download(url, CACHE / name)
    model_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'onnx/model_quantized.onnx']
    for name in model_files:
        sources['model/' + name] = download(f'https://huggingface.co/{MODEL}/resolve/{MODEL_REV}/{name}', CACHE / 'model' / name)
    corpus = parquet.read_table(CACHE / 'corpus.parquet', use_threads=False).to_pylist()
    queries = {row['_id']: row['text'] for row in parquet.read_table(CACHE / 'queries.parquet', use_threads=False).to_pylist()}
    qrels = defaultdict(dict)
    for line in (CACHE / 'test.tsv').read_text(encoding='utf-8').splitlines()[1:]:
        query_id, doc_id, score = line.split('\t')
        if int(score) > 0:
            qrels[query_id][doc_id] = int(score)
    ids = sorted(random.Random(SEED).sample(sorted(qrels), 100))
    assert len(corpus) == 5183 and len(qrels) == 300
    assert len({row['_id'] for row in corpus}) == len(corpus)
    corpus_ids = {row['_id'] for row in corpus}
    assert all(set(qrels[qid]) <= corpus_ids for qid in ids)
    protocol = {
        'scope': 'Exploratory 100-query ranking pilot over the full fixed 5,183-abstract SciFact corpus, not open-web search.',
        'created': '2026-09-20', 'seed': SEED,
        'dataset': 'BEIR SciFact', 'data_revision': DATA_REV, 'qrels_revision': QRELS_REV,
        'query_ids': ids, 'candidate_count': len(corpus), 'sources': sources,
        'query_selection': 'Random sample of 100 from sorted official test qids, using Python Random seed 45109; fixed before model scoring.',
        'human_provenance': 'Scientific abstracts are real papers; claims were authored by experts for research and evidence judgments are human annotations. These are not production search logs.',
        'model': MODEL, 'model_revision': MODEL_REV, 'dtype': 'q8 ONNX', 'layers': 6, 'dimensions': 384,
        'embedding': 'Attention-mask mean pooling of final hidden states, L2 normalization; truncate to 256 WordPieces including special tokens; no generation.',
        'bm25': {'k1': 1.2, 'b': 0.75, 'tokens': '[a-z0-9]+ lowercase', 'stopwords': False, 'stemming': False, 'document_input': 'full title plus abstract'},
        'hybrid': 'Reciprocal-rank fusion of complete BM25 and MiniLM rankings with fixed constant 60; no tuning.',
        'metrics': ['hit@1', 'recall@5', 'ndcg@10', 'mrr@10'],
        'timing': 'Three warm per-query repeats, method order rotates; includes query preparation/encoding, scoring all candidates, sorting and fusion. Excludes model load, corpus indexing/encoding and evaluation metrics; these setup costs are reported separately.',
        'candidate_scope': 'Every method ranks the same complete 5,183-document fixed corpus. No gold document is injected into a shortlist.',
        'limits': ['Relevant means evidence supporting OR refuting the claim, not that a claim is true.', 'Unjudged documents count as nonrelevant under benchmark scoring; judgments may be incomplete.', 'BM25 reads full abstracts; MiniLM truncates at 256 tokens. Report truncation; this is a method comparison, not an input-length-controlled ablation.', 'Pretraining may include related scientific text. No claims of unseen-document pretraining or external replication.', 'No method or hyperparameter selected on these test outcomes.'],
    }
    if (OUT / 'protocol.json').exists():
        assert json.loads((OUT / 'protocol.json').read_text(encoding='utf-8')) == protocol, 'Protocol changed; use a new study directory.'
    else:
        write('protocol.json', protocol)
    write('corpus.json', {'dataset': 'SciFact via BEIR', 'revision': DATA_REV, 'license': 'Abstract database: ODC-By 1.0 per original SciFact release; retain original sources.', 'documents': [{'id': row['_id'], 'title': row['title'], 'text': row['text']} for row in corpus]})
    write('queries.json', [{'id': qid, 'text': queries[qid], 'relevant': qrels[qid]} for qid in ids])
    print('Prepared', len(corpus), 'real abstracts and', len(ids), 'held-out human-authored claims.', flush=True)


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def bm25_index(documents):
    lengths, postings = [], defaultdict(list)
    for index, document in enumerate(documents):
        terms = Counter(tokenize(document['title'] + '\n' + document['text']))
        lengths.append(sum(terms.values()))
        for term, frequency in terms.items():
            postings[term].append((index, frequency))
    average = sum(lengths) / len(lengths)
    index = {}
    for term, values in postings.items():
        idf = math.log(1 + (len(lengths) - len(values) + .5) / (len(values) + .5))
        index[term] = [(i, idf * frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * lengths[i] / average))) for i, frequency in values]
    return index


def bm25_score(query, index, count):
    import numpy as np
    scores = np.zeros(count, dtype=np.float32)
    for term in set(tokenize(query)):
        for position, value in index.get(term, []):
            scores[position] += value
    return scores


def ordered(scores):
    import numpy as np
    return np.argsort(-scores, kind='stable')


def measures(order, relevant, documents):
    gains = [1 if documents[int(i)]['id'] in relevant else 0 for i in order[:10]]
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return {'hit@1': gains[0], 'recall@5': sum(gains[:5]) / len(relevant),
            'ndcg@10': sum(g / math.log2(i + 2) for i, g in enumerate(gains)) / ideal,
            'mrr@10': next((1 / (i + 1) for i, g in enumerate(gains) if g), 0)}


def run():
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
    import platform
    if (OUT / 'result.json').exists():
        raise RuntimeError('Completed study exists; do not overwrite measured outcomes.')
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    for name, source in protocol['sources'].items():
        assert digest(CACHE / name) == source['sha256'], name
    documents = json.loads((OUT / 'corpus.json').read_text(encoding='utf-8'))['documents']
    queries = json.loads((OUT / 'queries.json').read_text(encoding='utf-8'))
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(CACHE / 'model', local_files_only=True)
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(CACHE / 'model/onnx/model_quantized.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    input_names = {value.name for value in session.get_inputs()}
    load_seconds = time.perf_counter() - load_started
    def encode(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='np')
        feed = {key: value.astype(np.int64) for key, value in inputs.items() if key in input_names}
        hidden = session.run(None, feed)[0]
        mask = inputs['attention_mask'][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
    started = time.perf_counter(); index = bm25_index(documents); lexical_setup = time.perf_counter() - started
    started = time.perf_counter(); vectors = []; truncated = 0
    for offset in range(0, len(documents), 16):
        texts = [d['title'] + '\n' + d['text'] for d in documents[offset:offset + 16]]
        truncated += sum(len(tokenizer.encode(text, truncation=False)) > 256 for text in texts)
        vectors.extend(encode(texts))
        if offset % 512 == 0:
            print('Encoded', min(offset + 16, len(documents)), '/', len(documents), flush=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    embedding_setup = time.perf_counter() - started
    vectors.astype('<f4').tofile(OUT / 'embeddings.f32')
    # Separate warm-up; its text is not an evaluation claim.
    encode(['Find scientific evidence about cell growth.'])
    bm25_score('cell growth', index, len(documents))
    times = {method: [] for method in ['bm25', 'minilm', 'hybrid']}
    predictions = []
    query_truncation = 0
    for query_number, query in enumerate(queries):
        query_truncation += len(tokenizer.encode(query['text'], truncation=False)) > 256
        first_orders = {}
        for repeat in range(3):
            methods = ['bm25', 'minilm', 'hybrid']
            rotation = (query_number + repeat) % 3
            for method in methods[rotation:] + methods[:rotation]:
                started = time.perf_counter()
                if method == 'bm25':
                    ranking = ordered(bm25_score(query['text'], index, len(documents)))
                elif method == 'minilm':
                    ranking = ordered(vectors @ encode([query['text']])[0])
                else:
                    lexical = ordered(bm25_score(query['text'], index, len(documents)))
                    dense = ordered(vectors @ encode([query['text']])[0])
                    fusion = np.zeros(len(documents), dtype=np.float32)
                    fusion[lexical] += 1 / (60 + np.arange(1, len(documents) + 1))
                    fusion[dense] += 1 / (60 + np.arange(1, len(documents) + 1))
                    ranking = ordered(fusion)
                elapsed = (time.perf_counter() - started) * 1000
                times[method].append({'query_id': query['id'], 'repeat': repeat, 'ms': elapsed})
                if repeat == 0:
                    first_orders[method] = ranking.tolist()
                else:
                    assert ranking.tolist() == first_orders[method], 'Ranking changed across repeated deterministic inference.'
        predictions.append({'query_id': query['id'], 'methods': {method: {'top10': [documents[i]['id'] for i in order[:10]], 'metrics': measures(order, query['relevant'], documents)} for method, order in first_orders.items()}})
        if (query_number + 1) % 20 == 0:
            print('Scored', query_number + 1, '/ 100 queries.', flush=True)
    metrics = {}
    for method in times:
        values = np.array([row['ms'] for row in times[method]])
        metrics[method] = {key: float(np.mean([row['methods'][method]['metrics'][key] for row in predictions])) for key in protocol['metrics']}
        metrics[method]['correct_first'] = sum(row['methods'][method]['metrics']['hit@1'] for row in predictions)
        metrics[method]['latency_ms'] = {'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)), 'mean': float(np.mean(values)), 'repeats': 3, 'total_query_runs': len(values)}
    manifest = {'model': MODEL, 'revision': MODEL_REV, 'dimensions': 384, 'layers': 6, 'dtype': 'q8', 'maxTokens': 256,
                'count': len(documents), 'corpusSHA256': digest(OUT / 'corpus.json'), 'embeddingsSHA256': digest(OUT / 'embeddings.f32'),
                'corpusBytes': (OUT / 'corpus.json').stat().st_size, 'embeddingsBytes': (OUT / 'embeddings.f32').stat().st_size,
                'examples': [{'id': q['id'], 'text': q['text']} for q in queries[:3]], 'licenseURL': 'https://github.com/allenai/scifact/blob/master/LICENSE.md'}
    write('manifest.json', manifest)
    write('predictions.json', predictions); write('timings.json', times)
    write('result.json', {'scope': protocol['scope'], 'date': datetime.now(timezone.utc).isoformat(),
                         'queries': len(queries), 'candidates': len(documents), 'metrics': metrics,
                         'setup_seconds': {'model_load': load_seconds, 'bm25_index': lexical_setup, 'document_encoding': embedding_setup},
                         'truncated_documents': truncated, 'truncated_queries': query_truncation,
                         'runtime': {'python': platform.python_version(), 'platform': platform.platform(), 'processor': platform.processor(), 'onnxruntime': ort.__version__, 'numpy': np.__version__, 'intra_op_threads': 2, 'inter_op_threads': 1, 'execution': 'sequential'},
                         'protocol_sha256': digest(OUT / 'protocol.json'), 'script_sha256': digest(Path(__file__)),
                         'manifest_sha256': digest(OUT / 'manifest.json')})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    if args.run: run()

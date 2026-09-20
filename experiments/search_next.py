"""Frozen search follow-up: unseen 200 SciFact queries and a two-layer reranker.

Run --prepare, --develop, then --evaluate. Never overwrite completed stages.
All numerical pools are capped at two threads; methods execute sequentially.
"""
import search_ranking as base
import search_reranker as fusion
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

OUT = base.ROOT / 'results/search-next'
CACHE = base.ROOT / 'experiments/.cache/search-next'
MODEL = 'cross-encoder/ms-marco-TinyBERT-L2-v2'
REVISION = '81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc'
METHODS = ['bm25', 'minilm', 'fusion', 'rerank', 'gated']
THRESHOLDS = [0., .05, .1, .2, .3, .5, 1.01]


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def qrels(path):
    result = defaultdict(dict)
    for line in path.read_text(encoding='utf-8').splitlines()[1:]:
        qid, did, score = line.split('\t')
        if int(score) > 0:
            result[qid][did] = int(score)
    return result


def prepare():
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Prepared study exists; do not change its protocol.')
    old = read(base.OUT / 'protocol.json')
    learned = read(base.OUT / 'learned/protocol.json')
    # IDs alone define the untouched complement; no new-test scoring occurs here.
    test_ids = sorted(set(qrels(base.CACHE / 'test.tsv')) - set(old['query_ids']))
    assert len(test_ids) == 200
    assert not set(test_ids) & set(learned['train_ids'] + learned['development_ids'])
    files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'onnx/model.onnx', 'README.md']
    sources = {name: base.download(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}', CACHE / name) for name in files}
    dependencies = [base.ROOT / 'experiments/search_ranking.py', base.ROOT / 'experiments/search_reranker.py',
                    base.OUT / 'corpus.json', base.OUT / 'embeddings.f32', base.OUT / 'learned/model.json',
                    base.OUT / 'learned/protocol.json', base.CACHE / 'queries.parquet', base.CACHE / 'train.tsv', base.CACHE / 'test.tsv']
    dependencies += [base.CACHE / name for name in old['sources'] if name.startswith('model/')]
    write('protocol.json', {
        'created': datetime.now(timezone.utc).isoformat(),
        'scope': 'First evaluation on the remaining 200 official SciFact test queries; prior 100 excluded. Not independent external replication.',
        'development_ids': learned['development_ids'], 'test_ids': test_ids,
        'model': MODEL, 'revision': REVISION, 'sources': sources,
        'reranker': 'Unmodified pretrained MS MARCO TinyBERT, two layers, float32 ONNX CPU. Query plus title and abstract, max 256 tokens, longest-first truncation, batch all 20 pairs. Rank raw logits; no generated text.',
        'candidate_policy': 'BM25 top 20 from all 5183 abstracts. Never inject gold candidates. A reranker cannot recover a missing candidate.',
        'fusion': 'Previously frozen five-feature logistic model, trained on 600 claims and selected on 209 development claims. No refit.',
        'gate': {'signal': '(top BM25 score - second BM25 score) / max(top BM25 score, 1e-9)',
                 'action': 'Rerank only if signal < threshold; otherwise retain BM25.',
                 'threshold_grid': THRESHOLDS,
                 'selection': 'Among thresholds with development hit@1 at least always-rerank and nDCG@10 no more than .01 below it, minimize reranked queries; tie prefers smaller threshold. Always-rerank is eligible. Use all 209 old development queries.'},
        'timing': 'Three per-query warm repeats, five-method order rotated by query and repeat. Includes raw-query tokenization, dense encoding when used, candidate selection, pair tokenization/inference when used, sorting and gate. Excludes model load, existing document embedding creation and BM25 indexing. Concurrent host load may remain; no isolated hardware claim.',
        'pools': {'intra_op': 2, 'inter_op': 1, 'tokenizer_parallelism': False},
        'dependencies': {str(p.relative_to(base.ROOT)).replace('\\', '/'): base.digest(p) for p in dependencies},
        'script_sha256': base.digest(Path(__file__)),
        'limits': ['Expert-authored scientific claims, not production queries.', 'Incomplete relevance judgments; evidence can support or refute a claim.', 'Pretraining contamination unknown; official splits may share source papers.', 'Generic MS MARCO reranker receives no SciFact fine-tuning.', 'BM25 full text; dense and cross-encoder inputs bounded to 256 tokens.', 'One dataset, one frozen candidate model and one CPU machine. No deployment reliability guarantee.']})


class Runner:
    def __init__(self):
        import numpy as np
        import onnxruntime as ort
        import pyarrow.parquet as parquet
        from transformers import AutoTokenizer
        self.np, self.ort = np, ort
        self.protocol = read(OUT / 'protocol.json')
        assert base.digest(Path(__file__)) == self.protocol['script_sha256']
        for path, digest in self.protocol['dependencies'].items():
            assert base.digest(base.ROOT / path) == digest, path
        for name, source in self.protocol['sources'].items():
            assert base.digest(CACHE / name) == source['sha256'], name
        start = time.perf_counter()
        self.docs = read(base.OUT / 'corpus.json')['documents']
        self.index = base.bm25_index(self.docs)
        self.vectors = np.fromfile(base.OUT / 'embeddings.f32', dtype='<f4').reshape(len(self.docs), 384)
        self.index_load_seconds = time.perf_counter() - start
        self.texts = [d['title'] + '\n' + d['text'] for d in self.docs]
        self.queries = {r['_id']: r['text'] for r in parquet.read_table(base.CACHE / 'queries.parquet', use_threads=False).to_pylist()}
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 2
        settings.inter_op_num_threads = 1
        settings.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        start = time.perf_counter()
        self.dense_tok = AutoTokenizer.from_pretrained(base.CACHE / 'model', local_files_only=True)
        self.cross_tok = AutoTokenizer.from_pretrained(CACHE, local_files_only=True)
        self.dense = ort.InferenceSession(str(base.CACHE / 'model/onnx/model_quantized.onnx'), sess_options=settings, providers=['CPUExecutionProvider'])
        self.cross = ort.InferenceSession(str(CACHE / 'onnx/model.onnx'), sess_options=settings, providers=['CPUExecutionProvider'])
        self.model_load_seconds = time.perf_counter() - start
        self.weights = read(base.OUT / 'learned/model.json')
        self.threshold = None

    def encode(self, text):
        np = self.np
        tokens = self.dense_tok([text], padding=True, truncation=True, max_length=256, return_tensors='np')
        names = {i.name for i in self.dense.get_inputs()}
        hidden = self.dense.run(None, {k: v.astype(np.int64) for k, v in tokens.items() if k in names})[0]
        mask = tokens['attention_mask'][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        return pooled[0] / max(float(np.linalg.norm(pooled[0])), 1e-9)

    def cross_order(self, text, order):
        candidates = order[:20]
        tokens = self.cross_tok([text] * len(candidates), [self.texts[int(i)] for i in candidates], padding=True, truncation=True, max_length=256, return_tensors='np')
        names = {i.name for i in self.cross.get_inputs()}
        logits = self.cross.run(None, {k: v.astype(self.np.int64) for k, v in tokens.items() if k in names})[0].reshape(-1)
        ranked = candidates[base.ordered(logits)]
        return ranked, logits

    def rank(self, method, text):
        if method == 'minilm':
            return base.ordered(self.vectors @ self.encode(text)), False
        lexical = base.bm25_score(text, self.index, len(self.docs))
        order = base.ordered(lexical)
        if method == 'fusion':
            values, _, _ = fusion.features(lexical, self.vectors @ self.encode(text))
            return base.ordered(values @ self.np.asarray(self.weights['coefficients']) + self.weights['intercept']), False
        gap = float((lexical[order[0]] - lexical[order[1]]) / max(float(lexical[order[0]]), 1e-9))
        if method == 'rerank' or method == 'gated' and gap < self.threshold:
            return self.cross_order(text, order)[0], True
        return order, False


def develop():
    if (OUT / 'development.json').exists():
        raise RuntimeError('Development already frozen.')
    runner = Runner()
    labels = qrels(base.CACHE / 'train.tsv')
    rows = []
    for n, qid in enumerate(runner.protocol['development_ids']):
        text = runner.queries[qid]
        scores = base.bm25_score(text, runner.index, len(runner.docs))
        order = base.ordered(scores)
        ranked, logits = runner.cross_order(text, order)
        gap = float((scores[order[0]] - scores[order[1]]) / max(float(scores[order[0]]), 1e-9))
        rows.append({'id': qid, 'gap': gap,
                     'bm25': base.measures(order, labels[qid], runner.docs),
                     'rerank': base.measures(ranked, labels[qid], runner.docs),
                     'candidate_ids': [runner.docs[int(i)]['id'] for i in order[:20]], 'logits': logits.tolist()})
        if (n + 1) % 50 == 0:
            print('Development', n + 1, '/ 209', flush=True)
    ref = {k: sum(r['rerank'][k] for r in rows) / len(rows) for k in ['hit@1', 'ndcg@10']}
    candidates = []
    for threshold in THRESHOLDS:
        item = {'threshold': threshold, 'reranked': sum(r['gap'] < threshold for r in rows)}
        item.update({k: sum(r['rerank' if r['gap'] < threshold else 'bm25'][k] for r in rows) / len(rows) for k in ref})
        item['eligible'] = item['hit@1'] >= ref['hit@1'] - 1e-12 and item['ndcg@10'] >= ref['ndcg@10'] - .01 - 1e-12
        candidates.append(item)
    winner = min((c for c in candidates if c['eligible']), key=lambda c: (c['reranked'], c['threshold']))
    write('development.json', {'created': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': base.digest(OUT / 'protocol.json'), 'selected': winner, 'always_rerank': ref, 'candidates': candidates, 'queries': rows})
    print('Frozen gate', winner, flush=True)


def evaluate():
    if (OUT / 'result.json').exists():
        raise RuntimeError('Completed result exists; do not reuse the holdout.')
    runner = Runner()
    dev = read(OUT / 'development.json')
    assert dev['protocol_sha256'] == base.digest(OUT / 'protocol.json')
    runner.threshold = dev['selected']['threshold']
    # Persist exact selected artifact hash before evaluating new-test labels.
    seal = {'development_sha256': base.digest(OUT / 'development.json'), 'protocol_sha256': base.digest(OUT / 'protocol.json'), 'created': datetime.now(timezone.utc).isoformat()}
    if (OUT / 'evaluation-seal.json').exists():
        previous = read(OUT / 'evaluation-seal.json')
        assert all(previous[k] == seal[k] for k in ['development_sha256', 'protocol_sha256'])
    else:
        write('evaluation-seal.json', seal)
    labels = qrels(base.CACHE / 'test.tsv')
    for method in METHODS:
        runner.rank(method, 'Find scientific evidence about cell growth.')
    records, timings = [], []
    for n, qid in enumerate(runner.protocol['test_ids']):
        text = runner.queries[qid]
        saved = {}
        for repeat in range(3):
            rotation = (n + repeat) % len(METHODS)
            for method in METHODS[rotation:] + METHODS[:rotation]:
                start = time.perf_counter()
                ranking, reranked = runner.rank(method, text)
                ms = (time.perf_counter() - start) * 1000
                top10 = [runner.docs[int(i)]['id'] for i in ranking[:10]]
                timings.append({'id': qid, 'repeat': repeat, 'method': method, 'ms': ms, 'reranked': reranked})
                if repeat == 0:
                    saved[method] = {'top10': top10, 'reranked': reranked, **base.measures(ranking, labels[qid], runner.docs)}
                else:
                    assert top10 == saved[method]['top10'] and reranked == saved[method]['reranked'], 'Repeated inference changed.'
        candidates = base.ordered(base.bm25_score(text, runner.index, len(runner.docs)))[:20]
        candidate_ids = [runner.docs[int(i)]['id'] for i in candidates]
        # Truncation audit occurs outside all timed paths.
        lengths = [len(runner.cross_tok.encode(text, runner.texts[int(i)], truncation=False)) for i in candidates]
        records.append({'id': qid, 'text': text, 'relevant': labels[qid], 'bm25_top20': candidate_ids,
                        'candidate_recall': len(set(candidate_ids) & set(labels[qid])) / len(labels[qid]),
                        'candidate_has_relevant': bool(set(candidate_ids) & set(labels[qid])),
                        'cross_pairs_over_256': sum(length > 256 for length in lengths), 'methods': saved})
        if (n + 1) % 20 == 0:
            print('Fresh evaluation', n + 1, '/ 200', flush=True)
    metrics = {}
    for method in METHODS:
        values = runner.np.asarray([t['ms'] for t in timings if t['method'] == method])
        metrics[method] = {key: float(runner.np.mean([r['methods'][method][key] for r in records])) for key in ['hit@1', 'recall@5', 'ndcg@10', 'mrr@10']}
        metrics[method].update({'correct_first': sum(r['methods'][method]['hit@1'] for r in records),
                               'reranked_queries': sum(r['methods'][method]['reranked'] for r in records),
                               'latency_ms': {'median': float(runner.np.median(values)), 'mean': float(values.mean()), 'p95': float(runner.np.quantile(values, .95)), 'runs': len(values)}})
    write('predictions.json', records)
    write('timings.json', timings)
    write('result.json', {'created': datetime.now(timezone.utc).isoformat(), 'queries': len(records), 'candidates': len(runner.docs),
                          'metrics': metrics, 'gate_threshold': runner.threshold,
                          'candidate_recall@20': float(runner.np.mean([r['candidate_recall'] for r in records])),
                          'queries_with_relevant_candidate': sum(r['candidate_has_relevant'] for r in records),
                          'cross_pairs_truncated': sum(r['cross_pairs_over_256'] for r in records), 'cross_pairs': 20 * len(records),
                          'setup_seconds': {'model_load': runner.model_load_seconds, 'index_and_vector_load': runner.index_load_seconds, 'document_encoding': 'Reused frozen vectors; creation cost in prior pilot.'},
                          'runtime': {'python': platform.python_version(), 'platform': platform.platform(), 'onnxruntime': runner.ort.__version__, 'numpy': runner.np.__version__, 'intra_op_threads': 2, 'inter_op_threads': 1},
                          'protocol_sha256': base.digest(OUT / 'protocol.json'), 'development_sha256': base.digest(OUT / 'development.json'),
                          'script_sha256': base.digest(Path(__file__)), 'predictions_sha256': base.digest(OUT / 'predictions.json'), 'timings_sha256': base.digest(OUT / 'timings.json')})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--develop', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.develop:
        develop()
    if args.evaluate:
        evaluate()

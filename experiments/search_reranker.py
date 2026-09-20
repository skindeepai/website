"""One train-only learned-fusion follow-up; keeps the first SciFact pilot sealed.

Run directly. Quality only: no timing claims while other experiments may run.
"""
import search_ranking as base
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, str(base.ROOT / 'experiments/.cache/tooling'))
OUT = base.OUT / 'learned'
SEED = 45110
FEATURES = ['bm25_div_query_max', 'minilm_cosine', '60_div_60_plus_bm25_rank',
            '60_div_60_plus_dense_rank', 'normalized_bm25_times_cosine']


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def features(lexical, dense):
    import numpy as np
    left, right = base.ordered(lexical), base.ordered(dense)
    lr, dr = np.empty(len(left)), np.empty(len(right))
    lr[left] = np.arange(1, len(left) + 1)
    dr[right] = np.arange(1, len(right) + 1)
    normalized = lexical / max(float(lexical.max()), 1e-9)
    return np.column_stack([normalized, dense, 60 / (60 + lr), 60 / (60 + dr), normalized * dense]), left, right


def main():
    import numpy as np
    import onnxruntime as ort
    import pyarrow.parquet as parquet
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoTokenizer
    if (OUT / 'result.json').exists():
        raise RuntimeError('Completed follow-up exists; use a new study for further changes.')
    source = base.download(f'https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/{base.QRELS_REV}/train.tsv', base.CACHE / 'train.tsv')
    qrels = defaultdict(dict)
    for line in (base.CACHE / 'train.tsv').read_text(encoding='utf-8').splitlines()[1:]:
        qid, did, score = line.split('\t')
        if int(score) > 0:
            qrels[qid][did] = int(score)
    all_queries = {r['_id']: r['text'] for r in parquet.read_table(base.CACHE / 'queries.parquet', use_threads=False).to_pylist()}
    test = json.loads((base.OUT / 'queries.json').read_text(encoding='utf-8'))
    shuffled = sorted(qrels)
    random.Random(SEED).shuffle(shuffled)
    fit_ids, dev_ids = sorted(shuffled[:600]), sorted(shuffled[600:])
    assert len(qrels) == 809 and len(dev_ids) == 209
    assert not (set(fit_ids) & set(dev_ids))
    assert not (set(qrels) & {q['id'] for q in test})
    manifest = json.loads((base.OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert base.digest(base.OUT / 'corpus.json') == manifest['corpusSHA256']
    assert base.digest(base.OUT / 'embeddings.f32') == manifest['embeddingsSHA256']
    original = json.loads((base.OUT / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in original['sources'].items():
        assert base.digest(base.CACHE / name) == expected['sha256'], name
    protocol = {
        'study': 'Exploratory learned fusion after observing the first pilot; same frozen 100 test queries reused.',
        'created': datetime.now(timezone.utc).isoformat(), 'seed': SEED, 'source': source,
        'train_ids': fit_ids, 'development_ids': dev_ids, 'test_ids': [q['id'] for q in test],
        'features': FEATURES, 'regularization_C_grid': [.1, 1., 10.],
        'selection': 'Highest mean development nDCG@10; ties prefer smaller C. No refit on development. Test labels never choose C or features.',
        'fit_examples': 'All judged positives plus union of top 20 BM25 and top 20 MiniLM candidates per training query. Other candidates in this union are treated as negative. Balanced class weights.',
        'model': 'L2 logistic regression, lbfgs, max_iter=1000, random_state=45110; five features plus intercept. Scores are ranking signals, not calibrated probabilities.',
        'evaluation': 'Rank all 5,183 documents, without positive injection or candidate filtering. The same pretrained MiniLM and frozen document vectors as first pilot. No encoder fine-tuning.',
        'timing': 'Not measured; other experiments may run concurrently.',
        'limits': ['Same test reused after seeing baseline: exploratory, not fresh confirmation.', 'Claims about the same source paper may appear in official train and test; not a document-held-out study.', 'Unjudged hard negatives may be relevant; SciFact relevance judgments are incomplete.', 'Training examples inject positives for supervised fitting only; development and test candidates are always the full corpus.'],
        'baseline_protocol_sha256': base.digest(base.OUT / 'protocol.json'),
        'corpus_sha256': manifest['corpusSHA256'], 'embeddings_sha256': manifest['embeddingsSHA256'],
        'experiment_sha256': base.digest(base.ROOT / 'experiments/search_reranker.py'),
    }
    if (OUT / 'protocol.json').exists():
        old = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
        protocol['created'] = old['created']
        assert old == protocol, 'Prepared study changed.'
    else:
        write('protocol.json', protocol)
    documents = json.loads((base.OUT / 'corpus.json').read_text(encoding='utf-8'))['documents']
    doc_positions = {d['id']: i for i, d in enumerate(documents)}
    assert all(set(qrels[qid]) <= set(doc_positions) for qid in qrels)
    vectors = np.fromfile(base.OUT / 'embeddings.f32', dtype='<f4').reshape(len(documents), 384)
    index = base.bm25_index(documents)
    tokenizer = AutoTokenizer.from_pretrained(base.CACHE / 'model', local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(base.CACHE / 'model/onnx/model_quantized.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    input_names = {item.name for item in session.get_inputs()}

    def query_features(text):
        inputs = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors='np')
        hidden = session.run(None, {k: v.astype(np.int64) for k, v in inputs.items() if k in input_names})[0]
        mask = inputs['attention_mask'][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
        return features(base.bm25_score(text, index, len(documents)), vectors @ pooled[0])

    x, y = [], []
    for n, qid in enumerate(fit_ids):
        values, left, right = query_features(all_queries[qid])
        selected = sorted(set(map(int, left[:20])) | set(map(int, right[:20])) | {doc_positions[did] for did in qrels[qid]})
        x.extend(values[selected])
        y.extend(int(documents[i]['id'] in qrels[qid]) for i in selected)
        if n % 100 == 0:
            print('Training query features', n + 1, '/ 600', flush=True)
    x, y = np.asarray(x), np.asarray(y)
    models = [LogisticRegression(C=c, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=SEED).fit(x, y) for c in protocol['regularization_C_grid']]
    assert all(int(m.n_iter_[0]) < 1000 for m in models), 'A model did not converge.'
    dev_rows = []
    for qid in dev_ids:
        values, _, _ = query_features(all_queries[qid])
        dev_rows.append({'id': qid, 'candidates': [{ 'C': float(m.C), **base.measures(base.ordered(m.decision_function(values)), qrels[qid], documents)} for m in models]})
    dev_scores = [float(np.mean([r['candidates'][i]['ndcg@10'] for r in dev_rows])) for i in range(len(models))]
    winner = max(range(len(models)), key=lambda i: (dev_scores[i], -models[i].C))
    model = models[winner]
    # Freeze the selected model before accessing test relevance labels for scoring.
    write('model.json', {'features': FEATURES, 'coefficients': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]), 'C': float(model.C), 'ranking_only': True})
    write('development.json', {'selection_metric': 'ndcg@10', 'candidates': [{'C': float(m.C), 'ndcg@10': dev_scores[i]} for i, m in enumerate(models)], 'selected_C': float(model.C), 'queries': dev_rows})
    predictions = []
    for q in test:
        values, _, _ = query_features(q['text'])
        order = base.ordered(model.decision_function(values))
        predictions.append({'id': q['id'], 'top10': [documents[int(i)]['id'] for i in order[:10]], **base.measures(order, q['relevant'], documents)})
    metrics = {key: float(np.mean([r[key] for r in predictions])) for key in ['hit@1', 'recall@5', 'ndcg@10', 'mrr@10']}
    metrics['correct_first'] = sum(r['hit@1'] for r in predictions)
    baseline = json.loads((base.OUT / 'result.json').read_text(encoding='utf-8'))
    write('predictions.json', predictions)
    write('result.json', {'date': datetime.now(timezone.utc).isoformat(), 'queries': len(test), 'candidates': len(documents), 'train_queries': len(fit_ids), 'development_queries': len(dev_ids), 'training_pairs': len(y), 'positive_pairs': int(y.sum()), 'selected_C': float(model.C), 'metrics': metrics, 'timing': None, 'baseline_metrics': {k: {metric: v[metric] for metric in metrics} for k, v in baseline['metrics'].items()}, 'sklearn_version': sklearn.__version__, 'model_sha256': base.digest(OUT / 'model.json'), 'protocol_sha256': base.digest(OUT / 'protocol.json'), 'scope': protocol['study']})
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()

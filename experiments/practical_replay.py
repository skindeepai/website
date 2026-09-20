"""Independent exact-source practical reruns with retained numerical predictions.

Execute the sealed original source without changing training or evaluation logic.
Redirect its output and capture predict_proba returns to make outcomes inspectable.
This is reproduction on the same data, not another quality holdout or timing study.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/practical-replay'
SOURCES = {
    'privacy': 'results/practical-privacy/implementation.py',
    'receipts': 'experiments/practical_baselines.py',
    'routing': 'results/practical-routing/implementation.py',
}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def plan():
    return {
        'scope': 'Independent same-data reproduction, not new validation or comparative timing.',
        'threads': 2,
        'runner_sha256': sha(Path(__file__)),
        'sources': {task: {'path': path, 'sha256': sha(ROOT/path),
                          'original_protocol_sha256': sha(ROOT/f'results/practical-{task}/protocol.json'),
                          'original_result_sha256': sha(ROOT/f'results/practical-{task}/result.json')}
                    for task, path in SOURCES.items()},
        'execution': 'Compile exact archived/current source under original runner filename for path resolution. Replace only begin output directory and subclass classifiers to copy predict_proba returns after calling original implementation. No parameter or selection changes.',
        'diagnostics': 'Save probability matrices for every call. Privacy adds per-token gold labels; receipts add candidate amounts and selected/reference values; routing retains all class probabilities. No raw input text.',
        'comparison': 'Compare every original outcome after removing seconds fields. Retain any mismatch, never overwrite original artifacts.',
        'reproducibility_limit': 'Receipt liblinear did not set random_state originally; preserve that choice and report any difference rather than silently changing the training recipe.'
    }


def stripped(value):
    if isinstance(value, dict): return {k: stripped(v) for k, v in value.items() if k != 'seconds'}
    if isinstance(value, list): return [stripped(v) for v in value]
    return value


def run(task):
    assert read(OUT/'protocol.json') == plan(), 'Sealed replay inputs changed.'
    destination = OUT/task
    assert not destination.exists(), 'Preserve every previous replay.'
    original = ROOT/f'results/practical-{task}'
    source = ROOT/SOURCES[task]
    assert sha(source) == read(original/'protocol.json')['source_sha256']
    ns = {'__name__': 'independent_practical_replay', '__file__': str(ROOT/'experiments/practical_baselines.py')}
    exec(compile(source.read_text(encoding='utf-8'), ns['__file__'], 'exec'), ns)
    np = ns['np']; calls = []; fitted = []

    def capture(base):
        class Captured(base):
            def fit(self, *args, **kwargs):
                result = super().fit(*args, **kwargs); fitted.append(self); return result
            def predict_proba(self, *args, **kwargs):
                result = super().predict_proba(*args, **kwargs); calls.append(result.copy()); return result
        return Captured

    for name in ['LogisticRegression', 'SGDClassifier']:
        ns[name] = capture(ns[name])

    def begin(name, extra):
        assert name == task
        expected = read(original/'protocol.json')
        for key, value in extra.items(): assert value == expected[key], key
        destination.mkdir(parents=True)
        write(destination/'protocol.json', {'replay_protocol_sha256': sha(OUT/'protocol.json'),
                                          'original_protocol_sha256': sha(original/'protocol.json'),
                                          'source_sha256': sha(source), 'task': task})
        return destination

    ns['begin'] = begin
    with ns['threadpool_limits'](limits=2):
        ns[task]()
    records = read(destination/'records.json')
    arrays = {f'probabilities_{i}': value for i, value in enumerate(calls)}
    for i, model in enumerate(fitted):
        for name in ['coef_', 'intercept_', 'classes_', 'n_iter_']:
            arrays[f'model_{i}_{name}'] = getattr(model, name)
    diagnostics = []
    if task == 'privacy':
        docs = {d['doc_id']: d for d in read(ROOT/'experiments/.cache/practical/echr_test.json')}
        assert len(calls) == 1+len(records)
        threshold = read(destination/'result.json')['threshold']
        for i, record in enumerate(records, 1):
            doc = docs[record['id']]
            tokens = list(re.finditer(r'\w+|[^\w\s]', doc['text']))[:1200]
            spans = {(m['start_offset'], m['end_offset']) for a in doc['annotations'].values()
                     for m in a['entity_mentions'] if m['entity_type'] == 'PERSON'}
            labels = np.array([int(any(t.start() < end and t.end() > start for start, end in spans)) for t in tokens])
            arrays[f'labels_{i}'] = labels
            prediction = calls[i][:, 1] >= threshold
            actual = {'tp': int(((labels == 1)&prediction).sum()),
                      'fn': int(((labels == 1)&~prediction).sum()),
                      'fp': int(((labels == 0)&prediction).sum())}
            assert all(actual[k] == record['metrics'][k] for k in actual)
            diagnostics.append({'id': record['id'], 'probability_array': f'probabilities_{i}', 'label_array': f'labels_{i}'})
    elif task == 'receipts':
        assert len(calls) == len(records)
        folder = ROOT/'experiments/.cache/practical/receipts'
        for i, record in enumerate(records):
            amounts = []
            for line in (folder/(record['id']+'.csv')).read_text(encoding='utf-8').splitlines():
                fields = line.split(',', 8)
                if len(fields) == 9:
                    amounts.extend(float(m.group(1)) for m in re.finditer(r'(?<![\d.])(\d{1,6}\.\d{2})(?!\d)', fields[8]))
            assert len(amounts) == len(calls[i])
            gold_text = re.sub(r'[^\d.]', '', read(folder/(record['id']+'.json'))['total'])
            gold = float(gold_text) if gold_text else None
            chosen = amounts[int(calls[i][:, 1].argmax())]
            assert (chosen == gold) == record['learned_correct']
            diagnostics.append({'id': record['id'], 'candidate_amounts': amounts,
                                'probability_array': f'probabilities_{i}', 'chosen': chosen,
                                'largest': max(amounts), 'reference': gold})
    else:
        assert len(calls) == 2 and len(calls[1]) == len(records)
        for i, record in enumerate(records):
            assert int(calls[1][i].argmax()) == record['raw']
        diagnostics = {'development_probability_array': 'probabilities_0',
                       'test_probability_array': 'probabilities_1',
                       'test_ids': [r['id'] for r in records], 'class_order': fitted[0].classes_.tolist()}
    np.savez_compressed(destination/'numerical-predictions.npz', **arrays)
    write(destination/'diagnostics.json', diagnostics)
    comparison = {name: stripped(read(destination/name)) == stripped(read(original/name))
                  for name in ['result.json', 'records.json']}
    if (original/'selection.json').exists():
        comparison['selection.json'] = read(destination/'selection.json') == read(original/'selection.json')
    write(destination/'comparison.json', {'exact_same_outcomes': all(comparison.values()), 'checks': comparison,
          'probability_calls': len(calls), 'numerical_sha256': sha(destination/'numerical-predictions.npz'),
          'scope': 'Same source/training/data independently rerun. Timing includes capture instrumentation and is not benchmark evidence.'})
    print(task, comparison, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('task', choices=['prepare']+list(SOURCES)); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.task == 'prepare':
        assert not (OUT/'protocol.json').exists(); write(OUT/'protocol.json', plan())
    else:
        try: run(args.task)
        except Exception as error:
            write(OUT/(args.task+'-failure.json'), {'error': str(error), 'type': type(error).__name__}); raise

"""Timing-only follow-up on the fixed first 50 queries; launch when host jobs finish."""
import search_next as study
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

OUT = study.OUT / 'isolated'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Already prepared.'
    source = study.read(study.OUT / 'predictions.json')
    write('protocol.json', {'created': datetime.now(timezone.utc).isoformat(),
                           'ids': [r['id'] for r in source[:50]], 'methods': study.METHODS, 'repeats': 3,
                           'scope': 'Timing-only first 50 sorted queries from completed 200-query study. No new evaluation data or method selection.',
                           'execution': 'Launch after other experiment jobs finish; three rotating warm paired passes per query. Raw input preparation and all needed encoders/ranking included, setup excluded.',
                           'pools': {'intra_op': 2, 'inter_op': 1},
                           'dependencies': {name: study.base.digest(study.OUT / name) for name in ['protocol.json', 'development.json', 'predictions.json']},
                           'script_sha256': study.base.digest(Path(__file__))})


def run():
    assert not (OUT / 'result.json').exists(), 'Completed timing exists.'
    protocol = study.read(OUT / 'protocol.json')
    assert study.base.digest(Path(__file__)) == protocol['script_sha256']
    for name, digest in protocol['dependencies'].items():
        assert study.base.digest(study.OUT / name) == digest
    runner = study.Runner()
    runner.threshold = study.read(study.OUT / 'development.json')['selected']['threshold']
    references = {r['id']: r for r in study.read(study.OUT / 'predictions.json')}
    for method in study.METHODS:
        runner.rank(method, 'Find scientific evidence about cell growth.')
    records = []
    for n, qid in enumerate(protocol['ids']):
        for repeat in range(3):
            rotation = (n + repeat) % len(study.METHODS)
            for method in study.METHODS[rotation:] + study.METHODS[:rotation]:
                started = time.perf_counter()
                ranking, reranked = runner.rank(method, runner.queries[qid])
                ms = (time.perf_counter() - started) * 1000
                top10 = [runner.docs[int(i)]['id'] for i in ranking[:10]]
                expected = references[qid]['methods'][method]
                assert top10 == expected['top10'] and reranked == expected['reranked']
                records.append({'id': qid, 'repeat': repeat, 'method': method, 'ms': ms, 'reranked': reranked, 'top10': top10})
        if (n + 1) % 10 == 0:
            print('Timing', n + 1, '/ 50', flush=True)
    metrics = {}
    for method in study.METHODS:
        values = runner.np.asarray([r['ms'] for r in records if r['method'] == method])
        metrics[method] = {'mean_ms': float(values.mean()), 'median_ms': float(runner.np.median(values)), 'p95_ms': float(runner.np.quantile(values, .95)),
                           'correct_first': sum(references[qid]['methods'][method]['hit@1'] for qid in protocol['ids']),
                           'reranked_queries': sum(references[qid]['methods'][method]['reranked'] for qid in protocol['ids']), 'timed_calls': len(values)}
    write('records.json', records)
    write('result.json', {'created': datetime.now(timezone.utc).isoformat(), 'queries': len(protocol['ids']), 'metrics': metrics,
                          'all_top10_rankings_match': True, 'protocol_sha256': study.base.digest(OUT / 'protocol.json'),
                          'records_sha256': study.base.digest(OUT / 'records.json'), 'script_sha256': study.base.digest(Path(__file__)),
                          'runtime': {'onnxruntime': runner.ort.__version__, 'numpy': runner.np.__version__, 'intra_op': 2, 'inter_op': 1},
                          'limits': 'Warm timing on one CPU host; operating-system background load is uncontrolled. No new quality validation.'})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run()

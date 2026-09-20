"""Independently recompute saved ranking metrics and paired first-result changes."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/search-next'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def main():
    result, rows, timing = read('result.json'), read('predictions.json'), read('timings.json')
    protocol, dev = read('protocol.json'), read('development.json')
    assert len(rows) == 200 and len({r['id'] for r in rows}) == 200
    assert sorted(r['id'] for r in rows) == protocol['test_ids']
    assert not set(protocol['test_ids']) & set(protocol['development_ids'])
    old = json.loads((ROOT / 'results/search-ranking/protocol.json').read_text(encoding='utf-8'))
    assert not set(protocol['test_ids']) & set(old['query_ids'])
    assert len(timing) == 3000
    checks = []
    for method, summary in result['metrics'].items():
        totals = {key: [] for key in ['hit@1', 'recall@5', 'ndcg@10', 'mrr@10']}
        for row in rows:
            relevant = set(row['relevant'])
            pred = row['methods'][method]
            hits = [int(did in relevant) for did in pred['top10']]
            assert len(hits) == 10 and len(set(pred['top10'])) == 10
            ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
            measured = {'hit@1': hits[0], 'recall@5': sum(hits[:5]) / len(relevant),
                        'ndcg@10': sum(hit / math.log2(i + 2) for i, hit in enumerate(hits)) / ideal,
                        'mrr@10': next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0)}
            for key, value in measured.items():
                assert abs(value - pred[key]) < 1e-12
                totals[key].append(value)
            if pred['reranked']:
                assert set(pred['top10']) <= set(row['bm25_top20'])
        for key, values in totals.items():
            assert abs(statistics.mean(values) - summary[key]) < 1e-12
        assert sum(totals['hit@1']) == summary['correct_first']
        values = [t['ms'] for t in timing if t['method'] == method]
        assert len(values) == 600
        assert abs(statistics.mean(values) - summary['latency_ms']['mean']) < 1e-9
        assert abs(statistics.median(values) - summary['latency_ms']['median']) < 1e-9
        checks.append(method)
    paired = {}
    for method in result['metrics']:
        for reference in ['bm25', 'rerank']:
            if method == reference:
                continue
            gains = sum(r['methods'][method]['hit@1'] > r['methods'][reference]['hit@1'] for r in rows)
            losses = sum(r['methods'][method]['hit@1'] < r['methods'][reference]['hit@1'] for r in rows)
            paired[method + '_vs_' + reference] = {'corrected_first_results': gains, 'lost_first_results': losses, 'net': gains - losses}
    candidate_recall = statistics.mean(len(set(r['bm25_top20']) & set(r['relevant'])) / len(r['relevant']) for r in rows)
    assert abs(candidate_recall - result['candidate_recall@20']) < 1e-12
    candidate_hits = sum(bool(set(r['bm25_top20']) & set(r['relevant'])) for r in rows)
    assert candidate_hits == result['queries_with_relevant_candidate']
    eligible = [c for c in dev['candidates'] if c['eligible']]
    assert min(eligible, key=lambda c: (c['reranked'], c['threshold'])) == dev['selected']
    artifact = {'checks_passed': checks, 'predictions': len(rows), 'timed_calls': len(timing), 'paired_first_result_changes': paired,
                'candidate_recall@20': candidate_recall, 'queries_with_relevant_candidate': candidate_hits,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'limits': 'Recomputes retained records and invariants, not independent inference or new data.'}
    (OUT / 'audit.json').write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()

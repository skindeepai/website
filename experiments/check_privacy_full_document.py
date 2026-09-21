"""Independent Python reconstruction of browser classifier and name coverage."""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/privacy-full-document'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol, records, result, timing = [read(OUT / (name+'.json')) for name in ['protocol', 'records', 'result', 'timing']]
    for name, expected in protocol['hashes'].items():
        assert digest(ROOT / name) == expected, name
    assert digest(OUT / 'source.cjs') == protocol['hashes']['experiments/privacy_full_document.cjs']
    assert digest(OUT / 'core.cjs') == protocol['hashes']['scripts/practical-demo-core.js']
    assert digest(OUT / 'records.json') == result['record_sha256']
    assert digest(OUT / 'timing.json') == result['timing_sha256']
    assert [r['id'] for r in records] == protocol['documents']
    model = read(ROOT / 'models/practical/privacy.json')
    assert model['classes'] == [0, 1] and len(model['weights']) == 1
    weight = dict(zip(model['features'], model['weights'][0]))
    threshold = math.log(model['threshold'] / (1-model['threshold']))
    docs = {d['doc_id']: d for d in read(ROOT / 'experiments/.cache/practical/echr_test.json')}
    total_predictions = 0
    for record in records:
        doc = docs[record['id']]
        assert record['input_code_units'] == len(doc['text'].encode('utf-16-le')) // 2
        assert record['within_browser_limit'] == (record['input_code_units'] <= 30000)
        spans = sorted({(m['start_offset'], m['end_offset']) for a in doc['annotations'].values()
                        for m in a['entity_mentions'] if m['entity_type'] == 'PERSON'})
        all_tokens = list(re.finditer(r'\w+|[^\w\s]', doc['text']))
        for variant in ['prefix', 'full']:
            tokens = all_tokens[:1200] if variant == 'prefix' else all_tokens
            marked = []
            for i, token in enumerate(tokens):
                word = token.group()
                features = {'word': word.lower(), 'prev': tokens[i-1].group().lower() if i else '',
                            'next': tokens[i+1].group().lower() if i+1 < len(tokens) else '',
                            'title': word.istitle(), 'upper': word.isupper(), 'digit': word.isdigit(),
                            'suffix': word[-3:].lower(), 'prefix': word[:3].lower()}
                score = model['bias'][0]
                for key, value in features.items():
                    if isinstance(value, str):
                        score += weight.get(key+'='+value, 0)
                    else:
                        score += weight.get(key, 0)*value
                marked.append(score >= threshold)
            total_predictions += len(marked)
            labels = [any(t.start() < end and t.end() > start for start, end in spans) for t in tokens]
            confusion = {'tp': sum(y and p for y, p in zip(labels, marked)),
                         'fn': sum(y and not p for y, p in zip(labels, marked)),
                         'fp': sum(not y and p for y, p in zip(labels, marked))}
            observed = record[variant]
            assert observed['tokens'] == len(tokens)
            assert observed['confusion'] == confusion, (record['id'], variant)
            counts = dict.fromkeys(observed['counts'], 0)
            occurrences = []
            for start, end in spans:
                indexes = [i for i, t in enumerate(tokens) if t.start() < end and t.end() > start]
                count = sum(marked[i] for i in indexes)
                boundary = tokens[-1].end() if tokens else 0
                if start >= boundary:
                    status = 'beyond_prefix'
                elif end > boundary:
                    status = 'crosses_prefix'
                elif not indexes:
                    status = 'no_token_overlap'
                else:
                    status = 'fully_marked' if count == len(indexes) else 'partly_marked' if count else 'unmarked'
                counts[status] += 1
                occurrences.append(dict(start=start, end=end, status=status, observed_tokens=len(indexes), marked_tokens=count))
            assert occurrences == observed['occurrences'], (record['id'], variant)
            assert counts == observed['counts']
            assert observed['all_person_occurrences_marked'] == (bool(spans) and counts['fully_marked'] == len(spans))
    assert len(timing) == 200
    cursor = 0
    for repeat in range(2):
        for index, record in enumerate(records):
            order = ['full', 'prefix'] if (index+repeat) % 2 else ['prefix', 'full']
            for position, variant in enumerate(order, 1):
                measured = timing[cursor]
                assert [measured[k] for k in ['id', 'pass', 'position', 'variant', 'tokens']] == [record['id'], repeat+1, position, variant, record[variant]['tokens']]
                assert math.isfinite(measured['ms']) and measured['ms'] > 0
                cursor += 1
    for variant, summary in result['methods'].items():
        assert summary['tokens'] == sum(r[variant]['tokens'] for r in records)
        assert summary['complete_documents'] == sum(r[variant]['all_person_occurrences_marked'] for r in records)
        for kind, key in [('confusion', 'confusion'), ('span_counts', 'counts')]:
            for metric, value in summary[kind].items():
                assert value == sum(r[variant][key][metric] for r in records)
        assert abs(summary['mean_ms'] - sum(t['ms'] for t in timing if t['variant'] == variant)/100) < 1e-9
    assert result['documents'] == len(records) == 50
    assert result['person_occurrences'] == sum(len(r['full']['occurrences']) for r in records) == 445
    assert result['documents_exceeding_browser_limit'] == sum(not r['within_browser_limit'] for r in records)
    newly_complete = lost_complete = 0
    for r in records:
        for before, after in zip(r['prefix']['occurrences'], r['full']['occurrences']):
            newly_complete += after['status'] == 'fully_marked' and before['status'] != 'fully_marked'
            lost_complete += before['status'] == 'fully_marked' and after['status'] != 'fully_marked'
    assert newly_complete == result['new_complete_spans'] == 29
    assert lost_complete == result['lost_complete_spans'] == 0
    report = {'status': 'passed', 'independently_reconstructed_token_predictions': total_predictions,
              'documents': len(records), 'timed_calls': len(timing),
              'checks': ['All source, model and data hashes', 'Python Unicode feature reconstruction and scalar linear classification',
                         'Every per-document confusion count and name-span outcome', 'Exact paired execution order',
                         'All aggregate values, new detections and lost detections'],
              'source_sha256': digest(Path(__file__)), 'result_sha256': digest(OUT/'result.json')}
    (OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

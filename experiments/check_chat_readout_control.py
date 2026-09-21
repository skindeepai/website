"""Read-only inference-artifact audit; only the audit report is written."""
import hashlib
import json
import math
import statistics
from pathlib import Path

from chat_smoke_common import source_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/chat-readout-control'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (OUT / 'result.json').exists(), 'Wait for the complete inference run.'
    protocol = read(OUT / 'protocol.json')
    records = read(OUT / 'records.json')
    result = read(OUT / 'result.json')
    parent_path = ROOT / 'results/chat-output-steps/protocol.json'
    parent = read(parent_path)
    assert digest(parent_path) == protocol['parent_sha256']
    assert digest(OUT / 'source.py') == protocol['source_sha256']
    assert digest(ROOT / 'experiments/chat_readout_control.py') == protocol['source_sha256']
    assert protocol['data_sha256'] == parent['data_sha256']
    for split, expected in protocol['data_sha256'].items():
        assert digest(ROOT / f'experiments/.cache/toxicchat/toxic-chat_annotation_{split}.csv') == expected
    source = source_rows()
    assert protocol['evaluation'] == parent['evaluation']
    assert protocol['model'] == parent['model'] and protocol['revision'] == parent['revision']
    assert protocol['system_prompt'] == parent['prompts']['SAFE']
    methods = ['two_rows', 'full_vocabulary', 'two_rows_cache', 'generate_one']
    assert protocol['methods'] == methods and protocol['passes'] == 2
    assert len(protocol['evaluation']) == len({r['id'] for r in protocol['evaluation']}) == 50
    assert result['timed_calls'] == len(records) == 400
    assert result['unique_messages'] == 50
    assert result['all_layers_executed'] == 24 and result['layers_skipped'] == 0
    assert result['all_predictions_match'] is True
    assert result['all_selected_logits_within'] == .0005
    parent_records_path = ROOT / 'results/chat-output-steps/records.json'
    parent_result = read(ROOT / 'results/chat-output-steps/result.json')
    assert digest(parent_records_path) == parent_result['records_sha256']
    parent_rows = read(parent_records_path)
    references = {(r['id'], r['pass']): r for r in parent_rows if r['method'] == 'vocabulary2'}
    assert len(references) == 100
    cursor = 0
    maximum_logit_delta = 0.
    for repeat in range(2):
        for index, example in enumerate(protocol['evaluation']):
            shift = (index + repeat) % 4
            order = methods[shift:] + methods[:shift]
            if repeat == 1:
                order.reverse()
            group = records[cursor:cursor + 4]
            reference = references[(example['id'], repeat + 1)]
            assert source[example['id']]['label'] == example['label']
            for position, (row, method) in enumerate(zip(group, order), 1):
                assert (row['id'], row['label'], row['method'], row['pass'], row['position']) == (
                    example['id'], example['label'], method, repeat + 1, position)
                assert row['executed_layers'] == list(range(1, 25))
                assert row['vocabulary_calls'] == int(method in ['full_vocabulary', 'generate_one'])
                assert row['input_tokens'] == reference['input_tokens']
                assert row['prediction'] == reference['prediction']
                assert math.isfinite(row['ms']) and row['ms'] > 0
                assert all(math.isfinite(v) and v >= 0 for v in row['stages'].values())
                assert sum(row['stages'].values()) <= row['ms'] + 1e-6
                if method == 'generate_one':
                    assert row['raw'] in ['SAFE', 'BLOCK']
                    assert row['prediction'] == int(row['raw'] == 'BLOCK')
                    assert row['scores'] is None
                    assert set(row['stages']) == {'input_ms', 'generation_ms'}
                else:
                    assert len(row['scores']) == 2 and all(math.isfinite(v) for v in row['scores'])
                    assert row['prediction'] == int(row['scores'][1] > row['scores'][0])
                    assert row['raw'] is None
                    assert set(row['stages']) == {'input_ms', 'backbone_ms', 'readout_ms'}
                    delta = max(abs(a - b) for a, b in zip(row['scores'], reference['scores']))
                    assert delta < .0005
                    maximum_logit_delta = max(maximum_logit_delta, delta)
            cursor += 4
    assert cursor == len(records)
    for method in methods:
        selected = [r for r in records if r['method'] == method]
        first = [r for r in selected if r['pass'] == 1]
        summary = result['methods'][method]
        assert summary['n'] == 50
        expected = {
            'correct': sum(r['prediction'] == r['label'] for r in first),
            'missed_toxic': sum(r['label'] == 1 and r['prediction'] == 0 for r in first),
            'false_block': sum(r['label'] == 0 and r['prediction'] == 1 for r in first),
        }
        assert all(summary[k] == v for k, v in expected.items())
        assert abs(summary['mean_ms'] - statistics.mean(r['ms'] for r in selected)) < 1e-8
        for repeat in [1, 2]:
            assert abs(summary['pass_ms'][repeat - 1] - statistics.mean(
                r['ms'] for r in selected if r['pass'] == repeat)) < 1e-8
        for key, value in summary['mean_stages_ms'].items():
            assert abs(value - statistics.mean(r['stages'][key] for r in selected)) < 1e-8
    audit = {
        'status': 'passed', 'timed_calls': len(records), 'unique_messages': 50,
        'checks': ['Runner, parent and source-data hashes', 'Original human-annotated labels',
                   'Exact declared method ordering', 'Every block and vocabulary trace',
                   'Finite complete and stage timings', 'Independent confusion and timing arithmetic',
                   'Every decision matches both original timing passes', 'Stored logits match original two-row scores'],
        'maximum_logit_delta_vs_parent': maximum_logit_delta,
        'hashes': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in [
            Path(__file__), OUT / 'protocol.json', OUT / 'records.json', OUT / 'result.json',
            ROOT / 'results/chat-output-steps/records.json']},
        'limits': 'Internal code and saved-artifact review, not independent model or hardware replication. '
                  'Two/full vocabulary isolates projection. Generation also changes cache and API work; '
                  'the generation wrapper alone is not isolated. Manual cached-state deallocation falls outside timing.',
    }
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

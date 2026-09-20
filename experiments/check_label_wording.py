"""Independent record-level checks for a completed paired label-wording run."""
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / 'results/label-wording' / sys.argv[1]
protocol = json.loads((run / 'protocol.json').read_text())
records = json.loads((run / 'records.json').read_text())
result = json.loads((run / 'result.json').read_text())
assert hashlib.sha256((run / 'protocol.json').read_bytes()).hexdigest() == result['protocol_sha256']
assert hashlib.sha256((ROOT / 'experiments/label_wording.py').read_bytes()).hexdigest() == protocol['source_sha256']
assert protocol['prompts']['SAFE'].replace('SAFE', '{ALLOW}') == protocol['prompts']['OK'].replace('OK', '{ALLOW}')
assert all(len(ids) == 1 for ids in protocol['token_ids'].values())
gold = {r['id']: r['expected'] for r in protocol['evaluation']}
assert len(gold) == 50 and Counter(gold.values()) == {0: 25, 1: 25}
expected_order = [(p['repeat'], p['id'], allow, mode) for p in protocol['order'] for allow, mode in p['order']]
assert [(r['repeat'], r['id'], r['allow'], r['mode']) for r in records] == expected_order
assert len(records) == 600
by_key = {(r['id'], r['repeat'], r['mode'], r['allow']): r for r in records}
assert len(by_key) == len(records)
for r in records:
    assert r['expected'] == gold[r['id']]
    assert r['correct'] == (r['prediction'] == r['expected'])
    assert math.isfinite(r['total_ms']) and r['total_ms'] > 0
    assert abs(r['total_ms'] - r['preparation_ms'] - r['inference_and_readout_ms']) < 1e-6
    other = by_key[(r['id'], r['repeat'], r['mode'], 'OK' if r['allow'] == 'SAFE' else 'SAFE')]
    assert r['input_tokens'] == other['input_tokens']
    assert r['original_message_tokens'] == other['original_message_tokens']
    if r['mode'] == 'direct':
        assert r['prediction'] == int(r['scores'][1] > r['scores'][0])
        assert not r['generated_tokens']
    else:
        assert r['prediction'] == (1 if r['output'] == 'BLOCK' else 0 if r['output'] == r['allow'] else None)
        assert 1 <= len(r['generated_tokens']) <= 8
for key, summary in result['summaries'].items():
    mode, allow = key.split('_')
    rows = [r for r in records if r['mode'] == mode and r['allow'] == allow]
    first = [r for r in rows if r['repeat'] == 0]
    assert summary['correct'] == sum(r['correct'] for r in first)
    assert summary['invalid'] == sum(r['prediction'] is None for r in first)
    assert summary['toxic_missed'] == sum(r['expected'] == 1 and r['prediction'] != 1 for r in first)
    assert summary['safe_blocked'] == sum(r['expected'] == 0 and r['prediction'] == 1 for r in first)
    assert abs(summary['mean_ms'] - sum(r['total_ms'] for r in rows) / 150) < 1e-6
checks = dict(status='passed', calls=600, unique_messages=50, label_counts={'benign': 25, 'toxic': 25},
              checks=['source and protocol hashes', 'only allowed-label wording differs', 'single-token labels',
                      'exact preregistered order and sample IDs', 'equal paired input lengths', 'no duplicate calls',
                      'predictions agree with logits or raw replies', 'timing and accuracy arithmetic'],
              result_sha256=hashlib.sha256((run / 'result.json').read_bytes()).hexdigest())
(run / 'audit.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(checks, indent=2))

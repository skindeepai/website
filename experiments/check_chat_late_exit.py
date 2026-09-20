"""Independently recompute late-exit scores and timing summaries from saved data."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/chat-late-exit'
read = lambda name: json.loads((OUT / (name+'.json')).read_text(encoding='utf-8'))
protocol, result, rows = read('protocol'), read('result'), read('records')
assert hashlib.sha256((OUT / 'source.py').read_bytes()).hexdigest() == protocol['source_sha256']
assert hashlib.sha256((ROOT / 'experiments/chat_late_exit.py').read_bytes()).hexdigest() == protocol['source_sha256']
assert hashlib.sha256((ROOT / 'experiments/chat_smoke_common.py').read_bytes()).hexdigest() == protocol['shared_source_sha256']
assert hashlib.sha256((ROOT / 'results/chat-smoke/protocol.json').read_bytes()).hexdigest() == protocol['shared_protocol_sha256']
ids = protocol['splits']['evaluation']
assert len(ids) == len(set(ids)) == 50 and len(protocol['splits']['train']) == 384
assert not set(ids) & set(protocol['splits']['train'])
assert len(rows) == 450 and len({(r['id'], r['depth'], r['pass']) for r in rows}) == 450
cursor = 0
for repeat in range(3):
    for index, rid in enumerate(ids):
        depths = [22, 23, 24]
        shift = (index+repeat) % 3
        for depth in depths[shift:]+depths[:shift]:
            row = rows[cursor]
            assert (row['id'], row['depth'], row['pass']) == (rid, depth, repeat+1)
            cursor += 1
assert Counter(r['label'] for r in rows if r['depth'] == 24 and r['pass'] == 1) == {0:25, 1:25}
assert result['lm_head_calls'] == result['text_tokens_generated'] == 0
heads = np.load(OUT / 'heads.npz', allow_pickle=False)
features = np.load(OUT / 'features.npz', allow_pickle=False)
full = {r['id']: r for r in rows if r['depth'] == 24 and r['pass'] == 1}
for depth in [22, 23, 24]:
    prefix = str(depth)+'_'
    train = features[f'train_{depth}'].astype(np.float64)
    assert np.allclose(train.mean(0), heads[prefix+'mean'], atol=2e-5, rtol=1e-5)
    assert np.allclose(np.maximum(train.std(0, ddof=1), .05), heads[prefix+'std'], atol=2e-5, rtol=1e-5)
    x = features[f'evaluation_{depth}'].astype(np.float64)
    logits = ((x-heads[prefix+'mean'])/heads[prefix+'std']) @ heads[prefix+'weight'].T + heads[prefix+'bias']
    exps = np.exp(logits-logits.max(1, keepdims=True))
    probs = exps/exps.sum(1, keepdims=True)
    expected = dict(zip(ids, probs))
    selected = [r for r in rows if r['depth'] == depth]
    for row in selected:
        assert row['executed_layers'] == list(range(1, depth+1))
        assert row['prediction'] == int(expected[row['id']].argmax())
        assert abs(row['block_probability']-expected[row['id']][1]) < .0005
        assert np.isfinite(row['ms']) and row['ms'] > 0
    first = [r for r in selected if r['pass'] == 1]
    summary = result['methods'][str(depth)]
    calculated = {
        'correct': sum(r['prediction'] == r['label'] for r in first),
        'missed_toxic': sum(r['label'] == 1 and r['prediction'] == 0 for r in first),
        'false_block': sum(r['label'] == 0 and r['prediction'] == 1 for r in first),
        'added_errors': sum(full[r['id']]['prediction'] == r['label'] and r['prediction'] != r['label'] for r in first),
        'corrected_errors': sum(full[r['id']]['prediction'] != r['label'] and r['prediction'] == r['label'] for r in first),
        'additional_missed_toxic': sum(r['label'] == 1 and full[r['id']]['prediction'] == 1 and r['prediction'] == 0 for r in first),
    }
    assert all(summary[k] == v for k, v in calculated.items())
    totals = [sum(r['ms'] for r in selected if r['pass'] == p)/1000 for p in [1, 2, 3]]
    assert np.allclose(totals, summary['pass_total_seconds'])
    assert abs(statistics.mean(totals)*1000/50-summary['mean_ms_per_message']) < 1e-6
    assert summary['blocks_skipped'] == 24-depth
    assert abs(summary['fraction_blocks_skipped']-(24-depth)/24) < 1e-9
audit = {'status':'passed', 'timed_calls':len(rows), 'unique_evaluation_messages':len(ids),
    'checks':['Sealed source hashes', 'Train-only feature standardization', 'Independent NumPy classifier predictions',
              'All contiguous layer traces', 'Identical decisions across timing repeats', 'Per-class and added-error counts', 'Timing arithmetic'],
    'limits':'Artifact and implementation checks, not independent data or hardware replication.'}
(OUT / 'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(audit,indent=2))

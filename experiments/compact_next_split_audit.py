"""Independent split/selection audit: tokenize inputs, never load model weights."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/compact-next'
sys.path.insert(0, str(ROOT / 'experiments/.cache/replay-runtime'))
sys.path.insert(0, str(ROOT / 'experiments/.cache/tooling'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(metrics):
    return metrics['balanced_accuracy'], -metrics['missed_toxic'], metrics['correct']


def main():
    from transformers import AutoTokenizer
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol = read(OUT / 'protocol.json')
    old = read(ROOT / 'results/chat600/protocol.json')
    assert old['data_sha256'] == protocol['source_data_sha256']
    rows = {}
    for split in ['train', 'test']:
        path = ROOT / f'experiments/.cache/toxicchat/toxic-chat_annotation_{split}.csv'
        assert sha(path) == old['data_sha256'][split]
        with path.open(encoding='utf-8', newline='') as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                if row['human_annotation'] == 'True':
                    rows[f'{split}:{i}'] = row
    used = {rid for rid in rows if rid.startswith('train:')}
    for name, digest in protocol['historical_artifacts'].items():
        path = ROOT / name
        assert sha(path) == digest, name
        used.update(set(re.findall(r'"((?:train|test):\d+)"', path.read_text(encoding='utf-8'))) & set(rows))
    qt = AutoTokenizer.from_pretrained(old['model'], revision=old['model_revision'], local_files_only=True)
    bt = AutoTokenizer.from_pretrained(ROOT / 'experiments/.cache/bert-small', local_files_only=True)

    def keys(rid):
        row = rows[rid]
        ids = qt.encode(row['user_input'], add_special_tokens=False)
        bounded = qt.decode(ids[:256], skip_special_tokens=False) if len(ids) > 256 else row['user_input']
        return (row['user_input'].strip().casefold(), row['conv_id'],
                tuple(qt.encode(bounded, add_special_tokens=False)),
                tuple(bt(bounded, truncation=True, max_length=512)['input_ids']))

    seen = [set() for _ in range(4)]
    for rid in sorted(used):
        for bucket, key in zip(seen, keys(rid)):
            bucket.add(key)
    pool = []
    for rid in sorted(rows):
        if not rid.startswith('test:') or rid in used:
            continue
        values = keys(rid)
        if any(value in bucket for value, bucket in zip(values, seen)):
            continue
        pool.append(rid)
        for bucket, key in zip(seen, values):
            bucket.add(key)
    assert len(pool) == protocol['eligible_fresh'] == 730
    random.Random(protocol['fresh_seed']).shuffle(pool)
    assert pool[:500] == protocol['splits']['fresh']
    sets = {name: set(ids) for name, ids in protocol['splits'].items()}
    for name, ids in protocol['splits'].items():
        assert len(ids) == len(sets[name])
        for other in sets:
            if name != other:
                assert not sets[name] & sets[other]
    fit = read(OUT / 'fit.json')
    result = read(OUT / 'result.json')
    assert fit['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert result['fit_sha256'] == sha(OUT / 'fit.json')
    choices = {}
    for kind in protocol['variants']:
        for seed in protocol['seeds']:
            key = f'{kind}-{seed}'
            candidate = fit['candidates'][key]
            winner = max(candidate['history'], key=lambda row: rank(row['selected']['metrics']))
            assert winner == candidate['selected']
            assert sha(ROOT / 'experiments/.cache/compact-next' / (key + '.pt')) == candidate['weights_sha256']
            assert sha(OUT / (key + '-gate.npz')) == candidate['gate_sha256']
            choices[key] = winner['selected']['metrics']
    selected = max(choices, key=lambda key: rank(choices[key]))
    assert selected == fit['selected'] == result['selected'] == 'distill-9927'
    assert sha(ROOT / 'experiments/.cache/bert-small/joint-selected.pt') == protocol['original_weights_sha256']
    predictions = read(OUT / 'predictions.json')
    for row in predictions:
        assert row['label'] == int(rows[row['id']]['toxicity'])
    original = [r for r in predictions if r['candidate'] == 'original']
    assert len(original) == 500
    assert all(r['predictions']['confidence'] == r['predictions']['full'] for r in original)
    summary = {'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(OUT / 'protocol.json'),
               'data_hashes_verified': True, 'historical_artifacts_verified': len(protocol['historical_artifacts']),
               'excluded_prior_ids': len(used), 'eligible_reproduced': len(pool), 'fresh_ids_reproduced': 500,
               'exclusion_keys': ['casefold stripped text', 'conversation ID', 'bounded Qwen token IDs', 'bounded BERT token IDs'],
               'split_ids_disjoint': True, 'checkpoint_and_gate_hashes_verified': 6,
               'development_selected_candidate': selected, 'all_fresh_labels_match_source': len(predictions),
               'original_full_vs_confidence_all_predictions_identical': True,
               'original_early_count': sum(r['depths']['confidence'] == 2 for r in original),
               'limits': 'Independent input-token and arithmetic checks, no model inference. Does not detect semantic paraphrases or pretraining contamination. Selected candidate remains development winner, not post-hoc fresh-test winner.'}
    (OUT / 'split-selection-audit.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

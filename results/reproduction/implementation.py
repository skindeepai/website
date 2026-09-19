"""Separate implementation replay of recorded BANKING77 early exits.

This is a local reviewer replay, not an external replication or fresh-data test.
It imports no original experiment implementation and loads no feature cache.
The optional export reads the trusted local heads.pt once, using weights_only.
Ordinary replay uses portable arrays loaded with allow_pickle=False.
"""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'results/banking77'
OUT = ROOT / 'results/reproduction'
DEPTHS = (6, 12, 18, 24)
ARRAYS = BANK / 'seed17-heads.npz'
SCHEMA = BANK / 'seed17-heads.schema.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_heads():
    source = BANK / 'heads.pt'
    saved = torch.load(source, map_location='cpu', weights_only=True)
    arrays = {}
    for depth in DEPTHS:
        for field in ('weight', 'bias', 'mean', 'std', 'temperature'):
            value = saved[depth][field]
            arrays[f'layer_{depth}_{field}'] = value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.array(value, dtype=np.float64)
    np.savez_compressed(ARRAYS, **arrays)
    generator = torch.Generator().manual_seed(194109)
    roundtrip = []
    with np.load(ARRAYS, allow_pickle=False) as restored:
        assert set(restored.files) == set(arrays)
        for key in arrays:
            assert np.array_equal(arrays[key], restored[key]), key
        for depth in DEPTHS:
            old = saved[depth]
            new = {field: torch.from_numpy(restored[f'layer_{depth}_{field}'].copy()) for field in ('weight', 'bias', 'mean', 'std')}
            hidden = torch.randn((8, 896), generator=generator)
            expected = torch.nn.functional.linear((hidden-old['mean'])/old['std'], old['weight'], old['bias'])/old['temperature']
            actual = torch.nn.functional.linear((hidden-new['mean'])/new['std'], new['weight'], new['bias'])/float(restored[f'layer_{depth}_temperature'])
            assert torch.equal(expected, actual), depth
            roundtrip.append({'depth': depth, 'vectors': 8, 'logits_bitwise_equal': True})
    schema = {
        'format': 'NumPy NPZ; numeric arrays only; load using allow_pickle=False',
        'seed': 17, 'layers': list(DEPTHS), 'hidden_width': 896, 'labels': 77,
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in arrays.items()},
        'formula': 'softmax(((hidden - mean) / std) @ weight.T + bias, divided by temperature)',
        'readout': 'Final prompt token, after each block; layer 24 uses final RMS-normalized model output.',
        'category_order': 'data-manifest.json categories, zero indexed',
        'category_manifest_sha256': digest(BANK/'data-manifest.json'),
        'npz_sha256': digest(ARRAYS), 'trusted_local_source_sha256': digest(source),
        'roundtrip': roundtrip,
        'scope': 'Exact serialization of existing seed-17 heads, not new training or validation.'
    }
    write(SCHEMA, schema)


def load_heads():
    schema = read(SCHEMA)
    assert digest(ARRAYS) == schema['npz_sha256']
    assert digest(BANK/'data-manifest.json') == schema['category_manifest_sha256']
    heads = {}
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(schema['arrays'])
        for key, spec in schema['arrays'].items():
            assert list(arrays[key].shape) == spec['shape']
            assert str(arrays[key].dtype) == spec['dtype']
            assert np.isfinite(arrays[key]).all()
        for depth in DEPTHS:
            heads[depth] = {field: torch.from_numpy(arrays[f'layer_{depth}_{field}'].copy()) for field in ('weight', 'bias', 'mean', 'std')}
            heads[depth]['temperature'] = float(arrays[f'layer_{depth}_temperature'])
            assert (heads[depth]['std'] > 0).all()
            assert heads[depth]['temperature'] > 0
    return heads


def check_artifacts(manifest):
    source_rows = {}
    for split in ('train', 'test'):
        path = ROOT / f'experiments/.cache/banking77/{split}.csv'
        assert digest(path) == manifest['files'][f'{split}.csv']['sha256']
        with path.open(encoding='utf-8', newline='') as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                source_rows[f'{split}:{index}'] = {'text': row['text'], 'label': manifest['categories'].index(row['category'])}
    checks = []
    for directory in ('banking77', 'banking77-conservative'):
        folder = ROOT/'results'/directory
        predictions, result, timings = (read(folder/name) for name in ('predictions.json', 'result.json', 'timings.json'))
        key = 'seed' if directory == 'banking77' else 'split'
        depth_key = 'exit_layer' if key == 'seed' else 'depth'
        for group in sorted({row[key] for row in predictions}):
            rows = [row for row in predictions if row[key] == group]
            assert len({row['id'] for row in rows}) == len(rows)
            assert all(row['label'] == source_rows[row['id']]['label'] for row in rows)
            if key == 'seed':
                expected = next(run for run in result['runs'] if run['seed'] == group)['candidate_test']
            else:
                expected = result['results'][group]['candidate']
            computed = {
                'count': len(rows), 'correct': sum(row['candidate_prediction'] == row['label'] for row in rows),
                'harmful_exits': sum(row['full_prediction'] == row['label'] != row['candidate_prediction'] for row in rows),
                'corrected_full_errors': sum(row['full_prediction'] != row['label'] == row['candidate_prediction'] for row in rows),
                'mean_depth': statistics.mean(row[depth_key] for row in rows),
                'exit_counts': dict(Counter(str(row[depth_key]) for row in rows))
            }
            for field in ('count', 'correct', 'harmful_exits', 'corrected_full_errors', 'exit_counts'):
                assert computed[field] == expected[field], (directory, group, field)
            assert abs(computed['mean_depth']-expected['mean_depth']) < 0.000002
            assert abs(computed['correct']/len(rows)-expected['accuracy']) < 0.000001
            assert abs(1-computed['mean_depth']/24-expected['blocks_skipped_fraction']) < 0.000001
            exited = [row for row in rows if row[depth_key] < 24]
            computed['early_answers'] = len(exited)
            computed['wrong_early_answers'] = sum(row['candidate_prediction'] != row['label'] for row in exited)
            checks.append({'dataset': directory, 'group': group, 'verified': computed})
        for path in ('full_head', 'candidate_early_exit'):
            subset = [row for row in timings if row['path'] == path]
            assert all(row['executed_layers'] == list(range(1, row['depth']+1)) for row in subset)
            source = {row['id']: row for row in predictions if row.get('seed') == 17 or row.get('split') == 'test'}
            for row in subset:
                assert row['prediction'] == source[row['id']]['full_prediction' if path == 'full_head' else 'candidate_prediction']
            expected = result['timing'][path]['end_to_end_ms']
            assert abs(statistics.mean(row['end_to_end_ms'] for row in subset)-expected['mean']) < 0.000001
            assert abs(statistics.median(row['end_to_end_ms'] for row in subset)-expected['p50']) < 0.000001
    return source_rows, checks


def environment():
    package = Path(transformers.__file__).parent
    sources = {str(path.relative_to(package)).replace('\\', '/'): digest(path) for path in sorted(package.rglob('*.py'))}
    write(OUT/'transformers-source-hashes.json', sources)
    distribution = importlib.metadata.distribution('transformers')
    direct_url = distribution.read_text('direct_url.json')
    return {
        'python': platform.python_version(), 'platform': platform.platform(), 'processor': platform.processor(),
        'torch': torch.__version__, 'numpy': np.__version__, 'transformers': transformers.__version__,
        'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'transformers_direct_url': json.loads(direct_url) if direct_url else None,
        'transformers_python_files': len(sources), 'transformers_source_manifest_sha256': digest(OUT/'transformers-source-hashes.json'),
        'note': 'Source hashes identify the installed code. They do not themselves preserve installable source or prove external independence.'
    }


class ExitReached(Exception):
    def __init__(self, prediction, depth):
        self.prediction, self.depth = prediction, depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-heads', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(min(4, max(1, (os.cpu_count() or 2)//2)))
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.export_heads:
        export_heads()
    heads = load_heads()
    manifest = read(BANK/'data-manifest.json')
    recorded = read(BANK/'result.json')
    reference = {row['id']: row for row in read(BANK/'predictions.json') if row['seed'] == 17}
    source, checks = check_artifacts(manifest)
    settings = next(run for run in recorded['runs'] if run['seed'] == 17)['selected_policy']
    # Query selection is independent of predictions, labels, confidence and exit depth.
    selected = sorted(manifest['splits']['test'], key=lambda ident: hashlib.sha256(('reviewer-replay-v1:'+ident).encode()).hexdigest())[:24]
    protocol = {
        'scope': 'Same-machine, separate-agent implementation replay. No original experiment imports or feature caches.',
        'selection': 'First 24 official test IDs sorted by SHA256 of reviewer-replay-v1: plus the ID; no outcome-dependent selection.',
        'ids': selected, 'policy': {key: settings[key] for key in ('threshold', 'agreement', 'minimum')},
        'model': recorded['model'], 'revision': recorded['revision'], 'threads_max': 4,
        'implementation_sha256': digest(Path(__file__)), 'heads_sha256': digest(ARRAYS),
        'references_sha256': {name: digest(BANK/name) for name in ('predictions.json', 'result.json', 'data-manifest.json')},
        'written_before_replay_utc': datetime.now(timezone.utc).isoformat(),
        'limitations': 'Local protocol is overwritten by rerunning this script; not immutable external preregistration.'
    }
    write(OUT/'protocol.json', protocol)
    tokenizer = AutoTokenizer.from_pretrained(recorded['model'], revision=recorded['revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(recorded['model'], revision=recorded['revision'], local_files_only=True,
                                               torch_dtype=torch.float32, attn_implementation='eager').eval()
    assert len(model.model.layers) == 24 and model.config.hidden_size == 896
    env = environment()
    env['qwen_implementation_sha256'] = digest(Path(inspect.getfile(type(model))))

    def classify(hidden, depth):
        head = heads[depth]
        scores = torch.nn.functional.linear((hidden-head['mean'])/head['std'], head['weight'], head['bias'])/head['temperature']
        probability = torch.softmax(scores, dim=-1)[0]
        return int(probability.argmax()), float(probability.max())

    records = []
    with torch.inference_mode():
        for ident in selected:
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': recorded['prompt'].format(text=source[ident]['text'])}],
                                                 tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors='pt')
            for adaptive in (False, True):
                visited, assessments, handles = [], [], []

                def observe(depth, module, arguments, output):
                    visited.append(depth)
                    if adaptive and depth in (6, 12, 18):
                        prediction, confidence = classify(output[0][:, -1, :], depth)
                        agree = not settings['agreement'] or bool(assessments and assessments[-1]['prediction'] == prediction)
                        assessments.append({'depth': depth, 'prediction': prediction, 'confidence': confidence})
                        if depth >= settings['minimum'] and confidence >= settings['threshold'] and agree:
                            raise ExitReached(prediction, depth)

                for block_index, block in enumerate(model.model.layers, start=1):
                    handles.append(block.register_forward_hook(partial(observe, block_index)))
                start = time.perf_counter()
                try:
                    output = model.model(**inputs, use_cache=False)
                    prediction, confidence = classify(output.last_hidden_state[:, -1, :], 24)
                    depth = 24
                except ExitReached as exit_signal:
                    prediction, depth = exit_signal.prediction, exit_signal.depth
                finally:
                    for handle in handles:
                        handle.remove()
                elapsed = 1000*(time.perf_counter()-start)
                expected = reference[ident]
                assert visited == list(range(1, depth+1))
                assert depth == (expected['exit_layer'] if adaptive else 24), ident
                assert prediction == expected['candidate_prediction' if adaptive else 'full_prediction'], ident
                records.append({'id': ident, 'path': 'early' if adaptive else 'full', 'label': source[ident]['label'],
                                'prediction': prediction, 'depth': depth, 'executed_layers': visited,
                                'head_checks': assessments, 'matches_original': True, 'diagnostic_model_ms': elapsed})
            print(f'Replayed {ident}: full and early match; exit {records[-1]["depth"]}', flush=True)
    write(OUT/'predictions.json', records)
    write(OUT/'artifact-checks.json', checks)
    write(OUT/'environment.json', env)
    result = {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'queries': len(selected), 'forward_passes': len(records), 'all_predictions_and_depths_match': True,
        'early_exit_counts': dict(Counter(row['depth'] for row in records if row['path'] == 'early')),
        'all_execution_traces_contiguous': True, 'portable_head_roundtrip': read(SCHEMA)['roundtrip'],
        'artifact_summary_checks_passed': len(checks),
        'hashes': {name: digest(OUT/name) for name in ('protocol.json', 'predictions.json', 'artifact-checks.json', 'environment.json')},
        'verdict': 'A separate implementation reproduced saved outputs and genuine conditional block execution on 24 predetermined examples.',
        'limits': ['Reviewer shares the same machine, model, source dataset and trained heads; not external independent replication.',
                   'These are previously evaluated examples, not new evidence of out-of-distribution accuracy or guard acceptance.',
                   'Only 24 queries are rerun. Original full-dataset execution was not independently repeated.',
                   'Diagnostic durations overlap other work, lack a timing design, and must not support a speed claim.',
                   'Portable arrays and checksums improve inspection, not model quality or statistical reliability.']
    }
    write(OUT/'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

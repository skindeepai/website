"""Sealed, paired actual-runtime measurement of existing fixed-depth MLP heads.

Prepare first, then run only after other launched model jobs have finished.
This reuses inspected quality data; it does not establish generalization.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/chat-depth-timing'
DEPTHS = [6, 12, 18, 24]
PASSES = 3


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def plan():
    shared = read('results/chat-smoke/protocol.json')
    dependencies = ['experiments/chat_depth_timing.py', 'experiments/chat_smoke_common.py',
                    'results/chat-smoke/protocol.json', 'results/chat600/protocol.json',
                    'results/chat-smoke-heads/protocol.json', 'results/chat-smoke-heads/mlp-heads.npz',
                    'results/chat-smoke-heads/predictions.json']
    return {
        'scope': 'New timing of frozen existing classifiers on reused balanced100; not fresh quality validation.',
        'model': shared['qwen_model'], 'model_revision': shared['qwen_revision'],
        'head': 'Existing separately trained per-depth MLP: 896 inputs, 64 GELU hidden units, two labels; standardized, temperature calibrated. No retraining.',
        'head_training': '384 train messages, seed109; separate128 tune for temperature. See sealed original head protocol.',
        'depths': DEPTHS, 'passes': PASSES, 'evaluation_ids': shared['splits']['evaluation'],
        'threads': 4, 'interop_threads': 1, 'batch_size': 1,
        'runtime': 'Python3.13, torch2.6.0+cpu, transformers4.50.3, CPUfloat32, eager attention, no KV cache.',
        'timing_boundary': 'perf_counter: raw message truncation, chat templating and tokenization through hook registration, actual backbone execution, selected MLP probability and label, and hook removal. Excludes loading, training, assertions, aggregation and file writes.',
        'execution': 'Fixed6/12/18 raise from the post-block hook before any later block or final norm; fixed24 completes all24 blocks and final norm. No vocabulary projection or text generation. Same representations as original feature cache.',
        'warmup': 'One excluded execution per depth, first evaluation message; no timed early stopping based on observed latency.',
        'order': 'Each pass traverses identical100 IDs. At item i in pass p, rotate [6,12,18,24] left by (i+p)%4. Each depth occupies each position25times per pass.',
        'parity': 'Each measured probability compared against saved corresponding MLP probability, atol0.0005 rtol0; argmax must match. All executed blocks must equal range(1,depth+1).',
        'summary': 'Report every pass, mean100-message duration, pooled latency percentiles, per-depth class errors, prediction changes relative to full depth and actual layer traces. Repeats are not additional independent quality examples.',
        'limits': 'Single-machine warmed CPU measurement. Previously inspected sample and models; no independent acceptance, deployment-prevalence or no-quality-loss claim. Timing excludes process start and model load. Four tracing hooks paths include instrumentation overhead.',
        'source_sha256': {p: digest(p) for p in dependencies},
    }


class Stop(Exception):
    def __init__(self, probabilities):
        self.probabilities = probabilities


def run(protocol):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'experiments/.cache/replay-runtime'))
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from chat_smoke_common import prepare

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert transformers.__version__ == '4.50.3'
    assert torch.__version__ == '2.6.0+cpu'
    shared, splits = prepare()
    assert [r['id'] for r in splits['evaluation']] == protocol['evaluation_ids']
    model = AutoModelForCausalLM.from_pretrained(protocol['model'], revision=protocol['model_revision'],
        local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(protocol['model'], revision=protocol['model_revision'], local_files_only=True)
    assert len(model.model.layers) == 24
    assert model.config.hidden_size == 896
    arrays = np.load(ROOT / 'results/chat-smoke-heads/mlp-heads.npz', allow_pickle=False)
    keys = ['0.weight', '0.bias', '3.weight', '3.bias', 'mean', 'std', 'temperature']
    heads = {d: {k: torch.tensor(arrays[f'{d}_{k}']) for k in keys} for d in DEPTHS}
    expected = {r['id']: r for r in read('results/chat-smoke-heads/predictions.json')['mlp']['evaluation']}
    assert set(expected) == set(protocol['evaluation_ids'])

    def probability(depth, hidden):
        h = heads[depth]
        hidden = torch.nn.functional.linear((hidden-h['mean'])/h['std'], h['0.weight'], h['0.bias'])
        hidden = torch.nn.functional.gelu(hidden)
        return torch.nn.functional.linear(hidden, h['3.weight'], h['3.bias']).div(h['temperature']).softmax(1)

    def execute(row, depth):
        start = time.perf_counter()
        ids = tokenizer.encode(row['text'], add_special_tokens=False)
        text = tokenizer.decode(ids[:256], skip_special_tokens=False) if len(ids) > 256 else row['text']
        rendered = tokenizer.apply_chat_template([{'role': 'system', 'content': shared['system_prompt']},
            {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt')
        visited, hooks = [], []

        def after(d):
            def hook(mod, inp, out):
                visited.append(d)
                if d == depth and depth < 24:
                    raise Stop(probability(depth, out[0][:, -1, :]))
            return hook

        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(after(i+1)))
        try:
            hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
            probs = probability(24, hidden)
        except Stop as stopped:
            probs = stopped.probabilities
        finally:
            for hook in hooks:
                hook.remove()
        prediction = int(probs.argmax(1))
        elapsed_ms = (time.perf_counter()-start)*1000
        actual = probs[0].tolist()
        reference = expected[row['id']]['probabilities'][str(depth)]
        assert np.allclose(actual, reference, atol=.0005, rtol=0), (row['id'], depth, actual, reference)
        assert prediction == int(np.argmax(reference)), (row['id'], depth, prediction, reference)
        assert visited == list(range(1, depth+1)), (row['id'], depth, visited)
        assert row['label'] == expected[row['id']]['label']
        return {'id': row['id'], 'depth': depth, 'label': row['label'], 'prediction': prediction,
                'probabilities': actual, 'cached_probability_max_abs_delta': max(abs(a-b) for a,b in zip(actual,reference)),
                'executed_layers': visited, 'blocks_skipped': 24-depth, 'ms': elapsed_ms,
                'input_tokens': inputs['input_ids'].shape[1], 'message_was_truncated': len(ids)>256}

    records, warmup = [], []
    started = datetime.now(timezone.utc).isoformat()
    with torch.inference_mode():
        for depth in DEPTHS:
            warmup.append(execute(splits['evaluation'][0], depth))
        for repeat in range(PASSES):
            for i, row in enumerate(splits['evaluation']):
                shift = (i+repeat) % 4
                order = DEPTHS[shift:] + DEPTHS[:shift]
                for position, depth in enumerate(order):
                    record = execute(row, depth)
                    record.update({'pass': repeat+1, 'position': position+1})
                    records.append(record)
                if (i+1) % 20 == 0:
                    print(f'Fixed-depth pass {repeat+1}/{PASSES}: {i+1}/100', flush=True)
            write(OUT / 'progress.json', {'completed_passes': repeat+1, 'records': records})

    summary = {}
    full = {r['id']:r for r in records if r['pass']==1 and r['depth']==24}
    for depth in DEPTHS:
        subset = [r for r in records if r['depth']==depth]
        unique = [r for r in subset if r['pass']==1]
        times = [sum(r['ms'] for r in subset if r['pass']==p)/1000 for p in range(1,PASSES+1)]
        for row in unique:
            assert all(r['prediction']==row['prediction'] for r in subset if r['id']==row['id'])
        summary[str(depth)] = {
            'unique_messages':100, 'measured_passes':PASSES, 'measured_forwards':len(subset),
            'pass_sum_seconds':times, 'mean_100_message_seconds':statistics.mean(times),
            'mean_ms':statistics.mean(r['ms'] for r in subset),
            'median_ms':statistics.median(r['ms'] for r in subset),
            'p95_ms':float(np.percentile([r['ms'] for r in subset],95)),
            'correct':sum(r['prediction']==r['label'] for r in unique),
            'missed_toxic':sum(r['label']==1 and r['prediction']==0 for r in unique),
            'false_block':sum(r['label']==0 and r['prediction']==1 for r in unique),
            'added_errors_vs_full':sum(full[r['id']]['prediction']==r['label'] and r['prediction']!=r['label'] for r in unique),
            'corrected_errors_vs_full':sum(full[r['id']]['prediction']!=r['label'] and r['prediction']==r['label'] for r in unique),
            'additional_missed_toxic_vs_full':sum(r['label']==1 and full[r['id']]['prediction']==1 and r['prediction']==0 for r in unique),
            'blocks_executed':depth, 'blocks_skipped':24-depth, 'fraction_blocks_skipped':(24-depth)/24,
        }
    for depth in DEPTHS:
        summary[str(depth)]['relative_time_saving_vs_full'] = 1-summary[str(depth)]['mean_100_message_seconds']/summary['24']['mean_100_message_seconds']
    result = {'started_utc':started, 'completed_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':digest('results/chat-depth-timing/protocol.json'),
        'source_sha256':protocol['source_sha256'], 'runtime':{'python':sys.version, 'torch':torch.__version__,
            'transformers':transformers.__version__, 'numpy':np.__version__, 'platform':platform.platform(),
            'processor':platform.processor(), 'threads':torch.get_num_threads(), 'interop_threads':torch.get_num_interop_threads()},
        'actual_timed_forward_passes':len(records), 'depths':summary,
        'parity':{'all_cached_labels_match':True, 'all_contiguous_block_traces_match':True,
                  'max_abs_probability_delta':max(r['cached_probability_max_abs_delta'] for r in records)},
        'limits':protocol['limits']}
    assert len(records)==1200
    write(OUT/'warmup.json', warmup)
    write(OUT/'records.json', records)
    write(OUT/'result.json', result)
    (OUT/'progress.json').unlink()
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT/'protocol.json'
    protocol = plan()
    if args.prepare:
        if protocol_path.exists():
            assert read('results/chat-depth-timing/protocol.json') == protocol, 'Existing sealed protocol differs'
        else:
            write(protocol_path, protocol)
        print('Protocol sealed. No model execution. Await isolated timing window.')
        return
    assert protocol_path.exists(), 'Run --prepare before timing'
    assert not (OUT/'result.json').exists(), 'Preserve completed results; use a new study directory'
    assert not (OUT/'progress.json').exists(), 'Incomplete run exists; preserve and inspect before new attempt'
    assert read('results/chat-depth-timing/protocol.json') == protocol, 'Sealed inputs changed'
    run(protocol)


if __name__ == '__main__':
    main()

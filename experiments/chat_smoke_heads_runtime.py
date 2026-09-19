"""Actual-forward parity/timing for the exploratory MLP learned gate.

Run only after other model work has finished. Four CPU threads, one alternating
paired pass over the reused 100; a diagnostic timing, not quality validation.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/.cache/replay-runtime'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from chat_smoke_common import prepare

OUT = ROOT / 'results/chat-smoke-heads'
DEPTHS = [6, 12, 18, 24]


class Stop(Exception):
    def __init__(self, label, depth):
        self.label, self.depth = label, depth


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    protocol, splits = prepare()
    assert __import__('transformers').__version__ == '4.50.3'
    model = AutoModelForCausalLM.from_pretrained(protocol['qwen_model'], revision=protocol['qwen_revision'],
                                               local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(protocol['qwen_model'], revision=protocol['qwen_revision'], local_files_only=True)
    heads_np = np.load(OUT / 'mlp-heads.npz', allow_pickle=False)
    gates_np = np.load(OUT / 'mlp-gates.npz', allow_pickle=False)
    heads = {d: {k: torch.tensor(heads_np[f'{d}_{k}']) for k in ['0.weight', '0.bias', '3.weight', '3.bias', 'mean', 'std', 'temperature']} for d in DEPTHS}
    gates = {d: {k: torch.tensor(gates_np[f'{d}_{k}']) for k in ['weight', 'bias', 'mean', 'std']} for d in DEPTHS[:-1]}
    results = json.loads((OUT / 'result.json').read_text())
    policy = results['mlp']['learned']['policy']
    expected = {r['id']: r for r in json.loads((OUT / 'predictions.json').read_text())['mlp']['evaluation']}

    def probability(d, h):
        x = heads[d]
        h = torch.nn.functional.linear((h-x['mean'])/x['std'], x['0.weight'], x['0.bias'])
        h = torch.nn.functional.gelu(h)
        return torch.nn.functional.linear(h, x['3.weight'], x['3.bias']).div(x['temperature']).softmax(1)

    def execute(row, path):
        start = time.perf_counter()
        ids = tokenizer.encode(row['text'], add_special_tokens=False)
        text = tokenizer.decode(ids[:256], skip_special_tokens=False) if len(ids) > 256 else row['text']
        rendered = tokenizer.apply_chat_template([{'role': 'system', 'content': protocol['system_prompt']},
                                                  {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt')
        visited, hooks, checkpoint = [], [], {}

        def after(d):
            def hook(mod, inp, out):
                visited.append(d)
                if path != 'learned' or d not in DEPTHS[:-1]:
                    return
                p = probability(d, out[0][:, -1, :])
                checkpoint[d] = p
                confidence, label = p.max(1)
                entropy = -(p*p.clamp_min(1e-9).log()).sum(1)
                if d == 6:
                    previous_confidence = agreement = delta = torch.zeros_like(confidence)
                else:
                    prev = checkpoint[d-6]
                    previous_confidence = prev.max(1).values
                    agreement = (prev.argmax(1) == label).float()
                    delta = p[:, 1]-prev[:, 1]
                features = torch.stack([p[:, 1], confidence, entropy, previous_confidence, agreement, delta], 1)
                g = gates[d]
                risk = torch.nn.functional.linear((features-g['mean'])/g['std'], g['weight'], g['bias']).sigmoid()[0]
                if float(risk[0]) <= policy['error_limit'] and float(risk[1]) <= policy['benefit_limit']:
                    raise Stop(int(label), d)
            return hook

        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(after(i+1)))
        try:
            hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
            label, depth = int(probability(24, hidden).argmax(1)), 24
        except Stop as stopped:
            label, depth = stopped.label, stopped.depth
        finally:
            for hook in hooks:
                hook.remove()
        elapsed = (time.perf_counter()-start)*1000
        reference = expected[row['id']]
        prediction = reference['learned']['prediction'] if path == 'learned' else int(np.argmax(reference['probabilities']['24']))
        expected_depth = reference['learned']['depth'] if path == 'learned' else 24
        assert label == prediction and depth == expected_depth, (row['id'], path, label, depth, prediction, expected_depth)
        assert visited == list(range(1, depth+1))
        return {'id': row['id'], 'path': path, 'label': row['label'], 'prediction': label,
                'depth': depth, 'executed_layers': visited, 'ms': elapsed}

    records = []
    with torch.inference_mode():
        for path in ['full', 'learned']:
            execute(splits['evaluation'][0], path)
        for i, row in enumerate(splits['evaluation']):
            order = ['full', 'learned'] if i % 2 == 0 else ['learned', 'full']
            for path in order:
                records.append(execute(row, path))
            if (i+1) % 20 == 0:
                print(f'Actual paired smoke: {i+1}/100', flush=True)
    paths = {path: {'sum_seconds': sum(r['ms'] for r in records if r['path'] == path)/1000,
                    'mean_ms': statistics.mean(r['ms'] for r in records if r['path'] == path),
                    'p95_ms': float(np.percentile([r['ms'] for r in records if r['path'] == path], 95)),
                    'correct': sum(r['prediction'] == r['label'] for r in records if r['path'] == path)}
             for path in ['full', 'learned']}
    report = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'threads': 4,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'actual_timed_forward_passes': len(records), 'paths': paths,
              'relative_sum_time_saving': 1-paths['learned']['sum_seconds']/paths['full']['sum_seconds'],
              'parity': 'All 200 predictions, exit depths and contiguous block traces match cached evaluation.',
              'limits': 'Exploratory selection after inspecting reused100 quality. One warmed paired pass on one CPU, alternating path order; no concurrent launched model jobs. Includes tokenization, readout, gate and hook overhead; excludes model loading/training. No independent quality validation or deployment speed guarantee.'}
    (OUT / 'runtime-records.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    (OUT / 'runtime.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

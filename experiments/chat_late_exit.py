"""Matched linear classifiers at Qwen blocks 22, 23 and 24; exploratory only."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/.cache/replay-runtime'))
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from chat_smoke_common import prepare

OUT = ROOT / 'results/chat-late-exit'
DEPTHS = [22, 23, 24]


def write(name, value):
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


class Stop(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


def main():
    assert not OUT.exists(), 'Preserve completed/partial evidence; use a new output directory to repeat.'
    OUT.mkdir(parents=True)
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.__version__ == '2.6.0+cpu' and transformers.__version__ == '4.50.3'
    shared, original = prepare()
    # Balanced first 25 per class in the pre-existing fixed evaluation order.
    counts = {0: 0, 1: 0}
    evaluation = []
    for row in original['evaluation']:
        if counts[row['label']] < 25:
            evaluation.append(row)
            counts[row['label']] += 1
    splits = {'train': original['train'], 'evaluation': evaluation}
    assert counts == {0: 25, 1: 25}
    assert not {r['conversation'] for r in splits['train']} & {r['conversation'] for r in evaluation}
    protocol = {
        'scope': 'New matched linear readouts, reused exploratory data; not independent quality validation.',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'model': shared['qwen_model'], 'revision': shared['qwen_revision'],
        'depths': DEPTHS, 'splits': {k: [r['id'] for r in v] for k, v in splits.items()},
        'dataset': 'ToxicChat0124, human-annotated subset; balanced50 is not deployment prevalence.',
        'data_sha256': shared['source_sha256'], 'system_prompt': shared['system_prompt'],
        'input': 'Same Qwen chat template and first256 message tokens; final prompt-position representation.',
        'features': 'Raw post-block states at22/23; final RMS-normalized state at24. No vocabulary projection or text generation.',
        'training': 'Frozen backbone. Separately train standardized896-to2 linear readouts on identical384 examples. Seed109 reset at each depth;300 AdamW steps, lr.01, weight_decay.1, inverse-frequency class weights. Std floor.05. Fixed argmax, no threshold or hyperparameter tuning.',
        'evaluation_selection': 'First25 examples of each label in pre-existing balanced100 order, before inference; report all depths.',
        'timing': 'Three paired passes of50, batch1, CPUfloat32 eager attention, four compute threads/one interop. Rotate depths by(item+pass)%3. Includes truncation/template/tokenization, hook management, actual forward, classifier and enum. Excludes loading/training/validation/file IO. One warm-up per depth.',
        'checks': 'Every measured trace must be exactly1..depth; lm_head calls must be0; measured probabilities must match full-forward extraction within.0005. Timing repeats must produce identical labels.',
        'selection': 'No adaptive gate or method selected. Compare added errors and new toxic misses against full-depth classifier; accuracy alone is insufficient.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_source_sha256': hashlib.sha256((ROOT / 'experiments/chat_smoke_common.py').read_bytes()).hexdigest(),
        'shared_protocol_sha256': hashlib.sha256((ROOT / 'results/chat-smoke/protocol.json').read_bytes()).hexdigest(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
    }
    write('protocol', protocol)
    tokenizer = AutoTokenizer.from_pretrained(protocol['model'], revision=protocol['revision'], local_files_only=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(protocol['model'], revision=protocol['revision'], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').eval()
    assert len(model.model.layers) == 24 and model.config.hidden_size == 896

    def render(row):
        ids = tokenizer.encode(row['text'], add_special_tokens=False)
        message = tokenizer.decode(ids[:256], skip_special_tokens=False) if len(ids) > 256 else row['text']
        return tokenizer.apply_chat_template([{'role': 'system', 'content': shared['system_prompt']},
            {'role': 'user', 'content': message}], tokenize=False, add_generation_prompt=True)

    features = {}
    with torch.inference_mode():
        for split, rows in splits.items():
            features[split] = {d: torch.empty(len(rows), 896) for d in DEPTHS}
            texts = [render(r) for r in rows]
            order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
            for start in range(0, len(order), 8):
                indexes = order[start:start+8]
                inputs = tokenizer([texts[i] for i in indexes], padding=True, return_tensors='pt')
                captured, hooks = {}, []
                for depth in DEPTHS[:-1]:
                    hooks.append(model.model.layers[depth-1].register_forward_hook(
                        lambda mod, inp, out, d=depth: captured.update({d: out[0][:, -1, :].clone()})))
                try:
                    captured[24] = model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
                finally:
                    for hook in hooks:
                        hook.remove()
                for depth in DEPTHS:
                    features[split][depth][indexes] = captured[depth]
                print(f'features {split}: {min(start+8,len(rows))}/{len(rows)}', flush=True)
    heads, arrays = {}, {}
    y = torch.tensor([r['label'] for r in splits['train']])
    weights = torch.bincount(y, minlength=2).float().reciprocal()
    weights /= weights.mean()
    expected = {}
    for depth in DEPTHS:
        torch.manual_seed(109)
        x = features['train'][depth]
        mean, std = x.mean(0), x.std(0).clamp_min(.05)
        head = torch.nn.Linear(896, 2)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
        for _ in range(300):
            optimizer.zero_grad()
            loss = F.cross_entropy(head((x-mean)/std), y, weight=weights)
            loss.backward()
            optimizer.step()
        head.eval()
        heads[depth] = (head, mean, std)
        with torch.inference_mode():
            expected[depth] = head((features['evaluation'][depth]-mean)/std).softmax(1)
        for key, value in {**head.state_dict(), 'mean': mean, 'std': std}.items():
            arrays[f'{depth}_{key}'] = value.detach().numpy()
    np.savez_compressed(OUT / 'heads.npz', **arrays)
    np.savez_compressed(OUT / 'features.npz', **{f'{s}_{d}': x.numpy() for s, v in features.items() for d, x in v.items()})
    lm_calls = []
    lm_hook = model.lm_head.register_forward_hook(lambda *args: lm_calls.append(1))

    def execute(row, depth, index):
        start = time.perf_counter()
        inputs = tokenizer(render(row), return_tensors='pt')
        visited, hooks = [], []
        def after(d):
            def hook(mod, inp, out):
                visited.append(d)
                if d == depth and depth < 24:
                    raise Stop(out[0][:, -1, :])
            return hook
        for d, layer in enumerate(model.model.layers, 1):
            hooks.append(layer.register_forward_hook(after(d)))
        try:
            hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
        except Stop as stop:
            hidden = stop.hidden
        finally:
            for hook in hooks:
                hook.remove()
        head, mean, std = heads[depth]
        probability = head((hidden-mean)/std).softmax(1)
        prediction = int(probability.argmax(1))
        ms = (time.perf_counter()-start)*1000
        delta = float((probability[0]-expected[depth][index]).abs().max())
        assert visited == list(range(1, depth+1)) and not lm_calls
        assert delta < .0005 and prediction == int(expected[depth][index].argmax()), (row['id'], depth, delta)
        return {'id': row['id'], 'label': row['label'], 'depth': depth, 'prediction': prediction,
            'block_probability': float(probability[0, 1]), 'ms': ms, 'executed_layers': visited,
            'input_tokens': inputs['input_ids'].shape[1], 'cached_probability_max_abs_delta': delta}

    records = []
    with torch.inference_mode():
        write('warmup', [execute(evaluation[0], d, 0) for d in DEPTHS])
        for repeat in range(3):
            for i, row in enumerate(evaluation):
                shift = (i+repeat) % 3
                for depth in DEPTHS[shift:]+DEPTHS[:shift]:
                    record = execute(row, depth, i)
                    record['pass'] = repeat+1
                    records.append(record)
                if (i+1) % 10 == 0:
                    write('records', records)
                    print(f'timing pass{repeat+1}: {i+1}/50', flush=True)
    lm_hook.remove()
    write('records', records)
    full = {r['id']: r for r in records if r['depth'] == 24 and r['pass'] == 1}
    summaries = {}
    for depth in DEPTHS:
        rows = [r for r in records if r['depth'] == depth and r['pass'] == 1]
        by_id = {r['id']: r['prediction'] for r in rows}
        assert all(r['prediction'] == by_id[r['id']] for r in records if r['depth'] == depth)
        pass_ms = [sum(r['ms'] for r in records if r['depth'] == depth and r['pass'] == p) for p in [1, 2, 3]]
        summaries[str(depth)] = {
            'correct': sum(r['prediction'] == r['label'] for r in rows), 'n': len(rows), 'toxic': 25,
            'missed_toxic': sum(r['label'] == 1 and r['prediction'] == 0 for r in rows),
            'false_block': sum(r['label'] == 0 and r['prediction'] == 1 for r in rows),
            'added_errors': sum(full[r['id']]['prediction'] == r['label'] and r['prediction'] != r['label'] for r in rows),
            'corrected_errors': sum(full[r['id']]['prediction'] != r['label'] and r['prediction'] == r['label'] for r in rows),
            'additional_missed_toxic': sum(r['label'] == 1 and full[r['id']]['prediction'] == 1 and r['prediction'] == 0 for r in rows),
            'blocks_skipped': 24-depth, 'fraction_blocks_skipped': (24-depth)/24,
            'pass_total_seconds': [v/1000 for v in pass_ms], 'mean_ms_per_message': statistics.mean(pass_ms)/len(rows),
        }
    for value in summaries.values():
        value['time_reduction_vs_full'] = 1-value['mean_ms_per_message']/summaries['24']['mean_ms_per_message']
    write('result', {'scope': protocol['scope'], 'methods': summaries, 'timed_calls': len(records),
        'all_layer_traces_valid': True, 'lm_head_calls': len(lm_calls), 'text_tokens_generated': 0,
        'limitations': ['Single warmed CPU; head training and loading excluded.', 'Previously inspected balanced50; no independent acceptance or no-quality-loss claim.', 'Fixed depths, not adaptive readiness detection.', 'All three heads trained equally; numbers are not the older MLP comparison.']})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Isolate vocabulary projection, KV cache and generation wrapper costs."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '4'
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
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from chat_smoke_common import source_rows

OUT = ROOT / 'results/chat-readout-control'
METHODS = ['two_rows', 'full_vocabulary', 'two_rows_cache', 'generate_one']


def write(name, value):
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    assert not OUT.exists(), 'Preserve existing evidence.'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.__version__ == '2.6.0+cpu' and transformers.__version__ == '4.50.3'
    parent_path = ROOT / 'results/chat-output-steps/protocol.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    source = source_rows()
    rows = [source[r['id']] for r in parent['evaluation']]
    OUT.mkdir(parents=True)
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    protocol = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'How much of the two-row versus one-token timing difference comes from vocabulary projection, cache setup and generation wrapper?',
        'model': parent['model'], 'revision': parent['revision'],
        'evaluation': parent['evaluation'], 'methods': METHODS, 'passes': 2,
        'system_prompt': parent['prompts']['SAFE'],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parent_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        'data_sha256': parent['data_sha256'],
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__, 'compute_threads': 4, 'interop_threads': 1, 'device': 'CPU', 'dtype': 'float32', 'attention': 'eager'},
        'controls': 'Same backbone, prompt, messages, last-position output weights, label argmax, batch1. No retraining. Two/full vocabulary both disable KV cache. Cache control changes only use_cache. Generate uses cache, full vocabulary and restricts SAFE/BLOCK to one token.',
        'timing': 'Two passes, rotate method order by (message+pass)%4, reverse second pass. Includes input truncation/template/tokenization, hooks, inference and readout, plus detokenization for generate. Excludes loading, warmup, result validation and fileIO. Stage timings separate input, backbone and readout for manual paths; API total for generate. No concurrent launched model jobs.',
        'scope': 'Same reused balanced50, not fresh quality validation. Return all methods, not a selected winner. Repeats are timing observations, not extra quality samples.',
    }
    write('protocol', protocol)
    tokenizer = AutoTokenizer.from_pretrained(parent['model'], revision=parent['revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(parent['model'], revision=parent['revision'], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').eval()
    ids = [tokenizer.encode(word, add_special_tokens=False) for word in ['SAFE', 'BLOCK']]
    assert all(len(value) == 1 for value in ids)
    label_ids = [value[0] for value in ids]
    weights = model.lm_head.weight[label_ids].detach().clone()

    class Restrict(LogitsProcessor):
        def __call__(self, input_ids, scores):
            selected = scores[:, label_ids].clone()
            scores[:] = -float('inf')
            scores[:, label_ids] = selected
            return scores

    restriction = LogitsProcessorList([Restrict()])

    def execute(row, method):
        start = time.perf_counter()
        message_ids = tokenizer.encode(row['text'], add_special_tokens=False)
        message = tokenizer.decode(message_ids[:256], skip_special_tokens=False) if len(message_ids) > 256 else row['text']
        text = tokenizer.apply_chat_template([{'role': 'system', 'content': protocol['system_prompt']},
            {'role': 'user', 'content': message}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt')
        input_end = time.perf_counter()
        visited, vocab_calls, handles = [], [], []
        for depth, layer in enumerate(model.model.layers, 1):
            handles.append(layer.register_forward_hook(lambda mod, inp, out, d=depth: visited.append(d)))
        handles.append(model.lm_head.register_forward_hook(lambda *args: vocab_calls.append(1)))
        stages = {'input_ms': (input_end-start)*1000}
        scores, raw = None, None
        try:
            forward_start = time.perf_counter()
            if method == 'generate_one':
                output = model.generate(**inputs, do_sample=False, max_new_tokens=1, use_cache=True,
                    logits_processor=restriction, logits_to_keep=1, pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1., temperature=1., top_p=1., top_k=0)
                raw = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                prediction = int(raw == 'BLOCK')
                stages['generation_ms'] = (time.perf_counter()-forward_start)*1000
            else:
                state = model.model(**inputs, use_cache=method == 'two_rows_cache')
                hidden = state.last_hidden_state[:, -1, :]
                readout_start = time.perf_counter()
                stages['backbone_ms'] = (readout_start-forward_start)*1000
                logits = model.lm_head(hidden)[:, label_ids] if method == 'full_vocabulary' else F.linear(hidden, weights)
                prediction = int(logits.argmax(1))
                scores = logits[0].tolist()
                stages['readout_ms'] = (time.perf_counter()-readout_start)*1000
        finally:
            for handle in handles:
                handle.remove()
        ms = (time.perf_counter()-start)*1000
        assert visited == list(range(1, 25))
        assert len(vocab_calls) == int(method in ['full_vocabulary', 'generate_one'])
        assert method != 'generate_one' or raw in ['SAFE', 'BLOCK']
        return {'id': row['id'], 'label': row['label'], 'method': method, 'prediction': prediction,
            'scores': scores, 'raw': raw, 'ms': ms, 'stages': stages, 'executed_layers': visited,
            'vocabulary_calls': len(vocab_calls), 'input_tokens': inputs['input_ids'].shape[1]}

    records = []
    with torch.inference_mode():
        warmup = {'id': 'warmup', 'label': 0, 'text': 'Thank you for your help.'}
        write('warmup', [execute(warmup, m) for m in METHODS])
        for repeat in range(2):
            for i, row in enumerate(rows):
                shift = (i+repeat) % 4
                order = METHODS[shift:] + METHODS[:shift]
                if repeat == 1:
                    order.reverse()
                group = []
                for position, method in enumerate(order):
                    record = execute(row, method)
                    record.update({'pass': repeat+1, 'position': position+1})
                    group.append(record)
                assert len({r['prediction'] for r in group}) == 1
                reference = next(r for r in group if r['method'] == 'two_rows')
                for record in group:
                    if record['scores'] is not None:
                        assert max(abs(a-b) for a, b in zip(record['scores'], reference['scores'])) < .0005
                records.extend(group)
                if (i+1) % 10 == 0:
                    write('records', records)
                    print(f'control pass {repeat+1}: {i+1}/50', flush=True)
    write('records', records)
    summary = {}
    for method in METHODS:
        selected = [r for r in records if r['method'] == method]
        first = [r for r in selected if r['pass'] == 1]
        assert all(a['prediction'] == b['prediction'] for a, b in zip(first, selected[50:]))
        summary[method] = {'correct': sum(r['prediction'] == r['label'] for r in first), 'n': 50,
            'missed_toxic': sum(r['label'] == 1 and r['prediction'] == 0 for r in first),
            'false_block': sum(r['label'] == 0 and r['prediction'] == 1 for r in first),
            'mean_ms': statistics.mean(r['ms'] for r in selected),
            'pass_ms': [statistics.mean(r['ms'] for r in selected if r['pass'] == p) for p in [1, 2]],
            'mean_stages_ms': {key: statistics.mean(r['stages'][key] for r in selected) for key in first[0]['stages']}}
    write('result', {'methods': summary, 'timed_calls': len(records), 'unique_messages': 50,
        'all_predictions_match': True, 'all_selected_logits_within': .0005,
        'all_layers_executed': 24, 'layers_skipped': 0})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

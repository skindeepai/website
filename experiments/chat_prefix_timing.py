"""Fixed-instruction prefix reuse; isolated execution-equivalence diagnostic."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/chat-prefix-timing'
MARKER = '__SKINDEEP_FIXED_PREFIX_BOUNDARY__'
PATHS = ['full_input', 'prefix_reuse']


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def plan():
    parent = read('results/chat-refinement/protocol.json')
    original = read('results/chat600/protocol.json')
    dependencies = ['experiments/chat_prefix_timing.py', 'results/chat-refinement/protocol.json',
                    'results/chat-refinement/predictions.json', 'results/chat600/protocol.json',
                    'results/chat600/heads.npz',
                    'experiments/.cache/replay-runtime/transformers/cache_utils.py',
                    'experiments/.cache/replay-runtime/transformers/models/qwen2/modeling_qwen2.py']
    return {'scope': 'Execution equivalence and throughput on the same consumed100; no fresh quality evaluation or model selection.',
            'model': parent['qwen_model'], 'revision': parent['qwen_revision'],
            'evaluation_ids': parent['splits']['evaluation'], 'source_data_sha256': parent['data_sha256'],
            'system_prompt': original['system_prompt'], 'user_token_limit': 256,
            'prefix_rule': 'Render fixed system prompt and user content marker with the original chat template. Take text strictly before marker; strip trailing CR/LF only, unconditionally. Tokenize that fixed text. No user content or user-dependent prefix adaptation.',
            'prefix_marker': MARKER,
            'boundary': 'Every complete tokenized request must start with the exact fixed prefix token sequence and have a nonempty suffix. Any mismatch fails that attempt; never shorten or repair based on the request.',
            'cache': 'Build fixed-prefix KV once per100-message prefix workload INSIDE corpus timer. Preserve immutable24-pair tensor tuple; each request clones all K/V tensors and constructs a new DynamicCache. Never reuse a cache extended with user tokens.',
            'positions': 'Suffix attention_mask spans prefix+suffix; position_ids and cache_position cover prefix_length..full_length-1. No padding. Require default/no dynamic RoPE scaling.',
            'cache_checks': 'Assert per-request tensor storage differs from immutable prefix, all24 initial lengths equal prefix length, and all24 final lengths equal full length. Hash immutable prefix before/after each workload; include integrity/clone checks in timer.',
            'runtime': 'torch2.6.0+cpu, transformers4.50.3; CPUfloat32,eagerattention,4threads,1interop.',
            'paths': PATHS, 'passes': 3,
            'order': 'Whole100-message workloads alternate [full_input,prefix_reuse], [prefix_reuse,full_input], [full_input,prefix_reuse]. Manifest message order unchanged.',
            'timing': 'Corpus wall timer includes raw tokenization/truncation/template preparation, prefix derivation/build when used, per-request cache cloning, tensor preparation, all24-layer forwards, readout, hooks, record conversion and cache integrity checks. Excludes model/data loading, warmup, post-corpus prediction/logit validation, aggregation/file writes.',
            'warmup': 'One full request and one prefix-build-plus-suffix request; excluded. Every measured prefix workload builds its own new prefix cache.',
            'equivalence': 'All labels match sealed float_qwen outputs. First measured full_input workload supplies numerical logits; compare every later record without adjusting tolerance.',
            'logit_tolerance': {'atol': .001, 'rtol': .0001},
            'failure_policy': 'Keep all6 workload attempts, including errors, incomplete coverage, slower results and parity failures. Failed methods cannot claim equivalent speedup.',
            'limits': ['Fixed prompt, fixed model and fixed device only; changes invalidate a prefix cache.',
                       'All24 layers still process every suffix. Prefix-prefill also executes24 layers; no early exit.',
                       'No user-message KV is shared across requests.',
                       'One CPU, one sequence of three warm passes; no hardware replication.',
                       'Only100 consumed balanced messages; not independent moderation validation or live queue timing.'],
            'source_sha256': {path: sha(path) for path in dependencies}}


def run(protocol):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT/'experiments/.cache/replay-runtime'))
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.__version__ == '2.6.0+cpu' and transformers.__version__ == '4.50.3'
    source = ROOT/'experiments/.cache/toxicchat/toxic-chat_annotation_test.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol['source_data_sha256']['test']
    source_rows = list(csv.DictReader(source.open(encoding='utf-8', newline='')))
    rows = []
    for rid in protocol['evaluation_ids']:
        origin, index = rid.split(':'); assert origin == 'test'
        r = source_rows[int(index)]; assert r['human_annotation'] == 'True'
        rows.append({'id': rid, 'text': r['user_input'], 'label': int(r['toxicity'])})
    expected = {r['id']: r for r in read('results/chat-refinement/predictions.json') if r['path'] == 'float_qwen'}
    assert len(rows) == len(expected) == 100 and {r['id'] for r in rows} == set(expected)
    tokenizer = AutoTokenizer.from_pretrained(protocol['model'], revision=protocol['revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(protocol['model'], revision=protocol['revision'], local_files_only=True,
                                               torch_dtype=torch.float32, attn_implementation='eager').eval().model
    assert len(model.layers) == 24
    rope = model.config.rope_scaling
    assert rope is None or rope.get('rope_type', rope.get('type', 'default')) == 'default', 'Dynamic/nondefault RoPE not supported'
    saved = np.load(ROOT/'results/chat600/heads.npz', allow_pickle=False)
    head = {k: torch.tensor(saved['24_'+k]) for k in ['weight', 'bias', 'mean', 'std']}

    def template(text):
        return tokenizer.apply_chat_template([{'role': 'system', 'content': protocol['system_prompt']},
                                               {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)

    def fixed_prefix():
        text = template(MARKER)
        assert text.count(MARKER) == 1
        prefix_text = text.split(MARKER)[0].rstrip('\r\n')
        ids = tokenizer(prefix_text)['input_ids']
        assert len(ids) > 0
        return ids

    def prepare(items):
        prepared = []
        for row in items:
            ids = tokenizer.encode(row['text'], add_special_tokens=False)
            text = tokenizer.decode(ids[:256], skip_special_tokens=False) if len(ids) > 256 else row['text']
            prepared.append({'id': row['id'], 'label': row['label'], 'ids': tokenizer(template(text))['input_ids'],
                             'truncated': len(ids) > 256})
        return prepared

    def forward(ids, total_length, cache=None, start_position=0):
        positions = torch.arange(start_position, start_position+len(ids), dtype=torch.long)
        visited = []
        hooks = [layer.register_forward_hook(lambda m,i,o,d=d: visited.append(d)) for d,layer in enumerate(model.layers, 1)]
        try:
            out = model(input_ids=torch.tensor([ids], dtype=torch.long),
                        attention_mask=torch.ones(1, total_length, dtype=torch.long),
                        position_ids=positions.unsqueeze(0), cache_position=positions,
                        past_key_values=cache, use_cache=cache is not None)
        finally:
            for hook in hooks: hook.remove()
        assert visited == list(range(1, 25))
        return out, visited

    def prefix_digest(prefix):
        digest = hashlib.sha256()
        for pair in prefix:
            for tensor in pair: digest.update(tensor.detach().numpy().tobytes())
        return digest.hexdigest()

    def workload(path, items):
        records, prefill_trace = [], []
        record = {'path': path, 'records': records, 'complete': False}
        start = time.perf_counter()
        try:
            prepared = prepare(items)
            prefix, prefix_ids = None, []
            if path == 'prefix_reuse':
                prefix_ids = fixed_prefix()
                for row in prepared:
                    assert row['ids'][:len(prefix_ids)] == prefix_ids and len(row['ids']) > len(prefix_ids), 'Fixed token prefix mismatch: '+row['id']
                before = time.perf_counter()
                out, prefill_trace = forward(prefix_ids, len(prefix_ids), DynamicCache())
                prefix = out.past_key_values.to_legacy_cache()
                assert len(prefix) == 24 and all(k.shape[-2] == len(prefix_ids) and v.shape[-2] == len(prefix_ids) for k,v in prefix)
                record.update(prefix_build_seconds=time.perf_counter()-before,
                              prefix_tokens=len(prefix_ids), prefix_prefill_layers=prefill_trace,
                              prefix_ids=prefix_ids, prefix_cache_bytes=sum(t.numel()*t.element_size() for pair in prefix for t in pair))
                before_hash = prefix_digest(prefix)
                del out
            for row in prepared:
                before = time.perf_counter()
                ids = row['ids']
                clone_bytes = 0
                if prefix is not None:
                    pairs = tuple((k.clone(), v.clone()) for k,v in prefix)
                    assert all(c.data_ptr() != source.data_ptr() for pair, original in zip(pairs,prefix) for c,source in zip(pair,original))
                    cache = DynamicCache.from_legacy_cache(pairs)
                    assert all(cache.get_seq_length(d) == len(prefix_ids) for d in range(24))
                    assert all(k.shape[-2] == len(prefix_ids) and v.shape[-2] == len(prefix_ids) for k,v in prefix)
                    suffix = ids[len(prefix_ids):]
                    out, visited = forward(suffix, len(ids), cache, len(prefix_ids))
                    assert all(cache.get_seq_length(d) == len(ids) for d in range(24))
                    clone_bytes = record['prefix_cache_bytes']
                else:
                    suffix = ids
                    out, visited = forward(ids, len(ids))
                h = out.last_hidden_state[:, -1, :]
                logits = torch.nn.functional.linear((h-head['mean'])/head['std'], head['weight'], head['bias'])
                assert bool(torch.isfinite(logits).all()), 'Nonfinite logits: '+row['id']
                values = logits[0].tolist()
                records.append({'id': row['id'], 'label': row['label'], 'prediction': int(logits.argmax(1)), 'logits': values,
                                'full_tokens': len(ids), 'forward_tokens': len(suffix), 'cached_prefix_tokens': len(prefix_ids),
                                'cloned_cache_bytes': clone_bytes, 'executed_layers': visited, 'layers_skipped': 0,
                                'truncated': row['truncated'], 'request_seconds': time.perf_counter()-before})
                del out
            if prefix is not None:
                record.update(prefix_hash_before=before_hash, prefix_hash_after=prefix_digest(prefix))
                assert record['prefix_hash_before'] == record['prefix_hash_after'], 'Immutable prefix changed'
            record['complete'] = True
        except Exception as error:
            record.update(error=str(error), error_type=type(error).__name__)
        record['corpus_seconds'] = time.perf_counter()-start
        record['coverage_valid'] = len(records) == len(items) and {r['id'] for r in records} == {r['id'] for r in items}
        record['label_mismatches'] = [r['id'] for r in records if r['prediction'] != expected[r['id']]['prediction']]
        record['labels_valid'] = all(r['label'] == expected[r['id']]['label'] for r in records)
        record['traces_valid'] = all(r['executed_layers'] == list(range(1, 25)) for r in records)
        record['processed_tokens'] = record.get('prefix_tokens', 0)+sum(r['forward_tokens'] for r in records)
        record['full_input_tokens'] = sum(r['full_tokens'] for r in records)
        return record

    started = datetime.now(timezone.utc).isoformat()
    runs = []
    with torch.inference_mode():
        warmup = [workload(path, rows[:1]) for path in PATHS]
        write('warmup.json', warmup)
        for repeat in range(3):
            order = PATHS if repeat % 2 == 0 else list(reversed(PATHS))
            for position, path in enumerate(order):
                result = workload(path, rows)
                result.update(repetition=repeat, order_position=position)
                runs.append(result)
                write('progress.json', runs)
                print(f'{path} pass{repeat+1}: {result["corpus_seconds"]:.3f}s complete={result["complete"]}, differences={len(result["label_mismatches"])}', flush=True)
    reference_run = next((r for r in runs if r['path'] == 'full_input' and r['repetition'] == 0 and r['complete']), None)
    reference = {r['id']: r['logits'] for r in reference_run['records']} if reference_run else {}
    for attempt in runs:
        deltas = []
        for row in attempt['records']:
            target = reference.get(row['id'])
            close = target is not None and bool(np.allclose(row['logits'], target, **protocol['logit_tolerance']))
            row['logits_within_tolerance'] = close
            if target is not None:
                delta = max(abs(a-b) for a,b in zip(row['logits'], target))
                row['max_logit_delta'] = delta
                deltas.append(delta)
        attempt['max_logit_delta'] = max(deltas, default=None)
        attempt['equivalent_execution'] = all([attempt['complete'], attempt['coverage_valid'], attempt['labels_valid'],
            attempt['traces_valid'], not attempt['label_mismatches'], len(deltas) == 100,
            all(r['logits_within_tolerance'] for r in attempt['records'])])
    summary = {}
    for path in PATHS:
        attempts = [r for r in runs if r['path'] == path]
        complete = all(r['complete'] and r['coverage_valid'] for r in attempts)
        times = [r['corpus_seconds'] for r in attempts]
        summary[path] = {'all_complete': complete, 'all_equivalent': all(r['equivalent_execution'] for r in attempts),
                         'pass_seconds': times, 'mean_seconds': statistics.mean(times) if complete else None,
                         'correct_first_pass': sum(r['label'] == r['prediction'] for r in attempts[0]['records']),
                         'first_pass_n': len(attempts[0]['records']),
                         'processed_tokens_per_pass': [r['processed_tokens'] for r in attempts],
                         'full_input_tokens_per_pass': [r['full_input_tokens'] for r in attempts],
                         'max_logit_delta': max((r['max_logit_delta'] for r in attempts if r['max_logit_delta'] is not None), default=None),
                         'layers_skipped_per_message': 0}
    baseline = summary['full_input']['mean_seconds']
    for value in summary.values():
        value['relative_time_saving'] = 1-value['mean_seconds']/baseline if baseline and value['mean_seconds'] is not None else None
    write('records.json', runs)
    write('result.json', {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                         'protocol_sha256': sha('results/chat-prefix-timing/protocol.json'), 'paths': summary,
                         'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__, 'threads': torch.get_num_threads(), 'rope_scaling': rope},
                         'scope': protocol['scope'], 'limits': protocol['limits']})
    (OUT/'progress.json').unlink()
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = plan()
    if args.prepare:
        if (OUT/'protocol.json').exists(): assert read('results/chat-prefix-timing/protocol.json') == protocol
        else: write('protocol.json', protocol)
        print('Sealed protocol; no tokenizer/model computation. Wait for isolated timing window.')
        return
    assert read('results/chat-prefix-timing/protocol.json') == protocol, 'Sealed source/input changed'
    assert not any((OUT/name).exists() for name in ['warmup.json', 'progress.json', 'result.json']), 'Preserve prior attempt'
    run(protocol)


if __name__ == '__main__':
    main()

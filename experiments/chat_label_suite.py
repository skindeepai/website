"""Frozen six-method replay and generated-label controls. No fitting or selection."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/.cache/replay-runtime'))
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from chat_smoke_specialist import Specialist, CACHE
from chat_smoke_adaptation import LowRank

OUT = ROOT / 'results/chat-label-suite/replay-100'
DEPTHS = [6, 12, 18, 24]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def write(name, value):
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


class Stop(Exception):
    def __init__(self, probability, depth):
        self.probability, self.depth = probability, depth


def main():
    assert not OUT.exists(), 'Preserve evidence: use a new directory for another run.'
    OUT.mkdir(parents=True)
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    records = []
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        assert transformers.__version__ == '4.50.3'
        assert torch.__version__ == '2.6.0+cpu'
        common = read('results/chat-smoke/protocol.json')
        parent = read('results/chat600/protocol.json')
        assert common['system_prompt'] == parent['system_prompt']
        data_path = ROOT / 'experiments/.cache/toxicchat/toxic-chat_annotation_test.csv'
        assert sha(data_path) == parent['data_sha256']['test']
        with data_path.open(encoding='utf-8', newline='') as f:
            data = list(csv.DictReader(f))
        rows = []
        for rid in common['splits']['evaluation']:
            assert rid.startswith('test:')
            raw = data[int(rid.split(':')[1])]
            assert raw['human_annotation'] == 'True'
            rows.append(dict(id=rid, text=raw['user_input'], expected=int(raw['toxicity'])))
        assert len(rows) == len({r['id'] for r in rows}) == 100
        assert sum(r['expected'] for r in rows) == 50
        files = ['results/chat-smoke/protocol.json', 'results/chat600/protocol.json',
                 'results/chat-smoke-heads/mlp-heads.npz', 'results/chat-smoke-heads/mlp-gates.npz',
                 'results/chat-smoke-heads/result.json', 'results/chat-smoke-heads/predictions.json',
                 'results/chat-smoke-specialist/specialist.npz', 'results/chat-smoke-specialist/result.json',
                 'results/chat-smoke-specialist/predictions.json', 'results/chat600/heads.npz',
                 'results/chat-smoke-adaptation/distill-weights.npz', 'results/chat-smoke-adaptation/distill.json',
                 'experiments/chat_smoke_specialist.py', 'experiments/chat_smoke_adaptation.py',
                 'experiments/censor_decision_parsers.cjs']
        files += ['experiments/.cache/bert-tiny/' + x for x in ['config.json', 'vocab.txt', 'model.safetensors']]
        prompts = {allow: common['system_prompt'].replace('SAFE', allow) for allow in ['SAFE', 'OK']}
        variants = [(method, allow) for method in ['full', 'fixed12', 'learned', 'cascade', 'direct', 'generated'] for allow in prompts]
        variants.append(('tiny', 'unchanged'))
        orders = [variants[i % len(variants):] + variants[:i % len(variants)] for i in range(100)]
        distill_orders = [[('distill', 'SAFE'), ('distill', 'OK')][::1 if i % 2 == 0 else -1] for i in range(100)]
        protocol = dict(createdUTC=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__)),
            inputs={f: sha(ROOT / f) for f in files}, dataset_sha256=sha(data_path),
            dataset_url='https://huggingface.co/datasets/lmsys/toxic-chat',
            dataset_revision='29df8e4dba60e1f4af4b4075c0705c5b313548a8',
            model=common['qwen_model'], revision=common['qwen_revision'],
            torch=torch.__version__, transformers=transformers.__version__, platform=platform.platform(),
            threads=4, interop_threads=1, logical_processors=os.cpu_count(), device='CPU', dtype='float32', attention='eager',
            evaluation=[{k: r[k] for k in ['id', 'expected']} for r in rows], prompts=prompts,
            passes=1, order=orders, distill_order=distill_orders, planned_calls=1500,
            max_message_tokens=256, max_generated_tokens=8,
            timing='One warmed interleaved pass per variant; distillation in a separate final phase. Includes raw input preparation, actual model work, readout and trace hooks; excludes loading, warmup and JS parser (measured separately).',
            generated='Unrestricted greedy generation, normal EOS, at most 8 tokens. Raw output retained. All 24 layers per decoding step.',
            direct='Full Qwen vocabulary projection; compare SAFE/OK against BLOCK logits, no trained task head.',
            classifier='All six original rows emit numeric decisions. SAFE-to-OK changes the Qwen INPUT template, not a text-output parser. Tiny sees identical raw text and is run once.',
            training='Frozen original heads, adapters, decision thresholds and gates. No fitting, prompt selection or gate adjustment. OK is a template shift from SAFE training.',
            limitations=['Reused balanced 100-message smoke test, not fresh traffic or reliability validation.',
                'No recent chat, usernames or compact hints; different policy from the application prompt.',
                'Different training between methods; cascade fallback uses the original linear head, not the full-row MLP.',
                'Single CPU timing pass, not browser/GPU performance; comparisons to old timings are not speed improvements.',
                'Distillation timing phase is separate; no strict cross-phase speed comparison.',
                'Parser was designed using a subset of these messages; no claim of independent parser generalization.'])
        write('protocol', protocol)
        tokenizer = AutoTokenizer.from_pretrained(common['qwen_model'], revision=common['qwen_revision'], local_files_only=True)
        tokens = {word: tokenizer.encode(word, add_special_tokens=False) for word in ['SAFE', 'OK', 'BLOCK']}
        assert all(len(v) == 1 for v in tokens.values())
        model = AutoModelForCausalLM.from_pretrained(common['qwen_model'], revision=common['qwen_revision'],
            local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
        tiny = Specialist()
        arrays = np.load(ROOT / 'results/chat-smoke-specialist/specialist.npz', allow_pickle=False)
        tiny.load_state_dict({k: torch.tensor(arrays[k]) for k in arrays.files})
        tiny.eval()
        bt = AutoTokenizer.from_pretrained(CACHE, local_files_only=True)
        specialist = read('results/chat-smoke-specialist/result.json')
        a = np.load(ROOT / 'results/chat-smoke-heads/mlp-heads.npz', allow_pickle=False)
        heads = {d: {k: torch.tensor(a[f'{d}_{k}']) for k in ['0.weight', '0.bias', '3.weight', '3.bias', 'mean', 'std', 'temperature']} for d in DEPTHS}
        a = np.load(ROOT / 'results/chat-smoke-heads/mlp-gates.npz', allow_pickle=False)
        gates = {d: {k: torch.tensor(a[f'{d}_{k}']) for k in ['weight', 'bias', 'mean', 'std']} for d in DEPTHS[:-1]}
        policy = read('results/chat-smoke-heads/result.json')['mlp']['learned']['policy']
        a = np.load(ROOT / 'results/chat600/heads.npz', allow_pickle=False)
        fallback_head = {k: torch.tensor(a[f'24_{k}']) for k in ['weight', 'bias', 'mean', 'std']}
        expected_mlp = {r['id']: r for r in read('results/chat-smoke-heads/predictions.json')['mlp']['evaluation']}
        expected_tiny = {r['id']: r for r in read('results/chat-smoke-specialist/predictions.json') if r['split'] == 'evaluation'}
        expected_distill = {r['id']: r for r in read('results/chat-smoke-adaptation/distill.json')['predictions'] if r['split'] == 'evaluation'}
        distill_head = None

        def probability(d, h):
            x = heads[d]
            z = F.gelu(F.linear((h-x['mean'])/x['std'], x['0.weight'], x['0.bias']))
            return (F.linear(z, x['3.weight'], x['3.bias'])/x['temperature']).softmax(1)

        def execute(row, method, allow):
            start = time.perf_counter()
            ids = tokenizer.encode(row['text'], add_special_tokens=False)
            bounded = tokenizer.decode(ids[:256], skip_special_tokens=False) if len(ids) > 256 else row['text']
            visited, bert_visited, handles, checkpoints, risks = [], [], [], {}, {}
            p = None
            fallback = False
            generated = []
            scores = None
            input_length = None
            if method in ['tiny', 'cascade']:
                th = [layer.register_forward_hook(lambda m, i, o, d=d: bert_visited.append(d)) for d, layer in enumerate(tiny.encoder.encoder.layer, 1)]
                try:
                    p = float(tiny(bt(bounded, return_tensors='pt', truncation=True, max_length=512)).softmax(1)[0, 1])
                finally:
                    for h in th:
                        h.remove()
                if method == 'tiny':
                    pred = int(p >= specialist['decision_threshold'])
                else:
                    c = specialist['cascade']
                    fallback = not (p <= c['safe'] or p >= c['block'])
                    pred = int(p >= c['block'])

            def after(d):
                def hook(mod, inp, output):
                    visited.append(d)
                    if method == 'fixed12' and d == 12:
                        raise Stop(probability(d, output[0][:, -1, :]), d)
                    if method != 'learned' or d not in DEPTHS[:-1]:
                        return
                    probs = probability(d, output[0][:, -1, :])
                    checkpoints[d] = probs
                    confidence, label = probs.max(1)
                    entropy = -(probs*probs.clamp_min(1e-9).log()).sum(1)
                    if d == 6:
                        previous_confidence = agreement = delta = torch.zeros_like(confidence)
                    else:
                        previous = checkpoints[d-6]
                        previous_confidence = previous.max(1).values
                        agreement = (previous.argmax(1) == label).float()
                        delta = probs[:, 1]-previous[:, 1]
                    features = torch.stack([probs[:, 1], confidence, entropy, previous_confidence, agreement, delta], 1)
                    g = gates[d]
                    risk = F.linear((features-g['mean'])/g['std'], g['weight'], g['bias']).sigmoid()[0]
                    risks[str(d)] = risk.tolist()
                    if float(risk[0]) <= policy['error_limit'] and float(risk[1]) <= policy['benefit_limit']:
                        raise Stop(probs, d)
                return hook

            if method not in ['tiny', 'cascade'] or fallback:
                rendered = tokenizer.apply_chat_template([{'role': 'system', 'content': prompts[allow]},
                    {'role': 'user', 'content': bounded}], tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(rendered, return_tensors='pt')
                input_length = inputs['input_ids'].shape[1]
                handles = [layer.register_forward_hook(after(d)) for d, layer in enumerate(model.model.layers, 1)]
                try:
                    if method == 'generated':
                        output = model.generate(**inputs, do_sample=False, max_new_tokens=8, use_cache=True,
                            repetition_penalty=1., temperature=1., top_p=1., top_k=0, logits_to_keep=1, pad_token_id=tokenizer.eos_token_id)
                        generated = output[0, input_length:].tolist()
                        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
                        pred = 1 if raw == 'BLOCK' else 0 if raw == allow else None
                    elif method == 'direct':
                        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
                        scores = [float(logits[tokens[allow][0]]), float(logits[tokens['BLOCK'][0]])]
                        pred = int(scores[1] > scores[0])
                    else:
                        hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
                        if method in ['cascade', 'distill']:
                            h = fallback_head if method == 'cascade' else distill_head
                            probs = F.linear((hidden-h['mean'])/h['std'], h['weight'], h['bias']).softmax(1)
                        else:
                            probs = probability(24, hidden)
                        pred = int(probs.argmax(1))
                        scores = probs[0].tolist()
                except Stop as stopped:
                    pred = int(stopped.probability.argmax(1))
                    scores = stopped.probability[0].tolist()
                finally:
                    for h in handles:
                        h.remove()
            if method != 'generated':
                raw = 'BLOCK' if pred else 'SAFE' if allow == 'unchanged' else allow
            elapsed = (time.perf_counter()-start)*1000
            depth = max(visited, default=0)
            assert visited == list(range(1, depth+1)) * (len(generated) if method == 'generated' else 1)
            assert bert_visited == ([1, 2] if method in ['tiny', 'cascade'] else [])
            if row['id'] != 'warmup' and allow in ['SAFE', 'unchanged']:
                if method in ['full', 'fixed12', 'learned']:
                    ref = expected_mlp[row['id']]
                    target_depth = 12 if method == 'fixed12' else 24
                    target_pred = int(np.argmax(ref['probabilities'][str(target_depth)]))
                    if method == 'learned':
                        target_pred, target_depth = ref['learned']['prediction'], ref['learned']['depth']
                    assert (pred, depth) == (target_pred, target_depth), (row['id'], method, pred, depth)
                elif method in ['tiny', 'cascade']:
                    ref = expected_tiny[row['id']]
                    assert pred == ref['specialist' if method == 'tiny' else 'cascade']
                    if method == 'cascade':
                        assert fallback == ref['fallback']
                elif method == 'distill':
                    assert pred == expected_distill[row['id']]['heads']['24']['prediction']
            return dict(id=row['id'], expected=row['expected'], method=method, allow=allow, prediction=pred,
                output=raw, total_ms=elapsed, qwen_depth=depth, executed_qwen_layers=visited,
                bert_layers=bert_visited, fallback=fallback, tiny_block_probability=p, scores=scores,
                gate_risks=risks, generated_tokens=generated, input_tokens=input_length,
                original_message_tokens=len(ids), truncated=len(ids)>256)

        with torch.inference_mode():
            warm = dict(id='warmup', text='Thank you for your help.', expected=0)
            warmup = [execute(warm, method, allow) for method, allow in variants]
            write('warmup', warmup)
            for i, row in enumerate(rows):
                for method, allow in orders[i]:
                    records.append(execute(row, method, allow))
                if (i+1) % 5 == 0:
                    write('records', records)
                    print(f'Base methods: {i+1}/100 messages; {len(records)}/1500 calls', flush=True)
            # Preserve the base-model phase before installing the frozen adapters.
            a = np.load(ROOT / 'results/chat-smoke-adaptation/distill-weights.npz', allow_pickle=False)
            for layer in model.model.layers:
                for name in ['q_proj', 'v_proj']:
                    setattr(layer.self_attn, name, LowRank(getattr(layer.self_attn, name)))
            adapter = {k.removeprefix('adapter::'): torch.tensor(a[k]) for k in a.files if k.startswith('adapter::')}
            assert set(adapter) == {k for k in model.model.state_dict() if k.endswith(('.a', '.b'))}
            incompatible = model.model.load_state_dict(adapter, strict=False)
            assert not incompatible.unexpected_keys
            assert all(not k.endswith(('.a', '.b')) for k in incompatible.missing_keys)
            model.eval()
            distill_head = {k: torch.tensor(a['head::24.' + ('linear.'+k if k in ['weight', 'bias'] else k)]) for k in ['mean', 'std', 'weight', 'bias']}
            warmup += [execute(warm, 'distill', allow) for allow in prompts]
            write('warmup', warmup)
            for i, row in enumerate(rows):
                for method, allow in distill_orders[i]:
                    records.append(execute(row, method, allow))
                if (i+1) % 10 == 0:
                    write('records', records)
                    print(f'Distillation: {i+1}/100 messages; {len(records)}/1500 calls', flush=True)
        assert len(records) == 1500
        assert sha(Path(__file__)) == protocol['source_sha256']
        assert all(sha(ROOT / f) == h for f, h in protocol['inputs'].items())
        summary = {}
        for method, allow in variants + [('distill', 'SAFE'), ('distill', 'OK')]:
            group = [r for r in records if r['method'] == method and r['allow'] == allow]
            assert len(group) == 100
            summary[method+'_'+allow] = dict(n=100, correct=sum(r['prediction'] == r['expected'] for r in group),
                invalid=sum(r['prediction'] is None for r in group),
                toxic_not_blocked=sum(r['expected'] == 1 and r['prediction'] != 1 for r in group),
                false_block=sum(r['expected'] == 0 and r['prediction'] == 1 for r in group),
                mean_ms=sum(r['total_ms'] for r in group)/100,
                qwen_exit_counts=dict(Counter(r['qwen_depth'] for r in group)),
                mean_qwen_blocks=sum(len(r['executed_qwen_layers']) for r in group)/100,
                mean_generated_tokens=sum(len(r['generated_tokens']) for r in group)/100)
        write('records', records)
        write('result', dict(protocol_sha256=sha(OUT/'protocol.json'), calls=len(records),
            completedUTC=datetime.now(timezone.utc).isoformat(), original_six_prediction_parity=True, summaries=summary))
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        write('failure', dict(error=repr(error), completed_calls=len(records)))
        write('records', records)
        raise


if __name__ == '__main__':
    main()

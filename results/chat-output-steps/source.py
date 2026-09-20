"""Paired input-pass, first-token and longer-reply controls on fixed balanced50."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import hashlib
import json
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
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from chat_smoke_common import source_rows

OUT = ROOT / 'results/chat-output-steps'
LATE = ROOT / 'results/chat-late-exit'
VARIANTS = [('classifier', 'SAFE'), ('vocabulary2', 'SAFE'), ('constrained1', 'SAFE'),
            ('greedy1', 'SAFE'), ('greedy8', 'SAFE'), ('greedy1', 'OK'), ('greedy8', 'OK'),
            ('greedy1', 'THIS IS SAFE'), ('greedy8', 'THIS IS SAFE')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / (name+'.json')).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    assert (LATE / 'result.json').exists(), 'Finish the other model job before timing.'
    assert not OUT.exists(), 'Preserve evidence; use a new output directory to repeat.'
    OUT.mkdir(parents=True)
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.__version__ == '2.6.0+cpu' and transformers.__version__ == '4.50.3'
    previous = json.loads((LATE / 'protocol.json').read_text(encoding='utf-8'))
    data = source_rows()
    rows = [data[rid] for rid in previous['splits']['evaluation']]
    model = AutoModelForCausalLM.from_pretrained(previous['model'], revision=previous['revision'], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(previous['model'], revision=previous['revision'], local_files_only=True)
    tokens = {word: tokenizer.encode(word, add_special_tokens=False) for word in ['SAFE', 'BLOCK', 'OK', 'THIS IS SAFE']}
    assert all(len(tokens[word]) == 1 for word in ['SAFE', 'BLOCK', 'OK'])
    prompts = {word: previous['system_prompt'].replace('SAFE', word) for word in ['SAFE', 'OK', 'THIS IS SAFE']}
    protocol = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(), 'model': previous['model'], 'revision': previous['revision'],
        'scope': 'Reused balanced50; output-path smoke comparison, no independent quality acceptance.',
        'evaluation': [{'id': r['id'], 'label': r['label']} for r in rows], 'variants': VARIANTS, 'passes': 2,
        'prompts': prompts, 'label_token_ids': tokens, 'data_sha256': previous['data_sha256'],
        'trained_head': 'Frozen new layer24 linear classifier from chat-late-exit, trained on384. Other rows use untouched Qwen vocabulary weights, not equal task supervision.',
        'vocabulary2': 'One backbone input pass, multiply final state by only SAFE/BLOCK vocabulary rows; return argmax. No full vocabulary projection or generated tokens.',
        'constrained1': 'Ordinary generate API, full last-position vocabulary projection, restrict to SAFE/BLOCK, exactly1 output token. Same decision scores as vocabulary2 within numerical tolerance.',
        'greedy1': 'Unrestricted greedy output capped at1 token. Score its first non-whitespace character S/O/T versus B under the declared label contract. Unrecognized output blocks; also retain exact-label scoring.',
        'greedy8': 'Unrestricted greedy output, EOS stopping, at most8 tokens. Offline supplied/enhanced/exact parsers retained with equal blocking fallback.',
        'timing': 'Two passes, rotate9 methods by(item+pass)%9 and reverse second-pass order. CPUfloat32 eager,4 compute threads/1 interop,batch1. Includes input truncation/template/tokenization, layer hooks, forward/readout/generation and token-to-text conversion. Excludes setup/warmup/fileIO/offline parsing. No concurrent launched model jobs.',
        'warmup': 'One excluded call per variant on a separate harmless sentence.',
        'trace': 'Record every executed block and vocabulary-head call. First output token comes from input prefill; later steps use KV cache. Repeated24-block traces do not imply equal work per input/output pass.',
        'comparison': 'Primary chart reports50 unique examples; timing repeats are not new quality samples. Token spelling/prompt changes can change decisions even at the same token count. No CtrlVox benchmark.',
        'inputs': {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [LATE/'protocol.json', LATE/'heads.npz', LATE/'records.json', ROOT/'experiments/chat_smoke_common.py', ROOT/'experiments/censor_decision_parsers.cjs']},
        'source_sha256': sha(Path(__file__)), 'torch':torch.__version__, 'transformers':transformers.__version__,
    }
    write('protocol', protocol)
    arrays = np.load(LATE / 'heads.npz', allow_pickle=False)
    head = {key: torch.tensor(arrays['24_'+key]) for key in ['weight', 'bias', 'mean', 'std']}
    label_ids = [tokens[word][0] for word in ['SAFE', 'BLOCK']]
    vocabulary_rows = model.lm_head.weight[label_ids].detach().clone()
    reference = {r['id']:r['prediction'] for r in json.loads((LATE/'records.json').read_text()) if r['depth']==24 and r['pass']==1}

    class Restrict(LogitsProcessor):
        def __call__(self, input_ids, scores):
            values = scores[:, label_ids].clone()
            scores[:] = -float('inf')
            scores[:, label_ids] = values
            return scores
    restriction = LogitsProcessorList([Restrict()])

    def execute(row, method, allow):
        started = time.perf_counter()
        message_ids = tokenizer.encode(row['text'], add_special_tokens=False)
        message = tokenizer.decode(message_ids[:256], skip_special_tokens=False) if len(message_ids)>256 else row['text']
        text = tokenizer.apply_chat_template([{'role':'system','content':prompts[allow]}, {'role':'user','content':message}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt')
        visited, lm_calls, hooks = [], [], []
        for depth, layer in enumerate(model.model.layers, 1):
            hooks.append(layer.register_forward_hook(lambda mod, inp, out, d=depth: visited.append(d)))
        hooks.append(model.lm_head.register_forward_hook(lambda *args:lm_calls.append(1)))
        output_ids, raw, prediction, scores = [], None, None, None
        try:
            if method in ['classifier', 'vocabulary2']:
                hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
                logits = F.linear((hidden-head['mean'])/head['std'], head['weight'], head['bias']) if method=='classifier' else F.linear(hidden, vocabulary_rows)
                prediction = int(logits.argmax(1))
                scores = logits[0].tolist()
            else:
                options = {'logits_processor':restriction} if method=='constrained1' else {}
                output = model.generate(**inputs, do_sample=False, max_new_tokens=8 if method=='greedy8' else 1,
                    use_cache=True, repetition_penalty=1., temperature=1., top_p=1., top_k=0,
                    logits_to_keep=1, pad_token_id=tokenizer.eos_token_id, **options)
                output_ids = output[0, inputs['input_ids'].shape[1]:].tolist()
                raw = tokenizer.decode(output_ids, skip_special_tokens=True)
        finally:
            for hook in hooks:
                hook.remove()
        ms = (time.perf_counter()-started)*1000
        forwards = len(output_ids) if output_ids else 1
        assert visited == list(range(1,25))*forwards
        assert len(lm_calls) == (0 if method in ['classifier', 'vocabulary2'] else forwards)
        if method=='classifier' and row['id'] in reference:
            assert prediction == reference[row['id']]
        if method=='constrained1':
            assert raw in ['SAFE','BLOCK']
        return {'id':row['id'], 'label':row['label'], 'method':method, 'allow':allow,
            'prediction':prediction, 'scores':scores, 'raw':raw, 'output_ids':output_ids,
            'input_tokens':inputs['input_ids'].shape[1], 'ms':ms, 'forward_passes':forwards,
            'executed_layers':visited, 'vocabulary_calls':len(lm_calls)}

    records = []
    with torch.inference_mode():
        warmup = {'id':'warmup', 'text':'Thank you for your help.', 'label':0}
        write('warmup', [execute(warmup,*variant) for variant in VARIANTS])
        for repeat in range(2):
            for index, row in enumerate(rows):
                shift = (index+repeat)%len(VARIANTS)
                order = VARIANTS[shift:]+VARIANTS[:shift]
                if repeat == 1:
                    order = list(reversed(order))
                for position, variant in enumerate(order):
                    record = execute(row,*variant)
                    record.update({'pass':repeat+1, 'position':position+1})
                    records.append(record)
                if (index+1)%5==0:
                    write('records',records)
                    print(f'output paths pass{repeat+1}: {index+1}/50',flush=True)
    write('records', records)
    write('complete', {'timed_calls':len(records), 'unique_messages':len(rows), 'passes':2, 'all_layer_checks_passed':True})
    print('Completed900 timed output-path calls.',flush=True)


if __name__=='__main__':
    main()

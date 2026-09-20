"""Sealed throughput comparison for an available queue, not live request latency.

No training, quantization, new head or skipped layers. Prepare, review, then run
only in an isolated timing window. Preserve every outcome, including failures.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/chat-batch-timing'
BATCHES = [1, 4, 8]
PASSES = 3


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def plan():
    parent = read('results/chat-refinement/protocol.json')
    original = read('results/chat600/protocol.json')
    sources = ['experiments/chat_batch_timing.py', 'results/chat-refinement/protocol.json',
               'results/chat-refinement/predictions.json', 'results/chat600/protocol.json',
               'results/chat600/heads.npz']
    return {
        'scope': 'Queued-message throughput follow-up chosen after inspecting refinement failures. Existing100 reused; execution equivalence diagnostic, not a fresh quality holdout.',
        'model':parent['qwen_model'], 'model_revision':parent['qwen_revision'],
        'classifier':'Original float32 standardized linear896-to-2 head at block24, trained in chat600. No retraining or temperature needed for argmax.',
        'evaluation_ids':parent['splits']['evaluation'], 'data_sha256':parent['data_sha256'],
        'system_prompt':original['system_prompt'], 'message_limit':256,
        'batch_sizes':BATCHES, 'corpus_passes':PASSES, 'threads':4, 'interop_threads':1,
        'runtime':'torch2.6.0+cpu, transformers4.50.3, float32, eager attention, use_cache=False.',
        'order':'Each repetition executes a whole100-message corpus per batch size. Rotate batch-size order [1,4,8], [4,8,1], [8,1,4].',
        'queue':'The entire100-message queue is available before execution. For every batch size, freshly tokenize/bound/render all messages, then stable-sort by actual templated token length. Consecutive sorted groups form batches; final short batch is retained. No label-dependent sorting.',
        'padding':'Explicit left padding with tokenizer.pad_token_id; attention mask0 for pads and1 for text. position_ids=(attention_mask.cumsum(-1)-1).clamp(min=0). Every final column is a real token. Read final hidden state after all24 blocks and final norm.',
        'timing':'One corpus timer includes fresh truncation, chat templating, tokenization, stable length sort, batch padding/masks/positions, actual24-block forward, classifier, conversion to output records and hook removal. Excludes model/data loading, warmup, post-corpus parity validation, aggregation and file writes.',
        'warmup':'One excluded batch per requested batch size, from the first8 messages in manifest order sorted by length.',
        'reference':'Every output label compared with saved float_qwen refinement predictions. First measured batch1 corpus supplies same-runtime float logits; subsequent logits compared without retuning.',
        'logit_tolerance':{'atol':.001,'rtol':.0001},
        'failure_policy':'Record every method and pass, including slower results, numeric/parity failures and execution errors. Never silently reduce batch size or substitute outputs. Incomplete/error passes are not assigned complete-corpus throughput.',
        'traces':'Record contiguous blocks1..24 for every batch; each member consequently executes24layers, zero skipped. Fewer batch forward calls are not fewer layers per message.',
        'limits':['Reused previously evaluated balanced100 messages; no independent quality acceptance.',
                  'Queue already available; no arrival-to-response latency or batch-wait claim.',
                  'One CPU and one thread configuration. Memory and hardware affect batch behavior.',
                  'Identical predictions on100 cases do not guarantee universal exact equivalence.',
                  'No600-message timing extrapolation; only100 messages measured.'],
        'source_sha256':{path:sha(path) for path in sources},
    }


def source_rows(protocol):
    path = ROOT/'experiments/.cache/toxicchat/toxic-chat_annotation_test.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['data_sha256']['test']
    with path.open(encoding='utf-8',newline='') as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for rid in protocol['evaluation_ids']:
        source, index = rid.split(':')
        assert source=='test'
        row = raw[int(index)]
        assert row['human_annotation']=='True'
        rows.append({'id':rid,'text':row['user_input'],'label':int(row['toxicity'])})
    assert len(rows)==len({r['id'] for r in rows})==100
    return rows


def padded_inputs(items, pad_token_id, torch):
    lengths = [len(item['ids']) for item in items]
    width = max(lengths)
    ids = torch.full((len(items),width),pad_token_id,dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i,item in enumerate(items):
        ids[i,-lengths[i]:] = torch.tensor(item['ids'],dtype=torch.long)
        mask[i,-lengths[i]:] = 1
    positions = (mask.cumsum(-1)-1).clamp(min=0)
    return {'input_ids':ids,'attention_mask':mask,'position_ids':positions}


def run(protocol):
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT/'experiments/.cache/replay-runtime'))
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert torch.__version__=='2.6.0+cpu' and transformers.__version__=='4.50.3'
    rows = source_rows(protocol)
    expected = {r['id']:r for r in read('results/chat-refinement/predictions.json') if r['path']=='float_qwen'}
    assert set(expected)=={r['id'] for r in rows}
    model = AutoModelForCausalLM.from_pretrained(protocol['model'],revision=protocol['model_revision'],
        local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval().model
    tokenizer = AutoTokenizer.from_pretrained(protocol['model'],revision=protocol['model_revision'],local_files_only=True)
    assert len(model.layers)==24 and model.config.hidden_size==896
    assert tokenizer.pad_token_id is not None
    saved = np.load(ROOT/'results/chat600/heads.npz',allow_pickle=False)
    head = {k:torch.tensor(saved['24_'+k]) for k in ['weight','bias','mean','std']}

    def prepare(items):
        prepared = []
        for row in items:
            ids = tokenizer.encode(row['text'],add_special_tokens=False)
            text = tokenizer.decode(ids[:256],skip_special_tokens=False) if len(ids)>256 else row['text']
            rendered = tokenizer.apply_chat_template([{'role':'system','content':protocol['system_prompt']},
                {'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
            prepared.append({'id':row['id'],'label':row['label'],'ids':tokenizer(rendered)['input_ids'],
                             'original_message_tokens':len(ids),'truncated':len(ids)>256})
        return sorted(prepared,key=lambda item:len(item['ids']))

    def execute(items):
        start = time.perf_counter()
        inputs = padded_inputs(items,tokenizer.pad_token_id,torch)
        visited = []
        hooks = [layer.register_forward_hook(lambda m,i,o,d=d:visited.append(d)) for d,layer in enumerate(model.layers,1)]
        try:
            hidden = model(**inputs,use_cache=False).last_hidden_state[:,-1,:]
            logits = torch.nn.functional.linear((hidden-head['mean'])/head['std'],head['weight'],head['bias'])
            predictions = logits.argmax(1).tolist()
            values = logits.tolist()
            if not np.isfinite(values).all():
                raise FloatingPointError('Non-finite classifier logits: '+','.join(item['id'] for item in items))
        finally:
            for hook in hooks:hook.remove()
        records = [{'id':item['id'],'label':item['label'],'prediction':prediction,'logits':value,
                    'input_tokens':len(item['ids']),'original_message_tokens':item['original_message_tokens'],
                    'truncated':item['truncated'],'layers_executed':24,'layers_skipped':0}
                   for item,prediction,value in zip(items,predictions,values)]
        trace = {'ids':[item['id'] for item in items],'actual_batch_size':len(items),'executed_layers':visited,
                 'padded_width':inputs['input_ids'].shape[1],'real_tokens':sum(len(item['ids']) for item in items),
                 'padded_token_slots':inputs['input_ids'].numel(),'batch_seconds':time.perf_counter()-start}
        return records,trace

    started = datetime.now(timezone.utc).isoformat()
    runs, warmup = [], []
    with torch.inference_mode():
        for size in BATCHES:
            try:
                prepared = prepare(rows[:8])[:size]
                records,trace = execute(prepared)
                warmup.append({'batch_size':size,'records':records,'trace':trace})
            except Exception as error:
                warmup.append({'batch_size':size,'error':str(error),'error_type':type(error).__name__})
        write('warmup.json',warmup)
        for repeat in range(PASSES):
            order = BATCHES[repeat:]+BATCHES[:repeat]
            for position,size in enumerate(order):
                records,traces = [],[]
                record = {'pass':repeat+1,'requested_batch_size':size,'corpus_order_position':position+1,
                          'records':records,'batches':traces,'complete':False}
                before = time.perf_counter()
                try:
                    prepared = prepare(rows)
                    for offset in range(0,len(prepared),size):
                        batch_records,trace = execute(prepared[offset:offset+size])
                        records.extend(batch_records);traces.append(trace)
                    record['complete'] = True
                except Exception as error:
                    record.update(error=str(error),error_type=type(error).__name__)
                record['corpus_seconds'] = time.perf_counter()-before
                # Validation remains outside every corpus timer, including failed passes.
                record['label_mismatches'] = [r['id'] for r in records if r['prediction']!=expected[r['id']]['prediction']]
                record['traces_valid'] = all(b['executed_layers']==list(range(1,25)) for b in traces)
                record['coverage_valid'] = len(records)==100 and {r['id'] for r in records}==set(expected)
                record['labels_valid'] = all(r['label']==expected[r['id']]['label'] for r in records)
                runs.append(record)
                write('progress.json',{'completed_corpus_attempts':len(runs),'runs':runs})
                print(f'Batch{size} pass{repeat+1}: {record["corpus_seconds"]:.3f}s, complete={record["complete"]}, label differences={len(record["label_mismatches"])}',flush=True)

    reference_run = next((r for r in runs if r['requested_batch_size']==1 and r['pass']==1 and r['complete']),None)
    reference = {r['id']:r['logits'] for r in reference_run['records']} if reference_run else {}
    for attempt in runs:
        comparisons = []
        for r in attempt['records']:
            if r['id'] not in reference:continue
            target = reference[r['id']]
            delta = max(abs(a-b) for a,b in zip(r['logits'],target))
            close = bool(np.allclose(r['logits'],target,**protocol['logit_tolerance']))
            r.update(logit_max_abs_delta_vs_batch1=delta,logits_within_tolerance=close)
            comparisons.append((delta,close))
        attempt['max_abs_logit_delta'] = max((d for d,_ in comparisons),default=None)
        attempt['logits_within_tolerance'] = len(comparisons)==100 and all(ok for _,ok in comparisons)
        attempt['equivalent_execution'] = all([attempt['complete'],attempt['coverage_valid'],attempt['labels_valid'],
            attempt['traces_valid'],not attempt['label_mismatches'],attempt['logits_within_tolerance']])

    summary = {}
    for size in BATCHES:
        attempts = [r for r in runs if r['requested_batch_size']==size]
        complete = all(r['complete'] and r['coverage_valid'] for r in attempts)
        totals = [r['corpus_seconds'] for r in attempts]
        first = attempts[0]['records']
        summary[str(size)] = {'all_passes_complete':complete,'all_equivalent':all(r['equivalent_execution'] for r in attempts),
            'pass_seconds':totals,'mean_corpus_seconds':statistics.mean(totals) if complete else None,
            'messages_per_second':100/statistics.mean(totals) if complete else None,
            'correct_first_pass':sum(r['label']==r['prediction'] for r in first),'first_pass_n':len(first),
            'label_mismatches_per_pass':[len(r['label_mismatches']) for r in attempts],
            'max_abs_logit_delta':max((r['max_abs_logit_delta'] for r in attempts if r['max_abs_logit_delta'] is not None),default=None),
            'forward_calls_per_pass':[len(r['batches']) for r in attempts],
            'real_tokens_per_pass':[sum(b['real_tokens'] for b in r['batches']) for r in attempts],
            'padded_token_slots_per_pass':[sum(b['padded_token_slots'] for b in r['batches']) for r in attempts],
            'layers_per_message':24,'layers_skipped':0}
    base = summary['1']['mean_corpus_seconds']
    for size in BATCHES:
        item = summary[str(size)];duration = item['mean_corpus_seconds']
        item['relative_corpus_time_saving'] = 1-duration/base if duration is not None and base is not None else None
        item['throughput_ratio'] = base/duration if duration is not None and base is not None else None
    result = {'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha('results/chat-batch-timing/protocol.json'),'source_sha256':protocol['source_sha256'],
        'runtime':{'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
            'numpy':np.__version__,'platform':platform.platform(),'processor':platform.processor(),
            'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()},
        'batch_sizes':summary,'scope':protocol['scope'],'limits':protocol['limits']}
    write('records.json',runs);write('result.json',result)
    (OUT/'progress.json').unlink()
    print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol = plan()
    if args.prepare:
        if (OUT/'protocol.json').exists():assert read('results/chat-batch-timing/protocol.json')==protocol,'Sealed protocol differs'
        else:write('protocol.json',protocol)
        print('Protocol sealed. No model computation; wait for isolated timing window.')
        return
    assert (OUT/'protocol.json').exists(),'Prepare and review protocol first'
    assert read('results/chat-batch-timing/protocol.json')==protocol,'Sealed inputs changed'
    assert not any((OUT/name).exists() for name in ['result.json','progress.json','warmup.json']),'Preserve prior attempted/completed run'
    run(protocol)


if __name__=='__main__':main()

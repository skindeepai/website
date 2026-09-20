"""Locked fresh-message diagnostic: more specialist supervision and int8 Qwen.

Run prepare, fit, evaluate, then benchmark in order. Four CPU threads; no other
model jobs during benchmark. Never overwrite a completed stage or retune on test.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, copy, csv, hashlib, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/.cache/replay-runtime'))
import numpy as np
import torch
from scipy.stats import beta
from transformers import AutoModelForCausalLM, AutoTokenizer
from chat_smoke_specialist import Specialist, CACHE, metrics

OUT = ROOT/'results/chat-refinement'
DEP = ['experiments/chat_refinement.py', 'experiments/chat_smoke_specialist.py',
       'results/chat600/protocol.json', 'results/chat600/heads.npz',
       'results/chat-smoke-specialist/specialist.npz', 'results/chat-smoke-specialist/result.json']


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def guard(name):
    assert not (OUT/name).exists(), 'Preserve completed stage: '+name


def rows_and_tokenizers():
    old = read('results/chat600/protocol.json')
    rows = {}
    for source in ['train', 'test']:
        path = ROOT/f'experiments/.cache/toxicchat/toxic-chat_annotation_{source}.csv'
        assert sha(path) == old['data_sha256'][source]
        for i, row in enumerate(csv.DictReader(path.open(encoding='utf-8', newline=''))):
            if row['human_annotation'] == 'True':
                rid = f'{source}:{i}'
                rows[rid] = dict(id=rid, text=row['user_input'], label=int(row['toxicity']), conversation=row['conv_id'])
    qt = AutoTokenizer.from_pretrained(old['model'], revision=old['model_revision'], local_files_only=True)
    bt = AutoTokenizer.from_pretrained(CACHE, local_files_only=True)
    return old, rows, qt, bt


def bounded(text, qt):
    ids = qt.encode(text, add_special_tokens=False)
    return qt.decode(ids[:256], skip_special_tokens=False) if len(ids)>256 else text


def rendered(text, qt, old):
    return qt.apply_chat_template([{'role':'system','content':old['system_prompt']},
                                  {'role':'user','content':text}], tokenize=False, add_generation_prompt=True)


def prepare():
    guard('protocol.json')
    old, rows, qt, bt = rows_and_tokenizers()
    def keys(row):
        text = bounded(row['text'], qt)
        q = qt.encode(rendered(text, qt, old))
        b = bt(text, truncation=True, max_length=512)['input_ids']
        return (row['text'].strip().casefold(), hashlib.sha256(json.dumps(q).encode()).hexdigest(),
                hashlib.sha256(json.dumps(b).encode()).hexdigest(), row['conversation'])
    seen = [set() for _ in range(4)]
    splits, removed = {}, {}
    for split in ['train', 'tune', 'calibration']:
        splits[split], removed[split] = [], []
        for rid in old['splits'][split]:
            k = keys(rows[rid])
            if any(value in used for value, used in zip(k, seen)):
                removed[split].append(rid)
            else:
                splits[split].append(rid)
            for value, used in zip(k, seen): used.add(value)
    # Exclude every historical ID/effective input, including discarded duplicates.
    old_ids = {rid for ids in old['splits'].values() for rid in ids}
    for rid in old_ids:
        for value, used in zip(keys(rows[rid]), seen): used.add(value)
    # Match original global train-first normalized text dedup, including unused train rows.
    seen[0].update(row['text'].strip().casefold() for rid, row in rows.items() if rid.startswith('train:'))
    eligible = []
    for rid, row in rows.items():
        if not rid.startswith('test:') or rid in old_ids: continue
        k = keys(row)
        if any(value in used for value, used in zip(k, seen)): continue
        eligible.append(rid)
        for value, used in zip(k, seen): used.add(value)
    rng = random.Random(920317)
    selected = []
    for label in [0, 1]:
        pool = [rid for rid in eligible if rows[rid]['label']==label]
        rng.shuffle(pool)
        assert len(pool)>=50
        selected.extend(pool[:50])
    rng.shuffle(selected)
    splits['evaluation'] = selected
    truncation = {}
    for split, ids in splits.items():
        truncation[split] = dict(qwen_256=sum(len(qt.encode(rows[rid]['text'],add_special_tokens=False))>256 for rid in ids),
            bert_512_after_qwen_bound=sum(len(bt(bounded(rows[rid]['text'],qt),truncation=False)['input_ids'])>512 for rid in ids))
    fingerprint = hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
    feature_cache = f'experiments/.cache/toxicchat/{fingerprint}-calibration.pt'
    protocol = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        scope='Fresh to these local experiments, not unknown pretraining data; balanced 100-message diagnostic, not deployment acceptance.',
        source_sha256={p:sha(ROOT/p) for p in DEP}, data_sha256=old['data_sha256'],
        bert_files={p:sha(CACHE/p) for p in ['config.json','model.safetensors','vocab.txt']},
        qwen_model=old['model'], qwen_revision=old['model_revision'],
        splits=splits, sizes={s:len(ids) for s,ids in splits.items()}, removed_effective_duplicates=removed,
        truncation=truncation, calibration_features_sha256={feature_cache:sha(ROOT/feature_cache)},
        label_counts={s:{str(y):sum(rows[rid]['label']==y for rid in ids) for y in [0,1]} for s,ids in splits.items()},
        unused_pool=len(eligible), seed=920317, epochs=4, batch_size=16, learning_rate=.0001,
        weight_decay=.01, class_weights='inverse frequency normalized mean one',
        fit='All two-layer BERT parameters plus masked mean 128-to-2 head. Train historical training partition only.',
        selection='Epoch and standalone threshold maximize tune balanced accuracy, then fewer toxic misses, then more correct. First tie retained.',
        thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9],
        safe_thresholds=[-1,0,.001,.005,.01,.02,.05,.1,.15,.2,.3,.4],
        block_thresholds=[.6,.7,.8,.85,.9,.95,.98,.99,.995,.999,1,2],
        gate='Use calibration only. Maximize coverage subject to ZERO added errors relative to existing full Qwen; ties higher correct then fewer misses. All fallback is allowed. Freeze before fresh evaluation.',
        quantization='Dynamic int8 torch.nn.Linear modules in Qwen backbone only, x86 engine; embeddings/norms/frozen classifier float32. No layers skipped; no training.',
        evaluation='Report float Qwen, int8 Qwen, original BERT and cascade, newly trained BERT and cascade. Never select thresholds/weights using this evaluation.',
        quality='Report observed new errors/new toxic misses and exact one-sided 95% binomial upper bounds. n100 cannot establish a 1% added-error margin, even with zero errors.',
        timing=dict(paths=['float_qwen','int8_qwen','new_cascade'], repetitions=3, threads=4,
                    order='Rotate path order by message index plus repetition.',
                    boundary='Original text preparation, model execution, head, traces and all fallback; exclude loading, training, warmup.',
                    isolation='No other launched model jobs during recorded timing.'),
        limits=['Toxicity labels are not universal moderation rules.','Balanced sample does not estimate live prevalence.',
                'Only archived user message and first 256 Qwen tokens; no conversation history.',
                'Historical training/development data have been repeatedly inspected. Public data may occur in pretraining.',
                'No equal-quality or production-safety claim from 100 examples.'])
    write('protocol.json', protocol)
    print(json.dumps({k:protocol[k] for k in ['sizes','label_counts','unused_pool','removed_effective_duplicates']}), flush=True)


def load():
    protocol = read('results/chat-refinement/protocol.json')
    for p, h in protocol['source_sha256'].items(): assert sha(ROOT/p)==h,p
    for p, h in protocol['bert_files'].items(): assert sha(CACHE/p)==h,p
    for p, h in protocol['calibration_features_sha256'].items(): assert sha(ROOT/p)==h,p
    old, rows, qt, bt = rows_and_tokenizers()
    splits = {s:[rows[rid] for rid in ids] for s,ids in protocol['splits'].items()}
    return protocol, old, splits, qt, bt


def load_weights(model, path):
    with np.load(path, allow_pickle=False) as saved:
        model.load_state_dict({k:torch.tensor(saved[k]) for k in saved.files})
    return model.eval()


def train():
    guard('fit.json')
    p, old, splits, qt, bt = load()
    # No evaluation labels/predictions are consulted in this stage.
    random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed'])
    model = Specialist()
    for split in ['train','tune','calibration']:
        for row in splits[split]: row['bounded'] = bounded(row['text'], qt)
    def batch(rows): return bt([r['bounded'] for r in rows],padding=True,truncation=True,max_length=512,return_tensors='pt')
    def predict(rows):
        model.eval(); pieces=[]
        with torch.inference_mode():
            for start in range(0,len(rows),16): pieces.append(model(batch(rows[start:start+16])).softmax(1)[:,1].numpy())
        return np.concatenate(pieces)
    y = {s:np.array([r['label'] for r in splits[s]]) for s in ['train','tune','calibration']}
    counts=np.bincount(y['train'],minlength=2)
    weight=torch.tensor(1/counts,dtype=torch.float32);weight/=weight.mean()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    history=[];best=None
    for epoch in range(1,p['epochs']+1):
        model.train(); order=torch.randperm(len(y['train'])).tolist();losses=[]
        for start in range(0,len(order),16):
            ix=order[start:start+16];optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(batch([splits['train'][i] for i in ix])),torch.tensor(y['train'][ix]),weight=weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        prob=predict(splits['tune'])
        choices=[dict(threshold=t,metrics=metrics(y['tune'],prob>=t)) for t in p['thresholds']]
        rank=lambda c:(c['metrics']['balanced_accuracy'],-c['metrics']['missed_toxic'],c['metrics']['correct'])
        chosen=max(choices,key=rank)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),candidates=choices))
        if best is None or rank(chosen)>best[0]:
            best=(rank(chosen),epoch,chosen)
            np.savez_compressed(OUT/'specialist.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        print(json.dumps(dict(epoch=epoch,selected=chosen)),flush=True)
    load_weights(model,OUT/'specialist.npz')
    prob=predict(splits['calibration'])
    from chat_smoke_specialist import full_reference
    ref=full_reference(old,{'calibration':splits['calibration']})['calibration']
    candidates=[]
    for safe in p['safe_thresholds']:
        for block in p['block_thresholds']:
            accept=(prob<=safe)|(prob>=block)
            pred=np.where(prob<=safe,0,np.where(prob>=block,1,ref))
            m=metrics(y['calibration'],pred,ref)
            candidates.append(dict(safe=safe,block=block,coverage=float(accept.mean()),metrics=m))
    permitted=[c for c in candidates if c['metrics']['added_errors']==0]
    gate=max(permitted,key=lambda c:(c['coverage'],c['metrics']['correct'],-c['metrics']['missed_toxic']))
    write('development-predictions.json',[dict(id=r['id'],label=r['label'],probability=float(prob[i]),reference=int(ref[i])) for i,r in enumerate(splits['calibration'])])
    write('fit.json',dict(history=history,selected_epoch=best[1],standalone_threshold=best[2]['threshold'],gate=gate,gate_candidates=candidates,
        parameter_count=sum(v.numel() for v in model.parameters()),weights_sha256=sha(OUT/'specialist.npz'),protocol_sha256=sha(OUT/'protocol.json'),
        frozen_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(selected_epoch=best[1],gate=gate)),flush=True)


class Runner:
    def __init__(self,p,old,qt,bt):
        self.p,self.old,self.qt,self.bt=p,old,qt,bt
        self.fit=read('results/chat-refinement/fit.json')
        assert sha(OUT/'protocol.json')==self.fit['protocol_sha256']
        assert sha(OUT/'specialist.npz')==self.fit['weights_sha256']
        self.bert=load_weights(Specialist(),OUT/'specialist.npz')
        self.oldbert=load_weights(Specialist(),ROOT/'results/chat-smoke-specialist/specialist.npz')
        self.qwen=AutoModelForCausalLM.from_pretrained(old['model'],revision=old['model_revision'],local_files_only=True,
                    torch_dtype=torch.float32,attn_implementation='eager').eval().model
        torch.backends.quantized.engine='x86'
        self.int8=torch.ao.quantization.quantize_dynamic(copy.deepcopy(self.qwen),{torch.nn.Linear},dtype=torch.qint8).eval()
        self.module_counts=dict(float_linear=sum(isinstance(m,torch.nn.Linear) for m in self.qwen.modules()),
            int8_linear=sum(isinstance(m,torch.ao.nn.quantized.dynamic.Linear) for m in self.int8.modules()),
            float_layers=len(self.qwen.layers),int8_layers=len(self.int8.layers))
        assert self.module_counts==dict(float_linear=168,int8_linear=168,float_layers=24,int8_layers=24)
        saved=np.load(ROOT/'results/chat600/heads.npz',allow_pickle=False)
        self.head={k:torch.tensor(saved[f'24_{k}']) for k in ['weight','bias','mean','std']}
        self.oldfit=read('results/chat-smoke-specialist/result.json')

    def execute(self,row,path):
        start=time.perf_counter(); text=bounded(row['text'],self.qt)
        visited=[];bert_visited=[];prob=None;fallback=path in ['float_qwen','int8_qwen']
        if not fallback:
            model=self.oldbert if path.startswith('old_') else self.bert
            hooks=[layer.register_forward_hook(lambda m,i,o,d=d:bert_visited.append(d)) for d,layer in enumerate(model.encoder.encoder.layer,1)]
            try:prob=float(model(self.bt(text,return_tensors='pt',truncation=True,max_length=512)).softmax(1)[0,1])
            finally:
                for hook in hooks:hook.remove()
            assert bert_visited==[1,2]
            if path.endswith('bert'):
                threshold=self.oldfit['decision_threshold'] if path.startswith('old_') else self.fit['standalone_threshold']
                label=int(prob>=threshold)
            else:
                gate=self.oldfit['cascade'] if path.startswith('old_') else self.fit['gate']
                fallback=not(prob<=gate['safe'] or prob>=gate['block'])
                label=int(prob>=gate['block'])
        if fallback:
            model=self.int8 if path=='int8_qwen' else self.qwen
            hooks=[layer.register_forward_hook(lambda m,i,o,d=d:visited.append(d)) for d,layer in enumerate(model.layers,1)]
            try:h=model(**self.qt(rendered(text,self.qt,self.old),return_tensors='pt'),use_cache=False).last_hidden_state[:,-1,:]
            finally:
                for hook in hooks:hook.remove()
            x=self.head;label=int(torch.nn.functional.linear((h-x['mean'])/x['std'],x['weight'],x['bias']).argmax(1))
            assert visited==list(range(1,25))
        return dict(id=row['id'],label=row['label'],path=path,prediction=label,probability=prob,fallback=fallback,
                    bert_layers=bert_visited,qwen_layers=visited,seconds=time.perf_counter()-start)


def upper(k,n): return 1. if n==0 or k==n else float(beta.ppf(.95,k+1,n-k))


def evaluate(benchmark=False):
    guard('benchmark.json' if benchmark else 'result.json')
    p,old,splits,qt,bt=load()
    runner=Runner(p,old,qt,bt)
    records=[];rows=splits['evaluation']
    paths=p['timing']['paths'] if benchmark else ['float_qwen','int8_qwen','old_bert','old_cascade','new_bert','new_cascade']
    expected={(r['id'],r['path']):r for r in read('results/chat-refinement/predictions.json')} if benchmark else {}
    if benchmark:
        assert read('results/chat-refinement/result.json')['fit_sha256']==sha(OUT/'fit.json')
    with torch.inference_mode():
        for path in paths:runner.execute(rows[0],path)
        for repeat in range(p['timing']['repetitions'] if benchmark else 1):
            for i,row in enumerate(rows):
                offset=(i+repeat)%len(paths)
                for path in paths[offset:]+paths[:offset]:
                    r=runner.execute(row,path);r['repetition']=repeat
                    if benchmark:
                        ref=expected[(r['id'],path)]
                        assert all(r[k]==ref[k] for k in ['prediction','fallback','bert_layers','qwen_layers'])
                    records.append(r)
                if (i+1)%20==0:print(f'{"Timing" if benchmark else "Evaluation"} {repeat+1}: {i+1}/100',flush=True)
    if benchmark:
        totals={path:[sum(r['seconds'] for r in records if r['path']==path and r['repetition']==rep) for rep in range(p['timing']['repetitions'])] for path in paths}
        write('timings.json',records)
        write('benchmark.json',dict(totals_seconds=totals,n=len(rows),calls=len(records),protocol_sha256=sha(OUT/'protocol.json'),
            evaluation_sha256={name:sha(OUT/name) for name in ['result.json','predictions.json','fit.json']},
            parity='Every actual prediction, routing and contiguous layer trace matches evaluation.',
            versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,quantized_engine=torch.backends.quantized.engine),
            scope='Three warm paired repetitions on one CPU. Loading excluded; no layer skipping in int8; quality checked separately.'))
        print(json.dumps(totals),flush=True)
    else:
        lookup={(r['id'],r['path']):r for r in records}
        y=[r['label'] for r in rows];ref=[lookup[(r['id'],'float_qwen')]['prediction'] for r in rows]
        summary={}
        for path in paths:
            subset=[lookup[(r['id'],path)] for r in rows]
            m=metrics(y,[r['prediction'] for r in subset],ref)
            m.update(added_error_upper95=upper(m['added_errors'],len(rows)),additional_toxic_miss_upper95=upper(m['additional_missed_toxic'],sum(y)),
                     qwen_calls=sum(r['fallback'] for r in subset),qwen_blocks=sum(len(r['qwen_layers']) for r in subset),
                     bert_blocks=sum(len(r['bert_layers']) for r in subset))
            summary[path]=m
        routes={}
        for fallback in [False,True]:
            subset=[r for r in rows if lookup[(r['id'],'new_cascade')]['fallback']==fallback]
            routes['qwen' if fallback else 'bert']=dict(n=len(subset),
                cascade_correct=sum(lookup[(r['id'],'new_cascade')]['prediction']==r['label'] for r in subset),
                qwen_correct_on_same_messages=sum(lookup[(r['id'],'float_qwen')]['prediction']==r['label'] for r in subset))
        write('predictions.json',records)
        write('result.json',dict(metrics=summary,routes=routes,fit_sha256=sha(OUT/'fit.json'),
            module_counts=runner.module_counts,
            timing='Evaluation durations are diagnostics only; separate paired repeated timing follows.',
            limits=p['limits']))
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit','evaluate','benchmark']);args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert __import__('transformers').__version__=='4.50.3'
    if args.stage=='prepare':prepare()
    elif args.stage=='fit':train()
    else:evaluate(args.stage=='benchmark')

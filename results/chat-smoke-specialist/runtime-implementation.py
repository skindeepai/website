"""Exploratory tiny BERT specialist and Qwen fallback. Never a deployment gate."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, csv, hashlib, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/.cache/replay-runtime'))
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

OUT = ROOT/'results/chat-smoke-specialist'
CACHE = ROOT/'experiments/.cache/bert-tiny'
COMMON = ROOT/'results/chat-smoke/protocol.json'
MODEL = 'google/bert_uncased_L-2_H-128_A-2'
REV = '30b0a37ccaaa32f332884b96992754e246e48c5f'

def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(y, p, reference=None):
    y, p = np.asarray(y), np.asarray(p)
    tp, fn = int(((y == 1)&(p == 1)).sum()), int(((y == 1)&(p == 0)).sum())
    tn, fp = int(((y == 0)&(p == 0)).sum()), int(((y == 0)&(p == 1)).sum())
    result = dict(n=len(y), correct=tp+tn, true_block=tp, missed_toxic=fn,
                  true_safe=tn, false_block=fp, accuracy=(tp+tn)/len(y),
                  balanced_accuracy=(tp/max(1,tp+fn)+tn/max(1,tn+fp))/2)
    if reference is not None:
        r = np.asarray(reference)
        result.update(added_errors=int(((r == y)&(p != y)).sum()),
                      corrected_errors=int(((r != y)&(p == y)).sum()),
                      additional_missed_toxic=int(((y == 1)&(r == 1)&(p == 0)).sum()))
    return result

class Specialist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(CACHE, local_files_only=True, add_pooling_layer=False,
                                                attn_implementation='eager')
        self.head = torch.nn.Linear(128, 2)

    def forward(self, inputs):
        h = self.encoder(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (h*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.head(pooled)

def data():
    common = json.loads(COMMON.read_text())
    old = json.loads((ROOT/'results/chat600/protocol.json').read_text())
    rows = {}
    for source in ['train', 'test']:
        path = ROOT/f'experiments/.cache/toxicchat/toxic-chat_annotation_{source}.csv'
        assert sha(path) == old['data_sha256'][source]
        for index, row in enumerate(csv.DictReader(path.open(encoding='utf-8', newline=''))):
            rows[f'{source}:{index}'] = dict(id=f'{source}:{index}', text=row['user_input'], label=int(row['toxicity']))
    split_ids = common['splits']
    splits = {s: [rows[i] for i in ids] for s, ids in split_ids.items()}
    qt = AutoTokenizer.from_pretrained(old['model'], revision=old['model_revision'], local_files_only=True)
    for values in splits.values():
        for row in values:
            ids = qt.encode(row['text'], add_special_tokens=False)
            row['bounded'] = qt.decode(ids[:256], skip_special_tokens=False) if len(ids)>256 else row['text']
    return common, old, splits, qt

def full_reference(old, splits):
    fingerprint = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    weights = np.load(ROOT/'results/chat600/heads.npz', allow_pickle=False)
    result = {}
    for split, rows in splits.items():
        original = 'test' if rows[0]['id'].startswith('test:') else split
        x = torch.load(ROOT/f'experiments/.cache/toxicchat/{fingerprint}-{original}.pt', weights_only=True)[24].numpy()
        indexes = {rid: i for i, rid in enumerate(old['splits'][original])}
        x = x[[indexes[row['id']] for row in rows]]
        result[split] = (((x-weights['24_mean'])/weights['24_std'])@weights['24_weight'].T+weights['24_bias']).argmax(1)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    if args.fit and (OUT/'result.json').exists():
        raise RuntimeError('Preserve the completed smoke study; use a new output directory for further training.')
    if args.benchmark and (OUT/'benchmark.json').exists():
        raise RuntimeError('Preserve the completed timing pass; use a new output directory for repeats.')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    random.seed(771); np.random.seed(771); torch.manual_seed(771)
    common, old, splits, qt = data()
    bt = AutoTokenizer.from_pretrained(CACHE, local_files_only=True)
    model = Specialist()
    def batch(rows):
        return bt([r['bounded'] for r in rows], padding=True, truncation=True,
                  max_length=512, return_tensors='pt')
    def predict(rows):
        model.eval(); chunks=[]
        with torch.inference_mode():
            for start in range(0, len(rows), 16):
                chunks.append(model(batch(rows[start:start+16])).numpy())
        return np.concatenate(chunks)
    labels = {s: np.array([r['label'] for r in rows]) for s, rows in splits.items()}
    ref = full_reference(old, splits)
    protocol = dict(type='Exploratory, reused evaluation; not prospective validation',
        common_protocol_sha256=sha(COMMON), model=MODEL, revision=REV, license='Apache-2.0',
        model_files={f:sha(CACHE/f) for f in ['config.json','vocab.txt','model.safetensors']},
        seed=771, epochs=4, batch_size=16, learning_rate=.0001, weight_decay=.01,
        architecture='2 transformer encoder layers, masked mean pooling, linear 128-to-2 head; all parameters trainable',
        input='Same first 256 Qwen user tokens; no instruction template needed by specialist; additional BERT limit 512 WordPieces including special tokens',
        selection='Choose epoch and BLOCK probability threshold with highest tuning balanced accuracy; ties fewer missed toxic then more correct. Tune cascade asymmetric probability thresholds for maximum specialist coverage with zero additional missed toxic and <=1% added errors vs frozen Qwen full reference.',
        calibration='Shared calibration split is development data, not fresh calibration. Here report fixed selected policies without reselection. No claim of strict risk guarantee.',
        timing='Separate --benchmark when other model workloads finish. Actual batch-one tokenization + model + fallback if invoked; 3 passes of evaluation, rotating path order. Includes Qwen bounding tokenization, excludes load/training.',
        decision_thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9],
        safe_thresholds=[0,.005,.01,.02,.05,.1,.15,.2,.3,.4],
        block_thresholds=[.6,.7,.8,.85,.9,.95,.98,.99,.995,1.01],
        runtime=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,threads=4))
    if args.fit:
        if (OUT/'protocol.json').exists():
            assert json.loads((OUT/'protocol.json').read_text()) == protocol
        else: write('protocol.json', protocol)
        counts = np.bincount(labels['train'], minlength=2)
        weight = torch.tensor(1/counts, dtype=torch.float32); weight /= weight.mean()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
        history=[]; best=None
        for epoch in range(1,5):
            model.train(); indexes=torch.randperm(len(splits['train'])).tolist(); losses=[]
            for start in range(0,len(indexes),16):
                ix=indexes[start:start+16]; rows=[splits['train'][i] for i in ix]
                optimizer.zero_grad()
                loss=torch.nn.functional.cross_entropy(model(batch(rows)),torch.tensor(labels['train'][ix]),weight=weight)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
            logits=predict(splits['tune']); prob=torch.softmax(torch.tensor(logits),1).numpy()[:,1]
            choices=[]
            for threshold in protocol['decision_thresholds']:
                m=metrics(labels['tune'],prob>=threshold)
                choices.append(dict(threshold=threshold,metrics=m))
            selected=max(choices,key=lambda c:(c['metrics']['balanced_accuracy'],-c['metrics']['missed_toxic'],c['metrics']['correct']))
            rank=(selected['metrics']['balanced_accuracy'],-selected['metrics']['missed_toxic'],selected['metrics']['correct'])
            history.append(dict(epoch=epoch,mean_batch_loss=float(np.mean(losses)),selected=selected))
            if best is None or rank>best[0]:
                best=(rank,epoch,selected); torch.save(model.state_dict(),CACHE/'specialist.pt')
            print(json.dumps(history[-1]),flush=True)
        model.load_state_dict(torch.load(CACHE/'specialist.pt',weights_only=True))
        np.savez_compressed(OUT/'specialist.npz', **{k:v.numpy() for k,v in model.state_dict().items()})
        probabilities={s:torch.softmax(torch.tensor(predict(rows)),1).numpy()[:,1] for s,rows in splits.items()}
        candidates=[]
        p=probabilities['tune']
        for safe in protocol['safe_thresholds']:
            for block in protocol['block_thresholds']:
                accept=(p<=safe)|(p>=block); pred=np.where(p<=safe,0,np.where(p>=block,1,ref['tune']))
                m=metrics(labels['tune'],pred,ref['tune'])
                if m['additional_missed_toxic']==0 and m['added_errors']<=.01*len(p):
                    candidates.append(dict(safe=safe,block=block,coverage=float(accept.mean()),metrics=m))
        chosen=max(candidates,key=lambda c:(c['coverage'],-c['metrics']['added_errors'],c['metrics']['correct']))
        result=dict(parameter_count=sum(p.numel() for p in model.parameters()),history=history,
                    selected_epoch=best[1],decision_threshold=best[2]['threshold'],cascade=chosen,
                    checkpoint_sha256=sha(CACHE/'specialist.pt'),portable_weights_sha256=sha(OUT/'specialist.npz'),splits={},truncation={},input_overlap={},within_split_duplicate_inputs={})
        records=[]; effective={}
        for split,rows in splits.items():
            p=probabilities[split]; accept=(p<=chosen['safe'])|(p>=chosen['block'])
            pred=np.where(p<=chosen['safe'],0,np.where(p>=chosen['block'],1,ref[split]))
            specialist=(p>=result['decision_threshold']).astype(int)
            result['splits'][split]=dict(specialist=metrics(labels[split],specialist,ref[split]),
                cascade=metrics(labels[split],pred,ref[split]),full_qwen_reference=metrics(labels[split],ref[split]),
                specialist_coverage=float(accept.mean()),fallback_count=int((~accept).sum()))
            tokens=bt([r['bounded'] for r in rows],truncation=False)['input_ids']
            result['truncation'][split]=dict(secondary_bert_truncation=sum(len(t)>512 for t in tokens),n=len(rows))
            effective[split]={hashlib.sha256(json.dumps(bt(r['bounded'],truncation=True,max_length=512)['input_ids']).encode()).hexdigest() for r in rows}
            result['within_split_duplicate_inputs'][split]=len(rows)-len(effective[split])
            for i,row in enumerate(rows):
                records.append(dict(id=row['id'],split=split,label=row['label'],block_probability=float(p[i]),specialist=int(specialist[i]),cascade=int(pred[i]),fallback=bool(not accept[i]),full_qwen_reference=int(ref[split][i])))
        for i,a in enumerate(splits):
            for b in list(splits)[i+1:]:result['input_overlap'][f'{a}/{b}']=len(effective[a]&effective[b])
        write('predictions.json',records);write('result.json',result)
        print(json.dumps(result),flush=True)
    if args.benchmark:
        result=json.loads((OUT/'result.json').read_text())
        portable=np.load(OUT/'specialist.npz',allow_pickle=False)
        model.load_state_dict({k:torch.tensor(portable[k]) for k in portable.files});model.eval()
        if 'portable_weights_sha256' in result:assert sha(OUT/'specialist.npz')==result['portable_weights_sha256']
        eval_name=next(s for s,rows in splits.items() if rows[0]['id'].startswith('test:'))
        expected={r['id']:r for r in json.loads((OUT/'predictions.json').read_text()) if r['split']==eval_name}
        qm=AutoModelForCausalLM.from_pretrained(old['model'],revision=old['model_revision'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
        saved=np.load(ROOT/'results/chat600/heads.npz');weights={k:torch.tensor(saved[f'24_{k}']) for k in ['weight','bias','mean','std']}
        def execute(row,path):
            start=time.perf_counter(); ids=qt.encode(row['text'],add_special_tokens=False)
            bounded=qt.decode(ids[:256],skip_special_tokens=False) if len(ids)>256 else row['text']
            fallback=path=='full_qwen_reference'; p=None; bert_visited=[]
            if not fallback:
                hooks=[layer.register_forward_hook(lambda mod,inp,out,d=d:bert_visited.append(d)) for d,layer in enumerate(model.encoder.encoder.layer,1)]
                try:prob=torch.softmax(model(bt(bounded,return_tensors='pt',truncation=True,max_length=512)),1)[0,1].item();p=prob
                finally:
                    for hook in hooks:hook.remove()
                assert bert_visited==[1,2]
                if path=='specialist': label=int(prob>=result['decision_threshold'])
                else:
                    c=result['cascade']; fallback=not(prob<=c['safe'] or prob>=c['block']); label=int(prob>=c['block'])
            visited=[]
            if fallback:
                rendered=qt.apply_chat_template([{'role':'system','content':old['system_prompt']},{'role':'user','content':bounded}],tokenize=False,add_generation_prompt=True)
                hooks=[layer.register_forward_hook(lambda mod,inp,out,d=d:visited.append(d)) for d,layer in enumerate(qm.model.layers,1)]
                try:h=qm.model(**qt(rendered,return_tensors='pt'),use_cache=False).last_hidden_state[:,-1,:]
                finally:
                    for hook in hooks:hook.remove()
                label=int(torch.nn.functional.linear((h-weights['mean'])/weights['std'],weights['weight'],weights['bias']).argmax(1))
                assert visited==list(range(1,25))
            return dict(prediction=label,fallback=fallback,bert_layers=bert_visited,qwen_layers=visited,seconds=time.perf_counter()-start,block_probability=p)
        records=[];totals=[];paths=['specialist','cascade','full_qwen_reference']
        amendment=dict(reason='User authorized 50/100-sample smoke tests; reduce timing burden before any timing outcomes.',
            replaces='Original three full-corpus timing passes',repetitions=1,n=len(splits[eval_name]),
            order='Interleave three paths per message, rotating first path by message index; 100 messages gives positions 34/33/33.',
            warmup='One unrecorded call per path on first evaluation item; no cold start or loading estimate.',
            measurement='Sum measured request durations, including tokenizer, both actual models when fallback occurs, classifier readout, trace instrumentation. No repeated-run uncertainty estimate.')
        if (OUT/'timing-protocol.json').exists():assert json.loads((OUT/'timing-protocol.json').read_text())==amendment
        else:write('timing-protocol.json',amendment)
        with torch.inference_mode():
            for path in paths:execute(splits[eval_name][0],path)
            for i,row in enumerate(splits[eval_name]):
                start=i%3;order=paths[start:]+paths[:start]
                for path in order:
                    item=execute(row,path);assert item['prediction']==expected[row['id']][path]
                    if path=='cascade':assert item['fallback']==expected[row['id']]['fallback']
                    records.append(dict(id=row['id'],path=path,repetition=0,**item))
                if i%10==9:print(f'Timed {i+1}/{len(splits[eval_name])}',flush=True)
            totals=[dict(path=path,repetition=0,seconds=sum(r['seconds'] for r in records if r['path']==path)) for path in paths]
        write('timings.json',records)
        write('benchmark.json',dict(n=len(splits[eval_name]),repetitions=1,actual_calls=len(records),totals=totals,
            total_seconds={r['path']:r['seconds'] for r in totals},
            note='Exploratory same-machine timing. Totals sum measured per-request durations, not one whole-corpus wall timer. Single interleaved pass; no repeated-run uncertainty estimate. Loading/training excluded; actual fallback work included.'))

if __name__=='__main__':main()

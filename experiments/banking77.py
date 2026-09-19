"""Frozen Qwen classifiers and actual early exits on the public BANKING77 test set.

No remote model code; downloads only pinned, CC-BY-4.0 dataset files.
Feature caches and learned weights stay out of git. See docs/banking77-protocol.md.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, csv, hashlib, io, json, platform, random, time, urllib.request
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_decisions import MODEL, REVISION, StopAtHead

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/banking77'
CACHE = ROOT / 'experiments/.cache/banking77'
DATA_REV = '57ec275d8078af65b7731c2a98be812d844a6d6b'
DEPTHS = [6, 12, 18, 24]
PROMPT = 'Read this customer support request and identify its banking intent.\nRequest: {text}\nIntent:'

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def prepare_data():
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ['train.csv', 'test.csv', 'categories.json']:
        path = CACHE / name
        url = f'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{DATA_REV}/banking_data/{name}'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response: path.write_bytes(response.read())
        files[name] = {'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    categories = json.loads((CACHE/'categories.json').read_text())
    rows = {}
    for source in ['train', 'test']:
        rows[source] = [{'id': f'{source}:{i}', 'text': row['text'], 'label': categories.index(row['category'])}
                        for i, row in enumerate(csv.DictReader(io.StringIO((CACHE/f'{source}.csv').read_text(encoding='utf-8'))))]
    splits = {name: [] for name in ['train', 'tune', 'calibration', 'test']}
    for label in range(77):
        group = [r for r in rows['train'] if r['label'] == label]
        random.Random(20260919 + label).shuffle(group)
        for name, start, stop in [('train', 0, 32), ('tune', 32, 40), ('calibration', 40, 48)]: splits[name].extend(group[start:stop])
    splits['test'] = rows['test']
    for split in splits.values(): random.Random(20260919).shuffle(split)
    # Detect exact duplicated text across partitions; report, never silently drop test items.
    text_sets = {name: {r['text'].strip().casefold() for r in split} for name, split in splits.items()}
    overlaps = {a+'/'+b: len(text_sets[a]&text_sets[b]) for i,a in enumerate(splits) for b in list(splits)[i+1:]}
    manifest = {'dataset': 'BANKING77', 'authors': 'Casanueva et al., PolyAI, 2020', 'license': 'CC-BY-4.0',
                'source_revision': DATA_REV, 'files': files, 'categories': categories,
                'splits': {k: [r['id'] for r in v] for k,v in splits.items()}, 'exact_text_overlap': overlaps}
    write(OUT/'data-manifest.json', manifest)
    return splits, categories, manifest

def policy(probs, threshold, agreement=False, minimum=6):
    n = len(probs[24]); pred = probs[24].argmax(1).clone(); depth = torch.full((n,),24)
    previous = None
    for d in DEPTHS[:-1]:
        confidence, candidate = probs[d].max(1)
        accept = (confidence >= threshold) & (depth == 24) & (d >= minimum)
        if agreement: accept &= False if previous is None else candidate == previous
        pred[accept] = candidate[accept]; depth[accept] = d
        previous = candidate
    return pred, depth

def wilson(k, n, z=1.96):
    if not n: return [0.,1.]
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return [max(0,center-half),min(1,center+half)]

def summarize(pred, depth, labels, full):
    correct = pred == labels; harms = (full == labels) & ~correct; fixes = (full != labels) & correct
    return {'accuracy': float(correct.float().mean()), 'correct': int(correct.sum()), 'count': len(labels),
            'accuracy_ci95': wilson(int(correct.sum()),len(labels)), 'mean_depth': float(depth.float().mean()),
            'blocks_skipped_fraction': float((24-depth).float().mean()/24),
            'exit_counts': dict(Counter(map(int,depth.tolist()))), 'harmful_exits': int(harms.sum()), 'corrected_full_errors':int(fixes.sum()),
            'accuracy_delta_vs_full': float((correct.float()-(full==labels).float()).mean()),
            'harm_rate_upper95': wilson(int(harms.sum()),len(labels),1.645)[1]}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=8); parser.add_argument('--timing-samples',type=int,default=96)
    args=parser.parse_args(); threads=max(1,min(args.threads,(os.cpu_count() or 2)//2))
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    splits,categories,manifest=prepare_data()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    for p in model.parameters(): p.requires_grad_(False)
    def inputs(rows):
        texts=[tokenizer.apply_chat_template([{'role':'user','content':PROMPT.format(text=r['text'])}],tokenize=False,add_generation_prompt=True) for r in rows]
        return tokenizer(texts,return_tensors='pt',padding=True)
    identity=hashlib.sha256(json.dumps({'data':manifest,'prompt':PROMPT,'revision':REVISION,'depths':DEPTHS,'features':'last-token-v1'},sort_keys=True).encode()).hexdigest()
    cache_path=CACHE/(identity+'.pt')
    features={}; lengths={}; extraction_start=time.perf_counter()
    if cache_path.exists():
        cached=torch.load(cache_path,weights_only=True); features=cached['features']; lengths=cached['lengths']
        print('Using fingerprinted frozen-feature cache.',flush=True)
    else:
        with torch.inference_mode():
            for name,rows in splits.items():
                collected={d:[] for d in DEPTHS}; lengths[name]=[]
                for start in range(0,len(rows),args.batch_size):
                    batch=inputs(rows[start:start+args.batch_size]); lengths[name].extend(batch['attention_mask'].sum(1).tolist())
                    # Store only the final prompt-position vector; all layers execute for extraction.
                    handles=[]
                    for d in DEPTHS[:-1]:
                        handles.append(model.model.layers[d-1].register_forward_hook(lambda module,args,out,d=d: collected[d].append(out[0][:,-1,:].clone())))
                    try: out=model.model(**batch,use_cache=False)
                    finally:
                        for handle in handles: handle.remove()
                    collected[24].append(out.last_hidden_state[:,-1,:].clone())
                    if start%128==0: print(f'Features {name}: {min(start+args.batch_size,len(rows))}/{len(rows)}',flush=True)
                features[name]={d:torch.cat(v).clone() for d,v in collected.items()}
        torch.save({'features':features,'lengths':lengths},cache_path)
    features={s:{d:x.clone() for d,x in depths.items()} for s,depths in features.items()}
    labels={s:torch.tensor([r['label'] for r in rows]) for s,rows in splits.items()}
    extraction_seconds=time.perf_counter()-extraction_start
    results=[]; trace=[]; deployed=None
    for seed in [17,29,43]:
        torch.manual_seed(seed); heads={}; probabilities={s:{} for s in splits}; head_metrics=[]
        for d in DEPTHS:
            x=features['train'][d]; mean=x.mean(0); std=x.std(0).clamp_min(.05); x=(x-mean)/std
            head=torch.nn.Linear(x.shape[1],77); optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1)
            # Fixed optimization budget, identical architecture and training at every depth.
            for step in range(200):
                optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(head(x),labels['train']); loss.backward();optimizer.step()
            head.eval()
            with torch.inference_mode():
                tune=head((features['tune'][d]-mean)/std)
                temperature=min([.5,1.,1.5,2.,3.,4.,6.,8.],key=lambda t:float(torch.nn.functional.cross_entropy(tune/t,labels['tune'])))
                for s in splits: probabilities[s][d]=torch.softmax(head((features[s][d]-mean)/std)/temperature,1)
            heads[d]=(head,mean,std,temperature)
            head_metrics.append({'depth':d,'parameters':sum(p.numel() for p in head.parameters()),'temperature':temperature,
                                 'tune_accuracy':float((probabilities['tune'][d].argmax(1)==labels['tune']).float().mean()),
                                 'test_accuracy':float((probabilities['test'][d].argmax(1)==labels['test']).float().mean())})
        tune_full=probabilities['tune'][24].argmax(1)
        candidates=[]
        for agreement in [False,True]:
            for minimum in [6,12]:
                for threshold in [.5,.6,.7,.8,.9,.95,.975,.99,1.01]:
                    pred,depth=policy(probabilities['tune'],threshold,agreement,minimum)
                    m=summarize(pred,depth,labels['tune'],tune_full)
                    if m['accuracy_delta_vs_full']>=-.005:
                        candidates.append({'threshold':threshold,'agreement':agreement,'minimum':minimum,'tune':m})
        chosen=min(candidates,key=lambda c:(c['tune']['mean_depth'],-c['tune']['accuracy'], -c['threshold']))
        cal_pred,cal_depth=policy(probabilities['calibration'],chosen['threshold'],chosen['agreement'],chosen['minimum'])
        cal=summarize(cal_pred,cal_depth,labels['calibration'],probabilities['calibration'][24].argmax(1))
        # Independent guard: no test labels involved. On failure the production policy is full depth.
        passed=cal['harm_rate_upper95']<=.01
        pred,depth=policy(probabilities['test'],chosen['threshold'],chosen['agreement'],chosen['minimum'])
        full=probabilities['test'][24].argmax(1); test=summarize(pred,depth,labels['test'],full)
        fixed_depth=min(head_metrics,key=lambda row:(-row['tune_accuracy'],row['depth']))['depth']
        result={'seed':seed,'heads':head_metrics,'selected_policy':chosen,'calibration':cal,'calibration_guard_passed':passed,
                'fixed_depth_selected_on_tune':fixed_depth,
                'fixed_depth_test':summarize(probabilities['test'][fixed_depth].argmax(1),torch.full_like(depth,fixed_depth),labels['test'],full),
                'candidate_test':test,'deployed_test':test if passed else summarize(full,torch.full_like(depth,24),labels['test'],full)}
        results.append(result)
        print(json.dumps(result),flush=True)
        for i,row in enumerate(splits['test']):
            trace.append({'seed':seed,'id':row['id'],'label':row['label'],'full_prediction':int(full[i]),'candidate_prediction':int(pred[i]),'exit_layer':int(depth[i]),
                          'heads':{d:{'prediction':int(probabilities['test'][d][i].argmax()),'confidence':float(probabilities['test'][d][i].max())} for d in DEPTHS}})
        if seed==17: deployed=(heads,chosen,passed,probabilities['test'])
    heads,chosen,passed,expected=deployed
    torch.save({d:{'weight':h.weight,'bias':h.bias,'mean':m,'std':s,'temperature':t} for d,(h,m,s,t) in heads.items()},OUT/'heads.pt')
    def score(d,hidden):
        head,mean,std,temp=heads[d]; p=torch.softmax(head((hidden-mean)/std)/temp,1); c,y=p.max(1); return int(y.item()),float(c.item())
    def run(batch,adaptive):
        visited=[]; previous=[None]; handles=[]
        def hook(d):
            def check(module,args,out):
                visited.append(d)
                if adaptive and d in DEPTHS[:-1]:
                    pred,confidence=score(d,out[0][:,-1,:]); agreement=not chosen['agreement'] or previous[0]==pred
                    previous[0]=pred
                    if d>=chosen['minimum'] and confidence>=chosen['threshold'] and agreement: raise StopAtHead((pred,d,confidence))
            return check
        for i,layer in enumerate(model.model.layers): handles.append(layer.register_forward_hook(hook(i+1)))
        try:
            out=model.model(**batch,use_cache=False); pred,confidence=score(24,out.last_hidden_state[:,-1,:]); result=(pred,24,confidence)
        except StopAtHead as stop: result=stop.result
        finally:
            for handle in handles:handle.remove()
        assert visited==list(range(1,result[1]+1)),(visited,result)
        return result,visited
    timings=[]
    with torch.inference_mode():
        for _ in range(2):
            run(inputs([splits['test'][0]]),False);run(inputs([splits['test'][0]]),True)
        for i,row in enumerate(splits['test'][:args.timing_samples]):
            for repeat in range(2):
                for adaptive in ([False,True] if (i+repeat)%2==0 else [True,False]):
                    start=time.perf_counter();batch=inputs([row]); prepared=time.perf_counter()
                    (pred,depth,confidence),visited=run(batch,adaptive); end=time.perf_counter()
                    # Compare actual stopped runtime with cached full-depth readout policy.
                    ex_pred,ex_depth=policy({d:expected[d][i:i+1] for d in DEPTHS},chosen['threshold'],chosen['agreement'],chosen['minimum']) if adaptive else (expected[24][i:i+1].argmax(1),torch.tensor([24]))
                    assert pred==int(ex_pred[0]) and depth==int(ex_depth[0]),'Cached/batch-one execution disagreement'
                    timings.append({'id':row['id'],'repeat':repeat,'path':'candidate_early_exit' if adaptive else 'full_head','prediction':pred,'depth':depth,'executed_layers':visited,
                                    'input_tokens':int(batch['input_ids'].shape[1]),'tokenization_ms':(prepared-start)*1000,'model_ms':(end-prepared)*1000,'end_to_end_ms':(end-start)*1000})
            if i%12==0: print(f'Actual runtime timing: {i+1}/{args.timing_samples}',flush=True)
    timing_summary={path:{kind:{'mean':float(np.mean([r[kind] for r in timings if r['path']==path])), 'p50':float(np.median([r[kind] for r in timings if r['path']==path])), 'p95':float(np.percentile([r[kind] for r in timings if r['path']==path],95))} for kind in ['model_ms','end_to_end_ms']} for path in ['full_head','candidate_early_exit']}
    payload={'status':'completed','model':MODEL,'revision':REVISION,'hidden_size':model.config.hidden_size,'layers':24,'classes':77,
             'dataset':'BANKING77','dataset_revision':DATA_REV,'feature_fingerprint':identity,'prompt':PROMPT,
             'split_sizes':{s:len(rows) for s,rows in splits.items()},'threads':threads,'batch_size':args.batch_size,
             'torch':torch.__version__,'transformers':__import__('transformers').__version__,'hardware':platform.processor(),
             'feature_extraction_seconds_this_run':extraction_seconds,'runs':results,'timing':timing_summary,'timing_rows':len(timings),'timing_seed':17,
             'limitations':['Frozen specialist classifiers; does not establish arbitrary changing-rule instruction following.',
                            'Public benchmark may overlap unknown pretraining data; no contamination-free claim.',
                            'Only 32 training queries per intent; all 3080 official test queries evaluated.',
                            'Three classifier initialization seeds share one data split and frozen backbone.',
                            'Policy chosen on tune and independently guarded on calibration; timing shows candidate even if guard rejects it.',
                            'CPU warm batch-one timing includes tokenization, excludes loading/downloads. Full-depth features are not early-exit timing.',
                            'Frozen heads compared with each other; no claim of superiority to an equally trained text decoder.']}
    write(OUT/'result.json',payload);write(OUT/'predictions.json',trace);write(OUT/'timings.json',timings)
    print(json.dumps({'timing':timing_summary,'output':str(OUT)}),flush=True)

if __name__=='__main__':main()

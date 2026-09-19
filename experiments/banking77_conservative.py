"""Exploratory gate revision, confirmed on previously unused BANKING77 train rows.

Uses the already trained seed-17 heads. Does not train against reserve labels.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('MKL_NUM_THREADS','8');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse,csv,hashlib,json,random,time
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from banking77 import MODEL,REVISION,DEPTHS,PROMPT,CACHE,ROOT,policy,summarize,write,StopAtHead

OUT=ROOT/'results/banking77-conservative'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--time-runtime',action='store_true');args=parser.parse_args()
    torch.set_num_threads(min(8,max(1,(os.cpu_count() or 2)//2)));torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    original=json.loads((ROOT/'results/banking77/result.json').read_text());manifest=json.loads((ROOT/'results/banking77/data-manifest.json').read_text())
    cached=torch.load(CACHE/(original['feature_fingerprint']+'.pt'),weights_only=True)
    saved=torch.load(ROOT/'results/banking77/heads.pt',weights_only=True);heads={}
    for d,data in saved.items():
        head=torch.nn.Linear(896,77);head.load_state_dict({'weight':data['weight'],'bias':data['bias']});head.eval()
        heads[d]=(head,data['mean'],data['std'],data['temperature'])
    all_rows={}
    for source in ['train','test']:
        with (CACHE/f'{source}.csv').open(encoding='utf-8',newline='') as f:
            for i,r in enumerate(csv.DictReader(f)):all_rows[f'{source}:{i}']={'id':f'{source}:{i}','text':r['text'],'label':manifest['categories'].index(r['category'])}
    used_ids={i for ids in manifest['splits'].values() for i in ids}
    used_text={all_rows[i]['text'].strip().casefold() for i in used_ids}
    reserve={s:[] for s in ['calibration','test']};seen=set(used_text)
    for label in range(77):
        pool=[r for i,r in all_rows.items() if i.startswith('train:') and i not in used_ids and r['label']==label]
        random.Random(194102+label).shuffle(pool);unique=[]
        for r in pool:
            key=r['text'].strip().casefold()
            if key not in seen:seen.add(key);unique.append(r)
        # Interleave allocation so small classes can contribute to both sets.
        reserve['calibration'].extend(unique[:20:2]);reserve['test'].extend(unique[1:20:2])
    for rows in reserve.values():random.Random(194102).shuffle(rows)
    assert not ({r['id'] for rs in reserve.values() for r in rs}&used_ids)
    def probs(features):
        with torch.inference_mode():return {d:torch.softmax(h((features[d]-mean)/std)/temp,1) for d,(h,mean,std,temp) in heads.items()}
    tune=probs(cached['features']['tune']);tune_y=torch.tensor([all_rows[i]['label'] for i in manifest['splits']['tune']]);full=tune[24].argmax(1)
    candidates=[]
    for agreement in [False,True]:
        for minimum in [6,12]:
            for threshold in [.5,.6,.7,.8,.9,.95,.975,.99,1.01]:
                pred,depth=policy(tune,threshold,agreement,minimum);metrics=summarize(pred,depth,tune_y,full)
                if metrics['harm_rate_upper95']<=.005:candidates.append({'threshold':threshold,'agreement':agreement,'minimum':minimum,'tune':metrics})
    chosen=min(candidates,key=lambda c:(c['tune']['mean_depth'],-c['tune']['accuracy'],-c['threshold']))
    protocol={'revision_reason':'Original candidate preserved average accuracy but failed its harmful-exit guard. This is a post-first-test exploratory gate revision.',
              'selection':'Require tuning harmful-exit Wilson upper95 <=0.5%; select lowest mean depth. Confirm on fresh reserve calibration with upper95 <=1%.',
              'selected_before_reserve_inference':chosen,'source_feature_fingerprint':original['feature_fingerprint'],'seed':17,
              'reserve_ids':{s:[r['id'] for r in rows] for s,rows in reserve.items()},'sizes':{s:len(rows) for s,rows in reserve.items()},
              'classes':{s:len({r['label'] for r in rows}) for s,rows in reserve.items()},'excluded':'All original development/test IDs and exact normalized-text duplicates, including within the reserve.'}
    write(OUT/'protocol.json',protocol);print(json.dumps(protocol['selected_before_reserve_inference']),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    def inputs(rows):return tokenizer([tokenizer.apply_chat_template([{'role':'user','content':PROMPT.format(text=r['text'])}],tokenize=False,add_generation_prompt=True) for r in rows],return_tensors='pt',padding=True)
    identity=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest();cache_path=CACHE/('reserve-'+identity+'.pt')
    if cache_path.exists():features=torch.load(cache_path,weights_only=True)
    else:
        features={}
        with torch.inference_mode():
            for split,rows in reserve.items():
                collected={d:[] for d in DEPTHS}
                for start in range(0,len(rows),8):
                    handles=[model.model.layers[d-1].register_forward_hook(lambda module,args,out,d=d:collected[d].append(out[0][:,-1,:].clone())) for d in DEPTHS[:-1]]
                    try:out=model.model(**inputs(rows[start:start+8]),use_cache=False)
                    finally:
                        for h in handles:h.remove()
                    collected[24].append(out.last_hidden_state[:,-1,:].clone())
                    if start%128==0:print(f'Fresh reserve {split}: {min(start+8,len(rows))}/{len(rows)}',flush=True)
                features[split]={d:torch.cat(v) for d,v in collected.items()}
        torch.save(features,cache_path)
    result={'model':MODEL,'revision':REVISION,'seed':17,'protocol_hash':identity,'policy':chosen,'results':{}}
    traces=[]
    for split,rows in reserve.items():
        p=probs(features[split]);labels=torch.tensor([r['label'] for r in rows]);full=p[24].argmax(1);pred,depth=policy(p,chosen['threshold'],chosen['agreement'],chosen['minimum'])
        result['results'][split]={'full':summarize(full,torch.full_like(depth,24),labels,full),'candidate':summarize(pred,depth,labels,full)}
        traces.extend({'split':split,'id':r['id'],'label':r['label'],'full_prediction':int(full[i]),'candidate_prediction':int(pred[i]),'depth':int(depth[i])} for i,r in enumerate(rows))
    result['guard_passed']=result['results']['calibration']['candidate']['harm_rate_upper95']<=.01
    if args.time_runtime:
        def score(d,hidden):
            h,mean,std,temp=heads[d];confidence,pred=torch.softmax(h((hidden-mean)/std)/temp,1).max(1)
            return int(pred.item()),float(confidence.item())
        def execute(batch,adaptive):
            visited=[];handles=[];previous=[None]
            def hook(d):
                def check(module,args,out):
                    visited.append(d)
                    if adaptive and d in DEPTHS[:-1]:
                        pred,confidence=score(d,out[0][:,-1,:]);agree=not chosen['agreement'] or previous[0]==pred;previous[0]=pred
                        if d>=chosen['minimum'] and confidence>=chosen['threshold'] and agree:raise StopAtHead((pred,d))
                return check
            for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(hook(i+1)))
            try:
                output=model.model(**batch,use_cache=False);pred,_=score(24,output.last_hidden_state[:,-1,:]);answer=(pred,24)
            except StopAtHead as stop:answer=stop.result
            finally:
                for handle in handles:handle.remove()
            assert visited==list(range(1,answer[1]+1))
            return answer,visited
        expected=probs(features['test']);timings=[]
        with torch.inference_mode():
            for _ in range(2):
                execute(inputs([reserve['test'][0]]),False);execute(inputs([reserve['test'][0]]),True)
            for i,row in enumerate(reserve['test'][:96]):
                for repeat in range(2):
                    for adaptive in ([False,True] if (i+repeat)%2==0 else [True,False]):
                        start=time.perf_counter();batch=inputs([row]);prepared=time.perf_counter();(pred,depth),visited=execute(batch,adaptive);end=time.perf_counter()
                        ex_pred,ex_depth=policy({d:expected[d][i:i+1] for d in DEPTHS},chosen['threshold'],chosen['agreement'],chosen['minimum']) if adaptive else (expected[24][i:i+1].argmax(1),torch.tensor([24]))
                        assert pred==int(ex_pred[0]) and depth==int(ex_depth[0])
                        timings.append({'id':row['id'],'repeat':repeat,'path':'candidate_early_exit' if adaptive else 'full_head','prediction':pred,'depth':depth,'executed_layers':visited,
                                        'model_ms':(end-prepared)*1000,'end_to_end_ms':(end-start)*1000})
                if i%16==0:print(f'Conservative runtime {i+1}/96',flush=True)
        result['timing']={path:{kind:{'mean':float(np.mean([r[kind] for r in timings if r['path']==path])), 'p50':float(np.median([r[kind] for r in timings if r['path']==path])),
                                   'p95':float(np.percentile([r[kind] for r in timings if r['path']==path],95))} for kind in ['model_ms','end_to_end_ms']} for path in ['full_head','candidate_early_exit']}
        write(OUT/'timings.json',timings)
    result['limitations']=['Gate design changed after seeing first official-test results. Fresh reserves are disjoint from that run, but still the same public dataset.',
                           'Small original training classes may have no unused reserve; exact per-split class coverage is recorded.',
                           'One trained-head seed; no new backbone/head training. Depth metrics use cached full passes; only the optional timed runtime actually skips blocks.',
                           'Unknown pretraining or paraphrase overlap remains. Calibration is not a production safety guarantee.']
    write(OUT/'result.json',result);write(OUT/'predictions.json',traces);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

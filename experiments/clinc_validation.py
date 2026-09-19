"""Prospective second-dataset evaluation, with unknown requests and exact risk bounds.

Run --prepare first to seal the protocol before inference. No test labels select a gate.
This is a 30-intent CLINC150 subset, not the complete 150-intent benchmark.
"""
import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta
from transformers import AutoModelForCausalLM, AutoTokenizer
from banking77 import MODEL, REVISION, DEPTHS, policy

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'experiments/.cache/clinc'
OUT=ROOT/'results/clinc-validation'
DATA_REV='828f8093932c8fe6ca7936c3d2e52903b1c523de'
EXPECTED={'data_full.json':'36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0',
          'domains.json':'b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a'}
PROMPT='Read this assistant request and identify its intent.\nRequest: {text}\nIntent:'

def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def upper(k,n):
    return 1. if n==0 or k==n else float(beta.ppf(.95,k+1,n-k))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,digest in EXPECTED.items():
        assert hashlib.sha256((CACHE/name).read_bytes()).hexdigest()==digest, name
    data=json.loads((CACHE/'data_full.json').read_text())
    domains=json.loads((CACHE/'domains.json').read_text())
    categories=sorted(c for domain in sorted(domains) for c in sorted(domains[domain])[:3])
    splits={s:[] for s in ['train','tune','calibration','test','oos_tune','oos_calibration','oos_test','unsupported_test']}
    for label,name in enumerate(categories):
        for source in ['train','val','test']:
            rows=[{'id':f'{source}:{i}','text':text,'label':label,'intent':intent} for i,(text,intent) in enumerate(data[source]) if intent==name]
            random.Random(20260920+label).shuffle(rows)
            if source=='train':splits['train']+=rows[:32];splits['tune']+=rows[32:42]
            elif source=='val':splits['calibration']+=rows
            else:splits['test']+=rows
    for source,target in [('oos_train','oos_tune'),('oos_val','oos_calibration'),('oos_test','oos_test')]:
        splits[target]=[{'id':f'{source}:{i}','text':text,'label':-1,'intent':intent} for i,(text,intent) in enumerate(data[source])]
    # All official test examples from the other 120 intents are unsupported by these heads.
    # Predetermined one per intent keeps this secondary stress test bounded.
    for name in sorted(set(x[1] for x in data['test'])-set(categories)):
        i,(text,intent)=next((i,x) for i,x in enumerate(data['test']) if x[1]==name)
        splits['unsupported_test'].append({'id':f'test:{i}','text':text,'label':-1,'intent':intent})
    # Remove exact duplicates against earlier partitions before looking at model outputs.
    seen=set();removed=[]
    for split,rows in splits.items():
        clean=[]
        for row in rows:
            key=row['text'].strip().casefold()
            if key in seen:removed.append({'split':split,'id':row['id']});continue
            seen.add(key);clean.append(row)
        splits[split]=clean;random.Random(20260920).shuffle(clean)
    protocol={'record_type':'Local prospective protocol; not external preregistration','dataset_revision':DATA_REV,'data_sha256':EXPECTED,
              'categories':categories,'selection':'First three alphabetical intents in each of ten domains; all official test rows for these intents.',
              'ids':{s:[r['id'] for r in rows] for s,rows in splits.items()},'removed_exact_duplicates':removed,
              'sizes':{s:len(rows) for s,rows in splits.items()},'model':MODEL,'revision':REVISION,'prompt':PROMPT,
              'training':{'seed':61,'steps':200,'lr':.01,'weight_decay':.1,'depths':DEPTHS,'head':'standardized linear softmax'},
              'selection_rule':'On tuning: minimize mean depth with <=0.5pp net accuracy loss, <=5% absolute early error, <=5% early OOS acceptance, and >=5% in-scope early coverage. Fall back if no candidate.',
              'grid':{'threshold':[.5,.6,.7,.8,.9,.95,.975,.99,.995],'agreement':[False,True],'minimum':[6,12]},
              'calibration_rule':'Frozen selected gate; exact one-sided 95% Clopper-Pearson bounds: new errors/all <=1%, wrong early answers/early exits <=5%, early acceptance of OOS <=5%. Require >=5% in-scope early coverage. Bounds are individual, not a joint 95% guarantee.',
              'unknowns':'Early acceptance is an error for OOS. Full-depth classifier has no unknown class: continued unknown requests remain unresolved, not successfully rejected.',
              'comparators':'Identical-data TF-IDF logistic regression; fixed-depth heads; full-depth trained head. No ordinary generation speed claim.',
              'limits':['Crowdsourced benchmark, not production traffic.','30 selected intents, one seed, one frozen backbone.','Unknown pretraining overlap.','No new latency claim from full feature extraction.'],
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol changed: use a new experiment directory, not silent replacement.'
    else:write(path,protocol)
    return splits,protocol

def metrics(pred,depth,labels,full):
    early=depth<24;correct=pred==labels;harm=(full==labels)&~correct
    n=len(labels);exits=int(early.sum());wrong=int((early&~correct).sum());harms=int(harm.sum())
    return {'n':n,'correct':int(correct.sum()),'full_correct':int((full==labels).sum()),'early_count':exits,'early_wrong':wrong,
            'harmful':harms,'corrected':int(((full!=labels)&correct).sum()),'mean_depth':float(depth.float().mean()),
            'blocks_skipped':float((24-depth).float().mean()/24),'early_error_upper95':upper(wrong,exits),'harm_upper95':upper(harms,n),
            'early_coverage':exits/n,'exit_counts':{str(d):int((depth==d).sum()) for d in DEPTHS}}

def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare',action='store_true');opts=args.parse_args()
    torch.set_num_threads(min(8,max(1,(os.cpu_count() or 2)//2)));torch.set_num_interop_threads(1)
    splits,protocol=prepare()
    print(json.dumps({'protocol_saved':str(OUT/'protocol.json'),'sizes':protocol['sizes']}),flush=True)
    if opts.prepare:return
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    features={}
    with torch.inference_mode():
        for split,rows in splits.items():
            cache=CACHE/f'{fingerprint}-{split}.pt'
            if cache.exists():features[split]=torch.load(cache,weights_only=True);continue
            chunks={d:[] for d in DEPTHS}
            for start in range(0,len(rows),8):
                messages=[tokenizer.apply_chat_template([{'role':'user','content':PROMPT.format(text=r['text'])}],tokenize=False,add_generation_prompt=True) for r in rows[start:start+8]]
                batch=tokenizer(messages,padding=True,return_tensors='pt')
                hooks=[model.model.layers[d-1].register_forward_hook(lambda module,args,out,d=d:chunks[d].append(out[0][:,-1,:].clone())) for d in DEPTHS[:-1]]
                try:out=model.model(**batch,use_cache=False)
                finally:
                    for hook in hooks:hook.remove()
                chunks[24].append(out.last_hidden_state[:,-1,:].clone())
                if start%128==0:print(f'Features {split}: {min(start+8,len(rows))}/{len(rows)}',flush=True)
            features[split]={d:torch.cat(v).clone() for d,v in chunks.items()};torch.save(features[split],cache)
    del model
    labels={s:torch.tensor([r['label'] for r in rows]) for s,rows in splits.items()}
    torch.manual_seed(61);probabilities={s:{} for s in splits};portable={}
    for depth in DEPTHS:
        x=features['train'][depth];mean=x.mean(0);std=x.std(0).clamp_min(.05);x=(x-mean)/std
        head=torch.nn.Linear(896,len(protocol['categories']));opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1)
        for _ in range(200):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(x),labels['train']);loss.backward();opt.step()
        with torch.inference_mode():
            tune=head((features['tune'][depth]-mean)/std)
            temp=min([.5,1.,1.5,2.,3.,4.,6.,8.],key=lambda t:float(torch.nn.functional.cross_entropy(tune/t,labels['tune'])))
            for s in splits:probabilities[s][depth]=torch.softmax(head((features[s][depth]-mean)/std)/temp,1)
        for key,value in [('weight',head.weight),('bias',head.bias),('mean',mean),('std',std)]:portable[f'{depth}_{key}']=value.detach().numpy()
        portable[f'{depth}_temperature']=np.array(temp)
    np.savez_compressed(OUT/'heads.npz',**portable)
    candidates=[];tune=probabilities['tune'];tune_y=labels['tune'];full=tune[24].argmax(1)
    for minimum in [6,12]:
        for agreement in [False,True]:
            for threshold in protocol['grid']['threshold']:
                pred,depth=policy(tune,threshold,agreement,minimum);m=metrics(pred,depth,tune_y,full)
                _,ood_depth=policy(probabilities['oos_tune'],threshold,agreement,minimum)
                ood_rate=float((ood_depth<24).float().mean())
                if m['early_coverage']>=.05 and (m['correct']-m['full_correct'])/m['n']>=-.005 and m['early_wrong']/max(1,m['early_count'])<=.05 and ood_rate<=.05:
                    candidates.append({'threshold':threshold,'agreement':agreement,'minimum':minimum,'tune':m,'oos_tune_acceptance':ood_rate})
    chosen=min(candidates,key=lambda c:(c['tune']['mean_depth'],-c['threshold'])) if candidates else {'threshold':1.01,'agreement':False,'minimum':6,'reason':'No tuning candidate met constraints'}
    write(OUT/'selected-policy.json',chosen) # Freeze before any calibration/test policy evaluation.
    result={'protocol_sha256':fingerprint,'policy':chosen,'candidate_count':len(candidates),'splits':{},'fixed_heads':{}}
    traces=[]
    for split in ['calibration','test','oos_calibration','oos_test','unsupported_test']:
        probs=probabilities[split];pred,depth=policy(probs,chosen['threshold'],chosen['agreement'],chosen['minimum']);full=probs[24].argmax(1)
        if split in ['calibration','test']:result['splits'][split]=metrics(pred,depth,labels[split],full)
        else:
            accepted=int((depth<24).sum());result['splits'][split]={'n':len(depth),'early_accepted':accepted,'acceptance_upper95':upper(accepted,len(depth)),'continued':len(depth)-accepted}
        for i,row in enumerate(splits[split]):traces.append({'split':split,'id':row['id'],'label':row['label'],'prediction':int(pred[i]),'full_prediction':int(full[i]),'depth':int(depth[i]),'heads':{d:{'prediction':int(probs[d][i].argmax()),'confidence':float(probs[d][i].max())} for d in DEPTHS}})
    for d in DEPTHS:result['fixed_heads'][d]={'test_correct':int((probabilities['test'][d].argmax(1)==labels['test']).sum()),'tune_correct':int((probabilities['tune'][d].argmax(1)==labels['tune']).sum())}
    cal=result['splits']['calibration'];oos=result['splits']['oos_calibration']
    result['guard_checks']={'added_error':cal['harm_upper95']<=.01,'absolute_early_error':cal['early_error_upper95']<=.05,'unknown_acceptance':oos['acceptance_upper95']<=.05,'coverage':cal['early_coverage']>=.05}
    result['guard_passed']=all(result['guard_checks'].values())
    # Same training examples and labels for an inexpensive baseline; no timing comparison here.
    sys.path.insert(0,str(ROOT/'experiments/.cache/tooling'))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=12000,sublinear_tf=True)
        x=vectorizer.fit_transform([r['text'] for r in splits['train']]);lexical=LogisticRegression(C=4,max_iter=400).fit(x,labels['train'].numpy())
        lp=lexical.predict(vectorizer.transform([r['text'] for r in splits['test']]))
    result['lexical']={'correct':int((lp==labels['test'].numpy()).sum()),'n':len(lp),'converged':bool(max(lexical.n_iter_)<400),'predictions':lp.tolist()}
    write(OUT/'predictions.json',traces);write(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='lexical'}),flush=True)

if __name__=='__main__':main()

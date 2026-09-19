"""Paired changing-rule stress test on held-out public utterances.

Authored rules over CLINC labels: not real traffic and not general instruction validation.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from banking77 import MODEL,REVISION,DEPTHS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/changing-rules';CACHE=ROOT/'experiments/.cache/clinc'

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    torch.set_num_threads(min(4,max(1,(os.cpu_count() or 2)//2)));torch.set_num_interop_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    raw=json.loads((CACHE/'data_full.json').read_text());domains=json.loads((CACHE/'domains.json').read_text());splits={s:[] for s in ['train','tune','test']}
    used=set()
    for domain in sorted(domains):
        pair=sorted(domains[domain])[:2]
        for intent in pair:
            for split,source,count in [('train','train',8),('tune','val',2),('test','test',5)]:
                rows=[(i,text) for i,(text,label) in enumerate(raw[source]) if label==intent and text.strip().casefold() not in used]
                random.Random(194203).shuffle(rows)
                for i,text in rows[:count]:
                    used.add(text.strip().casefold())
                    for flip in [0,1]:
                        a,b=(pair if not flip else pair[::-1]);label=0 if intent==a else 1
                        if split=='test':rule=f'Choose one code. Code A means {a.replace("_"," ")}; code B means {b.replace("_"," ")}. Which code matches the request below?'
                        else:rule=f'Return A for requests about {a.replace("_"," ")}. Return B for requests about {b.replace("_"," ")}. Classify this request.'
                        splits[split].append({'id':f'{source}:{i}:{flip}','pair_id':f'{source}:{i}','domain':domain,'intent':intent,'label':label,'prompt':rule+'\nRequest: '+text+'\nCode:'})
    protocol={'type':'Paired authored-rule stress test, not production instruction following','source_revision':'828f8093932c8fe6ca7936c3d2e52903b1c523de',
              'source_sha256':hashlib.sha256((CACHE/'data_full.json').read_bytes()).hexdigest(),'seed':73,'depths':DEPTHS,
              'selection':'First two alphabetical intents in each of ten CLINC domains; fixed seeded source selection; each query paired with reversed A/B mapping.',
              'training':'Eight train queries per intent, two opposite rules each. Linear heads, 200 steps AdamW lr .01 wd .1. Temperatures are unnecessary for argmax.',
              'evaluation':'Two tuning queries and five test queries per intent; test uses a different authored rule template. Report both-correct paired accuracy and prediction flip rate.',
              'success_criterion':'At least 90% of held-out query pairs have both opposite-rule answers correct. This criterion alone does not establish arbitrary instructions.',
              'rows':splits,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:write(OUT/'protocol.json',protocol)
    print({s:len(r) for s,r in splits.items()},flush=True)
    if args.prepare:return
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    features={};fingerprint=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    with torch.inference_mode():
        for split,rows in splits.items():
            path=CACHE/f'rules-{fingerprint}-{split}.pt'
            if path.exists():features[split]=torch.load(path,weights_only=True);continue
            chunks={d:[] for d in DEPTHS}
            for start in range(0,len(rows),4):
                messages=[tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True) for r in rows[start:start+4]]
                batch=tokenizer(messages,padding=True,return_tensors='pt');hooks=[]
                for d in DEPTHS[:-1]:hooks.append(model.model.layers[d-1].register_forward_hook(lambda mod,inp,out,d=d:chunks[d].append(out[0][:,-1,:].clone())))
                try:out=model.model(**batch,use_cache=False)
                finally:
                    for h in hooks:h.remove()
                chunks[24].append(out.last_hidden_state[:,-1,:].clone())
                if start%80==0:print(f'{split} {start+4}/{len(rows)}',flush=True)
            features[split]={d:torch.cat(v).clone() for d,v in chunks.items()};torch.save(features[split],path)
    torch.manual_seed(73);labels={s:torch.tensor([r['label'] for r in rows]) for s,rows in splits.items()};result={};trace=[];weights={}
    for d in DEPTHS:
        x=features['train'][d];mean=x.mean(0);std=x.std(0).clamp_min(.05);head=torch.nn.Linear(896,2);opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1)
        for _ in range(200):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(head((x-mean)/std),labels['train']);loss.backward();opt.step()
        with torch.inference_mode():pred=head((features['test'][d]-mean)/std).argmax(1)
        pairs={}
        for i,row in enumerate(splits['test']):
            pairs.setdefault(row['pair_id'],[]).append((int(pred[i]),row['label']));trace.append({'depth':d,'id':row['id'],'label':row['label'],'prediction':int(pred[i])})
        both=sum(all(p==y for p,y in v) for v in pairs.values());changed=sum(v[0][0]!=v[1][0] for v in pairs.values())
        result[d]={'correct':int((pred==labels['test']).sum()),'n':len(pred),'pairs':len(pairs),'both_correct':both,'changed_prediction':changed,'criterion_passed':both/len(pairs)>=.9}
        for name,value in [('weight',head.weight),('bias',head.bias),('mean',mean),('std',std)]:weights[f'{d}_{name}']=value.detach().numpy()
    write(OUT/'result.json',{'protocol_sha256':fingerprint,'depths':result,'limits':['Public utterances with authored rules, one template shift, no human adjudication of inferred label names.','No stopping gate selected or latency claim.','Query groups are held out; arbitrary instruction following remains unvalidated.']})
    write(OUT/'predictions.json',trace);np.savez_compressed(OUT/'heads.npz',**weights);print(result,flush=True)

if __name__=='__main__':main()

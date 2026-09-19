"""Bounded CPU pilot: frozen Qwen readouts, calibrated exits, and real skipped layers.

Synthetic keyword-policy fixtures only. This is not a moderation benchmark.
Requires the pinned checkpoint; no remote model code is executed.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, hashlib, json, platform, random, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
DEPTHS = [6, 12, 18, 24]

def fixtures(words, seed):
    rng = random.Random(seed); rows = []
    for word in words:
        for action in ['ALLOW', 'BLOCK']:
            for matches in [True, False]:
                for contextual in [False, True]:
                    other = 'unrelated'
                    message = f'Please discuss {word if matches else other}.'
                    context = 'No previous message.'
                    if contextual:
                        context = message
                        message = 'Same topic as the previous message.'
                    label = (0 if action == 'ALLOW' else 1) if matches else 2
                    prompt = f'Rule: If the topic is {word}, return {action}. Otherwise return IGNORE.\nPrevious message: {context}\nCurrent message: {message}\nReturn exactly one code: A=ALLOW, B=BLOCK, C=IGNORE.'
                    rows.append({'prompt':prompt,'label':label,'group':word,'contextual':contextual})
    rng.shuffle(rows); return rows

class StopAtHead(Exception):
    def __init__(self, result): self.result = result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--threads',type=int,default=8); args=parser.parse_args()
    threads=max(1,min(args.threads,(os.cpu_count() or 2)//2)); torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.manual_seed(17); np.random.seed(17)
    output=ROOT/'results/qwen-decisions'; output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    load_seconds=time.perf_counter()-started
    for parameter in model.parameters(): parameter.requires_grad_(False)
    splits={
        'train':fixtures(['apples','coffee','music','tickets','painting','bicycles','weather','gardens','books','football'],11),
        'calibration':fixtures(['oranges','travel','photography','hockey'],23),
        'test':fixtures(['peaches','astronomy','ceramics','rowing','bread','theatre'],37)}
    def inputs(rows):
        texts=[tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True) for r in rows]
        return tokenizer(texts,return_tensors='pt',padding=True)
    features={}; labels={}; token_accuracy={}; vocab_ids=[tokenizer.encode(x,add_special_tokens=False)[0] for x in ['A','B','C']]
    assert all(len(tokenizer.encode(x,add_special_tokens=False))==1 for x in ['A','B','C'])
    with torch.inference_mode():
        for split,rows in splits.items():
            cache={d:[] for d in DEPTHS}; predictions=[]
            for start in range(0,len(rows),4):
                batch=inputs(rows[start:start+4]);out=model.model(**batch,output_hidden_states=True,use_cache=False)
                for depth in DEPTHS: cache[depth].append(out.hidden_states[depth][:,-1,:].cpu())
                # Actual full vocabulary projection, constrained to the three legal codes.
                predictions.extend(model.lm_head(out.last_hidden_state[:,-1,:])[:,vocab_ids].argmax(-1).tolist())
                print(f'{split}: {min(start+4,len(rows))}/{len(rows)}',flush=True)
            features[split]={d:torch.cat(v) for d,v in cache.items()}
            labels[split]=torch.tensor([r['label'] for r in rows])
            token_accuracy[split]=float((np.array(predictions)==labels[split].numpy()).mean())
    labels={key:value.clone() for key,value in labels.items()}
    heads={};metrics=[];thresholds={};traces=[]
    for depth in DEPTHS:
        x=features['train'][depth].clone();mean=x.mean(0);std=x.std(0).clamp_min(.05); x=(x-mean)/std
        head=torch.nn.Linear(x.shape[-1],3);optim=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1)
        for step in range(180):
            optim.zero_grad();loss=torch.nn.functional.cross_entropy(head(x),labels['train']);loss.backward();optim.step()
        head.eval()
        with torch.inference_mode():
            cal_logits=head((features['calibration'][depth]-mean)/std)
            # Temperature selection uses calibration data only.
            temperatures=[.5,1.,2.,4.,8.]
            temperature=min(temperatures,key=lambda t:float(torch.nn.functional.cross_entropy(cal_logits/t,labels['calibration'])))
            cp=torch.softmax(cal_logits/temperature,-1);confidence,pred=cp.max(-1)
            valid=[]
            for threshold in [.5,.6,.7,.8,.9,.95,.99]:
                accepted=confidence>=threshold
                if accepted.sum()>=8 and float((pred[accepted]==labels['calibration'][accepted]).float().mean())>=.95: valid.append(threshold)
            threshold=min(valid) if valid else 1.01
            testp=torch.softmax(head((features['test'][depth]-mean)/std)/temperature,-1)
            correct=testp.argmax(-1)==labels['test']
            metrics.append({'depth':depth,'test_accuracy':float(correct.float().mean()),'temperature':temperature,'exit_threshold':threshold,'calibration_samples':len(cp)})
            heads[depth]=(head,mean,std,temperature);thresholds[depth]=threshold
            torch.save({'weight':head.weight,'bias':head.bias,'mean':mean,'std':std,'temperature':temperature,'revision':REVISION}, output/f'head-{depth}.pt')
    def score(depth,hidden):
        head,mean,std,temp=heads[depth]; p=torch.softmax(head((hidden-mean)/std)/temp,-1); c,pred=p.max(-1);return float(c.item()),int(pred.item())
    def adaptive(batch):
        count=[0];handles=[]
        def hook(depth):
            def check(module,args,result):
                count[0]+=1
                if depth in DEPTHS[:-1]:
                    confidence,pred=score(depth,result[0][:,-1,:])
                    if confidence>=thresholds[depth]: raise StopAtHead((pred,depth,confidence))
            return check
        for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(hook(i+1)))
        try:
            out=model.model(**batch,use_cache=False)
            confidence,pred=score(24,out.last_hidden_state[:,-1,:]);result=(pred,24,confidence)
        except StopAtHead as stop:result=stop.result
        finally:
            for handle in handles:handle.remove()
        assert count[0]==result[1]
        return result
    times={'minimal_token':[],'direct_head':[],'adaptive_head':[]};baseline_predictions=[];adaptive_predictions=[]
    with torch.inference_mode():
        # Warm each path; rotate measured order to reduce systematic ordering effects.
        batch=inputs([splits['test'][0]]);model(**batch,use_cache=False);adaptive(batch)
        for index,row in enumerate(splits['test']):
            batch=inputs([row]);outputs={}
            order=list(times);order=order[index%3:]+order[:index%3]
            for key in order:
                start=time.perf_counter()
                if key=='minimal_token':
                    out=model.model(**batch,use_cache=False);outputs[key]=int(model.lm_head(out.last_hidden_state[:,-1,:])[:,vocab_ids].argmax(-1).item())
                elif key=='direct_head':
                    out=model.model(**batch,use_cache=False);outputs[key]=score(24,out.last_hidden_state[:,-1,:])[1]
                else:outputs[key]=adaptive(batch)
                times[key].append((time.perf_counter()-start)*1000)
            pred,depth,confidence=outputs['adaptive_head'];adaptive_predictions.append(pred);baseline_predictions.append(outputs['minimal_token'])
            traces.append({'index':index,'group':row['group'],'contextual':row['contextual'],'label':row['label'],'token_prediction':outputs['minimal_token'],'direct_prediction':outputs['direct_head'],'adaptive_prediction':pred,'depth':depth,'confidence':confidence})
            print(f'timing {index+1}/{len(splits["test"])}',flush=True)
    payload={'experiment_ids':['D01','D02','D03','D04','D05'],'status':'pilot','scope':'Synthetic keyword/topic policies; no human moderation validation; CPU batch one. Three-code constrained vocabulary baseline. Timing excludes tokenization and model loading, includes transformer/head and hooks. No JSON latency claim.',
        'model':MODEL,'revision':REVISION,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'hardware':platform.processor(),'threads':threads,'seed':17,'load_seconds':load_seconds,
        'split_sizes':{k:len(v) for k,v in splits.items()},'data_hash':hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest(),
        'heads':metrics,'token_accuracy':token_accuracy,'adaptive_accuracy':float((np.array(adaptive_predictions)==labels['test'].numpy()).mean()),
        'early_fraction':sum(t['depth']<24 for t in traces)/len(traces),'mean_depth':float(np.mean([t['depth'] for t in traces])),
        'timing_ms':{k:{'p50':float(np.median(v)),'p95':float(np.percentile(v,95)),'samples':len(v)} for k,v in times.items()},
        'limitations':['One model, seed and small synthetic split. Calibration thresholds are exploratory; no rare-error guarantees.','Rules differ by held-out topic words; rule grammar is shared. This does not establish arbitrary instruction generalization.','Intermediate-state extraction used full depth for probing. Only the timed hook path skipped layers.','No throughput, energy, GPU/NPU, or persistent-cache benchmark.']}
    (output/'result.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (output/'predictions.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    (output/'fixtures.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':main()

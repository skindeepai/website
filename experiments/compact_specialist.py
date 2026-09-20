"""Compare a larger compact specialist and a shared two/four-layer model.

Development selects epochs and thresholds. The consumed evaluation is diagnostic.
No runtime claims are made while other experiments are active.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,time,hashlib,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/.cache/replay-runtime'))
import numpy as np
import torch
import requests
from transformers import AutoModel,AutoTokenizer
from chat_refinement import rows_and_tokenizers,bounded
from chat_smoke_specialist import metrics
OUT=ROOT/'results/compact-specialist';CACHE=ROOT/'experiments/.cache/bert-small'
MODEL='google/bert_uncased_L-4_H-256_A-4';REV='387825ce42dbb39b87911cdf8e383ee3b25184f8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,d):(OUT/name).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
class Small(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=AutoModel.from_pretrained(CACHE,local_files_only=True,add_pooling_layer=False,attn_implementation='eager')
        self.heads=torch.nn.ModuleDict({str(d):torch.nn.Linear(256,2) for d in [2,4]})
    def forward(self,inputs):
        h=self.encoder(**inputs,output_hidden_states=True).hidden_states;mask=inputs['attention_mask'].unsqueeze(-1)
        return {d:self.heads[str(d)]((h[d]*mask).sum(1)/mask.sum(1).clamp_min(1)) for d in [2,4]}
def main():
    assert not (OUT/'result.json').exists(),'Preserve completed run.'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','vocab.txt','pytorch_model.bin']:
        target=CACHE/name
        if not target.exists():
            r=requests.get(f'https://huggingface.co/{MODEL}/resolve/{REV}/{name}',timeout=120);r.raise_for_status();target.write_bytes(r.content)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    parent=json.loads((ROOT/'results/chat-refinement/protocol.json').read_text());old,rows,qt,_=rows_and_tokenizers()
    bt=AutoTokenizer.from_pretrained(CACHE,local_files_only=True)
    protocol=dict(model=MODEL,revision=REV,files={p.name:sha(p) for p in CACHE.iterdir() if p.is_file()},
        source_sha256=sha(Path(__file__)),parent_sha256=sha(ROOT/'results/chat-refinement/protocol.json'),
        splits=parent['splits'],seed=9927,epochs=4,batch=16,lr=0.0001,threads=4,weight_decay=.01,
        variants=['full','joint'],loss='Inverse-frequency weighted human-label CE; full trains final4 only; joint averages layer2 and4 CE. All encoder parameters train.',
        selection='For each variant, epoch and final4 threshold maximize tune balanced accuracy, then fewer toxic misses, then more correct. Joint2 threshold selected on tune after best final4 epoch. Calibration sets asymmetric gate with zero added errors relative to own final4. No selection using evaluation.',
        scope='Exploratory model-size and multi-exit comparison on consumed100 evaluation, not new reliability validation; no isolated training-time claim.')
    write('protocol.json',protocol)
    encoded={};ys={}
    for s,ids in parent['splits'].items():
        encoded[s]=[bt(bounded(rows[rid]['text'],qt),truncation=True,max_length=512) for rid in ids]
        ys[s]=np.array([rows[rid]['label'] for rid in ids])
    def batch(s,ix):return bt.pad([encoded[s][i] for i in ix],padding=True,return_tensors='pt')
    def predict(model,s):
        out={2:[],4:[]};model.eval()
        with torch.inference_mode():
            for start in range(0,len(ys[s]),16):
                logits=model(batch(s,list(range(start,min(start+16,len(ys[s]))))))
                for d in out:out[d].extend(logits[d].softmax(1)[:,1].tolist())
        return {d:np.array(x) for d,x in out.items()}
    def select(prob,y):
        choices=[dict(threshold=t,metrics=metrics(y,prob>=t)) for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]]
        return max(choices,key=lambda c:(c['metrics']['balanced_accuracy'],-c['metrics']['missed_toxic'],c['metrics']['correct']))
    summary={};allrecords=[]
    for variant in protocol['variants']:
        torch.manual_seed(protocol['seed']);random.seed(protocol['seed']);np.random.seed(protocol['seed'])
        model=Small();counts=np.bincount(ys['train'],minlength=2);cw=torch.tensor(1/counts,dtype=torch.float32);cw/=cw.mean()
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=.01)
        best=None;history=[];save=CACHE/(variant+'-selected.pt')
        for epoch in range(1,5):
            model.train();ix=torch.randperm(len(ys['train'])).tolist();losses=[]
            for start in range(0,len(ix),16):
                chosen=ix[start:start+16];optimizer.zero_grad();logits=model(batch('train',chosen));labels=torch.tensor(ys['train'][chosen])
                ds=[4] if variant=='full' else [2,4]
                loss=sum(torch.nn.functional.cross_entropy(logits[d],labels,weight=cw) for d in ds)/len(ds)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
            tuned=select(predict(model,'tune')[4],ys['tune']);rank=(tuned['metrics']['balanced_accuracy'],-tuned['metrics']['missed_toxic'],tuned['metrics']['correct'])
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),selected=tuned));print(json.dumps(dict(variant=variant,**history[-1])),flush=True)
            if best is None or rank>best[0]:best=(rank,epoch,tuned);torch.save(model.state_dict(),save)
        model.load_state_dict(torch.load(save,weights_only=True));probs={s:predict(model,s) for s in ys}
        thresholds={4:best[2]['threshold'],2:select(probs['tune'][2],ys['tune'])['threshold']}
        candidates=[]
        if variant=='joint':
            for low in [-1,0,.001,.005,.01,.02,.05,.1,.2]:
                for high in [.8,.9,.95,.98,.99,.995,.999,1,2]:
                    p=probs['calibration'][2];full=(probs['calibration'][4]>=thresholds[4]).astype(int)
                    accept=(p<=low)|(p>=high);pred=np.where(p<=low,0,np.where(p>=high,1,full));m=metrics(ys['calibration'],pred,full)
                    if m['added_errors']==0:candidates.append(dict(low=low,high=high,coverage=float(accept.mean()),metrics=m))
        gate=max(candidates,key=lambda c:(c['coverage'],c['metrics']['correct'])) if candidates else dict(low=-1,high=2,coverage=0)
        summary[variant]=dict(parameters=sum(p.numel() for p in model.parameters()),history=history,selected_epoch=best[1],thresholds=thresholds,gate=gate,
            selected_weights_sha256=sha(save),splits={})
        for s,p in probs.items():
            full=(p[4]>=thresholds[4]).astype(int);accept=(p[2]<=gate['low'])|(p[2]>=gate['high'])
            routed=np.where(p[2]<=gate['low'],0,np.where(p[2]>=gate['high'],1,full))
            summary[variant]['splits'][s]=dict(full=metrics(ys[s],full),layer2=metrics(ys[s],p[2]>=thresholds[2]),
                cascade=metrics(ys[s],routed,full),early_count=int(accept.sum()),mean_depth=float(np.where(accept,2,4).mean()))
            for i,rid in enumerate(parent['splits'][s]):allrecords.append(dict(variant=variant,split=s,id=rid,label=int(ys[s][i]),
                layer2_probability=float(p[2][i]),layer4_probability=float(p[4][i]),full=int(full[i]),routed=int(routed[i]),early=bool(accept[i])))
    write('predictions.json',allrecords);write('result.json',dict(variants=summary,protocol_sha256=sha(OUT/'protocol.json'),
        runtime='Quality evaluation computed all layers. Depths are policy estimates until actual continuation timing is recorded separately.'))
if __name__=='__main__':main()

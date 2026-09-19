"""Short Qwen adapter, intermediate-loss, distillation and fixed-depth experiments."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,hashlib,random,time,argparse,copy,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/.cache/replay-runtime'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from chat_smoke_common import prepare
OUT=ROOT/'results/chat-smoke-reset';CACHE=ROOT/'experiments/.cache/chat-smoke-reset'
DEPTHS=[6,12,18,24]

def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def metrics(prob,labels):
    pred=prob.argmax(1);tp=int(((pred==1)&(labels==1)).sum());fn=int(((pred==0)&(labels==1)).sum());fp=int(((pred==1)&(labels==0)).sum());tn=int(((pred==0)&(labels==0)).sum())
    return {'n':len(labels),'correct':tp+tn,'missed_toxic':fn,'false_block':fp,'toxic_recall':tp/(tp+fn),'specificity':tn/(tn+fp),'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp))}

class LowRank(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.a=torch.nn.Parameter(torch.empty(4,base.in_features));self.b=torch.nn.Parameter(torch.zeros(base.out_features,4));torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):return self.base(x)+2*torch.nn.functional.linear(torch.nn.functional.linear(x,self.a),self.b)

class Head(torch.nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',mean);self.register_buffer('std',std);self.linear=torch.nn.Linear(896,2)
    def forward(self,x):return self.linear((x-self.mean)/self.std)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['full','joint','distill','fixed12'],required=True);parser.add_argument('--timing',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    common,splits=prepare();variant=args.variant;target=OUT/f'{variant}.json'
    for split,n in [('tune',16),('calibration',32)]:
        chosen={r['id'] for y in [0,1] for r in [r for r in splits[split] if r['label']==y][:n]}
        splits[split]=[r for r in splits[split] if r['id'] in chosen]
    if target.exists() and not args.timing:raise RuntimeError('Preserve completed run; use a new experiment for further tuning.')
    protocol={'common_sha256':hashlib.sha256((ROOT/'results/chat-smoke/protocol.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'variant':variant,
       'seed':73,'rank':4,'alpha':8,'adapter_modules':'q_proj/v_proj in every retained block','steps':32,'batch':4,'lr_adapter':.0005,'lr_head':.001,'weight_decay':.01,'head_initialization':'train384-only frozen-feature linear classifier,5AdamWsteps lr.01 wd.1, inverse-frequencyclassweights',
       'checkpoints':[16,32],'subset_ids':{s:[r['id'] for r in rows] for s,rows in splits.items()},'selection':'Highest tune32 balancedaccuracy at targetdepth24(or12fixed); tie earlier. Other heads are diagnostic, no eval choice.',
       'distillation':'joint CE on humanlabels plus KL(teacherfulladapter probabilities || eachhead probabilities), T2, 0.5CE+0.5KL*T^2; teacher uses same384labels, no evaluationlabels',
       'runtime':'publishedTransformers4.50.3 CPUfloat32 eager,4threads; training may overlap other jobs, not a timing claim','scope':'Exploratory short optimization budget, not a convergence or production-quality test.'}
    pp=OUT/f'{variant}-protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:write(pp.name,protocol)
    tokenizer=AutoTokenizer.from_pretrained(common['qwen_model'],revision=common['qwen_revision'],local_files_only=True,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
    encoded={};labels={}
    for split,rows in splits.items():
        encoded[split]=[];labels[split]=torch.tensor([r['label'] for r in rows])
        for r in rows:
            ids=tokenizer.encode(r['text'],add_special_tokens=False);message=tokenizer.decode(ids[:256],skip_special_tokens=False) if len(ids)>256 else r['text']
            text=tokenizer.apply_chat_template([{'role':'system','content':common['system_prompt']},{'role':'user','content':message}],tokenize=False,add_generation_prompt=True)
            encoded[split].append(tokenizer(text,add_special_tokens=True)['input_ids'])
    def batch(split,indexes):return tokenizer.pad({'input_ids':[encoded[split][i] for i in indexes]},padding=True,return_tensors='pt')
    old=json.loads((ROOT/'results/chat600/protocol.json').read_text());fingerprint=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest();frozen=torch.load(ROOT/f'experiments/.cache/toxicchat/{fingerprint}-train.pt',weights_only=True)
    indexes=[old['splits']['train'].index(r['id']) for r in splits['train']];counts=torch.bincount(labels['train']).float();weights=1/counts;weights/=weights.mean()
    torch.manual_seed(73);heads=torch.nn.ModuleDict();initial={}
    for d in DEPTHS:
        x=frozen[d][indexes];head=Head(x.mean(0),x.std(0).clamp_min(.05));opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1)
        for _ in range(5):opt.zero_grad();torch.nn.functional.cross_entropy(head(x),labels['train'],weight=weights).backward();opt.step()
        heads[str(d)]=head
    del frozen
    model=AutoModelForCausalLM.from_pretrained(common['qwen_model'],revision=common['qwen_revision'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').model
    for p in model.parameters():p.requires_grad_(False)
    if variant=='fixed12':model.layers=torch.nn.ModuleList(list(model.layers[:12]));model.config.num_hidden_layers=12
    torch.manual_seed(73)
    for layer in model.layers:
        for name in ['q_proj','v_proj']:setattr(layer.self_attn,name,LowRank(getattr(layer.self_attn,name)))
    active=[6,12] if variant=='fixed12' else DEPTHS;goal=active[-1];captured={};visited=[]
    def hook(d):
        def save(mod,inp,out):captured[d]=out[0][:,-1,:];visited.append(d)
        return save
    handles=[layer.register_forward_hook(hook(i+1)) for i,layer in enumerate(model.layers)]
    def forward(split,ix):
        captured.clear();visited.clear();out=model(**batch(split,ix),use_cache=False)
        assert visited==list(range(1,len(model.layers)+1))
        # Fixed12 uses raw block12, matching its frozen-head initialization.
        if variant!='fixed12':captured[24]=out.last_hidden_state[:,-1,:]
        return {d:heads[str(d)](captured[d]) for d in active}
    def evaluate(split):
        model.eval();heads.eval();values={d:torch.empty(len(splits[split]),2) for d in active};order=sorted(range(len(splits[split])),key=lambda i:len(encoded[split][i]))
        with torch.inference_mode():
            for start in range(0,len(order),8):
                ix=order[start:start+8]
                for d,logits in forward(split,ix).items():values[d][ix]=logits
        return values
    def pack():return {'adapter':{k:v.detach().clone() for k,v in model.state_dict().items() if k.endswith(('.a','.b'))},'heads':copy.deepcopy(heads.state_dict())}
    def restore(state):
        model.load_state_dict(state['adapter'],strict=False);heads.load_state_dict(state['heads'])
    if args.timing:
        state=torch.load(CACHE/f'{variant}.pt',weights_only=True);restore(state);model.eval();heads.eval();saved=json.loads(target.read_text());references={r['id']:r for r in saved['predictions'] if r['split']=='evaluation'};records=[]
        with torch.inference_mode():
            forward('evaluation',[0])
            for i,row in enumerate(splits['evaluation']):
                start=time.perf_counter();logits=forward('evaluation',[i]);pred=int(logits[goal].argmax(1));ms=(time.perf_counter()-start)*1000
                assert pred==references[row['id']]['heads'][str(goal)]['prediction']
                records.append({'id':row['id'],'prediction':pred,'depth':goal,'executed_layers':list(visited),'ms':ms})
        write(f'{variant}-timing.json',{'scope':'Single100-message isolated warmCPU smoke pass,4threads; not replicatedbenchmark. Includes tokenization-padding from pretokenized inputs, excludes originaltexttokenization andmodel loading.','seconds':sum(r['ms'] for r in records)/1000,'records':records});return
    baseline=evaluate('tune');best=None;history=[];teacher=None
    if variant=='distill':
        t=np.load(OUT/'full-teacher-train.npz',allow_pickle=False);assert list(t['ids'])==[r['id'] for r in splits['train']];teacher=torch.tensor(t['logits'])
    adapters=[p for n,p in model.named_parameters() if p.requires_grad];optimizer=torch.optim.AdamW([{'params':adapters,'lr':.0005},{'params':heads.parameters(),'lr':.001}],weight_decay=.01)
    order=sorted(range(len(splits['train'])),key=lambda i:len(encoded['train'][i]));batches=[order[i:i+4] for i in range(0,len(order),4)];random.Random(73).shuffle(batches)
    started=time.perf_counter();gradient_check=None
    for step in range(1,33):
        model.train();heads.train();ix=batches[(step-1)%len(batches)];optimizer.zero_grad();logits=forward('train',ix);depths=active if variant in ['joint','distill'] else [goal]
        ce=torch.stack([torch.nn.functional.cross_entropy(logits[d],labels['train'][ix],weight=weights) for d in depths]).mean();loss=ce
        if teacher is not None:
            kd=torch.stack([torch.nn.functional.kl_div(torch.log_softmax(logits[d]/2,1),torch.softmax(teacher[ix]/2,1),reduction='batchmean')*4 for d in depths]).mean();loss=.5*ce+.5*kd
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(adapters+list(heads.parameters()),1.)
        if step==2:
            gradient_check={'first_adapter_a':float(model.layers[0].self_attn.q_proj.a.grad.abs().sum()),'first_adapter_b':float(model.layers[0].self_attn.q_proj.b.grad.abs().sum()),'first_v_adapter_a':float(model.layers[0].self_attn.v_proj.a.grad.abs().sum()),'first_v_adapter_b':float(model.layers[0].self_attn.v_proj.b.grad.abs().sum()),'frozen_base_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad)}
            assert gradient_check['first_adapter_a']>0 and gradient_check['first_adapter_b']>0 and gradient_check['first_v_adapter_a']>0 and gradient_check['first_v_adapter_b']>0 and gradient_check['frozen_base_gradients']==0
        optimizer.step()
        if step%8==0:print(f'{variant}: step {step}/32 loss {float(loss.detach()):.4f} elapsed {time.perf_counter()-started:.1f}s',flush=True)
        if step in [16,32]:
            values=evaluate('tune');m=metrics(torch.softmax(values[goal],1),labels['tune']);history.append({'step':step,'tune':m})
            if best is None or m['balanced_accuracy']>best[0]:best=(m['balanced_accuracy'],step,pack())
    restore(best[2]);torch.save(best[2],CACHE/f'{variant}.pt')
    arrays={f'adapter::{k}':v.numpy() for k,v in best[2]['adapter'].items()};arrays.update({f'head::{k}':v.numpy() for k,v in best[2]['heads'].items()});np.savez_compressed(OUT/f'{variant}-weights.npz',**arrays)
    summaries={};predictions=[]
    for split in ['tune','calibration','evaluation']:
        values=evaluate(split);summaries[split]={str(d):metrics(torch.softmax(v,1),labels[split]) for d,v in values.items()}
        for i,row in enumerate(splits[split]):predictions.append({'split':split,'id':row['id'],'label':row['label'],'heads':{str(d):{'prediction':int(v[i].argmax()),'prob_block':float(torch.softmax(v[i],0)[1])} for d,v in values.items()}})
    if variant=='full':np.savez_compressed(OUT/'full-teacher-train.npz',ids=np.array([r['id'] for r in splits['train']]),logits=evaluate('train')[24].numpy())
    write(target.name,{'protocol':protocol,'selected_step':best[1],'training_history':history,'initial_tune':{str(d):metrics(torch.softmax(v,1),labels['tune']) for d,v in baseline.items()},'gradient_checks':gradient_check,'retained_blocks':len(model.layers),'trainable_adapter_parameters':sum(p.numel() for p in adapters),'metrics':summaries,'predictions':predictions,'elapsed_training_and_evaluation_seconds':time.perf_counter()-started,'note':'Concurrent training elapsed time is diagnostic. All evaluation reused and balanced; no acceptedqualityclaim.'})
    print(json.dumps({'variant':variant,'selected_step':best[1],'evaluation':summaries['evaluation']}),flush=True)
    for h in handles:h.remove()

if __name__=='__main__':main()

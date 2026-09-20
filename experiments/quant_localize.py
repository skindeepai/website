"""Development-only Qwen activation drift and selective dynamic INT8 diagnostic."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
import copy,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/.cache/replay-runtime'))
sys.path.insert(0,str(ROOT/'experiments/.cache/tooling'))
import torch
from chat_next_methods import Runtime,original,measures,data
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='2'
OUT=ROOT/'results/quant-localize'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def main():
    assert not (OUT/'protocol.json').exists(),'Preserve sealed diagnostic.'
    OUT.mkdir(parents=True,exist_ok=True)
    parent=json.loads((ROOT/'results/chat-next-methods/protocol.json').read_text())
    _,old,rows=data();ids=parent['quantization_splits']['development'][:32]
    p=dict(source_sha256=sha(__file__),parent_sha256=sha(ROOT/'results/chat-next-methods/protocol.json'),ids=ids,
           model=parent['model'],revision=parent['revision'],system_prompt=parent['system_prompt'],threads=2,
           methods=['all_default','attention_only','mlp_only','first12_only','last12_only','all_per_channel'],
           scope='32 previously inspected development messages, fixed first-half order. No new test, threshold adjustment, training, speed claim or chosen deployment configuration.',
           diagnostic='Identical batch1 token IDs, masks, eager CPU float32 reference. Capture last-token hidden state after each block, before final RMS norm, plus final normalized state. Compare cosine and RMS drift. Fixed original full-depth readout on final normalized state.',
           versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__))
    write('protocol.json',p);rt=Runtime(p);torch.backends.quantized.engine='x86'
    def run(model,rid):
        states={}; hooks=[]
        for d,l in enumerate(model.layers,1):hooks.append(l.register_forward_hook(lambda m,i,o,d=d:states.update({d:o[0][:,-1,:].detach().clone()})))
        try:
            tokens=rt.tokens(rows[rid]);out=model(input_ids=torch.tensor([tokens]),attention_mask=torch.ones(1,len(tokens),dtype=torch.long),use_cache=False)
            states[25]=out.last_hidden_state[:,-1,:].detach().clone()
        finally:
            for h in hooks:h.remove()
        logits=original(rt.heads,24,states[25]);return states,logits
    refs={};records=[];summary={}
    with torch.inference_mode():
        for rid in ids:refs[rid]=run(rt.model,rid)
        y=torch.tensor([rows[r]['label'] for r in ids]); full=torch.tensor([int(refs[r][1].argmax(1)) for r in ids])
        summary['float']=measures(y,full,full)
        for kind in p['methods']:
            qconfig=torch.ao.quantization.per_channel_dynamic_qconfig if kind=='all_per_channel' else torch.ao.quantization.default_dynamic_qconfig
            names=[]
            for name,module in rt.model.named_modules():
                if not isinstance(module,torch.nn.Linear):continue
                layer=int(name.split('.')[1]);accept=(kind.startswith('all_') or (kind=='attention_only' and '.self_attn.' in name) or (kind=='mlp_only' and '.mlp.' in name) or (kind=='first12_only' and layer<12) or (kind=='last12_only' and layer>=12))
                if accept:names.append(name)
            model=torch.ao.quantization.quantize_dynamic(copy.deepcopy(rt.model),{name:qconfig for name in names},inplace=True).eval()
            count=sum(isinstance(m,torch.ao.nn.quantized.dynamic.Linear) for m in model.modules());assert count==len(names)
            predictions=[];drifts={d:[] for d in range(1,26)}
            for rid in ids:
                states,logits=run(model,rid);pred=int(logits.argmax(1));predictions.append(pred);per_layer={}
                for d,state in states.items():
                    ref=refs[rid][0][d]; item=dict(cosine=float(torch.nn.functional.cosine_similarity(state,ref).mean()),rms_delta=float((state-ref).square().mean().sqrt()))
                    drifts[d].append(item);per_layer[str(d)]=item
                records.append(dict(id=rid,method=kind,label=rows[rid]['label'],prediction=pred,float_prediction=int(refs[rid][1].argmax(1)),float_logits=refs[rid][1][0].tolist(),logits=logits[0].tolist(),layers=per_layer))
            summary[kind]=dict(metrics=measures(y,predictions,full),quantized_linear_count=count,quantized_names=names,
                layers={str(d):{key:sum(v[key] for v in values)/len(values) for key in ['cosine','rms_delta']} for d,values in drifts.items()})
            print(json.dumps({kind:summary[kind]['metrics']}),flush=True)
            write('records.json',records);del model
    write('result.json',dict(protocol_sha256=sha(OUT/'protocol.json'),n=len(ids),methods=summary,scope=p['scope']))

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()

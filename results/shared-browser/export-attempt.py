"""Export shared prefix/suffix and an unsplit control; consumed-data parity only."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments/.cache/export-runtime'),str(ROOT/'experiments/.cache/replay-runtime')]
import numpy as np
import torch,onnx,onnxruntime as ort
from transformers import AutoModel,AutoTokenizer
from chat_refinement import rows_and_tokenizers,bounded
OUT=ROOT/'results/shared-browser';ASSETS=ROOT/'models/shared-moderation';CACHE=ROOT/'experiments/.cache/bert-small'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
class Small(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=AutoModel.from_pretrained(CACHE,local_files_only=True,add_pooling_layer=False,attn_implementation='eager')
        self.heads=torch.nn.ModuleDict({str(d):torch.nn.Linear(256,2) for d in [2,4]})
def pool(h,m):return (h*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
def mask(m):return (1-m[:,None,None,:].to(torch.float32))*torch.finfo(torch.float32).min
class Prefix(torch.nn.Module):
    def __init__(self,m):super().__init__();self.embeddings=m.encoder.embeddings;self.layers=m.encoder.encoder.layer[:2];self.head=m.heads['2']
    def forward(self,input_ids,attention_mask,token_type_ids):
        h=self.embeddings(input_ids=input_ids,token_type_ids=token_type_ids)
        for layer in self.layers:h=layer(h,attention_mask=mask(attention_mask))[0]
        return h,self.head(pool(h,attention_mask))
class Suffix(torch.nn.Module):
    def __init__(self,m):super().__init__();self.layers=m.encoder.encoder.layer[2:];self.head=m.heads['4']
    def forward(self,hidden_states,attention_mask):
        h=hidden_states
        for layer in self.layers:h=layer(h,attention_mask=mask(attention_mask))[0]
        return self.head(pool(h,attention_mask))
class Full(torch.nn.Module):
    def __init__(self,m):super().__init__();self.encoder=m.encoder;self.head=m.heads['4']
    def forward(self,input_ids,attention_mask,token_type_ids):
        h=self.encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).last_hidden_state
        return self.head(pool(h,attention_mask))
def main():
    assert not (OUT/'result.json').exists(),'Preserve completed result.'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    quality=json.loads((ROOT/'results/compact-specialist/result.json').read_text())['variants']['joint']
    weights=CACHE/'joint-selected.pt';assert sha(weights)==quality['selected_weights_sha256']
    parent=json.loads((ROOT/'results/compact-specialist/protocol.json').read_text())
    protocol=dict(source_sha256=sha(Path(__file__)),weights_sha256=sha(weights),parent_sha256=sha(ROOT/'results/compact-specialist/protocol.json'),evaluation_ids=parent['splits']['evaluation'],
        gate={k:quality['gate'][k] for k in ['low','high']},threshold=quality['thresholds']['4'],threads=2,opset=17,
        architecture='One jointly trained four-layer BERT. Prefix embeds and executes layers1/2; suffix receives hidden states and mask and executes layers3/4. Unsplit full graph is timing control.',
        scope='Consumed100 parity, no retraining or new quality claim. Browser timing predeclared: first50, three counterbalanced paired passes, tokenization through classifier including graph boundary; setup/download excluded.',
        tolerance=dict(atol=.0001,rtol=.0001),versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,onnx=onnx.__version__,onnxruntime=ort.__version__))
    write(OUT/'protocol.json',protocol)
    m=Small();m.load_state_dict(torch.load(weights,weights_only=True));m.eval();bt=AutoTokenizer.from_pretrained(CACHE,local_files_only=True);bt.save_pretrained(ASSETS)
    sample=bt('Please help me with my account.',return_tensors='pt');names=['input_ids','attention_mask','token_type_ids']
    prefix=Prefix(m).eval();suffix=Suffix(m).eval();full=Full(m).eval()
    with torch.inference_mode():hidden=prefix(*(sample[n] for n in names))[0]
    definitions=[('prefix',prefix,tuple(sample[n] for n in names),names,['hidden_states','logits2']),('suffix',suffix,(hidden,sample['attention_mask']),['hidden_states','attention_mask'],['logits4']),('full',full,tuple(sample[n] for n in names),names,['logits4'])]
    sessions={};options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    for name,wrapper,args,ins,outs in definitions:
        axes={n:{0:'batch',1:'sequence'} for n in ins};axes.update({n:({0:'batch',1:'sequence'} if n=='hidden_states' else {0:'batch'}) for n in outs})
        with torch.inference_mode():torch.onnx.export(wrapper,args,str(ASSETS/(name+'.onnx')),input_names=ins,output_names=outs,opset_version=17,dynamo=False,dynamic_axes=axes)
        onnx.checker.check_model(str(ASSETS/(name+'.onnx')))
        sessions[name]=ort.InferenceSession(str(ASSETS/(name+'.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
    old,rows,qt,_=rows_and_tokenizers();expected={r['id']:r for r in json.loads((ROOT/'results/compact-specialist/predictions.json').read_text()) if r['variant']=='joint' and r['split']=='evaluation'};records=[]
    for rid in protocol['evaluation_ids']:
        row=rows[rid];inputs=bt(bounded(row['text'],qt),truncation=True,max_length=512,return_tensors='pt');feeds={k:v.numpy() for k,v in inputs.items()}
        h,l2=sessions['prefix'].run(None,feeds);l4=sessions['suffix'].run(None,dict(hidden_states=h,attention_mask=feeds['attention_mask']));l4=l4[0];lf=sessions['full'].run(None,feeds)[0]
        with torch.inference_mode():ref=full(*(inputs[n] for n in names)).numpy()
        assert np.allclose(ref,l4,**protocol['tolerance']) and np.allclose(ref,lf,**protocol['tolerance'])
        p2=float(torch.tensor(l2).softmax(1)[0,1]);p4=float(torch.tensor(l4).softmax(1)[0,1]);early=p2<=protocol['gate']['low'] or p2>=protocol['gate']['high'];pred=int(p2>=protocol['gate']['high']) if early else int(p4>=protocol['threshold']);fp=int(p4>=protocol['threshold'])
        assert (pred,fp,early)==(expected[rid]['routed'],expected[rid]['full'],expected[rid]['early'])
        records.append(dict(id=rid,label=row['label'],prediction=pred,full_prediction=fp,early=early,layer2_probability=p2,layer4_probability=p4,max_logit_delta=float(max(np.abs(ref-l4).max(),np.abs(ref-lf).max())),input_ids=inputs['input_ids'][0].tolist()))
    models={n:dict(bytes=(ASSETS/(n+'.onnx')).stat().st_size,sha256=sha(ASSETS/(n+'.onnx'))) for n in sessions}
    result=dict(n=100,correct=sum(r['prediction']==r['label'] for r in records),early_count=sum(r['early'] for r in records),all_decisions_and_exits_match=True,max_logit_delta=max(r['max_logit_delta'] for r in records),models=models,protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'records.json',records);write(OUT/'result.json',result)
    write(ASSETS/'manifest.json',dict(model='Google BERT four-layer jointly trained moderation specialist',layers=4,parameters=sum(p.numel() for p in m.parameters()),gate=protocol['gate'],threshold=protocol['threshold'],models=models,evaluation=records,qwen_tokenizer=dict(model=old['model'],revision=old['model_revision']),dataset=dict(url='https://huggingface.co/datasets/lmsys/toxic-chat/resolve/29df8e4dba60e1f4af4b4075c0705c5b313548a8/data/0124/toxic-chat_annotation_test.csv',sha256=old['data_sha256']['test'])))
    (ASSETS/'NOTICE.md').write_text('Base model: google/bert_uncased_L-4_H-256_A-4 revision387825ce42dbb39b87911cdf8e383ee3b25184f8, Apache-2.0. Joint training used ToxicChat0124 (CC BY-NC 4.0); research artifact for noncommercial research. Public evaluation input token IDs are reversible, not anonymized. See ../../docs/shared-browser.md.\n',encoding='utf-8')
    (ASSETS/'LICENSE').write_bytes((ROOT/'models/moderation-tiny/LICENSE').read_bytes())
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

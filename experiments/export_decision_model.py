"""Export the frozen two-output specialist; verify runtime parity on consumed data."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[key]='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/.cache/export-runtime'))
sys.path.insert(0,str(ROOT/'experiments/.cache/replay-runtime'))
import numpy as np
import torch
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer
from chat_smoke_specialist import Specialist,CACHE,metrics
from chat_refinement import rows_and_tokenizers,bounded
OUT=ROOT/'results/decision-export'
ASSETS=ROOT/'models/moderation-tiny'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
class OutputOnly(torch.nn.Module):
    def __init__(self,model): super().__init__();self.model=model
    def forward(self,input_ids,attention_mask,token_type_ids):
        return self.model(dict(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids))
def main():
    assert not (OUT/'result.json').exists(),'Preserve completed export evaluation.'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    fit=json.loads((ROOT/'results/chat-refinement/fit.json').read_text())
    weights=ROOT/'results/chat-refinement/specialist.npz'
    assert sha(weights)==fit['weights_sha256']
    p=json.loads((ROOT/'results/chat-refinement/protocol.json').read_text())
    protocol=dict(source_weights_sha256=sha(weights),source_protocol_sha256=sha(ROOT/'results/chat-refinement/protocol.json'),
        script_sha256=sha(Path(__file__)),evaluation_ids=p['splits']['evaluation'],threshold=fit['standalone_threshold'],
        scope='Runtime/export equivalence on the consumed 100-message refinement sample; no retraining or new quality validation.',
        architecture='2-layer BERT, masked mean pooling, 128-to-2 classifier. No vocabulary projection or token generation.',
        tolerance=dict(atol=0.0001,rtol=0.0001),threads=2,opset=17,
        versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,onnx=onnx.__version__,onnxruntime=ort.__version__))
    write(OUT/'protocol.json',protocol)
    model=Specialist(); arr=np.load(weights,allow_pickle=False)
    model.load_state_dict({k:torch.from_numpy(arr[k]) for k in arr.files});model.eval()
    wrapper=OutputOnly(model).eval();bt=AutoTokenizer.from_pretrained(CACHE,local_files_only=True)
    bt.save_pretrained(ASSETS)
    sample=bt('Please help me with my account.',return_tensors='pt')
    with torch.inference_mode():
        torch.onnx.export(wrapper,tuple(sample[n] for n in ['input_ids','attention_mask','token_type_ids']),str(ASSETS/'model.onnx'),
            input_names=['input_ids','attention_mask','token_type_ids'],output_names=['logits'],opset_version=17,dynamo=False,
            dynamic_axes={n:{0:'batch',1:'sequence'} for n in ['input_ids','attention_mask','token_type_ids']}|{'logits':{0:'batch'}})
    onnx.checker.check_model(str(ASSETS/'model.onnx'))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(ASSETS/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    old,rows,qt,_=rows_and_tokenizers(); records=[]
    expected={r['id']:r for r in json.loads((ROOT/'results/chat-refinement/predictions.json').read_text()) if r['path']=='new_bert'}
    with torch.inference_mode():
        for rid in protocol['evaluation_ids']:
            row=rows[rid];text=bounded(row['text'],qt);inputs=bt(text,truncation=True,max_length=512,return_tensors='pt')
            ref=model(inputs).numpy();out=session.run(None,{k:v.numpy() for k,v in inputs.items()})[0]
            assert np.allclose(ref,out,**protocol['tolerance']),rid
            prob=float(torch.tensor(out).softmax(1)[0,1]);pred=int(prob>=protocol['threshold'])
            assert pred==expected[rid]['prediction'],rid
            records.append(dict(id=rid,label=row['label'],prediction=pred,block_probability=prob,
                                max_logit_delta=float(np.abs(ref-out).max()),input_ids=inputs['input_ids'][0].tolist()))
    result=dict(n=len(records),metrics=metrics([r['label'] for r in records],[r['prediction'] for r in records]),
        all_predictions_match=True,max_logit_delta=max(r['max_logit_delta'] for r in records),model_bytes=(ASSETS/'model.onnx').stat().st_size,
        model_sha256=sha(ASSETS/'model.onnx'),parameter_count=sum(v.numel() for v in model.parameters()),
        timing='This export check does not benchmark speed. Browser timing is recorded separately.',protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'records.json',records);write(OUT/'result.json',result)
    manifest=dict(model='Google BERT tiny, trained moderation specialist',layers=2,parameters=result['parameter_count'],threshold=protocol['threshold'],
        model_sha256=result['model_sha256'],model_bytes=result['model_bytes'],qwen_tokenizer=dict(model=old['model'],revision=old['model_revision']),
        dataset=dict(url='https://huggingface.co/datasets/lmsys/toxic-chat/resolve/29df8e4dba60e1f4af4b4075c0705c5b313548a8/data/0124/toxic-chat_annotation_test.csv',sha256=old['data_sha256']['test']),
        evaluation=[dict(id=r['id'],label=r['label'],prediction=r['prediction'],block_probability=r['block_probability'],input_ids=r['input_ids']) for r in records])
    write(ASSETS/'manifest.json',manifest)
    (ASSETS/'NOTICE.md').write_text('Base model: google/bert_uncased_L-2_H-128_A-2, Apache-2.0. Trained with ToxicChat0124 (CC BY-NC 4.0); this research artifact is offered for noncommercial research. Model provenance and evaluation: ../../docs/decision-export.md. No production moderation guarantee.\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

"""Additional browser assets; preserves every earlier model and result."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from export_shared_model import *
from onnxruntime.quantization import quantize_dynamic,QuantType

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    dest=ROOT/'models/method-lab';out=ROOT/'results/method-lab';dest.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    assert not (out/'export.json').exists(),'Preserve completed export.'
    fit=json.loads((ROOT/'results/compact-next/fit.json').read_text());key=fit['selected'];c=fit['candidates'][key]
    weights=ROOT/'experiments/.cache/compact-next'/(key+'.pt');assert sha(weights)==c['weights_sha256']
    write(out/'export-protocol.json',dict(source_sha256=sha(Path(__file__)),selected=key,weights_sha256=sha(weights),fit_sha256=sha(ROOT/'results/compact-next/fit.json'),threads=2,scope='Browser export and consumed100 parity, not fresh research. Quantization of BERT is a new illustrative conversion, not the Qwen diagnostic.',quantization='ONNX Runtime dynamic INT8 MatMul weights, per_channel=True, original four-layer BERT full graph. No threshold tuning.'))
    m=Small();m.load_state_dict(torch.load(weights,weights_only=True));m.eval();bt=AutoTokenizer.from_pretrained(CACHE,local_files_only=True)
    sample=bt('Please help with my account.',return_tensors='pt');names=['input_ids','attention_mask','token_type_ids'];wrapper=Full(m).eval()
    with torch.inference_mode():torch.onnx.export(wrapper,tuple(sample[n] for n in names),str(dest/'trained.onnx'),input_names=names,output_names=['logits4'],opset_version=17,dynamo=False,dynamic_axes={**{n:{0:'batch',1:'sequence'} for n in names},'logits4':{0:'batch'}})
    quantize_dynamic(str(ASSETS/'full.onnx'),str(dest/'int8.onnx'),weight_type=QuantType.QInt8,per_channel=True,op_types_to_quantize=['MatMul'])
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    sessions={n:ort.InferenceSession(str(dest/(n+'.onnx')),sess_options=opts,providers=['CPUExecutionProvider']) for n in ['trained','int8']}
    parent=json.loads((ASSETS/'manifest.json').read_text());rows=[]
    with torch.inference_mode():
        for e in parent['evaluation']:
            ids=np.array([e['input_ids']],dtype=np.int64);inputs=dict(input_ids=ids,attention_mask=np.ones_like(ids),token_type_ids=np.zeros_like(ids))
            logits={n:s.run(None,inputs)[0] for n,s in sessions.items()};ref=wrapper(*(torch.from_numpy(inputs[n]) for n in names)).numpy()
            assert np.allclose(ref,logits['trained'],atol=1e-4,rtol=1e-4)
            rows.append(dict(id=e['id'],label=e['label'],**{n:dict(probability=float(torch.tensor(v).softmax(1)[0,1]),prediction=int(float(torch.tensor(v).softmax(1)[0,1])>=(c['thresholds']['4'] if n=='trained' else parent['threshold']))) for n,v in logits.items()}))
    info={n:dict(bytes=(dest/(n+'.onnx')).stat().st_size,sha256=sha(dest/(n+'.onnx')),threshold=c['thresholds']['4'] if n=='trained' else parent['threshold']) for n in sessions}
    write(dest/'manifest.json',dict(models=info,selected=key,fixed2_threshold=.2,cascade=dict(low=.05,high=.95,scope='Illustrative conservative cutoff for the exported tiny model with four-layer BERT fallback; not the published tiny-to-Qwen cascade and not calibrated for this pairing.'),evaluation=rows))
    write(out/'export.json',dict(n=len(rows),trained_onnx_matches_pytorch=True,models=info,correct={n:sum(e[n]['prediction']==e['label'] for e in rows) for n in sessions},protocol_sha256=sha(out/'export-protocol.json')))
    (dest/'LICENSE.txt').write_bytes((ASSETS/'LICENSE.txt').read_bytes());(dest/'NOTICE.md').write_text('Apache-2.0 BERT models trained on ToxicChat0124 (CC BY-NC 4.0), noncommercial research artifacts. See ../../docs/method-demos.md.\n',encoding='utf-8')
    print(json.dumps(read_json(out/'export.json')))

def read_json(p):return json.loads(p.read_text())
if __name__=='__main__':main()


"""Execute the selected shared BERT model with real stop/continue control."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
import json,time,hashlib
from pathlib import Path
from compact_specialist import ROOT,CACHE,Small,rows_and_tokenizers,bounded,metrics
import torch
from transformers import AutoTokenizer
OUT=ROOT/'results/compact-specialist'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(OUT/n).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
class Stop(Exception):
    def __init__(self,label):self.label=label
def main():
    assert not (OUT/'runtime.json').exists(),'Preserve completed timing.'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    p=json.loads((OUT/'protocol.json').read_text());r=json.loads((OUT/'result.json').read_text());joint=r['variants']['joint']
    weights=CACHE/'joint-selected.pt';assert sha(weights)==joint['selected_weights_sha256']
    protocol=dict(source_sha256=sha(Path(__file__)),quality_sha256=sha(OUT/'result.json'),weights_sha256=sha(weights),
        ids=p['splits']['evaluation'][:50],passes=3,threads=2,paths=['full','layer2','adaptive'],
        boundary='Raw Qwen bounding, BERTtokenization, actualblockexecution, pooling/classifier/gate/hooks; loading/warmup excluded. Rotate methods by sample+pass. No other model jobs.',
        scope='Consumed50 diagnostic, same selected joint-trained backbone; early paths never recompute embeddings or lower layers.')
    write('runtime-protocol.json',protocol)
    model=Small();model.load_state_dict(torch.load(weights,weights_only=True));model.eval();bt=AutoTokenizer.from_pretrained(CACHE,local_files_only=True)
    old,rows,qt,_=rows_and_tokenizers();expected={v['id']:v for v in json.loads((OUT/'predictions.json').read_text()) if v['variant']=='joint' and v['split']=='evaluation'}
    records=[]
    def execute(rid,path):
        start=time.perf_counter();inputs=bt(bounded(rows[rid]['text'],qt),truncation=True,max_length=512,return_tensors='pt');mask=inputs['attention_mask'].unsqueeze(-1);visited=[];checked=False
        def classify(h,d):return float(model.heads[str(d)]((h*mask).sum(1)/mask.sum(1).clamp_min(1)).softmax(1)[0,1])
        def hook(d):
            def after(m,i,o):
                nonlocal checked
                visited.append(d)
                if d==2 and path!='full':
                    checked=True;prob=classify(o[0],2);g=joint['gate']
                    if path=='layer2':raise Stop(int(prob>=joint['thresholds']['2']))
                    if prob<=g['low']:raise Stop(0)
                    if prob>=g['high']:raise Stop(1)
            return after
        hooks=[l.register_forward_hook(hook(d)) for d,l in enumerate(model.encoder.encoder.layer,1)]
        try:h=model.encoder(**inputs).last_hidden_state;label=int(classify(h,4)>=joint['thresholds']['4'])
        except Stop as e:label=e.label
        finally:
            for handle in hooks:handle.remove()
        elapsed=time.perf_counter()-start;ref=expected[rid]
        expected_label=ref['full'] if path=='full' else ref['routed'] if path=='adaptive' else int(ref['layer2_probability']>=joint['thresholds']['2'])
        depth=4 if path=='full' else 2 if path=='layer2' or ref['early'] else 4
        assert label==expected_label and visited==list(range(1,depth+1)),(rid,path,visited,label,expected_label)
        return dict(id=rid,path=path,label=rows[rid]['label'],prediction=label,executed_layers=visited,checked_after2=checked,seconds=elapsed)
    with torch.inference_mode():
        for path in protocol['paths']:execute(protocol['ids'][0],path)
        for repeat in range(3):
            for i,rid in enumerate(protocol['ids']):
                offset=(i+repeat)%3
                for path in protocol['paths'][offset:]+protocol['paths'][:offset]:records.append(dict(repeat=repeat,**execute(rid,path)))
    summary={}
    for path in protocol['paths']:
        values=[x for x in records if x['path']==path];first=[x for x in values if x['repeat']==0]
        summary[path]=dict(seconds=[sum(x['seconds'] for x in values if x['repeat']==r) for r in range(3)],
            correct=sum(x['label']==x['prediction'] for x in first),exits={str(d):sum(len(x['executed_layers'])==d for x in first) for d in [2,4]},
            mean_depth=sum(len(x['executed_layers']) for x in first)/len(first))
    write('runtime-records.json',records);write('runtime.json',dict(n=50,passes=3,paths=summary,all_prediction_and_layer_checks_passed=True,
        protocol_sha256=sha(OUT/'runtime-protocol.json')));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

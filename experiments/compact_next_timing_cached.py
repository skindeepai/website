"""Follow-up timing with learned-gate arrays loaded once before inference.

Run after compact_next evaluate and after all other launched model work finishes.
"""
import time
import compact_next as study
from compact_next import torch,np,ROOT,OUT,LOCAL,Small,CACHE,sha,read,write,metrics,gate_risk

class Stop(Exception):
    def __init__(self,label): self.label=label

def main():
    study.guard('timing-cached.json'); p,old,rows,qt,bt=study.load();fit=read(OUT/'fit.json');quality=read(OUT/'result.json')
    key=fit['selected'];c=fit['candidates'][key];weights=LOCAL/(key+'.pt')
    assert sha(weights)==c['weights_sha256'];assert sha(OUT/'fit.json')==quality['fit_sha256']
    model=Small();model.load_state_dict(torch.load(weights,weights_only=True));model.eval()
    baseline=read(OUT/'frozen-original.json')['configuration'];assert sha(CACHE/'joint-selected.pt')==p['original_weights_sha256']
    original_model=Small();original_model.load_state_dict(torch.load(CACHE/'joint-selected.pt',weights_only=True));original_model.eval()
    models={key:model,'original':original_model}
    configs={key:c,'original':dict(thresholds=baseline['thresholds'],gates={'confidence':baseline['gate']})}
    assert sha(OUT/(key+'-gate.npz'))==c['gate_sha256']
    with np.load(OUT/(key+'-gate.npz')) as saved:
        state={name:saved[name] for name in saved.files}
    ids=p['splits']['fresh'][:50]
    expected={(r['candidate'],r['id']):r for r in read(OUT/'predictions.json')}
    paths=['original_full','original_confidence','full','confidence','risk'];records=[]
    protocol=dict(source_sha256=sha(__file__),quality_sha256=sha(OUT/'result.json'),ids=ids,paths=paths,passes=3,threads=2,
        boundary='Raw input bounding, BERT tokenization, hooks, actual block execution, pooling/readout, and learned PCA/scaler/logistic gate. Loading/warmup excluded. Rotate all five methods per message+pass.',
        isolation='Run only after other launched model jobs finish; CPU2. No lower block or embedding re-execution.',
        selected=key,
        parent_protocol_sha256=sha(OUT/'timing-protocol.json'),
        parent_result_sha256=sha(OUT/'timing.json'),
        failed_copy_attempt_sha256=sha(OUT/'timing-cache-copy-attempt.json'),
        amendment='Materialize the six saved gate arrays once before timing. The previous runner repeatedly decompressed its lazy NPZ archive inside the risk check. Same model, examples, thresholds, routes, and five-path timing protocol.')
    study.guard('timing-cached-protocol.json');write('timing-cached-protocol.json',protocol)
    def execute(rid,path):
        candidate='original' if path.startswith('original_') else key
        mode=path.removeprefix('original_');model=models[candidate];c=configs[candidate]
        start=time.perf_counter();inputs=bt(study.bounded(rows[rid]['text'],qt),truncation=True,max_length=512,return_tensors='pt')
        mask=inputs['attention_mask'].unsqueeze(-1);visited=[];score=None
        def pooled(h):return (h*mask).sum(1)/mask.sum(1).clamp_min(1)
        def hook(d):
            def after(module,args,output):
                nonlocal score
                visited.append(d)
                if d!=2 or mode=='full':return
                h=pooled(output[0]);prob=float(model.heads['2'](h).softmax(1)[0,1]);early=int(prob>=c['thresholds']['2']);g=c['gates'][mode]
                if mode=='risk':
                    score=float(gate_risk(dict(p2=np.array([prob]),hidden=h.numpy(),lengths=np.array([int(mask.sum())])),state)[0])
                    accept=score<=(g['high'] if early else g['low'])
                else:accept=prob<=g['low'] or prob>=g['high']
                if accept:raise Stop(early)
            return after
        handles=[layer.register_forward_hook(hook(d)) for d,layer in enumerate(model.encoder.encoder.layer,1)]
        try:
            h=model.encoder(**inputs).last_hidden_state;label=int(float(model.heads['4'](pooled(h)).softmax(1)[0,1])>=c['thresholds']['4'])
        except Stop as stop:label=stop.label
        finally:
            for handle in handles:handle.remove()
        seconds=time.perf_counter()-start;ref=expected[(candidate,rid)]
        assert label==ref['predictions'][mode] and visited==list(range(1,ref['depths'][mode]+1)),(rid,path,label,visited)
        return dict(id=rid,path=path,label=rows[rid]['label'],prediction=label,executed_layers=visited,risk=score,seconds=seconds)
    with torch.inference_mode():
        for path in paths:execute(ids[0],path)
        for repeat in range(3):
            for i,rid in enumerate(ids):
                offset=(i+repeat)%len(paths)
                for path in paths[offset:]+paths[:offset]:records.append(dict(repeat=repeat,**execute(rid,path)))
    prior={(r['repeat'],r['id'],r['path']):r for r in read(OUT/'timing-records.json')}
    for row in records:
        old=prior[(row['repeat'],row['id'],row['path'])]
        for field in ['prediction','executed_layers','risk']:
            assert row[field]==old[field],(field,row['id'],row['path'])
    methods={}
    for path in paths:
        first=[r for r in records if r['path']==path and r['repeat']==0]
        totals=[sum(r['seconds'] for r in records if r['path']==path and r['repeat']==repeat) for repeat in range(3)]
        methods[path]=dict(totals_seconds=totals,mean_seconds=float(np.mean(totals)),correct=sum(r['prediction']==r['label'] for r in first),early_count=sum(len(r['executed_layers'])==2 for r in first))
    write('timing-cached-records.json',records);write('timing-cached.json',dict(protocol_sha256=sha(OUT/'timing-cached-protocol.json'),n=50,methods=methods,all_prediction_and_layer_traces_match=True,all_750_outputs_depths_and_risk_scores_match_parent=True))
    print(study.json.dumps(methods),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()

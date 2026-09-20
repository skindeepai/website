"""Independent arithmetic checks and baseline-configuration integrity."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/compact-next'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
def measures(y,p,full):
    return dict(n=len(y),correct=sum(a==b for a,b in zip(y,p)),missed_toxic=sum(a==1 and b==0 for a,b in zip(y,p)),
        false_block=sum(a==0 and b==1 for a,b in zip(y,p)),added_errors=sum(a==c and a!=b for a,b,c in zip(y,p,full)),
        corrected_errors=sum(a!=c and a==b for a,b,c in zip(y,p,full)),additional_missed_toxic=sum(a==1 and c==1 and b==0 for a,b,c in zip(y,p,full)))
def preflight():
    p=read(OUT/'protocol.json');path=ROOT/'results/compact-specialist/result.json'
    prior_timing=read(ROOT/'results/compact-specialist/runtime-protocol.json')
    assert sha(path)==prior_timing['quality_sha256']
    assert sha(ROOT/'experiments/compact_next.py')==p['source_sha256']
    fit=read(OUT/'fit.json');assert fit['protocol_sha256']==sha(OUT/'protocol.json')
    assert not (OUT/'result.json').exists(),'Preflight must precede fresh evaluation.'
    write('frozen-original.json',dict(source_sha256=sha(path),configuration=read(path)['variants']['joint']))
    splits=p['splits'];sets={k:set(v) for k,v in splits.items()}
    assert len(sets['fresh'])==500
    for a in sets:
        assert len(sets[a])==len(splits[a])
        for b in sets:
            if a!=b:assert not sets[a]&sets[b]
    assert set(fit['candidates'])=={f'{kind}-{seed}' for kind in p['variants'] for seed in p['seeds']}
    def rank(k):
        m=fit['candidates'][k]['selected']['selected']['metrics'];return m['balanced_accuracy'],-m['missed_toxic'],m['correct']
    assert fit['selected']==max(fit['candidates'],key=rank)
    gate_checks=0
    for key,c in fit['candidates'].items():
        assert sha(OUT/(key+'-gate.npz'))==c['gate_sha256']
        dev=[v for v in read(OUT/(key+'-development.json')) if v['split']=='calibration']
        assert [v['id'] for v in dev]==splits['calibration']
        y=[v['label'] for v in dev];full=[int(v['p4']>=c['thresholds']['4']) for v in dev];early=[int(v['p2']>=c['thresholds']['2']) for v in dev]
        for kind,g in c['gates'].items():
            if kind=='risk':accepted=[s<=(g['high'] if e else g['low']) for s,e in zip(c['gate_info']['risk_calibration'],early)]
            else:accepted=[v['p2']<=g['low'] or v['p2']>=g['high'] for v in dev]
            pred=[e if a else f for e,a,f in zip(early,accepted,full)]
            m=measures(y,pred,full)
            assert all(m[k]==g['metrics'][k] for k in m)
            assert m['added_errors']==0 and sum(accepted)==g['accepted'];gate_checks+=1
    write('preflight-audit.json',dict(source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),fit_sha256=sha(OUT/'fit.json'),selected=fit['selected'],baseline_configuration_pinned=True,disjoint_ids=True,development_gates_recomputed=gate_checks))
    print('Preflight passed, no fresh predictions read.')
def audit():
    result=read(OUT/'result.json');rows=read(OUT/'predictions.json');protocol=read(OUT/'protocol.json')
    assert sha(OUT/'fit.json')==result['fit_sha256']
    fit=read(OUT/'fit.json');original=read(OUT/'frozen-original.json')['configuration']
    checks=0
    for key,paths in result['methods'].items():
        selected=[r for r in rows if r['candidate']==key]
        assert [r['id'] for r in selected]==protocol['splits']['fresh']
        y=[r['label'] for r in selected];full=[r['predictions']['full'] for r in selected]
        for path,summary in paths.items():
            predictions=[r['predictions'][path] for r in selected];m=measures(y,predictions,full)
            assert all(summary[k]==m[k] for k in m),(key,path)
            depth=[r['depths'][path] for r in selected];early=sum(d==2 for d in depth)
            assert summary['early_count']==early and math.isclose(summary['blocks_skipped'],early/1000)
            assert math.isclose(summary['mean_depth'],sum(depth)/500);checks+=1
            k=summary['added_errors'];upper=summary['added_error_upper95'];n=len(y)
            if k==n:assert upper==1.
            else:
                cdf=sum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)+j*math.log(upper)+(n-j)*math.log1p(-upper)) for j in range(k+1))
                assert math.isclose(cdf,.05,abs_tol=1e-9),(key,path,upper,cdf)
        c=original if key=='original' else fit['candidates'][key]
        for r in selected:
            full=int(r['p4']>=c['thresholds']['4']);early=int(r['p2']>=c['thresholds']['2'])
            assert r['predictions']['full']==full and r['predictions']['layer2']==early
            if 'confidence' in r['predictions']:
                g=c['gate'] if key=='original' else c['gates']['confidence']
                accept=r['p2']<=g['low'] or r['p2']>=g['high']
                assert r['predictions']['confidence']==(early if accept else full)
                assert r['depths']['confidence']==(2 if accept else 4)
                if key=='original':assert r['predictions']['confidence']==(0 if r['p2']<=g['low'] else 1 if r['p2']>=g['high'] else full)
            if 'risk' in r['predictions']:
                g=c['gates']['risk'];assert 0<=r['risk']<=1
                accept=r['risk']<=(g['high'] if early else g['low'])
                assert r['predictions']['risk']==(early if accept else full)
                assert r['depths']['risk']==(2 if accept else 4)
    write('audit.json',dict(source_sha256=sha(__file__),quality_sha256=sha(OUT/'result.json'),prediction_sha256=sha(OUT/'predictions.json'),method_paths_recomputed=checks,records=len(rows),all_passed=True))
    print('Fresh quality arithmetic passed:',checks,'paths')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','audit']);args=parser.parse_args();globals()[args.stage]()

"""Fresh, label-unstratified validation of training and learned continuation.

prepare -> fit -> evaluate. Existing results are never overwritten. Two CPU threads.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, hashlib, json, random, re, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/.cache/replay-runtime'))
sys.path.insert(0, str(ROOT/'experiments/.cache/tooling'))
from compact_specialist import Small, CACHE, rows_and_tokenizers, bounded, metrics
from chat_next_methods import cached, frozen_heads, original
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import beta
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
OUT = ROOT/'results/compact-next'
LOCAL = ROOT/'experiments/.cache/compact-next'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
def guard(name): assert not (OUT/name).exists(), 'Preserve '+name

def prepare():
    guard('protocol.json')
    old, rows, qt, _ = rows_and_tokenizers()
    bt = AutoTokenizer.from_pretrained(CACHE, local_files_only=True)
    parent = read(ROOT/'results/compact-specialist/protocol.json')
    # Conservative exclusion: every ToxicChat-shaped ID already published anywhere.
    used = set(); sources = {}
    for path in sorted((ROOT/'results').rglob('*.json')):
        if OUT in path.parents or path.name == 'provenance.json': continue
        ids = set(re.findall(r'"((?:train|test):\d+)"', path.read_text(encoding='utf-8')))
        ids &= set(rows)
        if ids: used.update(ids); sources[str(path.relative_to(ROOT)).replace('\\','/')] = sha(path)
    used.update(rid for rid in rows if rid.startswith('train:'))
    def keys(rid):
        row = rows[rid]; text = bounded(row['text'], qt)
        return [row['text'].strip().casefold(), row['conversation'],
                json.dumps(qt.encode(text, add_special_tokens=False)),
                json.dumps(bt(text, truncation=True, max_length=512)['input_ids'])]
    seen = [set() for _ in range(4)]
    for rid in sorted(used):
        for values, value in zip(seen, keys(rid)): values.add(value)
    pool = []
    for rid in sorted(rows):
        if not rid.startswith('test:') or rid in used: continue
        values = keys(rid)
        if any(value in s for value, s in zip(values, seen)): continue
        pool.append(rid)
        for value, s in zip(values, seen): s.add(value)
    random.Random(720916).shuffle(pool)
    assert len(pool) >= 500, len(pool)
    splits = {s: parent['splits'][s] for s in ['train','tune','calibration']}
    splits['fresh'] = pool[:500]
    p = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
             source_data_sha256=old['data_sha256'], historical_artifacts=sources,
             parent_sha256=sha(ROOT/'results/compact-specialist/protocol.json'),
             original_weights_sha256=sha(CACHE/'joint-selected.pt'), splits=splits,
             eligible_fresh=len(pool), fresh_seed=720916, seeds=[9927, 9928, 9929], epochs=3,
             batch=16, lr=.0001, threads=2, variants=['ce','distill'],
             training='All shared four-layer BERT parameters. Equal CE at layers2/4 with inverse-frequency class weights. Distill uses .75CE + .25KL(T2)*4 from frozen Qwen final classifier on training rows only.',
             selection='Each checkpoint/final threshold maximizes tune balanced accuracy, fewer toxic misses, correct count, first tie. Choose one model across both variants/seeds with same criterion before fresh evaluation.',
             learned_gate='Fit on tune500, out-of-sample for backbone weights (but reused for epoch selection). PCA16 of layer2 pooled states plus probability, entropy and input length. Balanced L2 logistic C1 predicts early wrong AND full correct. Separate calibration796 selects class-specific maximum risk thresholds with zero added errors versus own full answer. Never uses full state at runtime.',
             confidence_gate='Separate calibration796 selects SAFE/BLOCK probability extremes with zero added errors. No test tuning.',
             validation='500 label-unstratified human-annotated official test rows, excluding all historical published IDs, all training text, conversations and duplicate effective Qwen/BERT inputs. This estimates the eligible annotated remainder, not real deployment prevalence. No texts published.',
             evaluation='Report original frozen joint model and gates; selected new candidate and both gates; every seed final/early answer. Count new errors and corrected errors separately, one-sided95% upper bounds. Quality policy simulation, not timing.',
             limits=['Development reused historically; local audit is not external preregistration.', 'Three seeds are exploratory, not exhaustive tuning.', 'Public corpus may occur in pretraining.', 'Gate fit uses held-out backbone examples, not fully nested cross-validation: epoch selection also uses tune. Independent calibration and fresh test remain separate.'])
    write('protocol.json', p)
    print(json.dumps({'fresh':500,'eligible':len(pool),'seeds':p['seeds']}), flush=True)

def load():
    p = read(OUT/'protocol.json'); assert p['source_sha256'] == sha(__file__)
    old, rows, qt, _ = rows_and_tokenizers()
    assert old['data_sha256'] == p['source_data_sha256']
    bt = AutoTokenizer.from_pretrained(CACHE, local_files_only=True)
    return p, old, rows, qt, bt

def encode(ids, rows, qt, bt):
    return [bt(bounded(rows[rid]['text'], qt), truncation=True, max_length=512) for rid in ids]

def predict(model, encoded, bt):
    probs = {2:[],4:[]}; pooled = []; lengths = []; model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded), 16):
            inputs = bt.pad(encoded[start:start+16], padding=True, return_tensors='pt')
            states = model.encoder(**inputs, output_hidden_states=True).hidden_states
            mask = inputs['attention_mask'].unsqueeze(-1)
            for d in [2,4]:
                h = (states[d]*mask).sum(1)/mask.sum(1).clamp_min(1)
                probs[d].extend(model.heads[str(d)](h).softmax(1)[:,1].tolist())
                if d == 2: pooled.extend(h.tolist())
            lengths.extend(inputs['attention_mask'].sum(1).tolist())
    return dict(p2=np.array(probs[2]), p4=np.array(probs[4]), hidden=np.array(pooled), lengths=np.array(lengths))

def choose(prob, y):
    options = [dict(threshold=t, metrics=metrics(y, prob>=t)) for t in [.1,.2,.3,.4,.5,.6,.7,.8,.9]]
    return max(options, key=rank)
def rank(c):
    m = c['metrics']; return (m['balanced_accuracy'], -m['missed_toxic'], m['correct'])
def features(values, pca):
    p = np.clip(values['p2'], 1e-8, 1-1e-8)
    return np.column_stack([pca.transform(values['hidden']), p, -(p*np.log(p)+(1-p)*np.log(1-p)), values['lengths']/512])

def fit_gate(tune, calibration, yt, yc, thresholds):
    t2, t4 = thresholds['2'], thresholds['4']
    early_t, full_t = tune['p2']>=t2, tune['p4']>=t4
    harm = ((early_t != yt)&(full_t == yt)).astype(int)
    pca = PCA(n_components=16, svd_solver='full').fit(tune['hidden'])
    scaler = StandardScaler().fit(features(tune,pca))
    classifier = LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=720916).fit(scaler.transform(features(tune,pca)), harm)
    risk = classifier.predict_proba(scaler.transform(features(calibration,pca)))[:,1]
    early, full = calibration['p2']>=t2, calibration['p4']>=t4
    gates = {}
    for kind in ['confidence','risk']:
        candidates = []
        grid = [-1,0,.001,.005,.01,.02,.05,.1,.2,.3,.5,.7,1.01] if kind == 'risk' else [-1,0,.001,.005,.01,.02,.05,.1,.2,.3,.4]
        second = grid if kind == 'risk' else [.6,.7,.8,.9,.95,.98,.99,.995,.999,1,2]
        for low in grid:
            for high in second:
                accepted = np.where(early, risk<=high, risk<=low) if kind=='risk' else (calibration['p2']<=low)|(calibration['p2']>=high)
                pred = np.where(accepted, early, full)
                m = metrics(yc,pred,full)
                if m['added_errors']==0:
                    candidates.append(dict(low=low,high=high,accepted=int(accepted.sum()),metrics=m))
        gates[kind] = max(candidates, key=lambda x:(x['accepted'], x['metrics']['correct']))
    state = dict(pca_mean=pca.mean_,pca_components=pca.components_,scale_mean=scaler.mean_,scale_std=scaler.scale_,weight=classifier.coef_[0],bias=classifier.intercept_[0])
    return gates,state,dict(tune_harm_examples=int(harm.sum()),tune_n=len(yt),risk_calibration=risk.tolist())

def fit():
    guard('fit.json'); p, old, rows, qt, bt = load(); LOCAL.mkdir(parents=True,exist_ok=True)
    encoded = {s:encode(p['splits'][s],rows,qt,bt) for s in ['train','tune','calibration']}
    ys = {s:np.array([rows[r]['label'] for r in p['splits'][s]]) for s in encoded}
    teacher = original(frozen_heads(),24,cached(old,p['splits']['train'])[24]).detach()
    cw = torch.tensor(1/np.bincount(ys['train'],minlength=2),dtype=torch.float32); cw /= cw.mean()
    candidates = {}
    for kind in p['variants']:
        for seed in p['seeds']:
            key = f'{kind}-{seed}'; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = Small(); optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=.01)
            best = None; history = []; save = LOCAL/(key+'.pt')
            for epoch in range(1,p['epochs']+1):
                model.train(); order=torch.randperm(len(ys['train'])).tolist(); losses=[]
                for start in range(0,len(order),p['batch']):
                    ix=order[start:start+p['batch']]; inputs=bt.pad([encoded['train'][i] for i in ix],padding=True,return_tensors='pt')
                    optimizer.zero_grad(); logits=model(inputs); y=torch.tensor(ys['train'][ix])
                    ce=sum(torch.nn.functional.cross_entropy(logits[d],y,weight=cw) for d in [2,4])/2
                    loss=ce
                    if kind=='distill':
                        kd=sum(torch.nn.functional.kl_div((logits[d]/2).log_softmax(-1),(teacher[ix]/2).softmax(-1),reduction='batchmean')*4 for d in [2,4])/2
                        loss=.75*ce+.25*kd
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                tuned=choose(predict(model,encoded['tune'],bt)['p4'],ys['tune'])
                entry=dict(epoch=epoch,loss=float(np.mean(losses)),selected=tuned);history.append(entry)
                print(json.dumps(dict(candidate=key,**entry)),flush=True)
                if best is None or rank(tuned)>rank(best['selected']):
                    best=entry;torch.save(model.state_dict(),save)
            model.load_state_dict(torch.load(save,weights_only=True)); tune=predict(model,encoded['tune'],bt)
            calibration=predict(model,encoded['calibration'],bt)
            thresholds={'2':choose(tune['p2'],ys['tune'])['threshold'],'4':best['selected']['threshold']}
            gates,state,gate_info=fit_gate(tune,calibration,ys['tune'],ys['calibration'],thresholds)
            np.savez_compressed(OUT/(key+'-gate.npz'),**state)
            # Numeric development outcomes are public; no message text is copied.
            write(key+'-development.json',[dict(id=rid,split=s,label=int(ys[s][i]),p2=float(v['p2'][i]),p4=float(v['p4'][i])) for s,v in [('tune',tune),('calibration',calibration)] for i,rid in enumerate(p['splits'][s])])
            candidates[key]=dict(kind=kind,seed=seed,history=history,selected=best,thresholds=thresholds,gates=gates,gate_info=gate_info,weights_sha256=sha(save),gate_sha256=sha(OUT/(key+'-gate.npz')))
    chosen=max(candidates,key=lambda k:rank(candidates[k]['selected']['selected']))
    write('fit.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.json'),selected=chosen,candidates=candidates))
    print('Frozen candidate: '+chosen,flush=True)

def gate_risk(v,state):
    p=np.clip(v['p2'],1e-8,1-1e-8)
    z=(v['hidden']-state['pca_mean'])@state['pca_components'].T
    x=np.column_stack([z,p,-(p*np.log(p)+(1-p)*np.log(1-p)),v['lengths']/512])
    logits=((x-state['scale_mean'])/state['scale_std'])@state['weight']+state['bias']
    return 1/(1+np.exp(-np.clip(logits,-500,500)))

def evaluate():
    guard('result.json'); p,old,rows,qt,bt=load(); fit=read(OUT/'fit.json')
    assert fit['protocol_sha256']==sha(OUT/'protocol.json')
    ids=p['splits']['fresh']; encoded=encode(ids,rows,qt,bt);y=np.array([rows[r]['label'] for r in ids])
    result={}; records=[]
    original_config=read(ROOT/'results/compact-specialist/result.json')['variants']['joint']
    configs={'original':dict(thresholds=original_config['thresholds'],gates={'confidence':original_config['gate']},weights_sha256=p['original_weights_sha256'])}
    configs.update(fit['candidates'])
    for key,c in configs.items():
        path=CACHE/'joint-selected.pt' if key=='original' else LOCAL/(key+'.pt')
        assert sha(path)==c['weights_sha256']; model=Small();model.load_state_dict(torch.load(path,weights_only=True));v=predict(model,encoded,bt)
        full=(v['p4']>=c['thresholds']['4']).astype(int);early=(v['p2']>=c['thresholds']['2']).astype(int)
        paths={'full':(full,np.zeros(len(y),dtype=bool)),'layer2':(early,np.ones(len(y),dtype=bool))}
        risks=None
        if key==fit['selected']:
            assert sha(OUT/(key+'-gate.npz'))==c['gate_sha256']; risks=gate_risk(v,np.load(OUT/(key+'-gate.npz')))
        if key in ['original',fit['selected']]:
            for kind,g in c['gates'].items():
                accepted=np.where(early,risks<=g['high'],risks<=g['low']) if kind=='risk' else (v['p2']<=g['low'])|(v['p2']>=g['high'])
                paths[kind]=(np.where(accepted,early,full),accepted)
        result[key]={}
        for name,(pred,accepted) in paths.items():
            m=metrics(y,pred,full); k=m['added_errors']; n=len(y)
            m.update(early_count=int(accepted.sum()),mean_depth=float(np.where(accepted,2,4).mean()),blocks_skipped=float(accepted.mean()/2),added_error_upper95=float(beta.ppf(.95,k+1,n-k)) if k<n else 1.)
            result[key][name]=m
        for i,rid in enumerate(ids):
            records.append(dict(id=rid,candidate=key,label=int(y[i]),p2=float(v['p2'][i]),p4=float(v['p4'][i]),risk=None if risks is None else float(risks[i]),predictions={name:int(vv[0][i]) for name,vv in paths.items()},depths={name:2 if vv[1][i] else 4 for name,vv in paths.items()}))
        print(json.dumps({key:result[key]}),flush=True)
    write('predictions.json',records)
    write('result.json',dict(protocol_sha256=sha(OUT/'protocol.json'),fit_sha256=sha(OUT/'fit.json'),n=len(y),toxic=int(y.sum()),selected=fit['selected'],methods=result,timing='Not measured. Gate depths above are simulated from full-depth runs. Browser actual continuation is a separate study on the original model.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit','evaluate']);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2): globals()[args.stage]()

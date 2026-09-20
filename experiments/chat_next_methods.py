"""Bounded checkpoint-readout, prefix/batch, and int8 readout diagnostics."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, copy, csv, hashlib, json, random, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/.cache/replay-runtime'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

OUT = ROOT/'results/chat-next-methods'
CACHE = ROOT/'experiments/.cache/chat-next-methods'
DEPTHS = [6, 12, 18, 24]


def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, value): (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
def guard(name): assert not (OUT/name).exists(), 'Preserve completed output: '+name


def data():
    parent = read('results/chat-refinement/protocol.json'); old = read('results/chat600/protocol.json'); rows = {}
    for source in ['train', 'test']:
        path = ROOT/f'experiments/.cache/toxicchat/toxic-chat_annotation_{source}.csv'
        assert sha(path) == old['data_sha256'][source]
        for i, r in enumerate(csv.DictReader(path.open(encoding='utf-8', newline=''))):
            if r['human_annotation'] == 'True': rows[f'{source}:{i}'] = {'id': f'{source}:{i}', 'text': r['user_input'], 'label': int(r['toxicity'])}
    return parent, old, rows


def protocol():
    parent, old, rows = data(); rng = random.Random(41903); quant = {}
    for name, source, n in [('train', 'train', 64), ('development', 'tune', 32)]:
        ids = []
        for y in [0, 1]:
            pool = [i for i in parent['splits'][source] if rows[i]['label'] == y]; rng.shuffle(pool); ids += pool[:n]
        rng.shuffle(ids); quant[name] = ids
    paths = ['experiments/chat_next_methods.py', 'results/chat-refinement/protocol.json', 'results/chat-refinement/predictions.json', 'results/chat600/protocol.json', 'results/chat600/heads.npz']
    fingerprint = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    paths += [f'experiments/.cache/toxicchat/{fingerprint}-{s}.pt' for s in ['train', 'tune', 'calibration']]
    return {'scope': 'Bounded exploratory follow-up. Consumed100 are diagnostic, never gate/training selection. No backbone updates.',
            'splits': parent['splits'], 'timing_ids': parent['splits']['evaluation'][:50], 'quantization_splits': quant,
            'model': parent['qwen_model'], 'revision': parent['qwen_revision'], 'system_prompt': old['system_prompt'],
            'source_sha256': {p: sha(ROOT/p) for p in paths}, 'seed': 41903,
            'readouts': 'Train-only per-depth896-d normalization; shared896-to64 GELU/dropout.15/to2 head plus four learned64-d depth biases. JointCE vs .5CE+.5KL(teacherfull/student,T2)*4. Frozen originalfullhead teacher uses train examples only. AdamW300minibatch128steps,lr.002,wd.1,inverse-frequencyclassweights.',
            'temperature': 'Per-depth grid .5,1,1.5,2,3,4,6,8 minimizes oldtune500 NLL. No evaluation selection.',
            'gate': 'Oldclean796 calibration is DEVELOPMENT: choose lowest mean depth among zero added errors versus originalfullhead. Ties higher correct, then grid order. Separate SAFE/BLOCK thresholds; optional checkpoint agreement/minimum6or12. No minimumcoverage; disabled alwaysfallback allowed.',
            'thresholds': [.5, .7, .8, .9, .95, .975, .99, .995, 1.01],
            'continuation': 'One backbone forward processes layers once in order; hooks may stop after6/12/18, otherwise continue to originalfull24head without rerunning lower layers. Actual50-message single paired pass rotates full/jointCE/distilled paths. Includes raw preparation and gate work.',
            'combined': 'Two counterbalanced50-message corpus passes of full_b1/full_b4/prefix_b1/prefix_b4. Fresh raw preparation and stable length-sort for every path inside clock. Fixed marker-template prefix stripped trailingCR/LF; exact token-prefix match required. Buildprefix once inside each prefix workload; clone+expand KV privately perbatch. Leftpad suffixes, prefix+suffix mask, real-token RoPEpositions, cache_position indexes physical cache columns. No layer skipped.',
            'quantization': 'Default dynamicint8 linear projections, batch-one128train+64development only. Compare originalfloathead, float/quantmatched128-example linearrefits (200steps lr.01wd.1), and train-only scalar logit-threshold calibration. No new holdout or quantization speedclaim.',
            'numerical_tolerance': {'atol': .001, 'rtol': .0001},
            'limits': ['Historical development repeatedly inspected; no statistical acceptance.', 'Distillation/joint training changes shared readout only, not transformer.',
                       'Token reductions are not exact FLOP reductions; cached prefix remains attended to.', 'All timing must run after other model jobs finish; CPU4threads.']}


def load():
    p = read('results/chat-next-methods/protocol.json')
    for f, h in p['source_sha256'].items(): assert sha(ROOT/f) == h, f
    parent, old, rows = data()
    return p, old, rows


def frozen_heads():
    a = np.load(ROOT/'results/chat600/heads.npz', allow_pickle=False)
    return {d: {k: torch.tensor(a[f'{d}_{k}']) for k in ['weight', 'bias', 'mean', 'std']} for d in DEPTHS}


def original(heads, d, x):
    h = heads[d]; return torch.nn.functional.linear((x-h['mean'])/h['std'], h['weight'], h['bias'])


def cached(old, ids):
    fingerprint = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    where = {rid: (s, i) for s in ['train', 'tune', 'calibration'] for i, rid in enumerate(old['splits'][s])}
    arrays = {s: torch.load(ROOT/f'experiments/.cache/toxicchat/{fingerprint}-{s}.pt', weights_only=True) for s in ['train', 'tune', 'calibration']}
    return {d: torch.stack([arrays[where[rid][0]][d][where[rid][1]] for rid in ids]) for d in DEPTHS}


class Readout(torch.nn.Module):
    def __init__(self, means, stds):
        super().__init__(); self.register_buffer('means', means); self.register_buffer('stds', stds)
        self.input = torch.nn.Linear(896, 64); self.depth_bias = torch.nn.Parameter(torch.zeros(4, 64))
        self.dropout = torch.nn.Dropout(.15); self.output = torch.nn.Linear(64, 2)
    def forward(self, d, x):
        i = DEPTHS.index(d); h = self.input((x-self.means[i])/self.stds[i])+self.depth_bias[i]
        return self.output(self.dropout(torch.nn.functional.gelu(h)))


def load_readout(kind):
    a = np.load(OUT/f'{kind}.npz', allow_pickle=False); state = {k: torch.tensor(a[k]) for k in a.files}
    m = Readout(state['means'], state['stds']); m.load_state_dict(state); return m.eval()


def measures(y, prediction, full, depth=None):
    y, prediction, full = torch.as_tensor(y), torch.as_tensor(prediction), torch.as_tensor(full)
    value = {'n': len(y), 'correct': int((prediction == y).sum()), 'missed_toxic': int(((y == 1)&(prediction == 0)).sum()),
             'false_block': int(((y == 0)&(prediction == 1)).sum()), 'toxic': int((y == 1).sum()),
             'added_errors': int(((full == y)&(prediction != y)).sum()), 'corrected_errors': int(((full != y)&(prediction == y)).sum()),
             'additional_missed_toxic': int(((y == 1)&(full == 1)&(prediction == 0)).sum())}
    if depth is not None:
        value.update(mean_depth=float(depth.float().mean()), exit_counts={str(d): int((depth == d).sum()) for d in DEPTHS}, projected_blocks_skipped=float((24-depth).float().mean()/24))
    return value


def decisions(probs, full, gate):
    pred = full.clone(); depth = torch.full_like(full, 24); previous = None
    for d in DEPTHS[:-1]:
        confidence, label = probs[d].max(1); threshold = torch.where(label == 0, gate['safe'], gate['block'])
        accept = (depth == 24)&(d >= gate['minimum'])&(confidence >= threshold)
        if gate['agreement']: accept &= False if previous is None else label == previous
        pred[accept], depth[accept] = label[accept], d; previous = label
    return pred, depth


def fit():
    guard('fit.json'); p, old, rows = load(); heads = frozen_heads()
    x = {s: cached(old, p['splits'][s]) for s in ['train', 'tune', 'calibration']}
    y = {s: torch.tensor([rows[i]['label'] for i in p['splits'][s]]) for s in x}
    weight = torch.bincount(y['train'], minlength=2).float().reciprocal(); weight /= weight.mean()
    teacher = original(heads, 24, x['train'][24]).detach(); result = {}; predictions = {}
    for kind in ['joint_ce', 'joint_distill']:
        torch.manual_seed(p['seed']); model = Readout(torch.stack([x['train'][d].mean(0) for d in DEPTHS]), torch.stack([x['train'][d].std(0).clamp_min(.05) for d in DEPTHS]))
        opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.1)
        for step in range(300):
            ix = torch.randint(len(y['train']), (128,)); opt.zero_grad(); logits = torch.stack([model(d, x['train'][d][ix]) for d in DEPTHS])
            ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y['train'][ix].repeat(4), weight=weight); loss = ce
            if kind == 'joint_distill':
                kd = torch.nn.functional.kl_div((logits/2).log_softmax(-1), (teacher[ix]/2).softmax(-1).unsqueeze(0).expand_as(logits), reduction='sum')/ (4*len(ix))*4
                loss = .5*ce+.5*kd
            loss.backward(); opt.step()
        model.eval(); np.savez_compressed(OUT/f'{kind}.npz', **{k: v.detach().numpy() for k, v in model.state_dict().items()})
        probs = {s: {} for s in ['tune', 'calibration']}; temperatures = {}
        with torch.inference_mode():
            for d in DEPTHS:
                logits = model(d, x['tune'][d]); t = min([.5, 1., 1.5, 2., 3., 4., 6., 8.], key=lambda t: float(torch.nn.functional.cross_entropy(logits/t, y['tune'])))
                temperatures[str(d)] = t
                for s in probs: probs[s][d] = (model(d, x[s][d])/t).softmax(1)
        full = original(heads, 24, x['calibration'][24]).argmax(1); candidates = []
        for minimum in [6, 12]:
            for agreement in [False, True]:
                for safe in p['thresholds']:
                    for block in p['thresholds']:
                        gate = dict(safe=safe, block=block, minimum=minimum, agreement=agreement)
                        pred, depth = decisions(probs['calibration'], full, gate); m = measures(y['calibration'], pred, full, depth)
                        candidates.append({'gate': gate, 'metrics': m})
        eligible = [c for c in candidates if c['metrics']['added_errors'] == 0]
        chosen = min(eligible, key=lambda c: (c['metrics']['mean_depth'], -c['metrics']['correct']))
        result[kind] = {'temperatures': temperatures, 'selected': chosen, 'candidates': candidates,
                        'weights_sha256': sha(OUT/f'{kind}.npz'), 'fixed_heads': {s: {str(d): measures(y[s], probs[s][d].argmax(1), original(heads, 24, x[s][24]).argmax(1)) for d in DEPTHS} for s in probs}}
        predictions[kind] = {s: [{'id': rid, 'label': int(y[s][i]), 'full': int(original(heads, 24, x[s][24][i:i+1]).argmax(1)),
                                 'probabilities': {str(d): probs[s][d][i].tolist() for d in DEPTHS}} for i, rid in enumerate(p['splits'][s])] for s in probs}
        print(json.dumps({kind: chosen}), flush=True)
    write('development.json', predictions); write('fit.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'methods': result})


class Stop(Exception):
    def __init__(self, label, depth): self.label, self.depth = label, depth


class Runtime:
    def __init__(self, p):
        self.p = p; self.tokenizer = AutoTokenizer.from_pretrained(p['model'], revision=p['revision'], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(p['model'], revision=p['revision'], local_files_only=True, torch_dtype=torch.float32, attn_implementation='eager').eval().model
        assert len(self.model.layers) == 24 and self.model.config.rope_scaling is None
        self.heads = frozen_heads()
    def rendered(self, text):
        return self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.p['system_prompt']}, {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
    def tokens(self, row):
        ids = self.tokenizer.encode(row['text'], add_special_tokens=False)
        text = self.tokenizer.decode(ids[:256], skip_special_tokens=False) if len(ids) > 256 else row['text']
        return self.tokenizer(self.rendered(text))['input_ids']
    def features(self, row, model=None):
        captured = {}; hooks = [layer.register_forward_hook(lambda m,i,o,d=d: captured.update({d: o[0][:,-1,:].clone()})) for d,layer in enumerate((model or self.model).layers, 1) if d in DEPTHS[:-1]]
        try:
            ids = self.tokens(row); out = (model or self.model)(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1,len(ids),dtype=torch.long), use_cache=False)
            captured[24] = out.last_hidden_state[:,-1,:]
        finally:
            for h in hooks: h.remove()
        return captured


def collect():
    guard('quality.json'); p, old, rows = load(); fit_result = read('results/chat-next-methods/fit.json'); assert fit_result['protocol_sha256'] == sha(OUT/'protocol.json')
    rt = Runtime(p); ids = p['splits']['evaluation']; x = {d: [] for d in DEPTHS}
    with torch.inference_mode():
        for i, rid in enumerate(ids):
            h = rt.features(rows[rid])
            for d in DEPTHS: x[d].append(h[d][0])
            if (i+1)%20 == 0: print(f'Feature quality {i+1}/100', flush=True)
        x = {d: torch.stack(v) for d,v in x.items()}; full = original(rt.heads,24,x[24]).argmax(1)
        expected = {r['id']: r for r in read('results/chat-refinement/predictions.json') if r['path'] == 'float_qwen'}
        assert all(int(full[i]) == expected[rid]['prediction'] for i,rid in enumerate(ids))
        y = torch.tensor([rows[rid]['label'] for rid in ids]); result = {'full': measures(y,full,full)}; records = {}
        for kind, config in fit_result['methods'].items():
            assert sha(OUT/f'{kind}.npz') == config['weights_sha256']; model = load_readout(kind)
            probs = {d: (model(d,x[d])/config['temperatures'][str(d)]).softmax(1) for d in DEPTHS}
            pred,depth = decisions(probs,full,config['selected']['gate'])
            result[kind] = {'gated': measures(y,pred,full,depth), 'fixed': {str(d): measures(y,probs[d].argmax(1),full) for d in DEPTHS}}
            records[kind] = [{'id': rid, 'label': int(y[i]), 'full': int(full[i]), 'prediction': int(pred[i]), 'depth': int(depth[i]),
                              'probabilities': {str(d): probs[d][i].tolist() for d in DEPTHS}} for i,rid in enumerate(ids)]
    CACHE.mkdir(parents=True, exist_ok=True); torch.save(x,CACHE/'evaluation.pt')
    write('quality-predictions.json',records); write('quality.json', {'metrics':result,'fit_sha256':sha(OUT/'fit.json'),'cache_sha256':sha(CACHE/'evaluation.pt'),'scope':'Consumed100 diagnostic; gate choices frozen on olddevelopment.'})
    print(json.dumps(result),flush=True)


def continuation():
    guard('continuation-timing.json'); p,old,rows=load(); rt=Runtime(p); fit_result=read('results/chat-next-methods/fit.json')
    quality=read('results/chat-next-methods/quality.json'); assert quality['fit_sha256']==sha(OUT/'fit.json')
    expected=read('results/chat-next-methods/quality-predictions.json'); expected={k:{r['id']:r for r in v} for k,v in expected.items()}
    for kind,config in fit_result['methods'].items():assert sha(OUT/f'{kind}.npz')==config['weights_sha256']
    models={k:load_readout(k) for k in fit_result['methods']}; paths=['full','joint_ce','joint_distill']; records=[]
    def execute(row,path):
        start=time.perf_counter(); ids=rt.tokens(row); visited=[];checks=[];previous=None
        def hook(d):
            def after(m,i,o):
                nonlocal previous
                visited.append(d)
                if path=='full' or d not in DEPTHS[:-1]:return
                c=fit_result['methods'][path];g=c['selected']['gate'];prob=(models[path](d,o[0][:,-1,:])/c['temperatures'][str(d)]).softmax(1)
                confidence,label=prob.max(1);label=int(label);agree=not g['agreement'] or previous==label;previous=label
                accept=d>=g['minimum'] and float(confidence)>=(g['safe'] if label==0 else g['block']) and agree
                checks.append({'depth':d,'prediction':label,'accepted':accept})
                if accept:raise Stop(label,d)
            return after
        hooks=[layer.register_forward_hook(hook(d)) for d,layer in enumerate(rt.model.layers,1)]
        try:
            h=rt.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),use_cache=False).last_hidden_state[:,-1,:]
            label,depth=int(original(rt.heads,24,h).argmax(1)),24
        except Stop as stopped:label,depth=stopped.label,stopped.depth
        finally:
            for h in hooks:h.remove()
        seconds=time.perf_counter()-start; ref=expected['joint_ce' if path=='full' else path][row['id']]
        assert visited==list(range(1,depth+1)) and label==(ref['full'] if path=='full' else ref['prediction']) and depth==(24 if path=='full' else ref['depth'])
        return {'id':row['id'],'path':path,'label':row['label'],'prediction':label,'depth':depth,'executed_layers':visited,'checkpoints':checks,'seconds':seconds}
    with torch.inference_mode():
        for path in paths:execute(rows[p['timing_ids'][0]],path)
        for i,rid in enumerate(p['timing_ids']):
            for path in paths[i%3:]+paths[:i%3]:records.append(execute(rows[rid],path))
    totals={path:sum(r['seconds'] for r in records if r['path']==path) for path in paths}
    write('continuation-records.json',records);write('continuation-timing.json',{'n':len(p['timing_ids']),'totals_seconds':totals,'calls':len(records),'quality_sha256':sha(OUT/'quality.json'),'scope':'Single paired50-message pass; actual one-forward continuation, no repeated lower layers; no independent quality validation.'})
    print(json.dumps(totals),flush=True)


def combined():
    guard('combined-timing.json');p,old,rows=load();rt=Runtime(p); paths=['full_b1','full_b4','prefix_b1','prefix_b4']; runs=[]
    expected={r['id']:r for r in read('results/chat-refinement/predictions.json') if r['path']=='float_qwen'}
    def one(path,ids):
        records=[];traces=[];result={'path':path,'records':records,'traces':traces,'complete':False};start=time.perf_counter()
        try:
            prepared=sorted([{'id':rid,'ids':rt.tokens(rows[rid])} for rid in ids],key=lambda r:len(r['ids']));size=int(path[-1]);prefix=None;prefix_ids=[]
            if path.startswith('prefix'):
                marker='__NEXT_STATIC_PREFIX__';s=rt.rendered(marker);assert s.count(marker)==1;prefix_ids=rt.tokenizer(s.split(marker)[0].rstrip('\r\n'))['input_ids']
                assert all(r['ids'][:len(prefix_ids)]==prefix_ids and len(r['ids'])>len(prefix_ids) for r in prepared)
                visited=[];hooks=[l.register_forward_hook(lambda m,i,o,d=d:visited.append(d)) for d,l in enumerate(rt.model.layers,1)]
                try:out=rt.model(input_ids=torch.tensor([prefix_ids]),attention_mask=torch.ones(1,len(prefix_ids),dtype=torch.long),use_cache=True)
                finally:
                    for h in hooks:h.remove()
                assert visited==list(range(1,25));prefix=out.past_key_values.to_legacy_cache();del out
                digest=lambda:hashlib.sha256(b''.join(t.numpy().tobytes() for pair in prefix for t in pair)).hexdigest()
                before=digest();result.update(prefix_tokens=len(prefix_ids),prefix_prefill_layers=visited,prefix_hash_before=before)
            for offset in range(0,len(prepared),size):
                batch=prepared[offset:offset+size];suffixes=[r['ids'][len(prefix_ids):] for r in batch];width=max(map(len,suffixes));n=len(batch)
                tokens=torch.full((n,width),rt.tokenizer.pad_token_id,dtype=torch.long);mask=torch.zeros_like(tokens)
                for i,suffix in enumerate(suffixes):tokens[i,-len(suffix):]=torch.tensor(suffix);mask[i,-len(suffix):]=1
                positions=(mask.cumsum(-1)-1).clamp(min=0)+len(prefix_ids);attention=torch.cat([torch.ones(n,len(prefix_ids),dtype=torch.long),mask],1)
                cache=None
                if prefix is not None:
                    pairs=tuple((k.expand(n,-1,-1,-1).clone(),v.expand(n,-1,-1,-1).clone()) for k,v in prefix)
                    assert all(a.data_ptr()!=b.data_ptr() for pair,base in zip(pairs,prefix) for a,b in zip(pair,base))
                    cache=DynamicCache.from_legacy_cache(pairs);assert all(cache.get_seq_length(d)==len(prefix_ids) for d in range(24))
                visited=[];hooks=[l.register_forward_hook(lambda m,i,o,d=d:visited.append(d)) for d,l in enumerate(rt.model.layers,1)]
                try:
                    out=rt.model(input_ids=tokens,attention_mask=attention,position_ids=positions,
                                 cache_position=torch.arange(len(prefix_ids),len(prefix_ids)+width),past_key_values=cache,use_cache=cache is not None)
                    logits=original(rt.heads,24,out.last_hidden_state[:,-1,:]);assert bool(torch.isfinite(logits).all())
                finally:
                    for h in hooks:h.remove()
                assert visited==list(range(1,25))
                if cache is not None:assert all(cache.get_seq_length(d)==len(prefix_ids)+width for d in range(24))
                for i,row in enumerate(batch):records.append({'id':row['id'],'label':rows[row['id']]['label'],'prediction':int(logits[i].argmax()),'logits':logits[i].tolist(),'full_tokens':len(row['ids']),'suffix_tokens':len(suffixes[i])})
                traces.append({'ids':[r['id'] for r in batch],'executed_layers':visited,'batch_size':n,'suffix_width':width,'prefix_tokens':len(prefix_ids),'processed_slots':n*width,
                               'real_suffix_lengths':[len(s) for s in suffixes],'last_positions':positions[:,-1].tolist()})
            if prefix is not None:result['prefix_hash_after']=digest();assert result['prefix_hash_after']==before
            result.update(complete=True,traces=traces,processed_positions=len(prefix_ids)+sum(t['processed_slots'] for t in traces))
        except Exception as error:result.update(error=str(error),error_type=type(error).__name__)
        result['corpus_seconds']=time.perf_counter()-start;result['label_mismatches']=[r['id'] for r in records if r['prediction']!=expected[r['id']]['prediction']]
        return result
    with torch.inference_mode():
        write('combined-warmup.json',[one(path,p['timing_ids'][:4]) for path in paths])
        for repeat in range(2):
            for path in (paths if repeat==0 else list(reversed(paths))):
                result=one(path,p['timing_ids']);result['repeat']=repeat;runs.append(result);write('combined-progress.json',runs)
                print(f'Combined {path} {repeat}: {result["corpus_seconds"]:.3f}s, complete={result["complete"]}',flush=True)
    baseline={r['id']:r['logits'] for r in runs[0]['records']} if runs[0]['complete'] else {}
    for r in runs:
        r['numerical_equivalence']=len(r['records'])==50 and {x['id'] for x in r['records']}==set(p['timing_ids']) and all(x['id'] in baseline and np.allclose(x['logits'],baseline[x['id']],**p['numerical_tolerance']) for x in r['records'])
        r['equivalent']=r['complete'] and not r['label_mismatches'] and r['numerical_equivalence']
        r['max_logit_delta']=max((max(abs(a-b) for a,b in zip(x['logits'],baseline[x['id']])) for x in r['records'] if x['id'] in baseline),default=None)
    summary={path:{'seconds':[r['corpus_seconds'] for r in runs if r['path']==path], 'all_equivalent':all(r['equivalent'] for r in runs if r['path']==path)} for path in paths}
    for v in summary.values():v['mean_seconds']=statistics.mean(v['seconds'])
    write('combined-records.json',runs);write('combined-timing.json',{'n':50,'passes':2,'paths':summary,'protocol_sha256':sha(OUT/'protocol.json'),'scope':'Queued50-message execution diagnostic; all preprocessing/cachecopies/prefill included; no layer skipping.'})
    (OUT/'combined-progress.json').unlink();print(json.dumps(summary),flush=True)


def quant_diagnostic():
    guard('quantization.json');p,old,rows=load();rt=Runtime(p);torch.backends.quantized.engine='x86'
    quant=torch.ao.quantization.quantize_dynamic(copy.deepcopy(rt.model),{torch.nn.Linear},dtype=torch.qint8).eval()
    assert sum(isinstance(m,torch.ao.nn.quantized.dynamic.Linear) for m in quant.modules())==168
    qx={};fx={};ys={}
    with torch.inference_mode():
        for split,ids in p['quantization_splits'].items():
            fx[split]=cached(old,ids)[24];ys[split]=torch.tensor([rows[rid]['label'] for rid in ids]);values=[]
            for i,rid in enumerate(ids):
                values.append(rt.features(rows[rid],quant)[24][0])
                if (i+1)%32==0:print(f'Quant diagnostic {split}:{i+1}/{len(ids)}',flush=True)
            qx[split]=torch.stack(values)
    pred={};weights={}
    for kind,arrays in [('matched_float',fx),('refit_quant',qx)]:
        torch.manual_seed(193);mean=arrays['train'].mean(0);std=arrays['train'].std(0).clamp_min(.05);head=torch.nn.Linear(896,2);opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1)
        for _ in range(200):opt.zero_grad();torch.nn.functional.cross_entropy(head((arrays['train']-mean)/std),ys['train']).backward();opt.step()
        with torch.inference_mode():pred[kind]=head((arrays['development']-mean)/std).argmax(1)
        for k,v in [('mean',mean),('std',std),('weight',head.weight.detach()),('bias',head.bias.detach())]:weights[f'{kind}_{k}']=v.numpy()
    with torch.inference_mode():
        full=original(rt.heads,24,fx['development']).argmax(1);qlog=original(rt.heads,24,qx['development']);trainqlog=original(rt.heads,24,qx['train']);margin=trainqlog[:,1]-trainqlog[:,0]
        thresholds=[-float('inf')]+sorted(set(float(v) for v in margin))+[float('inf')]
        # Threshold and both refits use training128 only; development64 remains diagnostic.
        threshold=max(thresholds,key=lambda t:int(((margin>=t).long()==ys['train']).sum()))
        pred['original_float']=full;pred['unchanged_head_quant']=qlog.argmax(1);pred['threshold_quant']=((qlog[:,1]-qlog[:,0])>=threshold).long()
        drift=(qx['development']-fx['development']);cos=torch.nn.functional.cosine_similarity(qx['development'],fx['development'],dim=1)
        result={k:measures(ys['development'],v,full) for k,v in pred.items()}
    np.savez_compressed(OUT/'quant-refits.npz',**weights)
    records=[{'id':rid,'label':int(ys['development'][i]),'predictions':{k:int(v[i]) for k,v in pred.items()},
              'hidden_cosine':float(cos[i]),'hidden_rms_delta':float(drift[i].square().mean().sqrt()),
              'float_logits':original(rt.heads,24,fx['development'][i:i+1])[0].tolist(),'quant_logits':qlog[i].tolist()} for i,rid in enumerate(p['quantization_splits']['development'])]
    write('quantization-records.json',records);write('quantization.json',{'metrics':result,'protocol_sha256':sha(OUT/'protocol.json'),'threshold':threshold if np.isfinite(threshold) else str(threshold),
        'mean_hidden_cosine':float(cos.mean()),'mean_hidden_rms_delta':float(drift.square().mean().sqrt()),
        'scope':'128train/64olddevelopment, no fresh test or timing. Floatcachedbatch8 vs quantactualbatch1; quantfeatures collected separately permessage. Readout correction is not proof quantizedbackbone preserves all information.'})
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit','collect','continuation','combined','quant']);args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    assert __import__('transformers').__version__=='4.50.3'
    if args.stage=='prepare':
        guard('protocol.json');write('protocol.json',protocol());print('Protocol sealed; no model forward.');return
    try:globals()[{'collect':'collect','continuation':'continuation','combined':'combined','quant':'quant_diagnostic','fit':'fit'}[args.stage]]()
    except Exception as error:
        write(args.stage+'-failure.json',{'error':str(error),'type':type(error).__name__});raise


if __name__=='__main__':main()

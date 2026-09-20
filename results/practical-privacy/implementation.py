"""Small practical baselines: name tagging, receipt totals and request routing.

These establish task-specific reference points, not LLM speedup claims.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import sys,json,re,hashlib,random,time,argparse,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/.cache/tooling'))
import numpy as np
import requests
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits
CACHE=ROOT/'experiments/.cache/practical'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(out,name,d):(out/name).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
def begin(task,extra):
    out=ROOT/'results'/('practical-'+task);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists(),'Preserve completed experiment.'
    write(out,'protocol.json',dict(task=task,source_sha256=sha(Path(__file__)),threads=2,seed=190927,
        scope='Exploratory bounded task baseline, not production validation or measured LLM acceleration.',**extra));return out
def fetch(url,target):
    if not target.exists():
        r=requests.get(url,timeout=60);r.raise_for_status();target.write_bytes(r.content)
    return target
def privacy():
    rev='558e09e26d6b36f5f78440074e6a233946d98bd9';data={};paths={}
    for s in ['train','dev','test']:
        paths[s]=fetch(f'https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/{rev}/echr_{s}.json',CACHE/f'echr_{s}.json')
        data[s]=json.loads(paths[s].read_text(encoding='utf-8'))
    ids={s:{r['doc_id'] for r in rows} for s,rows in data.items()}
    assert not(ids['train']&ids['test'] or ids['dev']&ids['test'] or ids['train']&ids['dev'])
    rng=random.Random(190927)
    for s,n in [('train',100),('dev',25),('test',50)]:rng.shuffle(data[s]);data[s]=data[s][:n]
    out=begin('privacy',dict(dataset='TAB real ECHR documents',revision=rev,data_sha256={s:sha(p) for s,p in paths.items()},
        ids={s:[r['doc_id'] for r in x] for s,x in data.items()},target='PERSON token detection, union of human annotators; not complete anonymization or subject-specific masking.',
        limit='First1200 regex tokens per document; metrics only within that prefix.',training='SGD log-loss, alpha1e-5, balanced classes, max_iter30, random_state190927. Token/context features. Dev selects probability threshold for F2; no test selection.',
        baseline='Tokens following Mr/Mrs/Ms/Dr or capitalized tokens after those tokens, at most3tokens. Fixed before scoring.'))
    def features(doc):
        matches=list(re.finditer(r'\w+|[^\w\s]',doc['text']))[:1200];words=[m.group() for m in matches]
        spans={(m['start_offset'],m['end_offset']) for a in doc['annotations'].values() for m in a['entity_mentions'] if m['entity_type']=='PERSON'}
        labels=[int(any(m.start()<end and m.end()>start for start,end in spans)) for m in matches]
        feats=[];base=[];remaining=0
        for i,w in enumerate(words):
            prev=words[i-1] if i else '';nxt=words[i+1] if i+1<len(words) else ''
            if prev.lower() in ['mr','mrs','ms','dr']:remaining=3
            base.append(int(remaining>0 and (w[0].isupper() or w=='.')))
            if remaining:remaining=remaining-1 if (w[0].isupper() or w=='.') else 0
            feats.append(dict(word=w.lower(),prev=prev.lower(),next=nxt.lower(),title=w.istitle(),upper=w.isupper(),digit=w.isdigit(),suffix=w[-3:].lower(),prefix=w[:3].lower()))
        return feats,labels,base,len(list(re.finditer(r'\w+|[^\w\s]',doc['text'])))>1200
    train=[features(d) for d in data['train']];vec=DictVectorizer();x=vec.fit_transform([f for fs,_,_,_ in train for f in fs]);y=np.array([v for _,ls,_,_ in train for v in ls])
    model=SGDClassifier(loss='log_loss',alpha=1e-5,class_weight='balanced',max_iter=30,tol=1e-4,random_state=190927);model.fit(x,y)
    dev=[features(d) for d in data['dev']];dy=np.array([v for _,ls,_,_ in dev for v in ls]);dp=model.predict_proba(vec.transform([f for fs,_,_,_ in dev for f in fs]))[:,1]
    def score(y,p):
        precision,recall,f,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
        return dict(precision=float(precision),recall=float(recall),f1=float(f),f2=float(5*precision*recall/(4*precision+recall)) if precision+recall else 0,
            tp=int(((y==1)&(p==1)).sum()),fn=int(((y==1)&(p==0)).sum()),fp=int(((y==0)&(p==1)).sum()))
    threshold=max([.1,.2,.3,.4,.5,.6,.7,.8,.9],key=lambda t:score(dy,dp>=t)['f2']);write(out,'selection.json',dict(threshold=threshold,dev=score(dy,dp>=threshold)))
    ys=[];ps=[];bs=[];records=[]
    for doc in data['test']:
        start=time.perf_counter();fs,y,b,truncated=features(doc);p=(model.predict_proba(vec.transform(fs))[:,1]>=threshold).astype(int);elapsed=time.perf_counter()-start
        # This initial diagnostic includes gold-label alignment in preparation; not a deployment latency claim.
        records.append(dict(id=doc['doc_id'],tokens=len(y),truncated=truncated,metrics=score(np.array(y),p),seconds=elapsed));ys.extend(y);ps.extend(p.tolist());bs.extend(b)
    write(out,'records.json',records);write(out,'result.json',dict(documents=len(records),tokens=len(ys),truncated_documents=sum(r['truncated'] for r in records),
        learned=score(np.array(ys),np.array(ps)),title_rule=score(np.array(ys),np.array(bs)),threshold=threshold,
        timing='Diagnostic preparation includes annotation alignment; no inference-speed claim.',limitation='Token recall on truncated court documents is not document-level privacy protection; no live-chat validation.'))
    print('privacy',json.dumps(json.loads((out/'result.json').read_text())),flush=True)
def receipts():
    rev='27be4271b251c256f695acbade9a801bffe85994';folder=CACHE/'receipts';folder.mkdir(exist_ok=True)
    # The contestant mirror corrects original annotations. Treat it as a specific derived release, not the official test split.
    base=f'https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/{rev}/data/'
    names=[f'{i:03d}' for i in range(240)]
    def load(name):
        box=fetch(base+'box/'+name+'.csv',folder/(name+'.csv'));key=fetch(base+'key/'+name+'.json',folder/(name+'.json'))
        return dict(id=name,key=json.loads(key.read_text(encoding='utf-8')),box=box)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(load,names))
    groups={}
    for row in rows:
        merchant=re.sub(r'\W+','',row['key']['company'].casefold());groups.setdefault(merchant,[]).append(row)
    merchants=sorted(groups);random.Random(190927).shuffle(merchants);cut=int(len(merchants)*.7)
    train=[r for m in merchants[:cut] for r in groups[m]];test=[r for m in merchants[cut:] for r in groups[m]][:50]
    out=begin('receipts',dict(dataset='240 numbered SROIE receipts from a contestant-corrected annotation mirror',revision=rev,
        source='https://github.com/zzzDavid/ICDAR-2019-SROIE',data_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        ids=dict(train=[r['id'] for r in train],test=[r['id'] for r in test]),split='Seeded normalized-merchant groups70/30; first50heldout receipts, not official SROIE test.',
        task='Select total amount from provided text/boxes; OCR and image encoding excluded.',training='Fixed logistic regression C1 balancedclasses max_iter300. Weak candidate labels match gold total value. No threshold or test tuning.',
        baseline='Largest decimal amount anywhere on receipt.'))
    def candidates(row):
        lines=[]
        for raw in row['box'].read_text(encoding='utf-8').splitlines():
            a=raw.split(',',8)
            if len(a)==9:lines.append((list(map(float,a[:8])),a[8]))
        height=max((max(b[1::2]) for b,_ in lines),default=1);result=[]
        for i,(box,text) in enumerate(lines):
            y=sum(box[1::2])/4;near=' '.join(t for b,t in lines if abs(sum(b[1::2])/4-y)<35).lower()
            for match in re.finditer(r'(?<![\d.])(\d{1,6}\.\d{2})(?!\d)',text):
                amount=match.group(1);result.append((amount,dict(y=y/height,line=i/max(1,len(lines)),logamount=float(np.log1p(float(amount))),
                    total='total' in near,subtotal='subtotal' in near or 'sub total' in near,change='change' in near,cash='cash' in near,tax='tax' in near,
                    rounded='round' in near,alone=text.strip()==amount,context=near[:180])))
        return result
    vec=DictVectorizer();features=[];labels=[]
    for row in train:
        gold=float(row['key']['total'].replace(',',''))
        for amount,f in candidates(row):features.append(f);labels.append(int(float(amount)==gold))
    x=vec.fit_transform(features);model=LogisticRegression(C=1,class_weight='balanced',max_iter=300,solver='liblinear');model.fit(x,labels)
    records=[]
    for row in test:
        start=time.perf_counter();c=candidates(row)
        chosen=c[int(model.predict_proba(vec.transform([f for _,f in c]))[:,1].argmax())][0] if c else None
        baseline=max((a for a,_ in c),key=float,default=None);elapsed=time.perf_counter()-start;gold=float(row['key']['total'].replace(',',''))
        records.append(dict(id=row['id'],candidate_count=len(c),gold_present=any(float(a)==gold for a,_ in c),
            learned_correct=chosen is not None and float(chosen)==gold,largest_correct=baseline is not None and float(baseline)==gold,seconds=elapsed))
    write(out,'records.json',records);write(out,'result.json',dict(train_receipts=len(train),test_receipts=len(test),merchant_groups=len(groups),
        learned_correct=sum(r['learned_correct'] for r in records),largest_correct=sum(r['largest_correct'] for r in records),
        candidate_ceiling=sum(r['gold_present'] for r in records),timing='Preparation and classifier recorded during concurrent work; no image/OCR or LLM speed comparison.',
        limitations=['Small derived split, not official leaderboard score.','Normalized merchant names can miss chain/template relationships.','Human transcription/boxes, not noisy OCR.']))
    print('receipts',json.dumps(json.loads((out/'result.json').read_text())),flush=True)
def routing():
    parent=json.loads((ROOT/'results/clinc-validation/protocol.json').read_text());unknown=json.loads((ROOT/'results/clinc-unknown/protocol.json').read_text())
    source=ROOT/'experiments/.cache/clinc/data_full.json';raw=json.loads(source.read_text());categories=parent['categories']
    train=parent['ids']['train']+unknown['oos_train_ids'];tune=parent['ids']['tune']+unknown['oos_tune_ids']
    test=parent['ids']['test']+parent['ids']['oos_test']
    out=begin('routing',dict(dataset='CLINC150 crowdsourced requests, existing30-intent+UNKNOWNsubset',data_sha256=sha(source),
        ids=dict(train=train,tune=tune,test=test),categories=categories+['UNKNOWN'],training='Word1-2gramTFIDF maxfeatures12000, logistic C4balancedclasses max_iter400. Tunethreshold balances known-intent accuracy and unsupported rejection; freeze before consumedtest.',
        limitation='Same previously inspected test IDs; crowdsourced scenarios, not live requests.'))
    def get(ids):
        values=[raw[r.split(':')[0]][int(r.split(':')[1])] for r in ids]
        return [v[0] for v in values],np.array([categories.index(v[1]) if v[1] in categories else 30 for v in values])
    tx,ty=get(train);dx,dy=get(tune);vec=TfidfVectorizer(ngram_range=(1,2),max_features=12000,sublinear_tf=True)
    model=LogisticRegression(C=4,class_weight='balanced',max_iter=400);model.fit(vec.fit_transform(tx),ty)
    prob=model.predict_proba(vec.transform(dx));base=model.classes_[prob.argmax(1)]
    def summarize(y,p):
        known=y!=30;return dict(n=len(y),correct=int((y==p).sum()),known_correct=int(((y==p)&known).sum()),known=int(known.sum()),
            unsupported_rejected=int(((p==30)&~known).sum()),unsupported=int((~known).sum()),known_rejected=int(((p==30)&known).sum()))
    def rank(t):
        pred=np.where(prob.max(1)<t,30,base);m=summarize(dy,pred)
        return (m['known_correct']/m['known']+m['unsupported_rejected']/m['unsupported'])/2
    threshold=max([0,.1,.2,.3,.4,.5,.6,.7,.8,.9],key=rank);write(out,'selection.json',dict(threshold=threshold,dev=summarize(dy,np.where(prob.max(1)<threshold,30,base))))
    texts,y=get(test);start=time.perf_counter();p=model.predict_proba(vec.transform(texts));base=model.classes_[p.argmax(1)];pred=np.where(p.max(1)<threshold,30,base);elapsed=time.perf_counter()-start
    write(out,'records.json',[dict(id=rid,label=int(y[i]),raw=int(base[i]),with_rejection=int(pred[i]),confidence=float(p[i].max())) for i,rid in enumerate(test)])
    write(out,'result.json',dict(raw=summarize(y,base),with_rejection=summarize(y,pred),threshold=threshold,seconds=elapsed,
        timing='One batch, includes TFIDF transform and classifier; concurrent work, no paired Qwen speedup.',limitations=['Reused test data.','UNKNOWN training only80examples; unseen categories and natural inputs need broader validation.']))
    print('routing',json.dumps(json.loads((out/'result.json').read_text())),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['privacy','receipts','routing']);a=p.parse_args();CACHE.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):globals()[a.task]()

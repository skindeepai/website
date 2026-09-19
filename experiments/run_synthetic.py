"""Reproducible synthetic checks. No human-preference or real-generator claims."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json, platform, subprocess, time
import numpy as np
from scipy.optimize import minimize
from preference import fit, sigmoid, optimum, minimum_change
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'results/synthetic';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1701);cases=[];max_gap=0;reached=0
    for _ in range(80):
        w=rng.normal(size=4);b=float(rng.normal());ref=rng.uniform(-1,1,4)
        target=float(sigmoid(rng.uniform(b-np.abs(w).sum(),b+np.abs(w).sum())))
        z,status=minimum_change(w,b,ref,target);lt=np.log(target/(1-target))
        sol=minimize(lambda v:np.sum((v-ref)**2),ref,jac=lambda v:2*(v-ref),bounds=[(-1,1)]*4,constraints=[{'type':'ineq','fun':lambda v:w@v+b-lt,'jac':lambda v:w}],method='SLSQP',options={'ftol':1e-11,'maxiter':200})
        assert sol.success, sol.message
        gap=abs(np.sum((z-ref)**2)-sol.fun);max_gap=max(max_gap,float(gap))
        assert gap<1e-7 and np.abs(z).max()<=1 and w@z+b>=lt-1e-8
        reached+=status=='reached';cases.append({'w':w.tolist(),'b':b,'reference':ref.tolist(),'target':target,'expected':z.tolist(),'status':status})
    cases.extend([
        {'w':[0,0],'b':0,'reference':[.3,-.2],'target':.8,'expected':[.3,-.2],'status':'unreachable'},
        {'w':[1,0],'b':0,'reference':[1,.4],'target':.6,'expected':[1,.4],'status':'already-met'},
        {'w':[.001,2],'b':0,'reference':[0,0],'target':.999,'expected':[1,1],'status':'unreachable'}])
    (out/'math-fixtures.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    js="const fs=require('fs'),c=require('./scripts/preference-core.js'),assert=require('assert/strict');const cases=JSON.parse(fs.readFileSync('./results/synthetic/math-fixtures.json'));for(const f of cases){const m=c.makeModel(f.w.length);m.w.set(f.w);m.b=f.b;const r=c.transform(m,f.reference,f.target);assert.equal(r.status,f.status);r.z.forEach((v,i)=>assert(Math.abs(v-f.expected[i])<1e-7));assert(Math.abs(c.logit(m,c.idealZ(m,1))-(m.b+f.w.reduce((s,w)=>s+Math.abs(w),0)))<1e-9);}const m=c.makeModel(2);m.w.set([1,2]);m.b=3;c.train(m);assert.deepEqual(Array.from(m.w),[0,0]);assert.equal(m.b,0);console.log('Browser core fixtures passed: '+cases.length);"
    subprocess.run(['node','-e',js],cwd=ROOT,check=True)
    records=[];timings=[]
    for seed in [11,23,37,53,71]:
        r=np.random.default_rng(seed);test=r.uniform(-1,1,(1500,4));wtrue=np.array([1.,-.6,.3,0]);linear=(test@wtrue>0).astype(float)
        x=r.uniform(-1,1,(100,4)); y=(x@wtrue>0).astype(float)
        for n in [20,50,100]:
            start=time.perf_counter();w,b=fit(x[:n],y[:n]);ms=(time.perf_counter()-start)*1000
            p=sigmoid(test@w+b);records.append({'experiment':'P02','seed':seed,'n':n,'accuracy':float(((p>.5)==linear).mean()),'brier':float(np.mean((p-linear)**2)),'majority_accuracy':float(max(linear.mean(),1-linear.mean()))});timings.append(ms)
        # Two disjoint preferred regions; a quadratic feature head is explicitly named.
        def labels(v):return ((np.abs(v[:,0])>.55)&(np.abs(v[:,1])<.65)).astype(float)
        x=r.uniform(-1,1,(180,4));yt=labels(test);y=labels(x)
        def quadratic(v):return np.column_stack([v,v*v])
        for name,feature in [('linear',lambda v:v),('quadratic',quadratic)]:
            w,b=fit(feature(x),y,epochs=700,l2=.002);pred=sigmoid(feature(test)@w+b)>.5
            recall=float(pred[yt==1].mean());specificity=float((~pred[yt==0]).mean())
            records.append({'experiment':'P04','seed':seed,'head':name,'balanced_accuracy':(recall+specificity)/2,'positive_recall':recall,'specificity':specificity})
        for policy in ['random','uncertainty','mixed']:
            rr=np.random.default_rng(seed);xx=rr.uniform(-1,1,(8,4));yy=(xx@wtrue>0).astype(float)
            for step in range(12):
                w,b=fit(xx,yy);pool=rr.uniform(-1,1,(240,4));p=sigmoid(pool@w+b)
                if policy=='random': ids=rr.permutation(240)[:6]
                elif policy=='uncertainty':ids=np.argsort(np.abs(p-.5))[:6]
                else:ids=np.unique(np.r_[np.argsort(-p)[:2],np.argsort(np.abs(p-.5))[:2],rr.permutation(240)[:2]])
                # Equalize labels when the mixed selections overlap.
                if len(ids)<6:ids=np.r_[ids,np.setdiff1d(np.arange(240),ids)[:6-len(ids)]]
                xx=np.vstack([xx,pool[ids]]);yy=np.r_[yy,(pool[ids]@wtrue>0).astype(float)]
            w,b=fit(xx,yy);records.append({'experiment':'P03','seed':seed,'policy':policy,'ratings':len(yy),'accuracy':float(((test@w+b>0)==linear).mean())})
        # Distribution shift: recent labels invert a previous preference.
        old=r.uniform(-1,1,(80,4));new=r.uniform(-1,1,(40,4));oldy=(old@wtrue>0).astype(float);newy=(new@wtrue<0).astype(float)
        for name,xx,yy in [('static',old,oldy),('all-history',np.vstack([old,new]),np.r_[oldy,newy]),('recent-window',new,newy)]:
            w,b=fit(xx,yy);records.append({'experiment':'P07','seed':seed,'policy':name,'new_context_accuracy':float(((test@w+b>0)==(1-linear)).mean())})
        # Known synthetic utility saturates near a center: optimizing a misspecified linear fit can fail.
        x=r.uniform(-1,1,(60,4));center=np.array([.4,-.2,0,0]);utility=lambda v: -np.sum((v-center)**2,axis=-1)
        y=(utility(x)>-1).astype(float);w,b=fit(x,y);pool=r.uniform(-1,1,(100,4))
        for name,z in [('box-optimum',optimum(w)),('truncated',optimum(w,.5)),('reranked',pool[np.argmax(pool@w+b)]),('random',pool[0])]:
            records.append({'experiment':'P05','seed':seed,'method':name,'predicted':float(sigmoid(w@z+b)),'oracle_utility':float(utility(z))})
        # Independently known constraint x0 <= 0: hard feasibility versus learned negatives.
        x=r.uniform(-1,1,(100,4));y=((x[:,1]>0)&(x[:,0]<=0)).astype(float);w,b=fit(x,y)
        z=optimum(w);constrained=z.copy();constrained[0]=min(0,constrained[0])
        records.append({'experiment':'P10','seed':seed,'unconstrained_violation':bool(z[0]>0),'constrained_violation':bool(constrained[0]>0),'note':'Known toy half-space only; not a safety guarantee.'})
    payload={'status':'synthetic pilot','experiments':['P01','P02','P03','P04','P05','P06','P07','P10'],'seeds':[11,23,37,53,71],
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'threads':1},
        'math':{'random_cases':80,'browser_fixtures':len(cases),'max_squared_distance_gap_vs_scipy':max_gap,'tolerance':1e-7,'solver':'SciPy SLSQP'},
        'training_ms':{'p50':float(np.median(timings)),'p95':float(np.percentile(timings,95)),'scope':'Python CPU, 4 dimensions, 20/50/100 labels; 260 epochs; no rendering'},
        'records':records,'limitations':['Synthetic utilities and five seeds; no human evidence or real generator.','P03 final quality at a fixed budget; not user satisfaction or a proof of sample savings.','P04 quadratic features are a specified nonlinear baseline, not a trained generative model.','P06 establishes numerical distance in a box, not perceptual minimality.','P10 tests a known analytic constraint, not learned safety under distribution shift.']}
    (out/'result.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k!='records'},indent=2))

if __name__=='__main__':main()

"""Prospective 31-output variant: add an explicit unsupported-request category.

Uses the frozen CLINC features; a new head and protocol, not a revised test slice.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from banking77 import policy,DEPTHS
from clinc_validation import upper
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/clinc-unknown';BASE=ROOT/'results/clinc-validation';CACHE=ROOT/'experiments/.cache/clinc'
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    torch.set_num_threads(min(4,max(1,(os.cpu_count() or 2)//2)));torch.set_num_interop_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    parent=json.loads((BASE/'protocol.json').read_text());identity=hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest()
    protocol={'parent_protocol_sha256':identity,'design':'Add UNKNOWN=30 to30intent heads; train on80official OOS training queries, tune on remaining20. Original ID training960/tune300/cal600/test900 unchanged.',
              'oos_train_ids':parent['ids']['oos_tune'][:80],'oos_tune_ids':parent['ids']['oos_tune'][80:],
              'seed':83,'steps':200,'lr':.01,'weight_decay':.1,'selection':'Minimum mean depth on combined tuning with <=0.5pp net accuracy loss, <=5% error among early answers, >=5% early coverage. UNKNOWN is an explicit answer, not evidence of inability to reason.',
              'grid':{'threshold':[.7,.8,.9,.95,.975,.99,.995],'agreement':[False,True],'minimum':[6,12]},
              'guard':'Independent calibration: individual one-sided exact95% upper bounds <=1% added errors/all, <=5% wrong/early, <=5% OOS routed to a known intent early, <=5% known requests rejected as UNKNOWN early. Require5% early coverage. No joint95% claim.',
              'limits':'Same dataset/test partitions as30-output study; paired architecture comparison, not independent replication. Recorded before those result files existed.',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:
        assert not (BASE/'result.json').exists(),'Variant must be recorded before baseline results.'
        write('protocol.json',protocol)
    if args.prepare:print('Saved prospective UNKNOWN-head protocol.');return
    features={s:torch.load(CACHE/f'{identity}-{s}.pt',weights_only=True) for s in parent['ids']}
    raw=json.loads((CACHE/'data_full.json').read_text());categories=parent['categories']
    labels={}
    for split,ids in parent['ids'].items():
        labels[split]=torch.tensor([categories.index(raw[i.split(':')[0]][int(i.split(':')[1])][1]) if raw[i.split(':')[0]][int(i.split(':')[1])][1] in categories else 30 for i in ids])
    torch.manual_seed(83);probabilities={s:{} for s in parent['ids']};weights={}
    for d in DEPTHS:
        x=torch.cat([features['train'][d],features['oos_tune'][d][:80]])
        y=torch.cat([labels['train'],labels['oos_tune'][:80]])
        mean=x.mean(0);std=x.std(0).clamp_min(.05);head=torch.nn.Linear(896,31);opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.1)
        for _ in range(200):
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(head((x-mean)/std),y);loss.backward();opt.step()
        with torch.inference_mode():
            tx=torch.cat([features['tune'][d],features['oos_tune'][d][80:]]);ty=torch.cat([labels['tune'],labels['oos_tune'][80:]])
            logits=head((tx-mean)/std);temp=min([.5,1.,1.5,2.,3.,4.,6.,8.],key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,ty)))
            for s in probabilities:probabilities[s][d]=torch.softmax(head((features[s][d]-mean)/std)/temp,1)
        for name,value in [('weight',head.weight),('bias',head.bias),('mean',mean),('std',std)]:weights[f'{d}_{name}']=value.detach().numpy()
        weights[f'{d}_temperature']=np.array(temp)
    tune={d:torch.cat([probabilities['tune'][d],probabilities['oos_tune'][d][80:]]) for d in DEPTHS}
    y=torch.cat([labels['tune'],labels['oos_tune'][80:]]);full=tune[24].argmax(1);candidates=[]
    for minimum in [6,12]:
        for agreement in [False,True]:
            for threshold in protocol['grid']['threshold']:
                pred,depth=policy(tune,threshold,agreement,minimum);early=depth<24
                if float(early.float().mean())>=.05 and float(((pred==y).float()-(full==y).float()).mean())>=-.005 and float((pred[early]!=y[early]).float().mean())<=.05:
                    candidates.append({'threshold':threshold,'agreement':agreement,'minimum':minimum,'mean_depth':float(depth.float().mean())})
    chosen=min(candidates,key=lambda c:(c['mean_depth'],-c['threshold'])) if candidates else {'threshold':1.01,'agreement':False,'minimum':6,'reason':'No tuning candidate'}
    write('selected-policy.json',chosen);result={'policy':chosen,'splits':{}};traces=[]
    for split in ['calibration','test','oos_calibration','oos_test','unsupported_test']:
        p=probabilities[split];pred,depth=policy(p,chosen['threshold'],chosen['agreement'],chosen['minimum']);full=p[24].argmax(1);y=labels[split];early=depth<24
        result['splits'][split]={'n':len(y),'full_correct':int((full==y).sum()),'candidate_correct':int((pred==y).sum()),'early_count':int(early.sum()),
            'early_wrong':int((early&(pred!=y)).sum()),'added_errors':int(((full==y)&(pred!=y)).sum()),'early_unknown':int((early&(pred==30)).sum()),
            'early_known':int((early&(pred!=30)).sum()),'mean_depth':float(depth.float().mean()),'blocks_skipped':float((24-depth).float().mean()/24)}
        traces.extend({'split':split,'id':identifier,'label':int(y[i]),'full_prediction':int(full[i]),'prediction':int(pred[i]),'depth':int(depth[i])} for i,identifier in enumerate(parent['ids'][split]))
    a=result['splits']['calibration'];b=result['splits']['oos_calibration'];n=a['n']+b['n'];ec=a['early_count']+b['early_count']
    bounds={'added_error':upper(a['added_errors']+b['added_errors'],n),'absolute_early_error':upper(a['early_wrong']+b['early_wrong'],ec),
            'unknown_misrouted':upper(b['early_known'],b['n']),'known_rejected':upper(a['early_unknown'],a['n'])}
    limits={'added_error':.01,'absolute_early_error':.05,'unknown_misrouted':.05,'known_rejected':.05}
    result['upper95']=bounds;result['guard_checks']={k:v<=limits[k] for k,v in bounds.items()};result['guard_checks']['coverage']=ec/n>=.05
    result['guard_passed']=all(result['guard_checks'].values())
    write('result.json',result);write('predictions.json',traces);np.savez_compressed(OUT/'heads.npz',**weights);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

"""Retrospective confidence-as-presence diagnostic; no model inference/training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/coordinate-abstention-smoke'

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def presence_score(rows, threshold):
    present = [r for r in rows if r['box'] is not None]
    absent = [r for r in rows if r['box'] is None]
    accepted_present = sum(r['patch_probability'] >= threshold for r in present)
    rejected_absent = sum(r['patch_probability'] < threshold for r in absent)
    return (accepted_present/len(present)+rejected_absent/len(absent))/2

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'result.json').exists():
        raise RuntimeError('Preserve the completed diagnostic; use a new output directory.')
    source = ROOT/'results/coordinates/predictions.json'
    rows = json.loads(source.read_text())
    images = sorted({r['image'] for r in rows})
    assert len(images)==2 and len(rows)==8
    protocol = dict(scope='Retrospective exploratory reanalysis of eight already inspected local outputs. No new model inference or learned presence head.',
        source_sha256=sha(source), fold='Leave one screenshot out; choose threshold only on other screenshot, then evaluate held-out screenshot; reverse.',
        candidate_thresholds='0, 1.01, and midpoints between sorted distinct confidence values on training screenshot only.',
        selection='Maximize balanced present/absent classification accuracy on training screenshot; tie chooses smallest threshold. Accept coordinate when peak patch probability >= threshold.',
        target='Whether requested target exists, distinct from whether coordinate is correct.',
        limitations=['Only one authored UI at desktop/mobile sizes; same instructions and absent target recur across folds. Not independent task/generalization evidence.',
                     'One absent example per fold; no reliable error-rate estimate.',
                     'Peak patch probabilities depend on the patch grid and are not calibrated presence or correctness probabilities.',
                     'Saved ScreenSpot outputs have present targets only; rejection analysis cannot validate absent-target detection.'])
    write('protocol.json', protocol)
    folds=[]; decisions=[]
    for held_out in images:
        train=[r for r in rows if r['image']!=held_out]
        test=[r for r in rows if r['image']==held_out]
        assert len(train)==len(test)==4
        values=sorted({r['patch_probability'] for r in train})
        grid=[0.]+[(a+b)/2 for a,b in zip(values,values[1:])]+[1.01]
        threshold=max(grid,key=lambda t:(presence_score(train,t),-t))
        fold=dict(train_image=train[0]['image'],held_out_image=held_out,threshold=threshold,
                  train_presence_balanced_accuracy=presence_score(train,threshold),
                  held_out_presence_balanced_accuracy=presence_score(test,threshold),
                  train_grid=[dict(threshold=t,balanced_accuracy=presence_score(train,t)) for t in grid])
        folds.append(fold)
        for row in test:
            present=row['box'] is not None;accepted=row['patch_probability']>=threshold
            decisions.append(dict(id=row['id'],image=row['image'],confidence=row['patch_probability'],threshold=threshold,
                target_present=present,accepted=accepted,coordinate_hit=row['hit'],
                correct_absent_rejection=not present and not accepted,
                erroneous_present_rejection=present and not accepted,
                accepted_wrong_point=accepted and (not present or not row['hit']),
                successful_action=(not present and not accepted) or (present and accepted and row['hit'])))
    result=dict(n=8,present=6,absent=2,folds=folds,
        correct_absent_rejections=sum(r['correct_absent_rejection'] for r in decisions),
        erroneous_present_rejections=sum(r['erroneous_present_rejection'] for r in decisions),
        accepted_wrong_points=sum(r['accepted_wrong_point'] for r in decisions),
        accepted_correct_points=sum(r['accepted'] and r['coordinate_hit'] is True for r in decisions),
        successful_actions=sum(r['successful_action'] for r in decisions))
    ss_path=ROOT/'results/screenspot/predictions.json'
    screenspot=json.loads(ss_path.read_text())
    result['present_only_screenspot_rejection_diagnostic']=dict(source_sha256=sha(ss_path),n=len(screenspot),absent=0,policies=[])
    for fold in folds:
        threshold=fold['threshold'];accepted=[r for r in screenspot if r['peak_probability']>=threshold]
        rejected=[r for r in screenspot if r['peak_probability']<threshold]
        result['present_only_screenspot_rejection_diagnostic']['policies'].append(dict(threshold=threshold,
            accepted=len(accepted),erroneously_rejected_present=len(rejected),
            rejected_previously_correct_connected_points=sum(r['hits']['connected_region'] for r in rejected),
            accepted_correct_connected_points=sum(r['hits']['connected_region'] for r in accepted),
            accepted_wrong_connected_points=sum(not r['hits']['connected_region'] for r in accepted)))
    write('predictions.json',decisions);write('result.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()

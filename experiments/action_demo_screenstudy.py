"""Retrospective cross-platform click/review policies on saved real screenshots."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/action-demo/screenstudy'

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def family(row):return row['data_source'] if row['data_source'] in ['android','ios','macos','windows'] else 'web'

def protocol():
    return {'scope':'Retrospective reanalysis of30 already inspected real ScreenSpot outputs. No new vision inference, training or task-quality holdout.',
      'source_sha256':{p:sha(p) for p in ['experiments/action_demo_screenstudy.py','results/screenspot/predictions.json','results/screenspot/data-manifest.json','results/screenspot/result.json']},
      'methods':['always_click_connected_region','peak_gate','relative_peak_gate'],
      'fold':'Leave one platform family out: android,ios,macos,web,windows. Fit cutoff using other24 only; apply to held-out6.',
      'score':{'peak_gate':'Saved peak patch probability.','relative_peak_gate':'Saved peak patch probability times number of visual patches; dimensionless score relative to uniform mass, not calibrated correctness.'},
      'candidate_thresholds':'0, midpoints of distinct training scores, and reject-all sentinel2 for peak or577 for relative peak (grids capped576). Accept score>=cutoff.',
      'selection':'Maximum training coverage with zero accepted incorrect connected-region points. Ties choose smaller cutoff. Reject-all is allowed. No held-out target box enters selection.',
      'metrics':'Correct and incorrect automatic clicks; requests referred for human review; correctly localized requests withheld. Review is unresolved, not counted correct.',
      'limits':['All30 targets are present. This does not validate absent-target detection.','Only6 held-out examples per platform; no guaranteed error rate.','Same frozen pointer model and already inspected screenshots; neither new image inference nor latency improvement.','Peak confidence depends on image/grid; this study retains both predefined policies whether better or worse.']}

def run(p):
    rows=read('results/screenspot/predictions.json');manifest=read('results/screenspot/data-manifest.json')
    metadata={r['row_index']:r for r in manifest['samples']}
    assert len(rows)==30 and len({r['row_index'] for r in rows})==30
    assert all(0<=r['peak_probability']<=1 and r['patch_grid'][0]*r['patch_grid'][1]<=576 for r in rows)
    folds=[];decisions=[]
    for group in sorted({family(r) for r in rows}):
        train=[r for r in rows if family(r)!=group];test=[r for r in rows if family(r)==group]
        assert len(train)==24 and len(test)==6
        for method in ['peak_gate','relative_peak_gate']:
            def score(row):return row['peak_probability']*(row['patch_grid'][0]*row['patch_grid'][1] if method=='relative_peak_gate' else 1)
            values=sorted({score(r) for r in train});thresholds=[0]+[(a+b)/2 for a,b in zip(values,values[1:])]+[2 if method=='peak_gate' else 577]
            candidates=[]
            for threshold in thresholds:
                accepted=[r for r in train if score(r)>=threshold]
                candidates.append({'threshold':threshold,'accepted':len(accepted),'wrong':sum(not r['hits']['connected_region'] for r in accepted)})
            permitted=[c for c in candidates if c['wrong']==0]
            chosen=max(permitted,key=lambda c:(c['accepted'],-c['threshold']))
            folds.append({'held_out_platform':group,'method':method,'train_ids':[r['row_index'] for r in train],
                          'test_ids':[r['row_index'] for r in test],'chosen':chosen,'candidates':candidates})
            for row in test:
                accepted=score(row)>=chosen['threshold']
                decisions.append({'row_index':row['row_index'],'platform':group,'method':method,'score':score(row),
                    'threshold':chosen['threshold'],'accepted':accepted,'point_correct':row['hits']['connected_region'],
                    'correct_click':accepted and row['hits']['connected_region'],'wrong_click':accepted and not row['hits']['connected_region'],
                    'review':not accepted,'withheld_correct_point':not accepted and row['hits']['connected_region']})
    summary={}
    for method in ['peak_gate','relative_peak_gate']:
        subset=[r for r in decisions if r['method']==method]
        summary[method]={'n':30,'accepted':sum(r['accepted'] for r in subset),'correct_clicks':sum(r['correct_click'] for r in subset),
            'wrong_clicks':sum(r['wrong_click'] for r in subset),'review':sum(r['review'] for r in subset),
            'withheld_correct_points':sum(r['withheld_correct_point'] for r in subset)}
    summary['always_click_connected_region']={'n':30,'accepted':30,'correct_clicks':sum(r['hits']['connected_region'] for r in rows),'wrong_clicks':sum(not r['hits']['connected_region'] for r in rows),'review':0,'withheld_correct_points':0}
    gallery=[];destination=ROOT/'results/action-demo/screenshots';destination.mkdir(parents=True,exist_ok=True)
    for row in rows:
        m=metadata[row['row_index']];src=ROOT/'experiments/.cache/screenspot'/row['file_name']
        assert hashlib.sha256(src.read_bytes()).hexdigest()==m['sha256']
        shutil.copyfile(src,destination/row['file_name'])
        gallery.append({**row,'width':m['width'],'height':m['height'],'image_sha256':m['sha256'],
            'image':'results/action-demo/screenshots/'+row['file_name'],
            'review_policies':{method:next(r for r in decisions if r['row_index']==row['row_index'] and r['method']==method) for method in ['peak_gate','relative_peak_gate']}})
    write('folds.json',folds);write('decisions.json',decisions)
    write('result.json',{'protocol_sha256':sha('results/action-demo/screenstudy/protocol.json'),'methods':summary,
        'model':read('results/screenspot/result.json')['model'],'model_revision':read('results/screenspot/result.json')['revision'],
        'layers':28,'decoding_tokens':0,'limits':p['limits']})
    (ROOT/'results/action-demo/gallery.json').write_text(json.dumps({'mode':'Recorded outputs, not live model inference','source':'ScreenSpot','samples':gallery},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);p=protocol()
    if args.prepare:
        if (OUT/'protocol.json').exists():assert read('results/action-demo/screenstudy/protocol.json')==p
        else:write('protocol.json',p)
        print('Screen study sealed; no outcomes computed.');return
    assert not (OUT/'result.json').exists(),'Preserve completed results'
    assert read('results/action-demo/screenstudy/protocol.json')==p,'Sealed inputs changed'
    run(p)

if __name__=='__main__':main()

"""Shared, explicitly exploratory subset of the already inspected chat study."""
import csv,hashlib,json,random
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/chat-smoke'

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def source_rows():
    original=read('results/chat600/protocol.json');rows={}
    for source in ['train','test']:
        path=ROOT/f'experiments/.cache/toxicchat/toxic-chat_annotation_{source}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==original['data_sha256'][source]
        for i,r in enumerate(csv.DictReader(path.open(encoding='utf-8',newline=''))):
            if r['human_annotation']=='True':rows[f'{source}:{i}']={'id':f'{source}:{i}','text':r['user_input'],'label':int(r['toxicity']),'conversation':r['conv_id']}
    return rows

def prepare():
    rows=source_rows();old=read('results/chat600/protocol.json');rng=random.Random(39017);splits={}
    excluded={'train:230','train:295','train:4146'}
    for name,previous,counts in [('train','train',[256,128]),('tune','tune',[64,64]),('calibration','calibration',[64,64]),('evaluation','test',[50,50])]:
        ids=[]
        for label,n in enumerate(counts):
            pool=[i for i in old['splits'][previous] if i not in excluded and rows[i]['label']==label];rng.shuffle(pool);assert len(pool)>=n;ids+=pool[:n]
        rng.shuffle(ids);splits[name]=ids
    sets=[set(v) for v in splits.values()];assert sum(map(len,sets))==len(set.union(*sets))
    conversations=[{rows[i]['conversation'] for i in ids} for ids in splits.values()]
    assert sum(map(len,conversations))==len(set.union(*conversations))
    protocol={'scope':'Exploratory development smoke tests. All data were already inspected in the prior study; no new independent validation claim.',
      'seed':39017,'splits':splits,'sizes':{s:len(ids) for s,ids in splits.items()},
      'label_counts':{s:{str(y):sum(rows[i]['label']==y for i in ids) for y in [0,1]} for s,ids in splits.items()},
      'roles':{'train':'Fit model/head weights only.','tune':'Fit calibration, learned gate or select training checkpoint as declared by each method.','calibration':'Development selection of thresholds/gates; NOT independent calibration or acceptance.','evaluation':'Reused balanced100 diagnostic; never select model, epoch or gate on these outcomes.'},
      'input':'Same original256-Qwen-token bounded user message and fixed moderation instruction. Encoder tokenization differs and must report any extra truncation.',
      'excluded_effective_duplicates':sorted(excluded),'parent_protocol_sha256':hashlib.sha256((ROOT/'results/chat600/protocol.json').read_bytes()).hexdigest(),
      'rules':['Report both class errors, coverage and every tried method.','Balanced50/50 evaluation is not deployment prevalence or comparable overall accuracy to the prior600.','Cached hidden states describe frozen Qwen only; changed backbones require fresh forward passes.','No quality acceptance claims from this exploratory subset.','Concurrent model training timings are diagnostic only; isolate final smoke timing.'],
      'source_sha256':old['data_sha256'],'qwen_model':old['model'],'qwen_revision':old['model_revision'],'system_prompt':old['system_prompt']}
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    return protocol,{s:[rows[i] for i in ids] for s,ids in splits.items()}

if __name__=='__main__':
    p,_=prepare();print(json.dumps({'sizes':p['sizes'],'labels':p['label_counts'],'scope':p['scope']}))

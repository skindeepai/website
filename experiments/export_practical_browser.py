"""Export the established practical classifiers for actual browser inference."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/.cache/tooling'))
from threadpoolctl import threadpool_limits
OUT=ROOT/'results/practical-browser';ASSETS=ROOT/'models/practical';OUT.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def run(task,source):
 dest=OUT/task;assert not dest.exists(),'Preserve previous export.';dest.mkdir()
 ns={'__name__':'browser_export','__file__':str(ROOT/'experiments/practical_baselines.py')};code=ROOT/source
 assert sha(code)==json.loads((ROOT/f'results/practical-{task}/protocol.json').read_text())['source_sha256']
 exec(compile(code.read_text(encoding='utf-8'),ns['__file__'],'exec'),ns)
 vectors=[];models=[];examples=[]
 def capture_vec(base):
  class V(base):
   def fit_transform(self,*a,**kw):
    r=super().fit_transform(*a,**kw);vectors.append(self);return r
   def transform(self,x,*a,**kw):
    r=super().transform(x,*a,**kw)
    if len(examples)<3:examples.append((x,r.copy()))
    return r
  return V
 def capture_model(base):
  class M(base):
   def fit(self,*a,**kw):
    r=super().fit(*a,**kw);models.append(self);return r
  return M
 for n in ['DictVectorizer','TfidfVectorizer']:ns[n]=capture_vec(ns[n])
 for n in ['SGDClassifier','LogisticRegression']:ns[n]=capture_model(ns[n])
 def begin(name,extra):
  assert name==task;old=json.loads((ROOT/f'results/practical-{task}/protocol.json').read_text())
  for k,v in extra.items():assert old[k]==v,(k,'protocol differs')
  write(dest/'protocol.json',dict(source_sha256=sha(code),export_source_sha256=sha(Path(__file__)),parent_sha256=sha(ROOT/f'results/practical-{task}/protocol.json'),threads=2));return dest
 ns['begin']=begin
 with threadpool_limits(limits=2):ns[task]()
 model=models[-1];vec=vectors[-1];result=json.loads((dest/'result.json').read_text());meta=dict(kind=task,features=vec.get_feature_names_out().tolist(),weights=model.coef_.tolist(),bias=model.intercept_.tolist(),classes=model.classes_.tolist(),source=source,source_sha256=sha(code),threshold=result.get('threshold',.5))
 if task=='routing':meta.update(idf=vec.idf_.tolist(),categories=json.loads((ROOT/'results/practical-routing/protocol.json').read_text())['categories'])
 write(ASSETS/(task+'.json'),meta)
 fixtures=[]
 for values,x in examples:
  # Small examples validate the exported readout independently; not new quality evidence.
  count=min(4,x.shape[0]);prob=model.predict_proba(x[:count]).tolist()
  fixtures.extend(dict(features=values[i],probabilities=prob[i]) for i in range(count))
 write(dest/'fixtures.json',fixtures)
 old=json.loads((ROOT/f'results/practical-{task}/result.json').read_text());same=all(v==result[k] for k,v in old.items() if k!='seconds')
 write(dest/'export.json',dict(model_sha256=sha(ASSETS/(task+'.json')),bytes=(ASSETS/(task+'.json')).stat().st_size,original_summary_matches=same,scope='Same-data model re-creation and export; no new quality test. All earlier artifacts preserved.'))
 print(task,'exported',same,flush=True)
if __name__=='__main__':
 for task,source in [('privacy','results/practical-privacy/implementation.py'),('receipts','experiments/practical_baselines.py'),('routing','results/practical-routing/implementation.py')]:run(task,source)
 write(ASSETS/'manifest.json',{k:dict(sha256=sha(ASSETS/(k+'.json')),bytes=(ASSETS/(k+'.json')).stat().st_size) for k in ['privacy','receipts','routing']})

"""Fixed word/phrase baseline, using exactly the Qwen experiment's training IDs."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import csv, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/.cache/tooling'))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

def main():
    out=ROOT/'results/banking77';manifest=json.loads((out/'data-manifest.json').read_text())
    categories=manifest['categories']; data={}
    for split in ['train','test']:
        with (ROOT/f'experiments/.cache/banking77/{split}.csv').open(encoding='utf-8',newline='') as f:
            for i,row in enumerate(csv.DictReader(f)):data[f'{split}:{i}']=(row['text'],categories.index(row['category']))
    train=[data[i] for i in manifest['splits']['train']];test=[data[i] for i in manifest['splits']['test']]
    with threadpool_limits(limits=1):
        vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=12000,sublinear_tf=True)
        x=vectorizer.fit_transform([r[0] for r in train]);model=LogisticRegression(C=4,max_iter=400,solver='lbfgs')
        start=time.perf_counter();model.fit(x,[r[1] for r in train]);train_seconds=time.perf_counter()-start
        predictions=model.predict(vectorizer.transform([r[0] for r in test]));correct=predictions==[r[1] for r in test]
        times=[]
        for row in test[:96]:
            start=time.perf_counter();model.predict(vectorizer.transform([row[0]]));times.append((time.perf_counter()-start)*1000)
    result={'model':'TF-IDF word 1–2 grams + multinomial logistic regression','sklearn':__import__('sklearn').__version__,
            'training_samples':len(train),'test_samples':len(test),'features':len(vectorizer.vocabulary_),'C':4,'max_iter':400,
            'iterations':model.n_iter_.tolist(),'converged':bool(max(model.n_iter_)<400),'correct':int(correct.sum()),'accuracy':float(correct.mean()),
            'training_seconds':train_seconds,'diagnostic_end_to_end_p50_ms':float(np.median(times)),'threads':1,
            'limitations':['Fixed simple lexical baseline, not extensively tuned.','Same training IDs and known exact duplicates as the initial Qwen run.',
                           'CPU timings may overlap the model pilots; diagnostic only, no controlled speed ratio.'],
            'predictions':[{'id':i,'prediction':int(p),'label':r[1]} for i,p,r in zip(manifest['splits']['test'],predictions,test)]}
    (out/'lexical-baseline.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'}))

if __name__=='__main__':main()

"""Regression checks for the illustrative neural example's documented failures."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys, json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples'))
from core_implementation import GenerativeModel, PLGLCore, PreferenceSample
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(17)

class Identity(GenerativeModel):
    def generate(self,z):return z
    def encode(self,x):return x

def main():
    checks=[]
    for n,batch in [(10,32),(11,7),(41,8)]:
        core=PLGLCore(Identity(),4,4,device='cpu')
        x=torch.randn(n,4);samples=[PreferenceSample(z.unsqueeze(0),z.unsqueeze(0),float(z[0]>0)) for z in x]
        core.train_preference_model(samples,epochs=2,batch_size=batch)
        assert all(torch.isfinite(torch.tensor(h['train_loss'])) for h in core.training_history)
        assert core._diverse_sampling(3).shape==(3,4)
        assert core._uncertainty_sampling(2).shape==(2,4)
        try:core.generate_distribution(n_samples=1,optimization_steps=1,min_score_threshold=1.1,max_attempts=2)
        except RuntimeError as error:assert 'after 2 attempts' in str(error)
        else:raise AssertionError('Unattainable sampling threshold did not terminate explicitly')
        checks.append({'samples':n,'batch_size':batch,'epochs':2,'finite_loss':True,'bounded_sampling':True})
    out=ROOT/'results/reference';out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps({'status':'passed','scope':'CPU mock generator; small-batch and bounded-loop regressions, no domain validation','threads':1,'cases':checks},indent=2),encoding='utf-8')
    print('Reference regressions passed: three small-batch configurations and impossible sampling targets.')
if __name__=='__main__':main()

"""Run the unchanged sealed candidates in separate two-thread processes.

Only execution scheduling changes. All seeds, batches, objectives, epochs and
selection rules are those in compact_next.py and its original sealed protocol.
"""
import argparse,copy,shutil
import compact_next as study
from compact_next import ROOT,OUT,LOCAL,sha,read,write,torch

def run(key):
    p,old,rows,qt,bt=study.load();kind,seed=key.split('-');seed=int(seed)
    assert kind in p['variants'] and seed in p['seeds']
    path=OUT/'runs'/key;path.mkdir(parents=True,exist_ok=True)
    assert not (path/'fit.json').exists(),'Preserve completed candidate.'
    shutil.copyfile(OUT/'protocol.json',path/'protocol.json')
    filtered=copy.deepcopy(p);filtered['variants']=[kind];filtered['seeds']=[seed]
    study.OUT=path
    study.write('execution.json',dict(wrapper_sha256=sha(__file__),candidate=key,threads=2,
        original_protocol_sha256=sha(OUT/'protocol.json'),scope='Unchanged sealed candidate, executed independently. Final choice remains a global development-only comparison of all six.'))
    study.load=lambda:(filtered,old,rows,qt,bt)
    study.fit()

def merge():
    study.guard('fit.json');p,*_=study.load();candidates={};sources={}
    for kind in p['variants']:
        for seed in p['seeds']:
            key=f'{kind}-{seed}';path=OUT/'runs'/key;record=read(path/'fit.json')
            assert record['protocol_sha256']==sha(OUT/'protocol.json')
            assert set(record['candidates'])=={key}
            value=record['candidates'][key];assert sha(LOCAL/(key+'.pt'))==value['weights_sha256']
            candidates[key]=value;sources[key]=sha(path/'fit.json')
            for suffix in ['-gate.npz','-development.json']:
                shutil.copyfile(path/(key+suffix),OUT/(key+suffix))
    chosen=max(candidates,key=lambda k:study.rank(candidates[k]['selected']['selected']))
    write('fit.json',dict(frozen_utc=study.datetime.now(study.timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.json'),selected=chosen,candidates=candidates,execution=dict(wrapper_sha256=sha(__file__),run_sha256=sources)))
    print('Frozen global candidate: '+chosen,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('candidate');args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        if args.candidate=='merge':merge()
        else:run(args.candidate)

"""Frozen Qwen reference on the same fresh500, plus a trivial SAFE baseline.

prepare before fresh evaluation; run after candidate selection has been sealed.
"""
import argparse
import compact_next as study
from compact_next import ROOT,OUT,sha,read,write,torch,np,metrics
from chat_next_methods import Runtime,original

def prepare():
    study.guard('reference-protocol.json');assert not (OUT/'result.json').exists()
    p=read(OUT/'protocol.json');old=read(ROOT/'results/chat600/protocol.json')
    write('reference-protocol.json',dict(source_sha256=sha(__file__),parent_sha256=sha(OUT/'protocol.json'),ids=p['splits']['fresh'],
        model=old['model'],revision=old['model_revision'],system_prompt=old['system_prompt'],heads_sha256=sha(ROOT/'results/chat600/heads.npz'),
        batch=8,threads=2,scope='Frozen original Qwen classifier; same500 heldout inputs and256-Qwen-token bound. Left-padding and final nonpadding token. No training, threshold choice or timing. Also report always SAFE so the sample label mix is visible. Reference supplemental protocol frozen before newtest evaluation.'))
    print('Reference sealed.')

def run():
    study.guard('reference.json');p=read(OUT/'reference-protocol.json');assert p['source_sha256']==sha(__file__)
    assert p['parent_sha256']==sha(OUT/'protocol.json');fit=read(OUT/'fit.json')
    assert fit['protocol_sha256']==p['parent_sha256'];assert p['heads_sha256']==sha(ROOT/'results/chat600/heads.npz')
    _,_,rows,_,_=study.load();rt=Runtime(p);rt.tokenizer.padding_side='left'
    encoded={rid:rt.tokens(rows[rid]) for rid in p['ids']};ordered=sorted(p['ids'],key=lambda r:len(encoded[r]));records={}
    with torch.inference_mode():
        for start in range(0,len(ordered),p['batch']):
            ids=ordered[start:start+p['batch']]
            batch=rt.tokenizer.pad([{'input_ids':encoded[r]} for r in ids],padding=True,return_tensors='pt')
            assert bool((batch['attention_mask'][:,-1]==1).all())
            h=rt.model(**batch,use_cache=False).last_hidden_state[:,-1,:];logits=original(rt.heads,24,h)
            for i,rid in enumerate(ids):records[rid]=dict(id=rid,label=rows[rid]['label'],prediction=int(logits[i].argmax()),logits=logits[i].tolist())
            if start%80==0:print('Qwen fresh reference',start+len(ids),'/ 500',flush=True)
    values=[records[r] for r in p['ids']];y=np.array([r['label'] for r in values]);pred=np.array([r['prediction'] for r in values])
    write('reference-records.json',values);write('reference.json',dict(protocol_sha256=sha(OUT/'reference-protocol.json'),qwen=metrics(y,pred),always_safe=metrics(y,np.zeros_like(y))))
    print(study.json.dumps(read(OUT/'reference.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):globals()[args.stage]()

"""Actual target-readout runtime replay; no optimization or cached predictions."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/.cache/replay-runtime'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from chat_smoke_common import prepare
from chat_smoke_adaptation import LowRank
OUT=ROOT/'results/chat-smoke-adaptation'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--variant',choices=['full','joint','distill','fixed12'])
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    common,splits=prepare();rows=splits['evaluation'][:50]
    protocol=dict(scope='Single 50-message exploratory warm CPU pass per variant, not a replicated speed benchmark.',
        ids=[r['id'] for r in rows],selection='First 50 common evaluation IDs, fixed before any runtime outcomes.',
        common_sha256=sha(ROOT/'results/chat-smoke/protocol.json'),script_sha256=sha(Path(__file__)),
        boundary='Includes original user-text tokenization, first256 Qwen-token cap and decoding when needed, chat templating/re-tokenization, transformer blocks, one target head, argmax, execution trace overhead. Excludes loading, training and one unmeasured warmup call.',
        target='full/joint/distill: final normalized block24 state to head24 only. fixed12: physically12 blocks, raw block12 state to head12 only.',
        runtime='CPUfloat32, eagerattention,4threads, batchone, published Transformers4.50.3.',
        totals='Sum of per-request durations, not one whole-corpus wall timer. Different variants execute separately, not interleaved; no uncertainty estimate.')
    pp=OUT/'runtime-protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:write(pp.name,protocol)
    if args.prepare:return
    assert args.variant,'Select a variant.'
    variant=args.variant;target=OUT/f'{variant}-runtime.json'
    assert not target.exists(),'Preserve completed timing; use a new output directory for repeats.'
    training=json.loads((OUT/f'{variant}-protocol.json').read_text())
    assert training['common_sha256']==protocol['common_sha256']
    assert sha(ROOT/'experiments/chat_smoke_adaptation.py')==training['script_sha256']
    assert sha(OUT/'implementation.py')==training['script_sha256']
    saved=json.loads((OUT/f'{variant}.json').read_text())
    assert saved['protocol']==training
    weight_path=OUT/f'{variant}-weights.npz'
    replay=dict(variant=variant,training_protocol_sha256=sha(OUT/f'{variant}-protocol.json'),
                weights_sha256=sha(weight_path),result_sha256=sha(OUT/f'{variant}.json'))
    wp=OUT/f'{variant}-runtime-inputs.json'
    if wp.exists():assert json.loads(wp.read_text())==replay
    else:write(wp.name,replay)
    arrays=np.load(weight_path,allow_pickle=False)
    tokenizer=AutoTokenizer.from_pretrained(common['qwen_model'],revision=common['qwen_revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(common['qwen_model'],revision=common['qwen_revision'],
        local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').model
    goal=12 if variant=='fixed12' else 24
    if goal==12:model.layers=torch.nn.ModuleList(list(model.layers[:12]));model.config.num_hidden_layers=12
    for layer in model.layers:
        for name in ['q_proj','v_proj']:setattr(layer.self_attn,name,LowRank(getattr(layer.self_attn,name)))
    adapter={k.removeprefix('adapter::'):torch.tensor(arrays[k]) for k in arrays.files if k.startswith('adapter::')}
    assert set(adapter)=={k for k in model.state_dict() if k.endswith(('.a','.b'))}
    incompatible=model.load_state_dict(adapter,strict=False);assert not incompatible.unexpected_keys
    assert all(not k.endswith(('.a','.b')) for k in incompatible.missing_keys)
    h={k:torch.tensor(arrays[f'head::{goal}.{k}']) for k in ['mean','std','linear.weight','linear.bias']}
    assert all(torch.isfinite(v).all() for v in h.values())
    model.eval();visited=[];captured=[]
    def hook(d):
        def after(mod,inp,out):
            visited.append(d)
            if d==12 and goal==12:captured.append(out[0][:,-1,:])
        return after
    handles=[layer.register_forward_hook(hook(d)) for d,layer in enumerate(model.layers,1)]
    def execute(row):
        start=time.perf_counter();visited.clear();captured.clear()
        ids=tokenizer.encode(row['text'],add_special_tokens=False)
        message=tokenizer.decode(ids[:256],skip_special_tokens=False) if len(ids)>256 else row['text']
        text=tokenizer.apply_chat_template([{'role':'system','content':common['system_prompt']},
            {'role':'user','content':message}],tokenize=False,add_generation_prompt=True)
        out=model(**tokenizer(text,return_tensors='pt'),use_cache=False)
        state=captured[0] if goal==12 else out.last_hidden_state[:,-1,:]
        pred=int(torch.nn.functional.linear((state-h['mean'])/h['std'],h['linear.weight'],h['linear.bias']).argmax(1))
        duration=time.perf_counter()-start
        assert visited==list(range(1,goal+1))
        return dict(id=row['id'],label=row['label'],prediction=pred,executed_layers=list(visited),seconds=duration)
    expected={r['id']:r for r in saved['predictions'] if r['split']=='evaluation'}
    with torch.inference_mode():
        execute(rows[0]);records=[]
        for row in rows:
            item=execute(row)
            assert item['prediction']==expected[row['id']]['heads'][str(goal)]['prediction']
            records.append(item)
    for handle in handles:handle.remove()
    assert sha(weight_path)==replay['weights_sha256']
    write(target.name,dict(protocol_sha256=sha(pp),inputs=replay,n=len(records),retained_blocks=len(model.layers),
        executed_readout_depth=goal,seconds=sum(r['seconds'] for r in records),
        correct=sum(r['prediction']==r['label'] for r in records),toxic=sum(r['label']==1 for r in records),
        parity_passed=True,records=records))
    print(json.dumps(dict(variant=variant,n=len(records),seconds=sum(r['seconds'] for r in records),parity_passed=True)),flush=True)

if __name__=='__main__':main()

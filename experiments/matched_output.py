"""Matched supervised enum versus constrained one-token output.

Both use identical frozen representations and learned output rows. This controls
output format; it is not a comparison to Qwen's untouched vocabulary decoder.
Run alone for timings: eight CPU threads, 32 queries, two alternating repeats.
"""
import os
os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8';os.environ['TOKENIZERS_PARALLELISM']='false'
import csv,hashlib,json,time,random,statistics
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from banking77 import MODEL,REVISION,PROMPT
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/matched-output';BANK=ROOT/'results/banking77'

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(min(8,max(1,(os.cpu_count() or 2)//2)));torch.set_num_interop_threads(1)
    original=json.loads((BANK/'result.json').read_text());manifest=json.loads((BANK/'data-manifest.json').read_text())
    arrays=np.load(BANK/'seed17-heads.npz',allow_pickle=False)
    w=torch.from_numpy(arrays['layer_24_weight'].copy());b=torch.from_numpy(arrays['layer_24_bias'].copy())
    mean=torch.from_numpy(arrays['layer_24_mean'].copy());std=torch.from_numpy(arrays['layer_24_std'].copy())
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,padding_side='left');tokenizer.pad_token=tokenizer.eos_token
    symbols=[chr(i) for i in range(33,110)]
    symbol_to_label={symbol:i for i,symbol in enumerate(symbols)}
    token_ids=[tokenizer.encode(symbol,add_special_tokens=False) for symbol in symbols]
    assert all(len(x)==1 for x in token_ids) and len({x[0] for x in token_ids})==77
    token_ids=torch.tensor([x[0] for x in token_ids])
    def enum(hidden):return int(torch.nn.functional.linear((hidden-mean)/std,w,b).argmax(1).item())
    def text(hidden):
        # A constrained decoder only needs its 77 supported token rows, not the
        # entire vocabulary matrix. Same supervised rows are intentionally reused.
        logits=torch.nn.functional.linear((hidden-mean)/std,w,b)
        token=int(token_ids[logits.argmax(1)].item());decoded=tokenizer.decode([token],skip_special_tokens=True)
        return symbol_to_label[decoded],token,decoded
    cache=torch.load(ROOT/'experiments/.cache/banking77'/f'{original["feature_fingerprint"]}.pt',weights_only=True)
    cached=cache['features']['test'][24];reference={r['id']:r for r in json.loads((BANK/'predictions.json').read_text()) if r['seed']==17}
    predictions=[]
    with torch.inference_mode():
        for i,identifier in enumerate(manifest['splits']['test']):
            a=enum(cached[i:i+1]);c,t,decoded=text(cached[i:i+1]);assert a==c==reference[identifier]['full_prediction']
            predictions.append({'id':identifier,'label':reference[identifier]['label'],'enum':a,'token_result':c,'token_id':t,'text':decoded})
    ids=sorted(manifest['splits']['test'],key=lambda x:hashlib.sha256(('matched-output-20260919:'+x).encode()).hexdigest())[:32]
    protocol={'type':'Matched output-format control','ids':ids,'repeats':2,'paths':['enum','single_token'],'model':MODEL,'revision':REVISION,
              'heads_sha256':hashlib.sha256((BANK/'seed17-heads.npz').read_bytes()).hexdigest(),'token_ids':token_ids.tolist(),'label_to_symbol':symbols,
              'timing':'Warm model, includes tokenization, full24-block model call, learned readout, and token decoding for the text path. No other launched model jobs.',
              'limits':'Identical learned rows; not independently fine-tuned language generation, not arbitrary text or JSON. No early exit in either path.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    data={}
    with (ROOT/'experiments/.cache/banking77/test.csv').open(encoding='utf-8',newline='') as f:
        for i,row in enumerate(csv.DictReader(f)):data[f'test:{i}']=row['text']
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').eval()
    def run(identifier,path):
        start=time.perf_counter()
        rendered=tokenizer.apply_chat_template([{'role':'user','content':PROMPT.format(text=data[identifier])}],tokenize=False,add_generation_prompt=True)
        batch=tokenizer(rendered,return_tensors='pt');hidden=model.model(**batch,use_cache=False).last_hidden_state[:,-1,:]
        result=enum(hidden) if path=='enum' else text(hidden)[0]
        return result,(time.perf_counter()-start)*1000
    timings=[]
    with torch.inference_mode():
        for path in ['enum','single_token']:run(ids[0],path)
        for i,identifier in enumerate(ids):
            for repeat in range(2):
                for path in (['enum','single_token'] if (i+repeat)%2==0 else ['single_token','enum']):
                    prediction,elapsed=run(identifier,path);assert prediction==reference[identifier]['full_prediction']
                    timings.append({'id':identifier,'repeat':repeat,'path':path,'prediction':prediction,'end_to_end_ms':elapsed})
    pairs=[tuple(statistics.mean(r['end_to_end_ms'] for r in timings if r['id']==i and r['path']==p) for p in ['enum','single_token']) for i in ids]
    rng=random.Random(19);differences=sorted(statistics.mean(b-a for a,b in rng.choices(pairs,k=len(pairs))) for _ in range(2000))
    result={'queries':len(predictions),'correct':sum(r['label']==r['enum'] for r in predictions),'all_outputs_identical':True,
            'timing_queries':len(ids),'timing':{p:{'mean':float(np.mean([r['end_to_end_ms'] for r in timings if r['path']==p])),
                                                'p50':float(np.median([r['end_to_end_ms'] for r in timings if r['path']==p]))} for p in ['enum','single_token']},
            'paired_token_minus_enum_ms':{'mean':statistics.mean(b-a for a,b in pairs),'query_bootstrap_ci95':[differences[49],differences[1949]]},
            'interpretation':'Equality is an invariant of sharing learned rows, not new accuracy evidence. One constrained output token can implement the same trained classifier. Avoiding a token alone does not establish a useful speedup; early transformer termination is a separate optimization.'}
    for name,value in [('predictions',predictions),('timings',timings),('result',result)]:
        (OUT/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

"""Fixed public ScreenSpot sample; compare single-patch and connected-region readouts."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse, hashlib, json, random, time, urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from coordinates import Pointer, MODEL, REVISION

ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'experiments/.cache/screenspot'; OUT=ROOT/'results/screenspot'
DATA_REV='0be08781e2e188582f6131625ae1598d443b4d5d'

def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=45) as response:return response.read()
        except (OSError,TimeoutError):
            if attempt==3:raise
            time.sleep(2*(attempt+1))

def prepare():
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'data-manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if all((CACHE/r['file_name']).exists() for r in manifest['samples']):
            assert all(hashlib.sha256((CACHE/r['file_name']).read_bytes()).hexdigest()==r['sha256'] for r in manifest['samples']), 'Cached screenshot changed'
            return manifest
    rows=[]
    for offset in range(0,1272,100):
        url=f'https://datasets-server.huggingface.co/rows?dataset=bevaya/ScreenSpot&config=default&split=test&offset={offset}&length=100'
        page=CACHE/f'rows-{offset}.json'
        if not page.exists():page.write_bytes(fetch(url))
        response=json.loads(page.read_bytes()); assert response['num_rows_total']==1272
        rows.extend({'row_index':item['row_idx'],**item['row']} for item in response['rows'])
        print(f'Dataset metadata: {len(rows)}/1272',flush=True)
    groups=defaultdict(list)
    for r in rows:
        source=r['data_source'].lower()
        platform=source if source in ['windows','macos','ios','android'] else 'web'
        groups[(platform,r['data_type'])].append(r)
    selected=[]
    for key,group in sorted(groups.items()):
        random.Random('screenspot-20260919-'+str(key)).shuffle(group)
        selected.extend(group[:3])
    samples=[]
    for r in selected:
        url=r.pop('image')['src']; assert DATA_REV in url, 'Dataset server revision changed'
        image=fetch(url); path=CACHE/r['file_name'];path.write_bytes(image)
        with Image.open(path) as im:width,height=im.size
        samples.append({**r,'width':width,'height':height,'sha256':hashlib.sha256(image).hexdigest()})
    manifest={'dataset':'bevaya/ScreenSpot (formerly rootsautomation/ScreenSpot)','revision':DATA_REV,
              'authors':'Cheng et al., SeeClick, 2024','license':'Apache-2.0 dataset card; screenshots retain depicted third-party content',
              'selection':'Three per platform family and element type, deterministic shuffled sample before inference; no outcome selection.',
              'population':1272,'groups':{str(k):len(v) for k,v in groups.items()},'samples':samples}
    write(manifest_path,manifest);return manifest

def region_point(prob,width,height):
    # GUI-Actor's published 0.3-of-peak, four-neighbour connected-region rule.
    active=set(np.flatnonzero(prob>prob.max()*.3).tolist());regions=[]
    while active:
        start=min(active);active.remove(start);region=[start];stack=[start]
        while stack:
            i=stack.pop();x=i%width;y=i//width
            for nx,ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                j=ny*width+nx
                if 0<=nx<width and 0<=ny<height and j in active:active.remove(j);region.append(j);stack.append(j)
        regions.append(region)
    best=max(regions,key=lambda ids:float(prob[ids].mean()))
    weights=prob[best];return [float(sum((i%width+.5)/width*w for i,w in zip(best,weights))/weights.sum()),float(sum((i//width+.5)/height*w for i,w in zip(best,weights))/weights.sum())]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--threads',type=int,default=4);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    manifest=prepare();print(f'Fixed {len(manifest["samples"])} public samples.',flush=True)
    if args.prepare_only:return
    threads=max(1,min(args.threads,(os.cpu_count() or 2)//2));torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    checkpoint=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    processor=AutoProcessor.from_pretrained(checkpoint,min_pixels=256*28*28,max_pixels=576*28*28)
    # This installed Transformers revision reads size during preprocessing; the
    # legacy max_pixels attribute alone does not change the loaded size mapping.
    processor.image_processor.size={'shortest_edge':256*28*28,'longest_edge':576*28*28}
    model=Qwen2VLForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.float32,attn_implementation='eager').eval()
    pointer=Pointer(model.config.hidden_size).eval();state={}
    for shard in checkpoint.glob('*.safetensors'):
        with safe_open(shard,framework='pt') as reader:
            for key in reader.keys():
                if key.startswith('multi_patch_pointer_head.'):state[key.removeprefix('multi_patch_pointer_head.')]=reader.get_tensor(key).float()
    pointer.load_state_dict(state,strict=True)
    traces=json.loads((OUT/'predictions.json').read_text()) if args.resume and (OUT/'predictions.json').exists() else []
    for i,trace in enumerate(traces):
        assert trace['row_index']==manifest['samples'][i]['row_index'] and trace['bbox']==manifest['samples'][i]['bbox']
        assert trace['patch_grid'][0]*trace['patch_grid'][1]<=576
        trace.setdefault('threads',4)
    with torch.inference_mode():
        for index,row in enumerate(manifest['samples']):
            if index<len(traces):continue
            started=time.perf_counter()
            prompt='<|im_start|>system\nYou are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'+row['instruction']+'<|im_end|>\n<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)'
            batch=processor(text=[prompt],images=[Image.open(CACHE/row['file_name']).convert('RGB')],return_tensors='pt')
            assert int(batch['image_grid_thw'].prod())//4<=576, 'Visual-token cap was not applied'
            visual=model.visual(batch['pixel_values'].to(model.visual.dtype),grid_thw=batch['image_grid_thw'])
            ids=batch['input_ids']; embedded=model.model.embed_tokens(ids);mask=(ids==model.config.image_token_id).unsqueeze(-1).expand_as(embedded)
            embedded=embedded.masked_scatter(mask,visual);positions,_=model.get_rope_index(ids,batch['image_grid_thw'],None,batch['attention_mask'])
            hidden=model.model(inputs_embeds=embedded,attention_mask=batch['attention_mask'],position_ids=positions,use_cache=False).last_hidden_state
            prob=pointer(visual,hidden[ids==model.config.pointer_pad_token_id]).squeeze(0).numpy()
            _,height,width=batch['image_grid_thw'][0].tolist();height//=processor.image_processor.merge_size;width//=processor.image_processor.merge_size
            argmax=int(prob.argmax());points={'max_patch':[(argmax%width+.5)/width,(argmax//width+.5)/height],'connected_region':region_point(prob,width,height)}
            x1,y1,x2,y2=row['bbox'];hits={k:x1<=p[0]<=x2 and y1<=p[1]<=y2 for k,p in points.items()}
            traces.append({'row_index':row['row_index'],'file_name':row['file_name'],'instruction':row['instruction'],'data_source':row['data_source'],'data_type':row['data_type'],
                           'bbox':row['bbox'],'points':points,'hits':hits,'peak_probability':float(prob.max()),'patch_grid':[width,height],
                           'threads':threads,'elapsed_ms':(time.perf_counter()-started)*1000})
            write(OUT/'predictions.json',traces)
            print(f'{index+1}/{len(manifest["samples"])} {row["data_source"]} {row["data_type"]}: {hits}',flush=True)
    summary={'status':'completed','model':MODEL,'revision':REVISION,'dataset_revision':DATA_REV,'samples':len(traces),
             'hits':{key:sum(t['hits'][key] for t in traces) for key in ['max_patch','connected_region']},
             'by_type':{kind:{key:sum(t['hits'][key] for t in traces if t['data_type']==kind) for key in ['max_patch','connected_region']} for kind in ['text','icon']},
             'threads_used':sorted({r['threads'] for r in traces}),'max_visual_tokens':576,'transformer_layers':28,'decoding_tokens':0,
             'upstream_reference':'microsoft/GUI-Actor@d98d1bbd01862f9112114b83b032f492c365a173',
             'limitations':['Small stratified public benchmark sample, not the full leaderboard; possible unknown training overlap.',
                            'Frozen published pointer head, no new coordinate classifier training; no missing-target examples or abstention.',
                            'All 28 transformer blocks execute; no coordinate early exit.',
                            'Both readouts share one model pass. This is a localization comparison, not a text-versus-coordinate speed claim.',
                            'CPU elapsed times are diagnostic only; concurrent feature extraction and first-run effects may influence them.']}
    write(OUT/'result.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

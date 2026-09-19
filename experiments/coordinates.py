"""Local GUI-Actor pointer-head reproduction; no text decoding and no live clicks.

Architecture and pretrained weights: https://github.com/microsoft/GUI-Actor
This small, original inference adapter uses the published pointer architecture.
Fixtures are our own screenshots, not a benchmark. Prompt differs from upstream.

Historical reproduction: this runtime did not apply the legacy max_pixels option
to the saved processor size mapping. Actual grids are retained in predictions.
Use screenspot.py for the corrected, explicitly asserted visual-token cap.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ROOT=Path(__file__).resolve().parents[1]
MODEL='microsoft/GUI-Actor-2B-Qwen2-VL'
REVISION='8f87b366d004425a9823502553e2097c71116ece'

class Pointer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width=width
        for name in ['projection_enc','projection_dec']:
            setattr(self,name,nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width)))
        self.self_attention=nn.MultiheadAttention(width,8,batch_first=True)
        self.layer_norm=nn.LayerNorm(width)
    def forward(self, image, query):
        image=image.unsqueeze(0)
        context=self.self_attention(image,image,image,need_weights=False)[0]
        keys=self.projection_enc(self.layer_norm(image+context).squeeze(0))
        return (self.projection_dec(query)@keys.T/self.width**.5).softmax(-1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=8);args=parser.parse_args()
    threads=max(1,min(args.threads,(os.cpu_count() or 2)//2));torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    checkpoint=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    processor=AutoProcessor.from_pretrained(checkpoint,min_pixels=256*28*28,max_pixels=576*28*28)
    model=Qwen2VLForConditionalGeneration.from_pretrained(checkpoint,torch_dtype=torch.float32,attn_implementation='eager').eval()
    pointer=Pointer(model.config.hidden_size).eval();state={}
    for shard in checkpoint.glob('*.safetensors'):
        with safe_open(shard,framework='pt') as reader:
            for key in reader.keys():
                if key.startswith('multi_patch_pointer_head.'):
                    state[key.removeprefix('multi_patch_pointer_head.')]=reader.get_tensor(key).float()
    pointer.load_state_dict(state,strict=True)
    fixtures=json.loads((ROOT/'experiments/fixtures/coordinates.json').read_text())
    def prepare(row):
        # Literal prompt prefix exposes one pointer query; no autoregressive loop.
        prompt='<|im_start|>system\nLocate the requested GUI element in the screenshot. Return a click action using the special pointer tokens.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>'+row['instruction']+'<|im_end|>\n<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)'
        return processor(text=[prompt],images=[Image.open(ROOT/row['image']).convert('RGB')],return_tensors='pt')
    def forward(batch, vocabulary=False):
        ticks={};start=time.perf_counter()
        image=model.visual(batch['pixel_values'].to(model.visual.dtype),grid_thw=batch['image_grid_thw'])
        ticks['vision_ms']=(time.perf_counter()-start)*1000;start=time.perf_counter()
        ids=batch['input_ids'];embedded=model.model.embed_tokens(ids)
        mask=(ids==model.config.image_token_id).unsqueeze(-1).expand_as(embedded)
        embedded=embedded.masked_scatter(mask,image)
        positions,_=model.get_rope_index(ids,batch['image_grid_thw'],None,batch['attention_mask'])
        hidden=model.model(inputs_embeds=embedded,attention_mask=batch['attention_mask'],position_ids=positions,use_cache=False).last_hidden_state
        ticks['transformer_ms']=(time.perf_counter()-start)*1000;start=time.perf_counter()
        query=hidden[ids==model.config.pointer_pad_token_id]
        probabilities=pointer(image,query).squeeze(0)
        ticks['pointer_ms']=(time.perf_counter()-start)*1000
        if vocabulary:
            # Paired ablation: same pointer path plus exactly one full vocabulary projection.
            # It is not JSON generation or a separately trained coordinate-token baseline.
            model.lm_head(hidden[:,-1,:]).argmax(-1).item()
        return probabilities,ticks
    traces=[]
    with torch.inference_mode():
        forward(prepare(fixtures[0]));print('Pointer warm-up complete.',flush=True)
        for i,row in enumerate(fixtures):
            batch=prepare(row);outputs={};timings={};parts={}
            for path in (['direct','with_vocabulary'] if i%2==0 else ['with_vocabulary','direct']):
                started=time.perf_counter();outputs[path],parts[path]=forward(batch,path=='with_vocabulary');timings[path]=(time.perf_counter()-started)*1000
            assert torch.allclose(outputs['direct'],outputs['with_vocabulary'],atol=1e-6)
            p=outputs['direct'];index=int(p.argmax());merge=processor.image_processor.merge_size
            _,height,width=batch['image_grid_thw'][0].tolist();height//=merge;width//=merge
            x=(index%width+.5)/width;y=(index//width+.5)/height
            box=row['box'];px=x*row['width'];py=y*row['height']
            hit=None if box is None else box['x']<=px<=box['x']+box['width'] and box['y']<=py<=box['y']+box['height']
            traces.append({**row,'x':x,'y':y,'patch_probability':float(p.max()),'hit':hit,'no_target_supported':False,'patch_grid':[width,height],'timing_ms':timings,'components_ms':parts['direct']})
            print(f'{row["id"]}: ({x:.3f}, {y:.3f}) hit={hit}; {timings}',flush=True)
    target=[r for r in traces if r['box'] is not None]
    payload={'experiment_ids':['C01','C02'],'status':'reproduction pilot','model':MODEL,'revision':REVISION,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'hardware':platform.processor(),'threads':threads,'dtype':'float32','fixture_hash':hashlib.sha256(json.dumps(fixtures,sort_keys=True).encode()).hexdigest(),'target_count':len(target),'hits':sum(r['hit'] for r in target),'absent_targets':len(traces)-len(target),'timing_ms':{k:{'p50':float(np.median([r['timing_ms'][k] for r in traces])),'p95':float(np.percentile([r['timing_ms'][k] for r in traces],95)),'samples':len(traces)} for k in ['direct','with_vocabulary']},'limitations':['Eight fixtures from one locally authored interface at two viewports; not a GUI benchmark.','Pretrained GUI-Actor pointer weights; no new head was trained. Custom shortened prompt. Historical processor configuration did not enforce the intended 576-patch cap; actual grids are recorded in predictions.','Always chooses a visual patch; absent targets cannot be rejected. Patch probability is not calibrated correctness.','Both paths use the same pointer head. The paired vocabulary projection only isolates output-head overhead; it is not a text/JSON baseline.','CPU batch one, warm model. Timings exclude preprocessing/loading; eight samples are too few for stable tail latency.','All transformer layers run. Coordinate early exit and end-to-end task execution are not tested.']}
    out=ROOT/'results/coordinates';out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(traces,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':main()

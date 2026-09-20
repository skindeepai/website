"""Train and export a real tiny pixel-to-action model on synthetic arrow cards."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse,base64,hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/action-demo/arrows'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def plan():
    return {'scope':'Synthetic visual-instruction classifier. Not real screenshots, GUI understanding, a language model or a maze policy.',
        'source_sha256':{p:sha(p) for p in ['experiments/action_demo_arrows.py','scripts/action-demo-core.js']},
        'input':'Only1024 grayscale pixel intensities, normalized as1-gray/255. Seed, direction, style and render geometry never enter the classifier.',
        'labels':['UP','RIGHT','DOWN','LEFT'],'architecture':'1024→32ReLU→4 linear logits;32932 trained parameters; argmax returns one enum.',
        'splits':{'train':{'seed_start':100000,'n':2400,'style':'filled'},'tune':{'seed_start':200000,'n':400,'style':'filled'},
                  'test':{'seed_start':300000,'n':400,'style':'filled'},'outline_stress':{'seed_start':400000,'n':400,'style':'outline'}},
        'labels_rule':'Balanced labels index%4. Each card has a distinct render seed. Train/tune/test share the same procedural renderer; onlyoutline_stress has unseen style.',
        'training':{'torch_seed':390920,'epochs':20,'batch_size':64,'learning_rate':.003,'optimizer':'Adam','threads':2},
        'selection':'Highest tune accuracy, then lower tune cross-entropy; first exact tie retained. Never tune on test or outline outputs.',
        'baseline':'Untrained nearest-pixel template among four canonical centered black solid arrows. Same1024 input intensities, squared distance; no template alignment.',
        'deployment_parity':'Export weights to JSON; independently execute JS predictor on all800 test/stress images. Require same argmax asTorch and maxabsolute logit difference<0.0005.',
        'timing':'Three alternating JS classifier/template passes over400 test pixels, warmed each path. Input rasterization and model loading excluded. SingleNodeprocess; diagnostic local numerical-readout timing, not model-vs-LLM or browser speed claim.',
        'limits':['Synthetic pixels and shared renderer can be easy and unlike real images.','Onlyfour directions; no absent-sign or unknown-image class.','Fresh render seeds are not a fresh real-world dataset.','Nearest-template baseline is deliberately simple; no comparison with a stronger vision model.','Outline stress is held out from training and epoch selection; failures retained.']}

def fixtures(p):
    definitions=[]
    for split,cfg in p['splits'].items():
        definitions.extend({'id':f'{split}:{i}','split':split,'seed':cfg['seed_start']+i,'label':i%4,'style':cfg['style']} for i in range(cfg['n']))
    script="const A=require('./scripts/action-demo-core.js');let text='';process.stdin.on('data',x=>text+=x);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(text).map(r=>({...r,pixels:Buffer.from(A.renderCard(r.seed,r.label,r.style)).toString('base64')})))));"
    result=subprocess.run(['node','-e',script],cwd=ROOT,input=json.dumps(definitions),capture_output=True,text=True,check=True)
    return json.loads(result.stdout)

def run(p):
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(p['training']['torch_seed'])
    data=fixtures(p);assert len({r['seed'] for r in data})==len(data)
    arrays={}
    for split in p['splits']:
        subset=[r for r in data if r['split']==split]
        raw=np.array([list(base64.b64decode(r['pixels'])) for r in subset],dtype=np.float32)
        arrays[split]=(torch.from_numpy(1-raw/np.float32(255)),torch.tensor([r['label'] for r in subset]))
    model=torch.nn.Sequential(torch.nn.Linear(1024,32),torch.nn.ReLU(),torch.nn.Linear(32,4))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003);history=[];best=None;best_state=None
    started=time.perf_counter()
    for epoch in range(1,21):
        model.train();order=torch.randperm(len(arrays['train'][0]));losses=[]
        for offset in range(0,len(order),64):
            ix=order[offset:offset+64];optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(arrays['train'][0][ix]),arrays['train'][1][ix]);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits=model(arrays['tune'][0]);correct=int((logits.argmax(1)==arrays['tune'][1]).sum());loss=float(torch.nn.functional.cross_entropy(logits,arrays['tune'][1]))
        history.append({'epoch':epoch,'train_loss':sum(losses)/len(losses),'tune_correct':correct,'tune_loss':loss})
        rank=(correct,-loss)
        if best is None or rank>best:
            best=rank;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};selected_epoch=epoch
    model.load_state_dict(best_state);model.eval()
    trained_seconds=time.perf_counter()-started
    export={'kind':'Trained synthetic arrow-pixel MLP','input_size':32,'input_pixels':1024,'hidden':32,'labels':p['labels'],
            'parameter_count':sum(v.numel() for v in model.parameters()),'w1':model[0].weight.detach().tolist(),'b1':model[0].bias.detach().tolist(),
            'w2':model[2].weight.detach().tolist(),'b2':model[2].bias.detach().tolist(),'protocol_sha256':sha('results/action-demo/arrows/protocol.json')}
    write('model.json',export);write('training.json',{'history':history,'selected_epoch':selected_epoch,'training_seconds':trained_seconds})
    expected=[]
    with torch.inference_mode():
        for split in ['test','outline_stress']:
            subset=[r for r in data if r['split']==split];logits=model(arrays[split][0]);pred=logits.argmax(1).tolist()
            for r,values,guess in zip(subset,logits.tolist(),pred):
                expected.append({k:v for k,v in r.items() if k!='pixels'}|{'torch_logits':values,'torch_prediction':guess,'pixels_sha256':hashlib.sha256(base64.b64decode(r['pixels'])).hexdigest()})
    write('evaluation.json',expected)
    script="""const fs=require('fs'),A=require('./scripts/action-demo-core.js'),model=JSON.parse(fs.readFileSync('results/action-demo/arrows/model.json')),rows=JSON.parse(fs.readFileSync('results/action-demo/arrows/evaluation.json'));const measured=rows.map(r=>{const pixels=A.renderCard(r.seed,r.label,r.style),neural=A.predict(pixels,model),template=A.templatePredict(pixels);return{...r,js_logits:neural.scores,js_prediction:neural.index,template_prediction:template.index};});const test=rows.filter(r=>r.split==='test').map(r=>A.renderCard(r.seed,r.label,r.style));A.predict(test[0],model);A.templatePredict(test[0]);const times=[];let checksum=0;for(let repeat=0;repeat<3;repeat++){for(const method of repeat%2?['template','neural']:['neural','template']){const start=performance.now();for(const pixels of test)checksum+=(method==='neural'?A.predict(pixels,model):A.templatePredict(pixels)).index;times.push({repeat,method,ms:performance.now()-start});}}process.stdout.write(JSON.stringify({rows:measured,times,checksum,node:process.version}));"""
    checked=json.loads(subprocess.run(['node','-e',script],cwd=ROOT,capture_output=True,text=True,check=True).stdout)
    maxdelta=max(abs(a-b) for r in checked['rows'] for a,b in zip(r['torch_logits'],r['js_logits']))
    parity=all(r['torch_prediction']==r['js_prediction'] for r in checked['rows']) and maxdelta<.0005
    summary={}
    for split in ['test','outline_stress']:
        subset=[r for r in checked['rows'] if r['split']==split]
        summary[split]={'n':len(subset),'neural_correct':sum(r['js_prediction']==r['label'] for r in subset),'template_correct':sum(r['template_prediction']==r['label'] for r in subset),
                        'confusion':[[sum(r['label']==y and r['js_prediction']==pred for r in subset) for pred in range(4)] for y in range(4)]}
    write('predictions.json',checked['rows']);write('timings.json',checked['times'])
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha('results/action-demo/arrows/protocol.json'),
        'model_sha256':sha('results/action-demo/arrows/model.json'),'summary':summary,'deployment_parity':parity,'max_abs_logit_delta':maxdelta,
        'selected_epoch':selected_epoch,'parameters':export['parameter_count'],'threads':2,'node':checked['node'],'torch':torch.__version__,'limits':p['limits']}
    write('result.json',result)
    assert parity,'JS deployment parity failed; preserve results, do not publish model as validated'
    print(json.dumps(result))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);p=plan()
    if args.prepare:
        if (OUT/'protocol.json').exists():assert read('results/action-demo/arrows/protocol.json')==p
        else:write('protocol.json',p)
        print('Arrow protocol sealed; no training/evaluation.');return
    assert read('results/action-demo/arrows/protocol.json')==p,'Sealed source changed'
    assert not (OUT/'model.json').exists(),'Preserve previous attempt'
    run(p)

if __name__=='__main__':main()

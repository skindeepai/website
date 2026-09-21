"""Pin public OPT weights locally; preserve SSN baseline sources before edits."""
import concurrent.futures
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SSN = Path('C:/Users/steve/Code/social_stream')
OUT = ROOT/'results/moderation-transfer'
CACHE = ROOT/'experiments/.cache/qwen35-opt'
MODEL = 'onnx-community/Qwen3.5-0.8B-ONNX-OPT'
REVISION = 'fafab72d87a9e6be3925b38caf48286d2838f2d0'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = OUT/'original'
    original.mkdir(exist_ok=True)
    sources = ['ai.js','local-browser-model-worker.js','shared/ai/browserModelCatalog.js','shared/ai/localBrowserLLM.js']
    for source in sources:
        target=original/source
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((SSN/source).read_bytes())
    api=f'https://huggingface.co/api/models/{MODEL}/tree/{REVISION}/onnx'
    with urllib.request.urlopen(api,timeout=30) as r: entries=json.load(r)
    files={e['path']:e for e in entries if e['path'].endswith(('_q4.onnx','_q4.onnx_data'))}
    for name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']:
        files[name]={'path':name}
    CACHE.mkdir(parents=True,exist_ok=True)
    def download(entry):
        name=entry['path'];target=CACHE/name;target.parent.mkdir(parents=True,exist_ok=True)
        expected=entry.get('lfs',{}).get('oid')
        if not target.exists():
            old=SSN/'thirdparty/models/qwen3.5-0.8b-onnx'/name
            if expected and old.exists() and sha(old)==expected: shutil.copyfile(old,target)
            else:
                temp=target.with_name(target.name+'.partial')
                with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',timeout=120) as response,temp.open('wb') as f:
                    shutil.copyfileobj(response,f,1024*1024)
                temp.replace(target)
        digest=sha(target)
        if expected: assert digest==expected,name
        print('ready',name,target.stat().st_size,flush=True)
        return {'path':name,'bytes':target.stat().st_size,'sha256':digest}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: artifacts=list(pool.map(download,files.values()))
    record={'model':MODEL,'revision':REVISION,'source':'Public Hugging Face OPT export; no self-hosted deployment or Cloudflare access.',
            'files':artifacts,'original_ssn_sources':{p:sha(original/p) for p in sources}}
    (OUT/'assets.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__': main()

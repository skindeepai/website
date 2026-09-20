"""Deterministically package saved small maze policies; no training or data selection."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=ROOT/'results/maze-actions/protocol.json'
    p=json.loads(protocol.read_text(encoding='utf-8'));models={};sources={}
    for name in ['original_states','all_train_goal_pairs']:
        source=ROOT/'results/maze-smoke'/(name+'.npz');sources[str(source.relative_to(ROOT)).replace('\\','/')]=sha(source)
        with np.load(source) as saved:models[name]={key:saved[key].tolist() for key in saved.files}
    target=ROOT/'models/maze/model.json';target.parent.mkdir(exist_ok=True)
    data=(json.dumps(dict(models=models,mazes=p['mazes'],episode_ids=p['episode_ids']),separators=(',',':'))+'\n').encode()
    if target.exists():assert target.read_bytes()==data,'Existing asset differs; preserve it.'
    else:target.write_bytes(data)
    out=ROOT/'results/browser-demos/maze-export.json'
    record=dict(source_sha256=sha(Path(__file__)),parent_protocol_sha256=sha(protocol),weights=sources,asset_sha256=sha(target),scope='Same trained policies; runtime parity checked separately. Synthetic maze layouts, not real-world navigation.')
    out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
if __name__=='__main__':main()

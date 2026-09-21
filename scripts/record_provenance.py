"""Hash the finalized local research record; no remote access or git mutation."""
import hashlib,json,platform,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def main():
    paths=[]
    for directory in ['scripts','experiments','results','content','docs','examples','archive','images','models']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ['.py','.js','.cjs','.json','.html','.md','.txt','.png','.svg','.npz','.onnx','.bin','.f32'] and p.name!='provenance.json' and not {'.cache','__pycache__'}.intersection(p.parts))
    paths.extend(ROOT/n for n in ['PLAN.md','EXPERIMENTS.md','README.md','lab.css','demo.css','style.css','results.css','live-demos.css','use-cases.css','sitemap.xml','.gitattributes'])
    paths.extend(ROOT.glob('*.html'))
    paths.append(ROOT/'favicon.png')
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.DEVNULL)
    payload={'recorded_utc':datetime.now(timezone.utc).isoformat(),'record_type':'Finalized local working-tree record, not immutable pre-registration','git_base':git('rev-parse','HEAD').decode().strip(),'tracked_diff_sha256':hashlib.sha256(git('diff','--binary','HEAD','--','.',':(exclude)results/provenance.json')).hexdigest(),'tracked_diff_scope':'git diff --binary HEAD -- . :(exclude)results/provenance.json; untracked files are covered by the artifact hashes','python':platform.python_version(),'platform':platform.platform(),'sha256':hashes}
    (ROOT/'results/provenance.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8',newline='\n');print(f'Recorded {len(hashes)} source/artifact hashes.')
if __name__=='__main__':main()

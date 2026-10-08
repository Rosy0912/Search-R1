"""Download original PyPI wheels on target, validate public SHA, install only own env."""
import hashlib,json,os,subprocess,time
from pathlib import Path
BASE=Path('/dataset/shaoshitong/Search-G1-common8');PREP=BASE/'preparation'
meta=json.loads((PREP/'probe-extra-wheels-public.json').read_text())
root=BASE/'bootstrap-assets/probe-extra-wheels';root.mkdir(exist_ok=True)
pid=os.getpid();ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
status=PREP/'probe-extras-status.json'
def publish(stage,**kwargs):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=pid,start_ticks=ticks,unix=time.time(),**kwargs)));tmp.replace(status)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
try:
 publish('DOWNLOADING_ORIGINAL_PUBLIC_WHEELS')
 for item in meta['files']:
  dest=root/item['file'];partial=dest.with_suffix('.partial')
  if dest.exists():assert sha(dest)==item['sha256'];continue
  subprocess.run(['curl','--noproxy','*','--resolve','files.pythonhosted.org:443:'+meta['cdn_ip'],'--connect-timeout','10','--max-time','120','--fail','--location','--output',str(partial),item['url']],check=True)
  assert partial.stat().st_size==item['bytes'] and sha(partial)==item['sha256'];partial.replace(dest)
 publish('INSTALLING_PRIVATE_PROBE_DEPENDENCIES')
 subprocess.run(['/root/miniconda3/bin/python','-m','pip','--python',str(BASE/'env/bin/python'),'install','--no-index','--find-links',str(root),'--find-links',str(BASE/'bootstrap-assets/wheelhouse')]+[str(root/x['file']) for x in meta['files']],check=True)
 subprocess.run([str(BASE/'env/bin/python'),'-c','import sklearn,scipy,joblib,threadpoolctl; print(sklearn.__version__,scipy.__version__,joblib.__version__,threadpoolctl.__version__)'],check=True)
 publish('COMPLETE',accepted=True,files={x['file']:x['sha256'] for x in meta['files']})
except Exception as exc:
 publish('FAILED_RETAIN_PARTIAL',error=type(exc).__name__+': '+str(exc));raise

import subprocess,shlex,json,sys,hashlib,shutil,time
from pathlib import Path
sys.path.insert(0,"/Users/shaoshitong/Documents/remote_codex_control")
import control
root=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/gigpo-20261002')
root.mkdir(parents=True,exist_ok=True)
manifest=Path('/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/h20/assets.json')
assets=json.loads(manifest.read_text())
remote='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root'
def route(command):
 return control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote('env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(command))]
def send(local,target):
 sha=hashlib.file_digest(local.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else digest(local)
 command='mkdir -p '+shlex.quote(str(Path(target).parent))+' && cat > '+shlex.quote(target+'.transfer')+' && sha256sum '+shlex.quote(target+'.transfer')
 with local.open('rb') as f:res=subprocess.run(route(command),stdin=f,capture_output=True,text=True,timeout=120)
 if res.returncode or res.stdout.split()[0]!=sha:raise RuntimeError('Transfer verification failed '+target)
 subprocess.run(route('mv '+shlex.quote(target+'.transfer')+' '+shlex.quote(target)),check=True)
 return sha
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
ledger=root/'transfer-manifest.json'
record=json.loads(ledger.read_text()) if ledger.exists() else []
verified={row['path'] for row in record if row['verified']}
send(manifest,remote+'/assets.json')
for a in sorted(assets,key=lambda a:a.get('size') or 0):
 if a['path'] in verified:continue
 p=root/a['path'];p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  if shutil.disk_usage(root).free < (a.get('size') or 0)+2*1024**3:raise RuntimeError('Insufficient staging disk '+a['path'])
  prefix='/datasets/' if a['kind']=='dataset' else '/'
  url='https://hf-mirror.com'+prefix+a['repo']+'/resolve/'+a['revision']+'/'+a['file']
  print('DOWNLOAD',a['path'],a['size'],flush=True)
  result=subprocess.run(['curl','--proxy','http://127.0.0.1:7892','-fLsS','--connect-timeout','15','--speed-time','60','--speed-limit','1024','--retry','2','-C','-','-o',str(p),url])
  if result.returncode:raise RuntimeError('Download failed; staged copy retained '+a['path'])
 if a['size'] and p.stat().st_size!=a['size']:raise RuntimeError('Size mismatch '+a['path'])
 sha=digest(p)
 if a.get('sha256') and sha!=a['sha256']:raise RuntimeError('SHA mismatch '+a['path'])
 if a.get('git_blob_sha1'):
  h=hashlib.sha1(('blob '+str(p.stat().st_size)+'\0').encode());h.update(p.read_bytes())
  if h.hexdigest()!=a['git_blob_sha1']:raise RuntimeError('Blob mismatch '+a['path'])
 print('TRANSFER',a['path'],sha,flush=True)
 send(p,remote+'/assets/'+a['path'])
 record.append(dict(path=a['path'],sha256=sha,size=p.stat().st_size,target='h100-8',verified=True))
 (root/'transfer-manifest.json').write_text(json.dumps(record,indent=2))
 # The only required target for this relay is Agent 1. Preserve on any failed transfer.
 p.unlink()
 print('VERIFIED_AND_CLEANED',a['path'],flush=True)
print('ASSET_RELAY_COMPLETE',flush=True)

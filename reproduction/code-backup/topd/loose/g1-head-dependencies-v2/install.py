#!/usr/bin/env python3
"""Official wheel-only CPU dependency overlay; never modifies the fixed environment."""
import argparse,fcntl,hashlib,importlib.metadata,json,os,pathlib,socket,subprocess,sys,time,urllib.parse
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
HERE=pathlib.Path(__file__).resolve().parent
MANIFEST_SHA='52e8eebaf530af4051f86dda499685b225213600a406259aa830e97a0b7ecbab'
UV_SHA='a261804d9493a96e7eebf13bcffc1b7ecc87dbd7da9a2800af94c20182d1a70d'
PACKAGES={'scikit-learn':'1.5.2','scipy':'1.14.1','joblib':'1.4.2','threadpoolctl':'3.5.0'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(path,value):
 p=pathlib.Path(path);tmp=p.with_suffix('.atomic.tmp')
 with open(tmp,'x') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def records():
 manifest=HERE/'official-wheels.json'
 if sha(manifest)!=MANIFEST_SHA:raise ValueError('official metadata manifest bytes changed')
 r=json.loads(manifest.read_text())['records']
 if {x['package']:x['version'] for x in r}!=PACKAGES or len(r)!=4:raise ValueError('unexpected package set')
 for x in r:
  parsed=urllib.parse.urlsplit(x['url'])
  if parsed.scheme!='https' or parsed.hostname!='files.pythonhosted.org' or parsed.username or parsed.query or pathlib.Path(parsed.path).name!=x['filename']:raise ValueError('not pinned official static PyPI URL')
  metadata=HERE/(x['package']+'-'+x['version']+'-pypi.json')
  if sha(metadata)!=x['metadata_sha256']:raise ValueError('official metadata bytes drift')
  data=json.loads(metadata.read_text());match=[a for a in data['urls'] if a['filename']==x['filename']]
  if len(match)!=1 or match[0]['url']!=x['url'] or match[0]['size']!=x['official_size'] or match[0]['digests']['sha256']!=x['official_sha256']:raise ValueError('wheel official metadata binding drift')
 return r
def verify(path,item):
 p=pathlib.Path(path)
 if p.is_symlink() or p.stat().st_size!=item['official_size'] or sha(p)!=item['official_sha256']:raise ValueError('official full wheel size/SHA mismatch')
def download(item,wheels):
 import requests
 target=wheels/item['filename'];partial=target.with_suffix('.official.partial');events=[]
 if target.exists():verify(target,item);return {'official_record':item,'target_sha256':sha(target),'verified_existing':True,'events':events}
 session=requests.Session();session.trust_env=False
 try:
  for attempt in range(12):
   offset=partial.stat().st_size if partial.exists() else 0
   if partial.is_symlink() or offset>item['official_size']:raise ValueError('invalid own partial path/size')
   if offset==item['official_size']:break
   route='DIRECT_OFFICIAL_PYPI' if attempt==0 else 'OWN_WUHAN_OFFICIAL_H16_PROXY'
   proxy={} if attempt==0 else {'http':'http://127.0.0.1:18089','https':'http://127.0.0.1:18089'}
   try:
    with session.get(item['url'],headers={'Range':f'bytes={offset}-'} if offset else {},proxies=proxy,timeout=(10,30),stream=True) as response:
     response.raise_for_status()
     if response.status_code not in (200,206):raise ValueError('unexpected official HTTP status')
     if offset and response.status_code!=206:raise ValueError('official server refused exact resume')
     if response.status_code==206:
      expected=f'bytes {offset}-{item["official_size"]-1}/{item["official_size"]}'
      if response.headers.get('Content-Range')!=expected:raise ValueError('official resume range mismatch')
     with open(partial,'ab') as f:
      for block in response.iter_content(1024**2):
       if f.tell()+len(block)>item['official_size']:raise ValueError('official wheel exceeds pinned size')
       f.write(block)
      f.flush();os.fsync(f.fileno())
    events.append({'attempt':attempt,'route':route,'offset':offset,'bytes_after':partial.stat().st_size,'status':'HTTP_COMPLETE'})
    if partial.stat().st_size==item['official_size']:break
   except requests.RequestException as exc:
    events.append({'attempt':attempt,'route':route,'offset':offset,'bytes_after':partial.stat().st_size if partial.exists() else 0,'error':type(exc).__name__+': '+str(exc)})
    time.sleep(1)
  else:raise RuntimeError('official download retries exhausted; partial retained')
  verify(partial,item);os.replace(partial,target)
  return {'official_record':item,'target_sha256':sha(target),'target_size':target.stat().st_size,'events':events}
 finally:session.close()
def fixed_env_snapshot():
 result={}
 for name in ('torch','numpy','transformers'):
  d=importlib.metadata.distribution(name);m=d._path/'METADATA'
  result[name]={'version':d.version,'metadata_path':str(m),'metadata_sha256':sha(m)}
 return result
def pip_command(wheels,overlay):
 return [str(ROOT/'tools/uv'),'--no-cache','--no-config','--offline','--no-python-downloads','pip','install','--python',sys.executable,'--no-deps','--no-index','--no-build','--link-mode','copy','--target',str(overlay)]+[str(p) for p in wheels]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--files-sha256',required=True);a=ap.parse_args();os.umask(0o077)
 if socket.gethostname()!='h100-11' or os.environ.get('SLURM_JOB_NODELIST')!='h100-11' or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('SLURM_JOB_GPUS'):raise ValueError('own h11 CPU-only Slurm required')
 if not HERE.is_relative_to(ROOT/'tools') or not pathlib.Path(sys.prefix).resolve().is_relative_to(ROOT/'env'):raise ValueError('own tools/fixed env only')
 files=HERE/'files.json'
 if sha(files)!=a.files_sha256:raise ValueError('installer frozen files manifest drift')
 for name,r in json.loads(files.read_text())['files'].items():
  p=HERE/name
  if p.is_symlink() or p.stat().st_size!=r['size'] or sha(p)!=r['sha256']:raise ValueError('installer frozen bytes drift')
 lock=open(HERE/'install.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt_path=HERE/'install-receipt.json'
 if receipt_path.exists():raise FileExistsError('prior install receipt: no repeated installation')
 uv=ROOT/'tools/uv'
 if sha(uv)!=UV_SHA or subprocess.check_output([str(uv),'--version'],text=True).strip()!='uv 0.8.22':raise ValueError('fixed own uv source/version drift')
 before=fixed_env_snapshot();receipt={'scope':'OFFICIAL_HEAD_DEPS_CPU_OVERLAY_NOT_CALIBRATION_NO_TRAINING','status':'UNKNOWN_BEFORE_DOWNLOAD_INSTALL','job':os.environ['SLURM_JOB_ID'],'pid':os.getpid(),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'fixed_env_before':before,'files_sha256':a.files_sha256,'installer_uv_sha256':UV_SHA,'installer_uv_version':'0.8.22'};atomic(receipt_path,receipt)
 wheels=HERE/'wheels';wheels.mkdir(exist_ok=True);overlay=HERE/'overlay';build=HERE/('overlay.build-'+os.environ['SLURM_JOB_ID'])
 try:
  receipt['wheels']=[download(x,wheels) for x in records()]
  if overlay.exists() or build.exists():raise FileExistsError('do not overwrite overlay/build')
  command=pip_command([wheels/x['filename'] for x in records()],build)
  env=dict(os.environ,PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
  for key in list(env):
   if key.startswith('UV_') or key.startswith('PIP_'):env.pop(key,None)
  with open(HERE/'pip-install.log','wb') as log:
   result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  receipt.update(pip_returncode=result.returncode,pip_command=command,pip_log_sha256=sha(HERE/'pip-install.log'))
  if result.returncode:raise RuntimeError('isolated uv --no-deps target installation failed')
  if any(p.name.lower().startswith(('numpy','torch','transformers')) for p in build.iterdir()):raise ValueError('forbidden fixed environment package in overlay')
  os.replace(build,overlay)
  validation="import json,sklearn,scipy,joblib,threadpoolctl,numpy,torch,transformers; print(json.dumps({m.__name__:{'version':m.__version__,'path':m.__file__} for m in (sklearn,scipy,joblib,threadpoolctl,numpy,torch,transformers)})); assert not torch.cuda.is_initialized()"
  env['PYTHONPATH']=str(overlay)+os.pathsep+str(ROOT/'test-tools')
  result=subprocess.run([sys.executable,'-c',validation],env=env,capture_output=True,text=True,timeout=60)
  receipt.update(validation_returncode=result.returncode,validation_stdout=result.stdout,validation_stderr=result.stderr)
  if result.returncode:raise RuntimeError('isolated overlay import validation failed')
  imported=json.loads(result.stdout)
  for name,version in (('sklearn','1.5.2'),('scipy','1.14.1'),('joblib','1.4.2'),('threadpoolctl','3.5.0')):
   if imported[name]['version']!=version or not pathlib.Path(imported[name]['path']).resolve().is_relative_to(overlay.resolve()):raise ValueError('overlay package identity drift')
  for name in ('numpy','torch','transformers'):
   if not pathlib.Path(imported[name]['path']).resolve().is_relative_to((ROOT/'env').resolve()):raise ValueError('fixed library overridden')
  receipt['fixed_env_after']=fixed_env_snapshot()
  if before!=receipt['fixed_env_after']:raise ValueError('fixed env package metadata changed')
  receipt.update(status='COMPLETE_CPU_OVERLAY_ONLY',imports=imported,overlay=str(overlay),overlay_files={str(p.relative_to(overlay)):sha(p) for p in overlay.rglob('*') if p.is_file()})
 except BaseException as exc:receipt.update(status='FAILED_PARTIAL_RETAINED',error_type=type(exc).__name__,error=str(exc));raise
 finally:atomic(receipt_path,receipt);print(json.dumps({k:receipt[k] for k in ('scope','status','job')}))
if __name__=='__main__':main()

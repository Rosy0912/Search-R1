"""Bounded Xet alternative on HKUST1, isolated packages and remote files."""
import json,pathlib,shlex,subprocess,sys
worker=r'''
import os,sys,pathlib,json,time,hashlib
token=sys.stdin.readline().strip();root=pathlib.Path('/data/shaoshitong/topd-hf-route-probes-20261009/xet-teacher29');root.mkdir(exist_ok=True)
def save(d):root.joinpath('result.json').write_text(json.dumps(d))
t=time.time();save({'phase':'STARTED','epoch':t,'mac_used':False})
try:
 from huggingface_hub import hf_hub_download
 save({'phase':'HF_DOWNLOAD','epoch':time.time(),'mac_used':False})
 p=pathlib.Path(hf_hub_download('Sutoonq/TMP_MODEL','topd-migration-20261009/archives/teacher_model-00029-of-00029.tar.zst',token=token,endpoint='https://hf-mirror.com',local_dir=str(root)))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 save({'phase':'FINISHED','bytes':p.stat().st_size,'sha256':h.hexdigest(),'seconds':time.time()-t,'mac_used':False})
except Exception as e:save({'phase':'ERROR','type':type(e).__name__,'seconds':time.time()-t,'mac_used':False})
'''
remote=r'''
import pathlib,subprocess,os,sys,time,json,signal
token=sys.stdin.readline().strip();root=pathlib.Path('/data/shaoshitong/topd-hf-route-probes-20261009/xet-teacher29');root.mkdir(exist_ok=True)
script=root/'worker.py';script.write_text(WORKER)
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
env.update(PYTHONPATH='/data/shaoshitong/topd-hf-route-probes-20261009/xet-packages',HF_ENDPOINT='https://hf-mirror.com',HF_XET_HIGH_PERFORMANCE='1',HF_XET_LOG_FILE='/dev/null',HF_XET_CACHE=str(root/'xet-cache'),HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_VERBOSITY='error',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_ETAG_TIMEOUT='8',HF_HUB_DOWNLOAD_TIMEOUT='12')
p=subprocess.Popen(['/home/shaoshitong/anaconda3/bin/python',str(script)],env=env,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);p.stdin.write((token+'\n').encode());p.stdin.close()
try:p.wait(timeout=48)
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=5)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
out=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {'phase':'NO_RESULT'}
out['exit']=p.returncode;out['partial_preserved']=True;print(json.dumps(out))
'''.replace('WORKER',repr(worker))
token=sys.stdin.readline().strip();p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes','hkust1','python3 -c '+shlex.quote(remote)],input=token+'\n',capture_output=True,text=True,timeout=62)
out=json.loads(p.stdout) if p.returncode==0 else {'error':'remote_failed','exit':p.returncode}
pathlib.Path('outputs/hpc3-migration/xet-probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))

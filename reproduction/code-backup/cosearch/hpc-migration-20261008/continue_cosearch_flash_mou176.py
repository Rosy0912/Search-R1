"""Mac control only; credentials to remote RAM via stdin, no asset relay."""
import importlib.util,json,subprocess,shlex,re,sys,os,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('setup',Path(__file__).with_name('start_cosearch_flash_hf.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
secret=Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
password=re.split(r'密码\s*[:：]?\s*',next(l for l in secret.splitlines() if l.startswith('账号 mou176')),1)[1].strip().strip('`');secret=None
original_deploy=m.deploy

def hpc(command,payload=''):
 code="import os,sys,subprocess;env=os.environ.copy();env['SSHPASS']=sys.stdin.readline().rstrip('\\n');data=sys.stdin.read();r=subprocess.run(['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-oConnectTimeout=8','mou176@10.120.81.30',"+repr(command)+"],input=data,env=env,capture_output=True,text=True,timeout=60);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)"
 return m.campus('hkust3','python3 -c '+shlex.quote(code),input=password+'\n'+payload)

def deploy(host,root,files):
 if host!='hpc3':
  if host=='hkust3':
   # New account-specific entry point avoids replacing existing deployed metadata.
   files={'cosearch_flash_forward_mou176.py' if k=='cosearch_flash_forward_hkust3.py' else k:v for k,v in files.items()}
  return original_deploy(host,root,files)
 import base64
 code="import sys,json,base64;from pathlib import Path;r=Path("+repr(root)+");assert not r.is_symlink();r.mkdir(exist_ok=True);d=json.loads(sys.stdin.read())\nfor n,v in d.items():\n p=r/n;b=base64.b64decode(v)\n if p.exists():assert not p.is_symlink() and p.read_bytes()==b\n else:p.write_bytes(b)\nprint('DEPLOYED')"
 r=hpc('python3 -c '+shlex.quote(code),json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}));assert r.returncode==0,r.stderr
m.deploy=deploy
# Inspect controllers before any starts; never replace existing workers.
roots={'hkust2':'/data/shaoshitong/cosearch-flash-build-20261009','hkust3':'/backup/shaoshitong/cosearch-flash-build-20261009','mgmt-host':'/home/lichen/cosearch-flash-build-20261009'}
for host,root in roots.items():
 code="from pathlib import Path;import json,fcntl;r=Path("+repr(root)+");cs=list(r.glob('*controller.json'));assert not cs,'Existing controller: inspect before start'\nfor p in r.glob('*.lock'):\n f=p.open('r');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nprint('CONTROLLERS_ABSENT_LOCKS_FREE')"
 command='python3 -c '+shlex.quote(code)
 r=m.campus(host,command) if host!='mgmt-host' else subprocess.run(m.ctl.route(host,command),capture_output=True,text=True,timeout=45)
 assert r.returncode==0,(host,r.stderr);print(host,r.stdout.strip(),flush=True)
source=json.loads((m.E/'native-x-hpc-flash-source-preparation-20261009.json').read_text())['archive-verified.json'];plan=json.dumps({'repo':'Sutoonq/TMP_MODEL','records':[source]}).encode();work=m.BASE/'work';read=lambda n:(work/n).read_bytes()
for host,root in roots.items():
 files={'plan.json':plan}
 if host=='mgmt-host':files['cosearch_hf_stream_upload.py']=read('cosearch_hf_stream_upload.py')
 elif host=='hkust2':files.update({'download-network.json':json.dumps({'endpoint':'https://hf-mirror.com','proxy':None}).encode(),'cosearch_hf_download_hkust2.py':read('cosearch_hf_download_hkust2.py'),'cosearch_flash_forward_hkust2.py':read('cosearch_flash_forward_hkust2.py')})
 else:files['cosearch_flash_forward_hkust3.py']=read('cosearch_flash_forward_hkust3.py')
 deploy(host,root,files)
deploy('hpc3','/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/migration-flash-build-20261009',{'manifest.jsonl':(m.E/'native-x-hpc-flash-build-source-manifest.jsonl').read_bytes(),'cosearch_flash_hpc_receiver.py':read('cosearch_flash_hpc_receiver.py')})
import termios
settings=termios.tcgetattr(sys.stdin.fileno());quiet=settings.copy();quiet[3]&=~termios.ECHO;termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,quiet)
print('READY_FOR_TOKEN_STDIN',flush=True);token=sys.stdin.readline().strip();termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,settings);assert token
result={'epoch':time.time(),'upload':m.start('mgmt-host',roots['mgmt-host'],'cosearch_hf_stream_upload.py',token),'download':m.start('hkust2',roots['hkust2'],'cosearch_hf_download_hkust2.py',token),'forward2':m.start('hkust2',roots['hkust2'],'cosearch_flash_forward_hkust2.py')};token=None
root=roots['hkust3'];code="import subprocess,sys,json;from pathlib import Path;r=Path("+repr(root)+");assert not (r/'hpc-forward-controller.json').exists();f=(r/'mou176-forward.log').open('ab');p=subprocess.Popen([sys.executable,str(r/'cosearch_flash_forward_mou176.py')],stdin=subprocess.PIPE,stdout=f,stderr=f,start_new_session=True);p.stdin.write(sys.stdin.readline().encode());p.stdin.close();print(json.dumps({'pid':p.pid}))"
r=m.campus('hkust3','python3 -c '+shlex.quote(code),input=password+'\n');assert r.returncode==0,r.stderr;result['forward3']=json.loads(r.stdout);password=None
(m.E/'native-x-hpc-flash-mou176-controllers-20261009.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

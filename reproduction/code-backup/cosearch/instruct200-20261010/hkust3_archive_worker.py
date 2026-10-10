"""HKUST3-owned checkpoint puller. Password arrives via stdin and stays in RAM."""
import fcntl,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
ROOT='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-20261010'
DEST=Path('/backup/shaoshitong/cosearch-instruct200-20261010/checkpoints')
DEST.mkdir(parents=True,exist_ok=True)
lock=(DEST.parent/'archive.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ENV=dict(os.environ,SSHPASS=sys.stdin.readline().rstrip('\n'))
SSH=['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-oConnectTimeout=8','mou176@10.120.81.30']
def remote(code):
 r=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def state(**kw):
 (DEST.parent/'controller.json').write_text(json.dumps(dict(pid=os.getpid(),epoch=time.time(),**kw),indent=2))
while True:
 try:
  remote("from pathlib import Path;Path("+repr(ROOT+'/archive-worker-heartbeat.json')+").write_text("+repr(json.dumps({'epoch':time.time(),'pid':os.getpid(),'destination':str(DEST)}))+")")
  for step in (5,50,100,150,200):
   code="from pathlib import Path; p=Path("+repr(ROOT+'/archive-requests/'+str(step)+'.json')+");print(p.read_text() if p.exists() else 'null')"
   request=json.loads(remote(code))
   if request is None:continue
   target=DEST/('global_step_'+str(step));receiptpath=target/'hkust3-archive-receipt.json'
   if not receiptpath.exists():
    state(status='TRANSFERRING',step=step)
    target.mkdir(exist_ok=True)
    transport=' '.join(shlex.quote(x) for x in SSH[:-1])
    subprocess.run(['rsync','-a','--partial','--append-verify','-e',transport,'mou176@10.120.81.30:'+ROOT+'/checkpoints/global_step_'+str(step)+'/',str(target)+'/'],env=ENV,check=True)
    state(status='VERIFYING',step=step)
    for e in request['receipt']['files']:
     rel=e['path'].split('/global_step_'+str(step)+'/')[1];p=target/rel
     assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],str(p)
    receipt={'status':'HKUST3_ALL_CHECKPOINT_SHA_PASS','step':step,'path':str(target),'epoch':time.time(),'files':request['receipt']['files']}
    receiptpath.write_text(json.dumps(receipt,indent=2))
   receipt=json.loads(receiptpath.read_text())
   remote("from pathlib import Path;Path("+repr(ROOT+'/archive-verified-'+str(step)+'.json')+").write_text("+repr(json.dumps(receipt))+")")
   remote("from pathlib import Path;import shutil;p=Path("+repr(ROOT+'/checkpoints/global_step_'+str(step))+");safe=Path("+repr(ROOT+'/archive-delete-safe-'+str(step))+");shutil.rmtree(str(p)) if safe.exists() and p.exists() else None")
  state(status='WAITING_FOR_CHECKPOINTS')
 except Exception as e:state(status='RETRYING',error=str(e))
 time.sleep(15)

"""Explicitly authorized single-line temporary resolver repair; no process signals."""
import base64,datetime,fcntl,hashlib,json,os,socket,stat,subprocess,tempfile,time,sys
from pathlib import Path
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
P=Path('/etc/hosts'); LINE=b'10.33.0.8 h100-16 # BAPO_STOPPED3315_RESOLUTION_REPAIR_20261009_V1\n'
E=ROOT/'logs'/'authorized-hosts-repair-20261009'
def cmd(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=15);return {'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def birth():
 p=Path('/proc/755855');f=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert int(f[19])==1433986891 and int(f[1])==1 and p.stat().st_uid==0 and f[0] not in ('Z','X')
 assert (p/'cmdline').read_bytes().replace(b'\0',b' ').strip()==b'slurmstepd: [3315.0]'
 assert os.readlink(p/'exe')=='/usr/sbin/slurmstepd-wlm'
 assert hashlib.sha256((p/'exe').read_bytes()).hexdigest()=='be7770948fcfd24b074745d8add09005f77c1ae0ddb380b86abceb71d2bfdd6a'
def ident():
 s=P.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
 return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid,s.st_gid,stat.S_IMODE(s.st_mode))
def atomic(raw,expected,meta,attrs):
 assert P.read_bytes()==expected
 before=ident(); fd,name=tempfile.mkstemp(prefix='.bapo3315-',dir='/etc')
 try:
  with os.fdopen(fd,'wb') as h:
   h.write(raw);h.flush();os.fchown(h.fileno(),meta[5],meta[6]);os.fchmod(h.fileno(),meta[7]);
   for k,v in attrs.items():os.setxattr(h.fileno(),k,v)
   os.fsync(h.fileno())
  assert ident()==before and P.read_bytes()==expected
  os.replace(name,P)
  d=os.open('/etc',os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(d)
  finally:os.close(d)
  assert P.read_bytes()==raw
 finally:
  if os.path.exists(name):os.unlink(name)
def main():
 if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
  raise RuntimeError('Optimized Python disables admission assertions; refuse')
 assert socket.gethostname()=='h100-5'
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='9471cbc1-9ae1-4be5-8edb-1e84e850e2cc'
 E.mkdir(mode=0o700,exist_ok=False)
 lock=open(E/'lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 v={'authorization':'User: 可以，怎么快怎么来 (2026-10-09); temporary hosts alias and rollback only','start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'samples':[],'signals_sent':False}
 changed=False
 try:
  birth();assert os.path.samefile(P,Path('/proc/755855/root/etc/hosts'))
  meta=ident();original=P.read_bytes();attrs={k:os.getxattr(P,k) for k in os.listxattr(P)}
  assert hashlib.sha256(original).hexdigest()=='58600feec38db46e988e5f45fd0bc7c869be2015d6042dce771d60178b0fe565'
  assert original.endswith(b'\n') and b'BAPO_STOPPED3315_RESOLUTION_REPAIR' not in original
  assert not any('h100-16' in x.split('#',1)[0].split()[1:] for x in original.decode().splitlines())
  assert not any(x.split()[4]=='/etc/hosts' for x in Path('/proc/self/mountinfo').read_text().splitlines())
  assert hashlib.sha256(Path('/etc/slurm/slurm.conf').read_bytes()).hexdigest()=='8098a7a9eb5436522948607f18fccd891dc0b1ed3df14bb51432eb8d4b0fc0cc'
  v['controller']=cmd(['scontrol','ping']); assert v['controller']['exit']==0 and 'UP' in v['controller']['stdout'] and 'h100-8' in v['controller']['stdout']
  assert cmd(['getent','ahostsv4','h100-8'])['stdout'].split()[0]=='10.33.0.8'
  v['jobs']=cmd(['squeue','-h','-w','h100-5','-o','%i|%T']);assert v['jobs']['exit']==0
  for x in v['jobs']['stdout'].splitlines():
   jid,state=x.split('|');assert jid in ('3315','3345') and state in ('COMPLETING','PENDING')
  v['oldjob']=cmd(['scontrol','show','job','3315','-o']);assert 'JobState=COMPLETING' in v['oldjob']['stdout'] and 'bab8981d9fd14afc98923eb4ebb89e33' in v['oldjob']['stdout']
  v['newjob']=cmd(['scontrol','show','job','3345','-o']);assert 'JobState=PENDING' in v['newjob']['stdout'] and '13e07285c97044fd97974d94f53fea1b' in v['newjob']['stdout']
  gpu=cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']);assert gpu['exit']==0 and not gpu['stdout'].strip()
  v['before_sha256']=hashlib.sha256(original).hexdigest();v['original_metadata']=meta
  with (E/'original-hosts').open('xb') as h:h.write(original);h.flush();os.fsync(h.fileno())
  (E/'metadata.json').write_text(json.dumps({'metadata':meta,'xattrs':{k:base64.b64encode(x).decode() for k,x in attrs.items()}}))
  birth();assert ident()==meta
  changed=True
  atomic(original+LINE,original,meta,attrs)
  v['alias']=cmd(['getent','ahostsv4','h100-16']);assert v['alias']['stdout'].split()[0]=='10.33.0.8'
  until=time.monotonic()+300
  while True:
   row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old':cmd(['scontrol','show','job','3315','-o']),'new':cmd(['scontrol','show','job','3345','-o']),'node':cmd(['scontrol','show','node','h100-5','-o']),'daemon_exists':Path('/proc/755855').exists()};v['samples'].append(row)
   (E/'progress.json').write_text(json.dumps(v))
   if 'JobState=PENDING' not in row['new']['stdout'] or not row['daemon_exists'] or time.monotonic()>=until:break
   time.sleep(min(30,max(0,until-time.monotonic())))
 except BaseException as ex:v['error']=repr(ex)
 finally:
  if changed:
   current=P.read_bytes()
   if current==original: v['rollback']='already original'
   elif current.count(LINE)==1:
    restored=current.replace(LINE,b'',1);atomic(restored,current,ident(),{k:os.getxattr(P,k) for k in os.listxattr(P)});v['rollback']='removed only own line'
   else:v['rollback']='FAILED: own line absent or ambiguous; no blind overwrite'
   v['after_sha256']=hashlib.sha256(P.read_bytes()).hexdigest();v['original_bytes_restored']=P.read_bytes()==original
  v['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(E/'result.json').write_text(json.dumps(v));print(json.dumps(v))
if __name__=='__main__':main()

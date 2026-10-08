"""Exclusive isolated source publication; no training, imports, schedulers or GPU."""
import os,pathlib,stat,subprocess,hashlib,json,sys,base64,datetime,socket
ROOT=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
TRAIN=ROOT/'repo-fcecbea5d939';TARGET=ROOT/'repo-final200-eval-c5bf4c8db335';PUB=ROOT/'tools/final200-eval-source-publication-v1'
COMMIT='c5bf4c8db33562a259fa227a940dc69543901b98';PARENT='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
assert socket.gethostname()=='h100-10' and os.getuid()==0
for p in [ROOT,ROOT/'tools',TRAIN]:assert p.resolve(strict=True)==p and p.stat().st_uid==0
assert not TARGET.exists() and not PUB.exists()
payload=json.load(sys.stdin);inv=payload['inventory'];bundle=base64.b64decode(payload['bundle_base64']);publisher=base64.b64decode(payload['publisher_base64'])
assert hashlib.sha256(bundle).hexdigest()==payload['bundle_sha256'] and hashlib.sha256(publisher).hexdigest()==payload['publisher_sha256']
assert inv['version']==1 and inv['kind']=='ACTUAL_FIXED_EVAL_SOURCE_INVENTORY' and inv['source_commit']==COMMIT and inv['parent_commit']==PARENT and inv['member_count']==len(inv['files'])==857 and inv['remote_repository'] is None
rows=[]
def command(argv):
 r=subprocess.run(argv,capture_output=True,timeout=150);v={'argv':argv,'returncode':r.returncode,'waited':True,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()};rows.append(v)
 if r.returncode:raise RuntimeError(v)
 return r.stdout

def train_check():
 a=command(['git','--no-optional-locks','-C',str(TRAIN),'rev-parse','HEAD']);b=command(['git','--no-optional-locks','-C',str(TRAIN),'status','--porcelain']);assert a.decode().strip()==PARENT and b==b'';return {'HEAD':PARENT,'clean':True}
before=train_check();PUB.mkdir(mode=0o700)
def put(path,raw,mode=0o444):
 fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert path.read_bytes()==raw
put(PUB/'source.bundle',bundle);put(PUB/'publisher-source.original.py',publisher)
try:
 command(['git','-c','core.hooksPath=/dev/null','clone','--no-hardlinks','--no-checkout',str(TRAIN),str(TARGET)])
 command(['git','-c','core.hooksPath=/dev/null','-C',str(TARGET),'fetch','--no-tags',str(PUB/'source.bundle'),'refs/heads/codex/opsd-final200-paired-eval'])
 command(['git','-c','core.hooksPath=/dev/null','-C',str(TARGET),'checkout','--detach',COMMIT])
 assert command(['git','--no-optional-locks','-C',str(TARGET),'rev-parse','HEAD']).decode().strip()==COMMIT
 assert command(['git','--no-optional-locks','-C',str(TARGET),'rev-parse','HEAD^']).decode().strip()==PARENT
 assert command(['git','--no-optional-locks','-C',str(TARGET),'status','--porcelain'])==b''
 tree=command(['git','--no-optional-locks','-C',str(TARGET),'ls-tree','-rz',COMMIT]);actual={}
 for row in tree.split(b'\0'):
  if not row:continue
  head,name=row.split(b'\t',1);mode,kind,blob=head.decode().split();name=name.decode();assert name in inv['files'] and kind=='blob' and mode in ['100644','100755'];p=TARGET/name;expected=inv['files'][name]
  assert p.resolve(strict=True)==p and not p.is_symlink();st=p.stat();assert stat.S_ISREG(st.st_mode) and st.st_uid==0 and st.st_nlink==1
  with p.open('rb') as f:
   raw=f.read();opened=os.fstat(f.fileno());f.seek(0);assert f.read()==raw
  after=p.stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==(opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
  assert hashlib.sha256(raw).hexdigest()==expected['sha256'] and len(raw)==expected['bytes'] and mode==expected['git_mode'] and blob==expected['git_blob'] and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==blob
  actual[name]={'sha256':expected['sha256'],'bytes':expected['bytes'],'git_mode':mode,'git_blob':blob,'device':st.st_dev,'inode':st.st_ino,'uid':st.st_uid,'nlink':st.st_nlink}
 assert set(actual)==set(inv['files']) and len(actual)==857
 after_train=train_check();assert before==after_train
 actualinv={k:inv[k] for k in ['version','kind','source_commit','parent_commit','member_count','files']};actualinv['remote_repository']=str(TARGET)
 rawinv=json.dumps(actualinv,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\n';put(PUB/'source-inventory.actual.json',rawinv)
 result={'version':1,'kind':'ACTUAL_ISOLATED_C5BF_SOURCE_PUBLICATION_NOT_NATIVE_OR_EVALUATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'pid':os.getpid(),'uid':os.getuid(),'source_commit':COMMIT,'parent_training_commit':PARENT,'repository':str(TARGET),'member_count':857,'inventory':str(PUB/'source-inventory.actual.json'),'inventory_sha256':hashlib.sha256(rawinv).hexdigest(),'bundle_sha256':payload['bundle_sha256'],'publisher_sha256':payload['publisher_sha256'],'training_source_before':before,'training_source_after':after_train,'files':actual,'commands':rows,'native_execution':None,'GPU_evaluation_job_id':None,'actual200':None,'scope':'Only actual isolated publication-time source bytes/Git tree/physical paths. No imports, model/GPU/RPC, scheduler, newjob, signature or training modification.'}
 put(PUB/'publication.original.json',json.dumps(result,sort_keys=True,separators=(',',':')).encode()+b'\n')
 for n in [PUB,TARGET]:
  fd=os.open(str(n),os.O_DIRECTORY|os.O_RDONLY);os.fsync(fd);os.close(fd)
 print(json.dumps(result,sort_keys=True))
except BaseException as e:
 fail={'error_type':type(e).__name__,'error':str(e),'commands':rows,'unknown_not_retried':True}
 put(PUB/'failure.original.json',json.dumps(fail,sort_keys=True).encode()+b'\n');raise

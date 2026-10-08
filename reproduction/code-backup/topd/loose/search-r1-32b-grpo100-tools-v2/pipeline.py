import hashlib,json,os,pathlib
from checkpoint_inventory import sha
from ipc import identity
def atomic(path,value):
 p=pathlib.Path(path);temp=p.with_name(p.name+'.tmp')
 fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
 d=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(d)
 finally:os.close(d)

def verify_files(root,manifest):
 root=pathlib.Path(root).resolve();actual=set()
 for directory,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   if pathlib.Path(directory,name).is_symlink():raise ValueError('source alias rejected')
  for name in files:actual.add(str(pathlib.Path(directory,name).relative_to(root)))
 expected=set(manifest)
 # Source manifest itself is an explicitly bound outside field, not circular.
 if actual!=expected:raise ValueError('exact source file set drift')
 for name,record in manifest.items():
  p=root/name
  if p.stat().st_size!=record['size'] or sha(p)!=record['sha256']:raise ValueError('source bytes drift: '+name)

def verify_overlay(root,receipt_path,expected):
 if sha(receipt_path)!=expected:raise ValueError('native overlay receipt changed')
 receipt=json.loads(pathlib.Path(receipt_path).read_text());files=receipt['overlay_files']
 # Native installer records may use SHA strings or size/SHA dictionaries.
 actual=set()
 for directory,dirs,names in os.walk(root,followlinks=False):
  for name in dirs+names:
   if pathlib.Path(directory,name).is_symlink():raise ValueError('overlay symlink rejected')
  actual.update(str(pathlib.Path(directory,name).relative_to(root)) for name in names)
 if actual!=set(files):raise ValueError('native overlay exact file set changed')
 for name,record in files.items():
  expected_sha=record if isinstance(record,str) else record['sha256']
  if sha(pathlib.Path(root)/name)!=expected_sha:raise ValueError('overlay bytes changed')
 return receipt

def validate_peer(pid,uid,trainer_pid,job_cgroup):
 if uid!=os.getuid():raise ValueError('foreign IPC uid')
 actual=identity(pid)
 if actual['cgroup']!=job_cgroup:raise ValueError('foreign job peer')
 current=pid;seen=set()
 while current not in seen and current>1:
  if current==trainer_pid:return actual
  seen.add(current);current=int(pathlib.Path('/proc',str(current),'stat').read_text().rsplit(')',1)[1].split()[1])
 raise ValueError('peer not owned trainer descendant')


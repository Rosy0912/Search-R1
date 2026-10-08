"""CPU-only fresh reuse of the actual64 frozen retrieval inventory."""
import json,os,pathlib
from checkpoint_inventory import sha
BINDING_SHA='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'
DIRECTORIES=('assets-root/assets/e5-base-v2','retriever-gpu-lib')
REQUIRED={'assets-root/assets/wiki/e5_Flat.index','assets-root/assets/wiki/wiki-18.jsonl'}
def verify_retrieval(root,reference,server):
 root=pathlib.Path(root).resolve();p=pathlib.Path(reference['path'])
 if p.is_symlink() or root not in p.resolve().parents or reference['sha256']!=BINDING_SHA or sha(p)!=BINDING_SHA:raise ValueError('actual64 retrieval binding drift')
 binding=json.loads(p.read_text())
 return verify_inventory(root,binding['runtime_files'],server,binding['source_files']['search_r1/search/retrieval_server.py'])
def verify_inventory(root,files,server,server_sha):
 root=pathlib.Path(root).resolve();server=pathlib.Path(server)
 if server.is_symlink() or root not in server.resolve().parents or sha(server)!=server_sha:raise ValueError('retrieval server bytes drift')
 if not REQUIRED<=set(files) or not any(n.startswith(DIRECTORIES[0]+'/') and n.endswith(('.safetensors','.bin')) for n in files) or not any(n.startswith(DIRECTORIES[1]+'/') and n.endswith('.so') for n in files):raise ValueError('incomplete GPU fullwiki inventory')
 for directory in DIRECTORIES:
  base=root/directory
  if not base.is_dir() or base.is_symlink():raise ValueError('retrieval dependency missing/alias')
  actual=set()
  for current,dirs,names in os.walk(base,followlinks=False):
   for n in dirs+names:
    if pathlib.Path(current,n).is_symlink():raise ValueError('retrieval dependency symlink')
   # These are interpreter/cache outputs excluded by the original actual64 gate.
   dirs[:]=[n for n in dirs if n not in ('__pycache__','.cache')]
   actual.update(str(pathlib.Path(current,n).relative_to(root)) for n in names)
  expected={n for n in files if n.startswith(directory+'/')}
  if actual!=expected:raise ValueError('retrieval dependency exact file set drift')
 verified={}
 for n,expected in files.items():
  p=root/n
  if p.is_symlink() or root not in p.resolve().parents or not p.is_file() or sha(p)!=expected:raise ValueError('retrieval asset bytes drift: '+n)
  verified[n]={'size':p.stat().st_size,'sha256':expected}
 return {'status':'ACTUAL_FRESH_RETRIEVAL_BYTES_VERIFIED_NOT_RESOURCE_PROOF','files':verified,'server':{'path':str(server),'sha256':server_sha}}

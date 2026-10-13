"""CPU-only fresh reuse of the actual64 frozen retrieval inventory."""
import ast,json,os,pathlib
from checkpoint_inventory import sha
BINDING_SHA='ea13a3b8352699212ee83185d2c1db4f9cdd080db03444581af5d641113a6fdf'
DIRECTORIES=('assets-root/assets/e5-base-v2','retriever-gpu-lib')
REQUIRED={'assets-root/assets/wiki/e5_Flat.index','assets-root/assets/wiki/wiki-18.jsonl'}
def verify_retrieval(root,reference,server):
 root=pathlib.Path(root).resolve();p=pathlib.Path(reference['path'])
 if p.is_symlink() or root not in p.resolve().parents or reference['sha256']!=BINDING_SHA or sha(p)!=BINDING_SHA:raise ValueError('actual64 retrieval binding drift')
 binding=json.loads(p.read_text())
 original=binding['source_files']['search_r1/search/retrieval_server.py']
 if original!='b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052':raise ValueError('Original actual64 source identity drift')
 expected_server=pathlib.Path(__file__).with_name('retrieval_server.py').resolve()
 if pathlib.Path(server).resolve()!=expected_server:raise ValueError('Only isolated original-G1 RPC compatibility server permitted')
 result=verify_inventory(root,binding['runtime_files'],server,'972bbb955190b11c45b54640f5054a7fcb62fdca15c2f0a0f4d3c139b525cec2')
 result['original_actual64_server_sha256']=original
 result['original_g1_rpc_restoration']=verify_restoration(pathlib.Path(server))
 result['compatibility_scope']='ORIGINAL_G1_BOOLEAN_RESPONSE_BRANCH_ONLY_NOT_DENSE_ENCODER_OR_INDEX_CHANGE'
 return result
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

def verify_restoration(server):
 original=server.with_name('original-native-server.py')
 if sha(original)!='b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052':raise ValueError('Original retrieval source drift')
 if sha(server)!='972bbb955190b11c45b54640f5054a7fcb62fdca15c2f0a0f4d3c139b525cec2':raise ValueError('Frozen approved RPC adapter drift')
 native_tree=ast.parse(original.read_text());patched_tree=ast.parse(server.read_text())
 for tree in (native_tree,patched_tree):
  tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='retrieve_endpoint']
 if ast.dump(native_tree)!=ast.dump(patched_tree):raise ValueError('Dense encoder/index math changed')
 return {'scope':'FROZEN_RPC_ADAPTER_AND_UNCHANGED_DENSE_ENCODER_AST','original_sha256':sha(original),'server_sha256':sha(server)}

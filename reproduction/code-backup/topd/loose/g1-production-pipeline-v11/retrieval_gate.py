"""CPU-only fresh reuse of the actual64 frozen retrieval inventory."""
import ast,json,os,pathlib
from checkpoint_inventory import sha
BINDING_SHA='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'
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
 original=server.with_name('original-native-server.py');g1=server.with_name('original-g1-server.py')
 originals={original:'b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052',g1:'0f770791898c4fcbd9ac077540c9afbec94e13d18c8dd869dfbd7fa0ef056ec0'}
 for path,expected in originals.items():
  if path.is_symlink() or sha(path)!=expected:raise ValueError('Original RPC source snapshot drift')
 native_tree=ast.parse(original.read_text());patched_tree=ast.parse(server.read_text());g1_tree=ast.parse(g1.read_text())
 def endpoint(tree):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='retrieve_endpoint')
 patch_endpoint=endpoint(patched_tree);g1_endpoint=endpoint(g1_tree)
 if ast.dump(ast.Module(body=patch_endpoint.body,type_ignores=[]))!=ast.dump(ast.Module(body=g1_endpoint.body,type_ignores=[])):raise ValueError('RPC endpoint not original G1 body')
 native_endpoint=endpoint(native_tree)
 native_tree.body=[n for n in native_tree.body if n is not native_endpoint];patched_tree.body=[n for n in patched_tree.body if n is not patch_endpoint]
 if ast.dump(native_tree)!=ast.dump(patched_tree):raise ValueError('Dense/Encoder/GPU retrieval math changed')
 return {'original_native_server_sha256':sha(original),'patched_server_sha256':sha(server),'original_g1_server_sha256':sha(g1),'endpoint_body_matches_original_g1':True,'all_other_ast_unchanged':True,'scope':'SOURCE_AST_RESTORATION_NOT_ACTUAL_GPU_PROTOCOL_PROOF'}

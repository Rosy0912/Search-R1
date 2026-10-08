"""Authorized readonly native package/source evidence. No CUDA/model/step."""
import sys,pathlib,json,base64,hashlib
sys.dont_write_bytecode=True
sys.path.insert(0,str(pathlib.Path(__file__).parent/'g1-native-cpu-deploy-v6'))
from stream_transport import send
root='/dataset/shaoshitong/search-r1-rl/topd'
inner=r'''
import os,sys,inspect,importlib,json,pathlib,hashlib,base64,subprocess,ast,importlib.metadata,time
assert sys.executable=='/dataset/shaoshitong/search-r1-rl/topd/env/bin/python'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
adamw=importlib.import_module('torch.optim.adamw');optimizer=importlib.import_module('torch.optim.optimizer')
assert not torch.cuda.is_initialized()
repo=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/repo-search-r1-32b-grpo100')
files=[]
for label,path in [('torch.optim.adamw',pathlib.Path(adamw.__file__)),('torch.optim.optimizer',pathlib.Path(optimizer.__file__)),('verl.utils.fsdp_utils',repo/'verl/utils/fsdp_utils.py')]:
 b=path.read_bytes();assert len(b)<1024*1024
 files.append(dict(label=label,path=str(path),size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
functions={}
for name,fn in [('_single_tensor_adamw',adamw._single_tensor_adamw),('adamw',adamw.adamw),('AdamW._init_group',adamw.AdamW._init_group),('AdamW.step',adamw.AdamW.step),('AdamW.step.unwrap',inspect.unwrap(adamw.AdamW.step)),('_use_grad_for_differentiable',optimizer._use_grad_for_differentiable),('Optimizer.profile_hook_step',optimizer.Optimizer.profile_hook_step)]:
 source=inspect.getsource(fn);functions[name]=dict(signature=str(inspect.signature(fn)),source_sha256=hashlib.sha256(source.encode()).hexdigest(),source=source,definition_line=inspect.getsourcelines(fn)[1],module=fn.__module__)
fsdp_tree=ast.parse((repo/'verl/utils/fsdp_utils.py').read_text());fsdp_meta={}
for node in fsdp_tree.body:
 if isinstance(node,ast.FunctionDef) and node.name.startswith(('load_fsdp','offload_fsdp')):
  fsdp_meta[node.name]=dict(line=node.lineno,end_line=node.end_lineno,arguments=ast.unparse(node.args))
versions={name:importlib.metadata.version(name) for name in ('torch','transformers','tensordict','vllm')}
git=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,timeout=5)
manifest=repo/'r1-files.json'
meta=dict(scope='NATIVE_PACKAGE_SOURCE_ONLY_NO_OPTIMIZER_STEP_NO_MODEL_NO_GPU',epoch=time.time(),hostname=__import__('socket').gethostname(),python=sys.version,executable=sys.executable,torch_version=torch.__version__,torch_cuda_package_version=torch.version.cuda,cuda_initialized=torch.cuda.is_initialized(),package_versions=versions,repository=str(repo),repository_git_observation=dict(returncode=git.returncode,stdout=git.stdout,stderr=git.stderr),repository_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),adamw_functions=functions,fsdp_function_definitions=fsdp_meta)
print(json.dumps(dict(metadata=meta,files=files)))
'''
remote="import subprocess,os\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')\nr=subprocess.run(["+repr(root+'/env/bin/python')+",'-B','-c',"+repr(inner)+"],env=env,capture_output=True,timeout=90)\nprint(r.stdout.decode(),end='');__import__('sys').stderr.write(r.stderr.decode());raise SystemExit(r.returncode)\n"
r=send(remote,attempts=3,retry_safe=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
rows=json.loads(r.stdout);out=pathlib.Path('outputs/remote-evidence/standard3170-torch-adamw-source');out.mkdir(parents=True,exist_ok=False)
for row in rows['files']:
 raw=base64.b64decode(row.pop('data'),validate=True)
 assert len(raw)==row['size'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 name={'torch.optim.adamw':'torch_optim_adamw.py','torch.optim.optimizer':'torch_optim_optimizer.py','verl.utils.fsdp_utils':'verl_utils_fsdp_utils.py'}[row['label']]
 (out/name).write_bytes(raw);row['local_file']=name
(out/'native-source-receipt.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(directory=str(out),metadata=rows['metadata']['package_versions'],files=rows['files'])))

import pathlib,shlex,subprocess,json
source=r'''import sys
sys.dont_write_bytecode=True
import os,json,pathlib,time,hashlib
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
folder=root/'logs'/'entropy3212-native-cpu-backend-cache-v1'
folder.mkdir(mode=0o700,exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['TORCHINDUCTOR_CACHE_DIR']=str(folder/'inductor');os.environ['TRITON_CACHE_DIR']=str(folder/'triton');os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE']='0'
import torch,torch._dynamo
assert torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized()
options={'triton.autotune_pointwise':False,'max_autotune':False,'max_autotune_pointwise':False,'coordinate_descent_tuning':False,'dynamic_scale_rblock':False}
source_path=root/'tools/search-r1-32b-grpo100-source-v5/verl/utils/torch_functional.py'
import ast
node=next(n for n in ast.parse(source_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='entropy_from_logits')
ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source_path),'exec'),ns);fn=ns['entropy_from_logits']
original=torch.compile(fn,dynamic=True);candidate=torch.compile(fn,dynamic=True,options=options)
configs={name:{k:f.get_compiler_config()[k] for k in options} for name,f in [('original',original),('candidate',candidate)]}
events=[]
def backend(gm,inputs,**kw):
 events.append({'kwargs':kw,'graph_code':gm.code,'epoch':time.time()});return gm.forward
# Same native Dynamo wrapper/backend path with a CPU eager graph executor.
# No Inductor kernel compilation, no CUDA device or model is exercised here.
torch._dynamo.reset()
a=torch.compile(fn,backend=backend,dynamic=True)
b=torch.compile(fn,backend=backend,dynamic=True,options=options)
x=torch.randn((2,17),requires_grad=True)
ya=a(x);counts1=len(events);yb=b(x);counts2=len(events)
same_code_run={'after_original':counts1,'after_candidate':counts2,'events':list(events),'equal':torch.equal(ya,yb)}
torch._dynamo.reset();events.clear();yc=b(x)
reset_run={'after_cold_candidate':len(events),'events':list(events),'equal':torch.equal(ya,yc)}
result={'scope':'ACTUAL_NATIVE_TORCH24_CPU_DYNAMO_BACKEND_CACHE_REPRO_NOT_GPU_NOT_CAPACITY_NOT_TRAINING','torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized(),'configs':configs,'same_callable_same_backend_run':same_code_run,'after_reset_candidate_run':reset_run,'source_file_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'epoch':time.time()}
with (folder/'result.json').open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
'''
command="env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /dataset/shaoshitong/search-r1-rl/topd/env/bin/python -B -c "+shlex.quote(source)
p=pathlib.Path('outputs/entropy3212-native-cpu-backend-cache-repro.json')
with p.open('x') as out:r=subprocess.run(['python3','work/remote_topd.py'],input=command,text=True,stdout=out,stderr=subprocess.PIPE,timeout=55)
assert r.returncode==0,r.stderr
obj=json.loads(p.read_text());print(json.dumps({'scope':obj['scope'],'configs':obj['configs'],'same_counts':[obj['same_callable_same_backend_run']['after_original'],obj['same_callable_same_backend_run']['after_candidate']],'cold_candidate_count':obj['after_reset_candidate_run']['after_cold_candidate'],'cuda_initialized':obj['cuda_initialized']}))

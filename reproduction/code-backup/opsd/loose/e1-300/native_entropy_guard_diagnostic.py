"""External read-only native CPU diagnosis. No source/GPU/job operations."""
from pathlib import Path
import json,sys,shlex,subprocess
H=Path(__file__).resolve().parent
CHILD=r'''
import sys,os,json,pathlib,importlib.util,hashlib,types
assert os.environ['CUDA_VISIBLE_DEVICES']==''
r=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/repo-89beac273f78');sys.path.insert(0,str(r))
import torch
assert torch.__version__.split('+')[0]=='2.4.0' and not torch.cuda.is_initialized()
import verl.utils.torch_functional as f
p=r/'reproduction/baselines/e1_import_bootstrap.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(p)=='82433ba203fa8b8ed02cc3c4d0be4d0d0ceb7de096ebe657561f1fd726194164';assert sha(pathlib.Path(f.__file__))=='2822127917dd538fc6f08cb7967d9c5c93d7718df77aab4e15a154fe2a54a8b34'
sp=importlib.util.spec_from_file_location('actual_native_guard',p);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
allowed={str(p):g.compiled_codes(p),f.__file__:g.compiled_codes(f.__file__)}
for name in g.LIBRARY_MODULES:
 q=g.source_path(r,name);allowed[str(q)]=g.compiled_codes(q)
before=g.check_live_code(f,allowed)
mode=sys.argv[1];x=torch.tensor([[-1.3,.7,2.1,-.2],[.01,1.25,-.5,3.]],dtype=torch.float64,requires_grad=True)
fn=torch.compile(f.entropy_from_logits,dynamic=True,backend='eager') if mode=='compiled' else f.entropy_from_logits
y=fn(x);grad=torch.autograd.grad(y.sum(),x)[0]
z=x.detach().clone().requires_grad_();logp=torch.log_softmax(z,dim=-1);ref=-(logp.exp()*logp).sum(-1);refgrad=torch.autograd.grad(ref.sum(),z)[0]
assert torch.allclose(y,ref,atol=1e-12,rtol=1e-12) and torch.allclose(grad,refgrad,atol=1e-12,rtol=1e-12)
generated={k:dict(filename=v.__code__.co_filename,code_name=v.__code__.co_name,qualname=v.__qualname__,firstlineno=v.__code__.co_firstlineno,fingerprint_sha256=hashlib.sha256(repr(g.code_fingerprint(v.__code__)).encode()).hexdigest()) for k,v in vars(f).items() if isinstance(v,types.FunctionType) and k.startswith('__resume_at_')}
try:after=g.check_live_code(f,allowed);error=None
except ValueError as e:after=None;error=str(e)
if mode=='compiled':assert error and 'torch_dynamo_resume_in_entropy_from_logits_at_97' in error and generated
else:assert error is None and not generated
# Strict same-path replacement negative: validfilename/projectmodule preserved,
# but replacement body cannot be accepted as original source bytecode.
namespace={};exec(compile('def entropy_from_logits(logits):\n return logits.sum(-1)\n',f.__file__,'exec'),namespace);bad=namespace['entropy_from_logits'];bad.__module__=f.__name__;original=f.entropy_from_logits;f.entropy_from_logits=bad
try:g.check_live_code(f,allowed)
except ValueError as e:tamper_error=str(e)
else:raise AssertionError('Same-path altered entropy accepted')
f.entropy_from_logits=original
assert not torch.cuda.is_initialized()
print(json.dumps(dict(mode=mode,torch=torch.__version__,pid=os.getpid(),uid=os.getuid(),repo=str(r),source_commit='89beac273f783d8e48d7d3e34da040f753514d90',source_checker_sha256=sha(p),entropy_source_sha256=sha(pathlib.Path(f.__file__)),before_function_count=before,after_function_count=after,actual_guard_error=error,generated_functions=generated,entropy_values=y.detach().tolist(),entropy_gradients=grad.tolist(),entropy_reference_max_abs_delta=float((y-ref).abs().max()),gradient_reference_max_abs_delta=float((grad-refgrad).abs().max()),samepath_tamper_rejected=True,tamper_error=tamper_error,cuda_is_initialized=False,scope='Actual nativeTorch24 CPU backend=eager diagnosis only; no model/state/GPU/training acceptance'),sort_keys=True))
'''
# Fill the exact previously frozen source SHA from committed local source, not a guessed value.
repo=H.parent/'opsd-e1-300'
import hashlib
CHILD=CHILD.replace('2822127917dd538fc6f08cb7967d9c5c93d7718df77aab4e15a154fe2a54a8b34',hashlib.sha256((repo/'verl/utils/torch_functional.py').read_bytes()).hexdigest())
outer='''import os,subprocess,json,pathlib,socket
r=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/repo-89beac273f78');head=subprocess.check_output(['git','-C',str(r),'rev-parse','HEAD'],text=True).strip();assert head=='89beac273f783d8e48d7d3e34da040f753514d90';assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain'],text=True).strip()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONPATH=str(r));rows=[]
for mode in ('plain','compiled'):
 p=subprocess.run(['/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python','-c',CHILD,mode],env=env,capture_output=True,text=True,timeout=45);assert p.returncode==0,p.stderr;rows.append(json.loads(p.stdout))
assert rows[0]['entropy_values']==rows[1]['entropy_values'] and rows[0]['entropy_gradients']==rows[1]['entropy_gradients'];assert not subprocess.check_output(['git','-C',str(r),'status','--porcelain'],text=True).strip()
print(json.dumps({'node':socket.gethostname(),'source_head':head,'source_clean_before_after':True,'uid':os.getuid(),'plain_compiled_loss_gradient_numeric_exact':True,'cases':rows,'scope':'Read-only nativeCPU diagnostic after failed3198; noGPU/newjob/no source mutations'},sort_keys=True))
'''.replace('CHILD',repr(CHILD))
p=subprocess.run([sys.executable,str(H.parent/'remote.py'),'python3 -c '+shlex.quote(outer)],capture_output=True,timeout=70)
(H/'native-entropy-guard-diagnostic-original.json').write_bytes(p.stdout);(H/'native-entropy-guard-diagnostic-stderr.txt').write_bytes(p.stderr)
assert p.returncode==0,p.stderr.decode();v=json.loads(p.stdout);assert v['node']=='h100-10';print(json.dumps({'actual_native_cpu_pass':True,'cases':[x['mode'] for x in v['cases']],'cuda_is_initialized':[x['cuda_is_initialized'] for x in v['cases']]}))

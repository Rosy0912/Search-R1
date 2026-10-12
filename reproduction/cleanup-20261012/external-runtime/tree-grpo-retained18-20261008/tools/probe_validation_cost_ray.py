"""Fresh actual Ray subprocess proof with CPU fake generation, not GPU evidence."""
import argparse,hashlib,json,os,pathlib,sys,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--repo',required=True);a=p.parse_args()
sys.path.insert(0,a.repo)
import ray
from validation_cost_hooks import measured_remote,wrap_trainer
base=pathlib.Path(a.output).resolve();base.mkdir(parents=True,exist_ok=False)
entry_pid=os.getpid();started=time.time()
# Match the fixed task decorator's default resource options.
@ray.remote
def original_task(config):
 from validation_cost_ray_fixture import setup_case,Manager,reset,rng,assert_rng
 import torch
 reset();T=setup_case();t=T();result=t._validate()
 return dict(pid=os.getpid(),result=result,rng_torch_sha256=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),task_id=ray.get_runtime_context().get_task_id())
assert original_task._default_options=={}

def installer():
 # A CPU fixture substitutes a tiny trainer before installing the actual hooks.
 import validation_cost_ray_fixture as fixture
 import validation_cost_hooks as hooks
 old=fixture.setup_case
 def make(*args,**kwargs):
  T=old(*args,**kwargs);hooks.wrap_trainer(T,fixture.Manager,base/'observed');return T
 fixture.setup_case=make

try:
 ray.init(num_cpus=2,include_dashboard=False,object_store_memory=128*1024**2,_temp_dir='/tmp/tcproof-'+str(os.getpid()),runtime_env={'env_vars':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
 off=ray.get(original_task.remote({}))
 observed=measured_remote(original_task._function,installer)
 on=ray.get(observed.remote({}))
 assert off['result']==on['result'] and off['rng_torch_sha256']==on['rng_torch_sha256']
 reports=list((base/'observed').glob('*.json'));assert len(reports)==1
 report=json.loads(reports[0].read_text());assert report['pid']==on['pid']!=entry_pid and report['status']=='VALIDATION_COST_COMPLETE' and report['counters']['returned_nonpad_model_token_events']==48
 out={'status':'ACTUAL_FRESH_RAY_REMOTE_OBSERVER_CPU_FIXTURE_PASSED','not_real_gpu_or_formal_validation':True,'ray_version':ray.__version__,'entry_pid':entry_pid,'off':off,'on':on,'actual_observer_report':str(reports[0]),'source_sha256':{str(pathlib.Path(__file__).resolve()):hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},'wall_seconds':time.time()-started}
 (base/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
finally:ray.shutdown()

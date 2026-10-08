"""Native CPU configuration and saved data-state restoration gate, no GPU work."""
import os,json,hashlib,ast,sys,types
from pathlib import Path
import torch
from omegaconf import OmegaConf
from launch_observed import compose_launch_config
from continuation import source_checkpoint,equal_state
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
c,reload=compose_launch_config();source,step,checkpoint=source_checkpoint()
assert not reload and step==5 and c.trainer.resume_mode=='resume_path'
assert c.trainer.resume_from_path==str(checkpoint) and not c.trainer.del_local_ckpt_after_load
assert c.trainer.max_actor_ckpt_to_keep==2
assert c.trainer.total_training_steps is None and c.trainer.test_freq==50 and c.trainer.save_freq==1
assert len(json.loads(Path(__file__).with_name('author-cli.json').read_text()))==102
assert hashlib.sha256(Path(__file__).with_name('author-cli.json').read_bytes()).hexdigest()=='7b8ccb02f43454832afc07f3e16d137ea5aeeb8b6ffd0638e3bb487bbaa90f7a'
# Actual native dataloader construction and actual native load_state_dict without advancing iterator.
from verl.trainer.ppo.search_r1_reranker_ray_trainer import CoSearchRayTrainer
from verl.trainer.main_ppo import create_rl_dataset,create_rl_sampler
from transformers import AutoTokenizer
from verl.utils.dataset.rl_dataset import collate_fn
tr=CoSearchRayTrainer.__new__(CoSearchRayTrainer);tr.config=c
tr.tokenizer=AutoTokenizer.from_pretrained(c.actor_rollout_ref.model.path,local_files_only=True)
train=create_rl_dataset(c.data.train_files,c.data,tr.tokenizer,None,is_train=True)
val=create_rl_dataset(c.data.val_files,c.data,tr.tokenizer,None,is_train=False)
sampler=create_rl_sampler(c.data,train)
tr._create_dataloader(train,val,collate_fn,sampler)
assert len(train)==166222 and len(val)==64 and tr.total_training_steps==324
saved=torch.load(checkpoint/'data.pt',map_location='cpu',weights_only=False)
tr.train_dataloader.load_state_dict(saved)
rng=torch.get_rng_state().clone()
try:assert equal_state(tr.train_dataloader.state_dict(),saved)
finally:torch.set_rng_state(rng)
assert torch.equal(rng,torch.get_rng_state())
# Extract the actual submitted subclass and run its wrapper call chain with CPU fakes.
tree=ast.parse(Path(__file__).with_name('launch_observed.py').read_text())
klass=next(x for x in ast.walk(tree) if isinstance(x,ast.ClassDef) and x.name=='FiveTrainer')
loads=[];events=[]
class Original:
 def _load_checkpoint(self):loads.append('load');self.global_steps=5
 def fit(self):
  self.global_steps=0;self._load_checkpoint();self.global_steps+=1
  assert self.global_steps==6 and self.total_training_steps==300
  return 'original-fit-called'
class Manager:
 def generate_sequences(self,b):return b
 def _aggregate_reranker_metrics(self,*a):return None
module=sys.modules['verl.trainer.ppo.search_r1_reranker_ray_trainer'];old=module.process_single_agent_ppo_step;module.process_single_agent_ppo_step=object()
phase=types.ModuleType('phase_sampling');phase.install_aggregation=lambda manager,write:(manager._aggregate_reranker_metrics,[])
oldphase=sys.modules.get('phase_sampling');sys.modules['phase_sampling']=phase
namespace=dict(original_class=Original,continue_to=300,reload_only=False,after_original_load=lambda trainer,writer:events.append(('resume',trainer.global_steps)),write=lambda kind,record:events.append((kind,record)),validate_acks=lambda *a:[],os=os,ObservedRemoteStep=lambda x:x)
exec(compile(ast.Module(body=[klass],type_ignores=[]),'actual-Q-subclass','exec'),namespace)
fake=namespace['FiveTrainer']();fake.total_training_steps=324;fake.val_dataset=[None]*64;fake.config=c;fake.async_rollout_manager=Manager()
os.environ.setdefault('SLURM_JOB_ID','cpu-gate');os.environ.setdefault('COSEARCH_NATIVE_EVIDENCE_ROOT','/tmp/cpu-only-Q')
assert fake.fit()=='original-fit-called' and loads==['load'] and ('resume',5) in events
module.process_single_agent_ppo_step=old
if oldphase is None:sys.modules.pop('phase_sampling')
else:sys.modules['phase_sampling']=oldphase
result=dict(passed=True,cuda_initialized=torch.cuda.is_initialized(),updates=0,source=source,source_step=5,target_step=300,next_step=6,natural_horizon=324,argv_count=102,train_rows=len(train),val_rows=len(val),data_state_equal=True,data_sha256=hashlib.sha256((checkpoint/'data.pt').read_bytes()).hexdigest(),single_original_load=True,readonly_validation_cpu_rng_preserved=True,save_freq=1,test_freq=50,original_save_path=True)
assert not result['cuda_initialized']
Path(__file__).with_name('cpu-continue-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

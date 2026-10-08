import json,pathlib,types,sys
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import resource_audit as r

def ready_fixture(tmp):
 config={'uuids':['uuid'+str(i) for i in range(8)],'job':'test','cgroup':'actual-job-cgroup'};apps=[]
 for role in ('actor','ref'):
  for rank in range(8):
   record=dict(role=role,rank=rank,pid=100+rank,starttime_ticks=rank+1,job='test',hostname='h100-11',cgroup=config['cgroup'],gpu_uuids=[config['uuids'][rank]],allocated_bytes=12,reserved_bytes=16)
   (tmp/('%s-ready-rank%d.json'%(role,rank))).write_text(json.dumps(record))
   if role=='actor':apps.append({'pid':100+rank,'uuid':config['uuids'][rank]})
 tools=types.SimpleNamespace(process_identity=lambda pid:{'starttime_ticks':pid-99})
 peers=[]
 def peer(pid,uid,trainer,cgroup):peers.append(pid)
 return config,tools,lambda:{'applications':apps},peer,peers

def test_actual_live_init_all16_receipts_before_ready(tmp_path):
 c,t,s,p,peers=ready_fixture(tmp_path)
 out=r.verify_ready(tmp_path,c,55,t,s,p)
 assert len(out['records'])==16 and len(peers)==16
 (tmp_path/'ref-ready-rank7.json').unlink()
 assert r.verify_ready(tmp_path,c,55,t,s,p) is None

def test_actual_ready_pid_reuse_and_nvml_drift_reject(tmp_path):
 c,t,s,p,_=ready_fixture(tmp_path);t.process_identity=lambda pid:{'starttime_ticks':0}
 with pytest.raises(ValueError,match='identity changed'):r.verify_ready(tmp_path,c,55,t,s,p)
 t.process_identity=lambda pid:{'starttime_ticks':pid-99}
 with pytest.raises(ValueError,match='GPU/allocator'):r.verify_ready(tmp_path,c,55,t,lambda:{'applications':[]},p)

def test_actual_verify_updates_requires_tail_correct_counts_and_finite(tmp_path):
 u=dict(local_rows=80,input_finite={'old_log_probs':True,'advantages':True,'loss_mask':True,'ref_log_prob':True},micro_forward_records=[{'finite':True}]*10,optimizer_step_records=[{'finite':True,'grad_norm':1.}]*3,parameter_sample_before='old',parameter_sample_after='new')
 records=[dict(rank=i,updates=[u,u]) for i in range(8)]
 assert r.verify_updates(records,2)['optimizer_steps_per_rank']==6
 records[0]=dict(rank=0,updates=[dict(u,micro_forward_records=[{'finite':True}]*8),u])
 with pytest.raises(ValueError,match='micro count'):r.verify_updates(records,2)

def test_actual_global_ready_polled_before_supervisor_closes_startup():
 source=pathlib.Path(r.__file__).with_name('supervise.py').read_text()
 block=source[source.index('while trainer.poll() is None:'):source.index('expected_refresh=')]
 assert block.index('verify_ready(')<block.index('guardian.close_ready()')
 assert 'actual-global-workers-ready.json' in block


def test_actual_offloaded_init_zero_allocator_with_live_gpu_context_is_valid(tmp_path):
 c,t,s,p,_=ready_fixture(tmp_path)
 for file in tmp_path.glob('*-ready-*.json'):
  v=json.loads(file.read_text());v.update(allocated_bytes=0,reserved_bytes=0);file.write_text(json.dumps(v))
 assert len(r.verify_ready(tmp_path,c,55,t,s,p)['records'])==16

def test_actual_original_update_method_observed_cpu_calls_not_synthetic_counts(monkeypatch):
 import ast,torch
 candidate=pathlib.Path(__file__).parents[1]/'g1-32b-fallback-candidate-v3'
 def function(path,name,ns):
  tree=ast.parse((candidate/path).read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);return ns[name]
 masked=function('verl/utils/torch_functional.py','masked_mean',{'torch':torch})
 core=function('verl/trainer/ppo/core_algos.py','compute_policy_loss',{'torch':torch,'verl_F':types.SimpleNamespace(masked_mean=masked)})
 def append(metrics,values):
  for k,v in values.items():metrics.setdefault(k,[]).append(v)
 original=function('verl/workers/actor/dp_actor.py','update_policy',dict(DataProto=object,torch=torch,core_algos=types.SimpleNamespace(compute_policy_loss=core),verl_F=types.SimpleNamespace(masked_mean=masked),append_to_dict=append))
 class Batch(dict):
  def __len__(self):return len(self['responses'])
  def split(self,n):return [Batch({k:v[i:i+n] for k,v in self.items()}) for i in range(0,len(self),n)]
  def cuda(self):return self
 class Data:
  meta_info={'temperature':1.}
  batch=Batch(responses=torch.zeros(80,1,dtype=torch.long),input_ids=torch.zeros(80,2,dtype=torch.long),attention_mask=torch.ones(80,2),position_ids=torch.zeros(80,2),old_log_probs=torch.zeros(80,1),advantages=torch.ones(80,1),loss_mask=torch.ones(80,1),ref_log_prob=torch.zeros(80,1))
  def __len__(self):return len(self.batch)
  def select(self,batch_keys):return types.SimpleNamespace(batch=Batch({k:self.batch[k] for k in batch_keys}))
 model=torch.nn.Linear(1,1,bias=False);model.weight.data.zero_();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
 def forward(micro_batch,temperature):
  logp=model(torch.ones(len(micro_batch),1)*.1);return logp*0,logp
 def step():
  norm=torch.linalg.vector_norm(model.weight.grad);optimizer.step();return norm
 actor=types.SimpleNamespace(actor_module=model,actor_optimizer=optimizer,_forward_micro_batch=forward,_optimizer_step=step,config=types.SimpleNamespace(ppo_mini_batch_size=32,ppo_micro_batch_size=8,use_dynamic_bsz=False,state_masking=True,use_kl_loss=False,clip_ratio=.2,entropy_coeff=0.))
 monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0);monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
 output,receipt=r.observed_update(actor,lambda data:original(actor,data),Data())
 assert len(receipt['optimizer_step_records'])==3 and len(receipt['micro_forward_records'])==10
 assert receipt['parameter_sample_before']!=receipt['parameter_sample_after'] and all(x['finite'] and x['grad_norm']>0 for x in receipt['optimizer_step_records'])
 assert actor._forward_micro_batch is forward and actor._optimizer_step is step
 # CPU shim asserts real production AST invocation counts; not GPU memory evidence.
 assert receipt['peak_allocated_bytes']==0


def test_actual_resume_must_have_current_round_not_copied_initial(tmp_path):
 c,t,s,p,_=ready_fixture(tmp_path)
 for file in list(tmp_path.glob('*-ready-*.json')):
  v=json.loads(file.read_text());v['outer_step']=25;(tmp_path/('round25-'+file.name)).write_text(json.dumps(v))
 assert len(r.verify_ready(tmp_path,c,55,t,s,p,prefix='round25-')['records'])==16
 file=tmp_path/'round25-ref-ready-rank7.json';v=json.loads(file.read_text());v['outer_step']=50;file.write_text(json.dumps(v))
 with pytest.raises(ValueError,match='resume outer'):r.verify_ready(tmp_path,c,55,t,s,p,prefix='round25-')


def test_actual_charged_cgroup_counter_raw_not_invented_when_missing(tmp_path):
 assert r.charged_memory(tmp_path)['available'] is False
 (tmp_path/'memory.peak').write_text('12345\n');(tmp_path/'memory.max').write_text('max\n')
 actual=r.charged_memory(tmp_path);assert actual['values']['memory.peak']['raw']=='12345' and actual['values']['memory.max']['raw']=='max'

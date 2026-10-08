import ast,copy,importlib.util,json,pathlib,types
import pytest
import torch
HERE=pathlib.Path(__file__).resolve().parent
import os
SOURCE=pathlib.Path(os.environ.get('R1_SOURCE',str(HERE.parent/'search-r1-32b-grpo100-source-v4')))
from prepare import derive
from supervise import verify_recipe
from resource_audit import verify_updates
from r1_observation import record_outer

def method(path,classname,name,namespace):
 tree=ast.parse(path.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname);fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==name);fn.decorator_list=[]
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]

def prior():
 return json.loads((HERE/'assets-template.json').read_text())
@pytest.mark.parametrize('mode,steps',[('profile2',2),('train100',100)])
def test_concrete_original_recipe(mode,steps):
 c=derive(prior(),mode,'/dataset/shaoshitong/search-r1-rl/topd/logs/r1-test')
 assert verify_recipe(c)==steps
 assert not any('calibration' in x['path'] for x in c['input_bindings'])
 assert not any(k in c for k in ('producer','refit_args','probe_url','overlay'))
 assert c['training_command'][:3][-1]=='verl.trainer.main_ppo'
@pytest.mark.parametrize('key,value',[('data.train_batch_size','128'),('actor_rollout_ref.rollout.n','5'),('actor_rollout_ref.actor.ppo_micro_batch_size','8'),('trainer.total_training_steps','100')])
def test_bad_geometry_rejected(key,value):
 c=derive(prior(),'train100','/dataset/shaoshitong/search-r1-rl/topd/logs/r1-test')
 c['training_command']=[key+'='+value if x.split('=',1)[0].lstrip('+')==key else x for x in c['training_command']]
 with pytest.raises(ValueError):verify_recipe(c)

def cpu_stream_modules():
 import sys
 package=types.ModuleType('_r1_v6_cpu24');package.__path__=[str(HERE)];sys.modules[package.__name__]=package
 for key,name in [('optimizer','native_torch24_optimizer.py'),('adamw','native_torch24_adamw.py')]:
  spec=importlib.util.spec_from_file_location(package.__name__+'.'+key,HERE/name);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 spec=importlib.util.spec_from_file_location('r1_v6_streamed',SOURCE/'verl/workers/actor/streamed_adamw.py');stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)
 return sys.modules[package.__name__+'.adamw'],stream

def actor_method(step,offload):
 return method(SOURCE/'verl/workers/actor/dp_actor.py','DataParallelPPOActor','_optimizer_step',{'torch':torch,'FSDP':type('NotFSDP',(),{}),'step_cpu_offloaded_adamw':step,'load_fsdp_optimizer':lambda *args,**kw:pytest.fail('No whole-state moment H2D'), 'offload_fsdp_optimizer':offload})

def test_actual_method_adam_two_steps_same_as_original_cpu(monkeypatch):
 monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
 original,stream=cpu_stream_modules();a=torch.nn.Linear(3,2,bias=False);b=copy.deepcopy(a);oa=original.AdamW(a.parameters(),lr=1e-3,foreach=False);ob=original.AdamW(b.parameters(),lr=1e-3,foreach=False);events=[]
 def streamed(opt):
  events.append('original-streamed-step')
  with stream._namespace(opt,original):opt.step()
 fn=actor_method(streamed,lambda opt:events.append('offload'))
 worker=types.SimpleNamespace(config=types.SimpleNamespace(grad_clip=1.,fsdp_config={'optimizer_offload':True}),actor_module=a,actor_optimizer=oa)
 for scale in (1.,0.,.5):
  oa.zero_grad();ob.zero_grad();inp=torch.ones(2,3);(a(inp).sum()*scale).backward();(b(inp).sum()*scale).backward()
  norm=fn(worker);torch.nn.utils.clip_grad_norm_(b.parameters(),max_norm=1.);ob.step()
  assert torch.isfinite(norm) and norm>=0
  for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,atol=1e-7,rtol=1e-7)
  for sa,sb in zip(oa.state.values(),ob.state.values()):
   for key in sa:torch.testing.assert_close(sa[key],sb[key],atol=1e-7,rtol=1e-7)
 assert events==['original-streamed-step','offload']*3

def test_partial_moment_migration_failure_still_offloads(monkeypatch):
 monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
 module=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(module.parameters(),lr=.01,foreach=False);module(torch.ones(1,3)).sum().backward();optimizer.step();events=[]
 def partial(opt):
  state=next(iter(opt.state.values()));state['exp_avg']=state['exp_avg'].clone();events.append('first-state-staged');raise RuntimeError('injected partial migration')
 def offload(opt):
  for state in opt.state.values():
   for key,t in state.items():state[key]=t.cpu()
  events.append('offloaded-all-state')
 fn=actor_method(partial,offload);w=types.SimpleNamespace(config=types.SimpleNamespace(grad_clip=1.,fsdp_config={'optimizer_offload':True}),actor_module=module,actor_optimizer=optimizer)
 with pytest.raises(RuntimeError,match='partial migration'):fn(w)
 assert events==['first-state-staged','offloaded-all-state']
 assert all(t.device.type=='cpu' for state in optimizer.state.values() for t in state.values())

def records(outer):
 u={'local_rows':320,'input_finite':dict.fromkeys(('old_log_probs','advantages','loss_mask','ref_log_prob'),True),'micro_forward_records':[{'finite':True}]*40,'optimizer_step_records':[{'finite':True,'grad_norm':0.}]*10,'parameter_sample_before':'a','parameter_sample_after':'b'}
 return [{'rank':rank,'updates':[dict(copy.deepcopy(u),actual_outer_update_ordinal=step) for step in range(1,outer+1)]} for rank in range(8)]
def test_actual_rank_geometry_zero_norm_accepted():
 result=verify_updates(records(2),2);assert result['optimizer_steps_per_rank']==20 and result['micro_forwards_per_rank']==80
@pytest.mark.parametrize('field,value',[('local_rows',80),('optimizer_step_records',[]),('micro_forward_records',[])])
def test_wrong_actual_geometry_rejected(field,value):
 r=records(2);r[3]['updates'][1][field]=value
 with pytest.raises(ValueError):verify_updates(r,2)
@pytest.mark.parametrize('norm',[float('nan'),float('inf'),-1.])
def test_invalid_norm_rejected(norm):
 r=records(2);r[0]['updates'][0]['optimizer_step_records'][0]['grad_norm']=norm
 with pytest.raises(ValueError):verify_updates(r,2)

def test_actual_completed_outer_sequence(tmp_path):
 p=tmp_path/'outer.json';metrics={'actor/grad_norm':0.,'actor/lr':0.}
 record_outer(p,1,metrics,str(tmp_path));record_outer(p,2,metrics,str(tmp_path));assert json.loads(p.read_text())['actual_completed_outer_steps']==[1,2]
 with pytest.raises(ValueError):record_outer(p,2,metrics,str(tmp_path))

def test_latest_upstream_loss_and_terminal_reward_exact():
 import hashlib
 originals=json.loads((HERE/'original-core-files.json').read_text())
 for name,expected in originals.items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==expected
 tree=ast.parse((SOURCE/'verl/workers/actor/dp_actor.py').read_text());text=ast.unparse(tree)
 assert 'torch.compile(verl_F.entropy_from_logits, dynamic=True)' in text
 assert 'loss = policy_loss / self.gradient_accumulation' in text


def test_actual_hydra_compose_standard_override_fields():
 from hydra import initialize_config_dir,compose
 c=derive(prior(),'profile2','/dataset/shaoshitong/search-r1-rl/topd/logs/r1-test')
 with initialize_config_dir(config_dir=str(SOURCE/'verl/trainer/config'),version_base=None):cfg=compose(config_name='ppo_trainer',overrides=c['training_command'][3:])
 assert cfg.algorithm.no_think_rl is False and cfg.actor_rollout_ref.actor.ppo_epochs==1
 assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size==64 and cfg.actor_rollout_ref.rollout.tensor_model_parallel_size==8
 assert cfg.trainer.total_training_steps==3

def test_compensating_outer_counts_rejected():
 r=records(2)
 r[0]['updates'][0]['optimizer_step_records'].pop();r[0]['updates'][1]['optimizer_step_records'].append({'finite':True,'grad_norm':0.})
 with pytest.raises(ValueError):verify_updates(r,2)

@pytest.mark.parametrize('norm',[float('nan'),-1.])
def test_bad_outer_metrics_fail_before_publish(tmp_path,norm):
 p=tmp_path/'outer.json'
 with pytest.raises(ValueError):record_outer(p,1,{'actor/grad_norm':norm,'actor/lr':0.},str(tmp_path))
 assert not p.exists()


def test_real_reaped_child_sample_is_non_authority():
 import subprocess,sys,os,importlib.util
 if not pathlib.Path('/proc/self/status').is_file():pytest.skip('Linux actual procfs required')
 spec=importlib.util.spec_from_file_location('readonly_support',HERE/'owner_support.py');tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
 child=subprocess.Popen([sys.executable,'-c','pass']);pid=child.pid;child.wait()
 from process_observation import observe
 r=observe(tools,pid);assert r['scope']=='OBSERVED_GONE_NON_AUTHORITY' and r['fresh_proc_absent']
 assert 'identity' not in r and not hasattr(r,'signal')

def test_live_uid_drift_rejected(tmp_path):
 from process_observation import observe
 d=tmp_path/'12';d.mkdir();(d/'status').write_text('Uid: 1000 1000 1000 1000\n')
 tools=types.SimpleNamespace(process_identity=lambda pid:{'pid':pid,'starttime_ticks':3,'cgroup':['x'],'state':'S'},process_sample=lambda pid:{'pid':pid,'memory':[]})
 with pytest.raises(RuntimeError,match='UID drift'):observe(tools,12,{'starttime_ticks':3,'uid':1001},proc=tmp_path)


def test_numeric_vllm_mapping_preserves_allocated_uuid_order():
 from supervise import numeric_visible
 inventory=[str(i)+', GPU-'+str(i)+', 0, 80000, 0' for i in range(8)]
 assert numeric_visible(inventory,['GPU-'+str(i) for i in range(7,-1,-1)])=='7,6,5,4,3,2,1,0'
 with pytest.raises(ValueError):numeric_visible(inventory,['GPU-'+str(i) for i in range(7)]+['FOREIGN'])


def test_actual3144_transition_remains_inconclusive_not_authority(tmp_path):
 from process_observation import observe
 fixture=json.loads((HERE/'actual3144-nonauthority.fixture.json').read_text());r=fixture['actual_row']['census_identity'];before=r['initial'];after=r['second'];pid=before['pid']
 assert before['state']==after['state']=='R' and before['starttime_ticks']==after['starttime_ticks'] and before['uid']==after['uid']
 d=tmp_path/str(pid);d.mkdir();(d/'status').write_text('Uid: 0 0 0 0\n')
 rows=iter([before,after]);tools=types.SimpleNamespace(process_identity=lambda p:copy.deepcopy(next(rows)),process_sample=lambda p:{'pid':p,'memory':['VmRSS: 1kB']})
 result=observe(tools,pid,proc=tmp_path)
 assert result['scope']=='OBSERVED_IDENTITY_TRANSITION_NON_AUTHORITY' and result['trusted_rss_sample'] is None and result['inconclusive']
 assert 'fresh_proc_absent' not in result and 'identity' not in result

def test_after_status_uid_drift_rejected(tmp_path):
 from process_observation import observe
 d=tmp_path/'12';d.mkdir();status=d/'status';status.write_text('Uid: 1000 1000 1000 1000\n');calls=[]
 def identity(p):
  calls.append(p)
  if len(calls)==2:status.write_text('Uid: 1001 1001 1001 1001\n')
  return {'pid':p,'starttime_ticks':3,'cgroup':['x'],'state':'S'}
 tools=types.SimpleNamespace(process_identity=identity,process_sample=lambda p:{'pid':p,'memory':[]})
 with pytest.raises(RuntimeError,match='UID drift'):observe(tools,12,proc=tmp_path)

def test_actual_supervisor_cache_failure_drains_started_thread_and_cleanup(tmp_path):
 import threading,os
 from test_final_cleanup import context
 import supervise
 from pipeline import atomic
 from checkpoint_inventory import sha
 tree=ast.parse((HERE/'supervise.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');statement=next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody)
 stop=threading.Event();monitor=threading.Thread(target=lambda:stop.wait(30));args,done=context(tmp_path)
 # Execute the real pre-GPU/cache body and actual except/finally from main.
 body=[]
 for node in statement.body:
  body.append(node)
  if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='mkdir':break
 selected=ast.Try(body=body,handlers=statement.handlers,orelse=[],finalbody=statement.finalbody)
 (tmp_path/'cache').write_text('forces actual mkdir FileExistsError')
 lock=os.open(tmp_path/'lock',os.O_CREAT|os.O_RDWR,0o600)
 ns=dict(args);ns.update(os=os,pathlib=pathlib,json=json,__file__=str(HERE/'supervise.py'),ROOT=tmp_path,header={'cpath':'x'},c={'candidate':'x','job':'test'},root=tmp_path,error=None,report=None,monitor=monitor,stop=stop,connection=None,listener=None,sockpath=None,terminal_drained=False,lock=lock,atomic=atomic,sha=sha)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[selected],type_ignores=[])),str(HERE/'supervise.py'),'exec'),ns)
 assert done==['cleanup'] and not monitor.is_alive() and stop.is_set()
 assert isinstance(ns['error'],FileExistsError)
 record=json.loads((tmp_path/'owned-cleanup.json').read_text());assert record['primary_error']['error_type']=='FileExistsError'

@pytest.mark.parametrize('mode',['profile2','train100'])
def test_cache_budget_actual3159_engineering_adaptation(mode):
 c=derive(prior(),mode,'/dataset/shaoshitong/search-r1-rl/topd/logs/r1-cache-v4-test')
 args={x.split('=',1)[0].lstrip('+'):x.split('=',1)[1] for x in c['training_command'][3:]}
 assert args['actor_rollout_ref.rollout.gpu_memory_utilization']=='0.4'
 assert args['actor_rollout_ref.rollout.tensor_model_parallel_size']=='8'
 assert args['data.train_batch_size']=='512'
 assert args['actor_rollout_ref.actor.ppo_micro_batch_size']=='64'
 assert args['data.max_prompt_length']=='4096' and args['data.max_response_length']=='500'
 assert verify_recipe(c)==(2 if mode=='profile2' else 100)
 c['training_command']=[x.replace('gpu_memory_utilization=0.4','gpu_memory_utilization=0.6') for x in c['training_command']]
 with pytest.raises(ValueError,match='gpu_memory_utilization'):verify_recipe(c)


def test_actual_launch_source_v2_binding_and_recipe():
 import hashlib
 from source_binding import check,SOURCE_SHA
 c=derive(prior(),'profile2','/dataset/shaoshitong/search-r1-rl/topd/logs/r1-v6-test');c['source_manifest']['sha256']=SOURCE_SHA
 assert check(c)==hashlib.sha256((SOURCE/'r1-files.json').read_bytes()).hexdigest()
 assert verify_recipe(c)==2
 for filename in ('supervise.py','train_entry.py'):
  tree=ast.parse((HERE/filename).read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
  assert any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='check_source_v4' for x in ast.walk(main))
@pytest.mark.parametrize('field,value',[('sha256','50fcaec86dcc0f7266f49cdf482a17d3b414b54a439f43a10617e49573320a25'),('candidate','/dataset/shaoshitong/search-r1-rl/topd/repo-search-r1-32b-grpo100')])
def test_old_whole_moment_source_rejected(field,value):
 from source_binding import check,SOURCE_SHA
 c=derive(prior(),'profile2','/dataset/shaoshitong/search-r1-rl/topd/logs/r1-v6-test');c['source_manifest']['sha256']=SOURCE_SHA
 if field=='candidate':c[field]=value
 else:c['source_manifest'][field]=value
 with pytest.raises(ValueError,match='sourceV4'):check(c)

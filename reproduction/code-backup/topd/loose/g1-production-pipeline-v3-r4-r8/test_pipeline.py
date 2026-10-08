"""CPU contracts only; fake allocator receipts never claim GPU hardware acceptance."""
import hashlib,importlib.util,json,os,pathlib,socket,struct,sys,threading,types
import pytest
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import pipeline,ipc,checkpoint_inventory,supervise
CANDIDATE=HERE.parent/'g1-32b-fallback-candidate-v3';sys.path.insert(0,str(CANDIDATE))
from search_r1.g1_owned_refresh import refresh_owned_pipeline,record_actual_outer

def records(role):
 return [dict(rank=i,pid=100+i,starttime_ticks=123,cgroup='own',job='1',host='h100-11',gpu_uuid='GPU-'+str(i),process_gpu_uuids=['GPU-'+str(i)],status='RELEASED_ZERO',allocated=0,reserved=0,parameter_devices={'module':['cpu']},buffer_devices={'module':['cpu']},role=role) for i in range(8)]
def config(tmp):
 return {'uuids':['GPU-'+str(i) for i in range(8)],'job':'1','cgroup':'own','checkpoint_root':str(tmp),'evidence_root':str(tmp),'base_model':str(tmp/'base')}
def patch_identity(monkeypatch):monkeypatch.setattr(pipeline,'identity',lambda pid:{'pid':pid,'starttime_ticks':123,'cgroup':'own'})
def test_sync_full_order_and_complete_ack(tmp_path,monkeypatch):
 patch_identity(monkeypatch);events=[];cfg=config(tmp_path);checkpoint=tmp_path/'actor';checkpoint.mkdir()
 def http(endpoint):
  events.append(endpoint)
  if endpoint=='barrier_pause':return {'status':'PAUSED_ENCODER_RELEASED','release':{str(i):{'allocated':0,'reserved':0} for i in range(8)}}
  return {'status':'READY','round':25,'encoder':str(checkpoint),'manifest_sha256':'hash'}
 def round_call(cfg,checkpoint,step,env,root):
  events.extend(['A_TRACE_EXIT_ZERO','B_HIDDEN_EXIT_ZERO','ORIGINAL_CPU_HEADS_EXIT','FROZEN_PUBLISH']);(root/'frozen-calibration.json').write_text('{}');return {'manifest_file_sha256':'hash'}
 owner=pipeline.Supervisor(cfg,{},http,round_call)
 result=owner.refresh({'outer_step':25,'checkpoint':str(checkpoint),'worker_releases':{role:records(role) for role in ('actor_rollout','ref')}})
 assert result['status']=='COMPLETE' and owner.completed==[25]
 assert events==['barrier_pause','A_TRACE_EXIT_ZERO','B_HIDDEN_EXIT_ZERO','ORIGINAL_CPU_HEADS_EXIT','FROZEN_PUBLISH','barrier_resume']
def test_allocator_failure_never_starts_round(tmp_path,monkeypatch):
 patch_identity(monkeypatch);r=records('actor_rollout');r[3]['reserved']=32*1024**2;called=[]
 owner=pipeline.Supervisor(config(tmp_path),{},lambda x:called.append(x))
 with pytest.raises(ValueError,match='not zero'):owner.refresh({'outer_step':25,'checkpoint':str(tmp_path),'worker_releases':{'actor_rollout':r,'ref':records('ref')}})
 assert not called and not owner.completed
@pytest.mark.parametrize('step',[0,50,100])
def test_refresh_wrong_order_rejects(tmp_path,step):
 with pytest.raises(ValueError,match='original order'):pipeline.Supervisor(config(tmp_path),{},None).refresh({'outer_step':step})
def test_stage_failure_no_reload_or_step_advance(tmp_path,monkeypatch):
 patch_identity(monkeypatch);called=[]
 def http(x):called.append(x);return {'status':'PAUSED_ENCODER_RELEASED','release':{str(i):{'allocated':0,'reserved':0} for i in range(8)}}
 def fail(*a):raise RuntimeError('B actual failed')
 owner=pipeline.Supervisor(config(tmp_path),{},http,fail)
 with pytest.raises(RuntimeError):owner.refresh({'outer_step':25,'checkpoint':str(tmp_path),'worker_releases':{role:records(role) for role in ('actor_rollout','ref')}})
 assert called==['barrier_pause'] and not owner.completed

def test_actual_worker_caller_all_groups_drain_before_allocator_checks(tmp_path,monkeypatch):
 import search_r1.g1_pipeline_ipc as ci
 events=[]
 class Group:
  def __init__(self,name):self.name=name
  def g1_calibration_quiesce(self):events.append('drain'+self.name)
  def g1_calibration_release_receipt(self):events.append('zero'+self.name);return records(self.name)
  def g1_calibration_resume(self):events.append('resume'+self.name)
 binding=tmp_path/'binding';binding.write_text(json.dumps({'uuids':config(tmp_path)['uuids']}))
 monkeypatch.setattr(ci,'client',lambda *a:events.append('COMPLETE') or {'status':'COMPLETE'})
 refresh_owned_pipeline(binding,tmp_path,25,Group('actor_rollout'),Group('ref'))
 assert events==['drainactor_rollout','drainref','zeroactor_rollout','zeroref','COMPLETE','resumeactor_rollout','resumeref']
def test_unknown_ack_does_not_resume(tmp_path,monkeypatch):
 import search_r1.g1_pipeline_ipc as ci
 events=[]
 class Group:
  def g1_calibration_quiesce(self):pass
  def g1_calibration_release_receipt(self):return records('actor_rollout')
  def g1_calibration_resume(self):events.append('BAD_RESUME')
 binding=tmp_path/'binding';binding.write_text(json.dumps({'uuids':config(tmp_path)['uuids']}))
 def failed(*a):raise EOFError('unknown ACK')
 monkeypatch.setattr(ci,'client',failed)
 with pytest.raises(EOFError):refresh_owned_pipeline(binding,tmp_path,25,Group())
 assert not events

def test_real_socket_bounded_frame_and_truncated_ack():
 a,b=socket.socketpair()
 try:
  ipc.send(a,{'outer_step':25});assert ipc.read(b)=={'outer_step':25}
  with pytest.raises(ValueError):ipc.send(a,{'large':'x'*(ipc.MAX_FRAME+1)})
  a.sendall(struct.pack('!I',8)+b'{');a.shutdown(socket.SHUT_WR)
  with pytest.raises(EOFError):ipc.read(b)
 finally:a.close();b.close()
def test_exact_overlay_rejects_added_pth(tmp_path):
 overlay=tmp_path/'overlay';overlay.mkdir();(overlay/'a').write_text('one');receipt=tmp_path/'receipt';receipt.write_text(json.dumps({'overlay_files':{'a':checkpoint_inventory.sha(overlay/'a')}}))
 pipeline.verify_overlay(overlay,receipt,checkpoint_inventory.sha(receipt));(overlay/'attack.pth').write_text('bad')
 with pytest.raises(ValueError,match='exact file set'):pipeline.verify_overlay(overlay,receipt,checkpoint_inventory.sha(receipt))
def test_actual_outer_order_and_export_required(tmp_path):
 p=tmp_path/'outer.json';record_actual_outer(p,1,False,tmp_path)
 with pytest.raises(ValueError,match='order'):record_actual_outer(p,3,False,tmp_path)
 with pytest.raises(ValueError,match='export absent'):record_actual_outer(p,2,True,tmp_path)
 export=tmp_path/'actor/global_step_2';export.mkdir(parents=True);(export/'config.json').write_text('{}');record_actual_outer(p,2,True,tmp_path)
 assert json.loads(p.read_text())['actual_completed_outer_steps']==[1,2]
def test_qwen32b_complete_tensor_schema():
 shapes=checkpoint_inventory.expected_shapes({'num_attention_heads':40,'num_key_value_heads':8,'intermediate_size':27648})
 assert len(shapes)==771 and shapes['model.layers.63.mlp.down_proj.weight']==[5120,27648]
 assert sum(__import__('math').prod(v) for v in shapes.values())==32763876352

def test_real_original_refit_publication_intercept_failure_restores_old(tmp_path,monkeypatch):
 # Executes the unchanged original coevolve.refit function, substitutes only child processes.
 producer=HERE.parent/'g1-staged-calibration-producer-v3';sys.path.insert(0,str(producer))
 import refit_with_owner as refit
 args={'work_dir':str(tmp_path/'work'),'manifest_root':str(tmp_path/'manifests'),'data':str(tmp_path/'data'),'tag':'nq','nq':300,'holdout_offset':0,'group_size':5,'layers':'21,24,27','gpu_mem':.55,'commit_position':'Xa','commit_label':'flip'}
 cfg={'producer':str(producer),'base_model':str(tmp_path/'base'),'experiment':'search_g1','refit_args':args,'owner_context':{'config':{'stage_timeout_seconds':5,'leases':[]},'support_sha256':'fixture','evidence_root':str(tmp_path)}}
 manifest=tmp_path/'manifests/live_search_g1/manifest.json';manifest.parent.mkdir(parents=True);manifest.write_text('OLD');roundroot=tmp_path/'round';roundroot.mkdir()
 monkeypatch.setattr(pipeline,'inventory',lambda path:{'scope':'CPU_FIXTURE'})
 monkeypatch.setattr(pipeline,'tokenizer_compatible',lambda *a:{'scope':'CPU_FIXTURE'})
 calls=[]
 def run(command,env,ctx,pass_fds=()):
  calls.append(command)
  if '--dump-only' in command:raise RuntimeError('A real child failure fixture')
  return types.SimpleNamespace(returncode=0)
 monkeypatch.setattr(refit,'owned_command',run)
 with pytest.raises(RuntimeError,match='A real child failure'):pipeline.original_round(cfg,tmp_path/'model',25,{},roundroot,refit)
 assert manifest.read_text()=='OLD' and len(calls)==1 and '--dump-only' in calls[0]
 assert not (roundroot/'frozen-calibration.json').exists()

@pytest.mark.parametrize('head_rc',[0,1])
def test_actual_original_refit_complete_or_head_exit_fail_closed(tmp_path,monkeypatch,head_rc):
 producer=HERE.parent/'g1-staged-calibration-producer-v3';sys.path.insert(0,str(producer));import refit_with_owner as refit
 args={'work_dir':str(tmp_path/'work'),'manifest_root':str(tmp_path/'manifests'),'data':str(tmp_path/'data'),'tag':'nq','nq':300,'holdout_offset':0,'group_size':5,'layers':'21,24,27','gpu_mem':.55,'commit_position':'Xa','commit_label':'flip'}
 cfg={'producer':str(producer),'base_model':str(tmp_path/'base'),'experiment':'search_g1','refit_args':args,'owner_context':{'config':{'stage_timeout_seconds':5,'leases':[]},'support_sha256':'fixture','evidence_root':str(tmp_path)}}
 manifest=tmp_path/'manifests/live_search_g1/manifest.json';manifest.parent.mkdir(parents=True);manifest.write_text('OLD');roundroot=tmp_path/'round';roundroot.mkdir()
 monkeypatch.setattr(pipeline,'inventory',lambda path:{'scope':'CPU_FIXTURE'});monkeypatch.setattr(pipeline,'tokenizer_compatible',lambda *a:{'scope':'CPU_FIXTURE'})
 calls=[]
 def run(command,env,ctx,pass_fds=()):
  calls.append(command)
  if '--dump-only' in command:
   directory=pathlib.Path(command[command.index('--directory')+1]);directory.mkdir();source=json.loads(pathlib.Path(command[command.index('--config')+1]).read_text());inputfile=directory/'frozen-input-config.json';inputfile.write_text(json.dumps(source));inputsha=checkpoint_inventory.sha(inputfile)
   dump=directory/'dump';dump.mkdir();npz=dump/'nq_0.npz'
   import numpy as np
   np.savez(npz,q_cb=np.array([1.,0.]));npz.with_suffix('.jsonl').write_text('{}\n')
   result={'complete':True,'status':'DUMP_COMPLETE_HARDWARE_NOT_INDEPENDENTLY_ACCEPTED','config_sha256':inputsha,'npz_sha256':checkpoint_inventory.sha(npz),'jsonl_sha256':checkpoint_inventory.sha(npz.with_suffix('.jsonl'))}
   (directory/'result.json').write_text(json.dumps(result))
   for phase in ('A','B'):(directory/f'phase-{phase}-owner-barrier.json').write_text(json.dumps({'scope':'CPU_FIXTURE_NO_GPU_PROOF','returncode':0,'own_processes_remaining':[],'own_gpu_records_remaining':[],'baseline_preserved':True,'original_global_gpu_baseline_preserved':True,'config_sha256':inputsha}))
  else:
   heads=pathlib.Path(command[command.index('--save')+1]);heads.mkdir(parents=True)
   for name in ('commit_probe.pkl','boundary_probe.pkl'):(heads/name).write_bytes(b'CPU_FIXTURE_NOT_REAL_HEAD')
   (roundroot/'heads-1.json').write_text(json.dumps({'scope':'CPU_FIXTURE_NO_GPU_PROOF','returncode':head_rc,'own_processes_remaining':[],'own_gpu_records_remaining':[],'baseline_preserved':True}))
  # Failure receipt is injected even when command claims exit0: publish must reread evidence.
  return types.SimpleNamespace(returncode=0,receipt_path='CPU_FIXTURE',receipt_sha256='CPU_FIXTURE')
 monkeypatch.setattr(refit,'owned_command',run)
 if head_rc:
  with pytest.raises(ValueError,match='CPU head exit'):pipeline.original_round(cfg,tmp_path/'model',25,{},roundroot,refit)
  assert manifest.read_text()=='OLD' and not (roundroot/'frozen-calibration.json').exists()
 else:
  frozen=pipeline.original_round(cfg,tmp_path/'model',25,{},roundroot,refit)
  assert checkpoint_inventory.sha(manifest)==frozen['manifest_file_sha256']
  assert json.loads(manifest.read_text())['round']==25
  assert len(calls)==2 and calls[1][calls[1].index('--commit-position')+1]=='Xa' and calls[1][calls[1].index('--commit-label')+1]=='flip'

def test_old_hardware_scope_does_not_authorize_new32b(tmp_path):
 gate=tmp_path/'gate';config={'independent_gate':{'path':str(gate),'sha256':''}}
 gate.write_text(json.dumps({'status':'PASS','scope':'HARDWARE_SCORER_READY_NO_TRAINING','reviewed_static_recipe_sha256':supervise.reviewed_recipe_sha(config)}))
 with pytest.raises(ValueError,match='new32B'):supervise.verify_gate({'independent_gate':{'path':str(gate),'sha256':checkpoint_inventory.sha(gate)}},'train100')

def test_actual_worker_quiesce_clears_completed_gradient_preserves_parameters(monkeypatch):
 import torch
 import search_r1.g1_phase_resources as resources
 model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters());model(torch.ones(1,2)).sum().backward();optimizer.step();before=[p.detach().clone() for p in model.parameters()]
 worker=types.SimpleNamespace(rank=0,role='actor',_is_rollout=False,actor_module_fsdp=model,actor_optimizer=optimizer)
 cuda=types.SimpleNamespace(synchronize=lambda:None);fake=types.SimpleNamespace(cuda=cuda)
 monkeypatch.setattr(resources,'identity',lambda w:{'scope':'CPU_FIXTURE'})
 record=resources.quiesce(worker,fake,lambda module,offload_grad:None,lambda optimizer:None)
 assert all(p.grad is None for p in model.parameters()) and all(torch.equal(p,b) for p,b in zip(model.parameters(),before))
 assert record['status']=='OFFLOADED_PENDING_ALL_ROLES_ALLOCATOR_CHECK'
 assert resources.resume(worker,fake)['status']=='BUFFERS_RESTORED_ORIGINAL_CALLER_LOADS_WEIGHTS'

def test_actual_worker_release_rejects_nonzero_workspace(monkeypatch):
 import torch
 import search_r1.g1_phase_resources as resources
 worker=types.SimpleNamespace(rank=0,role='actor',_is_rollout=False,actor_module_fsdp=torch.nn.Linear(2,1),_g1_buffer_restore=[])
 fake=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,memory_allocated=lambda:0,memory_reserved=lambda:33554432),_C=types.SimpleNamespace(_cuda_clearCublasWorkspaces=lambda:None))
 monkeypatch.setattr(resources,'identity',lambda w:{'scope':'CPU_FIXTURE'});monkeypatch.setattr(resources,'actual_uuid',lambda t:('GPU-fixture',['GPU-fixture']))
 with pytest.raises(RuntimeError,match='not zero'):resources.release_receipt(worker,fake)

def test_reviewed_whole_recipe_rejects_command_or_localurl_drift(tmp_path):
 proof=tmp_path/'gate';raw=tmp_path/'raw';review=tmp_path/'review';raw.write_text('{}');review.write_text('{"status":"PASS"}')
 cfg={key:'bound' for key in ('candidate_manifest_sha256','producer_manifest_sha256','pipeline_manifest_sha256','base64_result_sha256','base_model_inventory_sha256')}
 cfg.update(training_command=['actual-static-command'],retriever_url='http://127.0.0.1:8000/retrieve',independent_gate={'path':str(proof),'sha256':''})
 value={key:cfg[key] for key in ('candidate_manifest_sha256','producer_manifest_sha256','pipeline_manifest_sha256','base64_result_sha256','base_model_inventory_sha256')}
 value.update(status='PASS',scope='G1_32B_INITIAL_CALIBRATION_SOURCE_AND_ALLOCATION_GATE',actual_gpu_count=8,host_memory_request_gib=768,global_actor_microbatch=64,reviewed_static_recipe_sha256=supervise.reviewed_recipe_sha(cfg),actual_raw_evidence={'path':str(raw),'sha256':checkpoint_inventory.sha(raw)},independent_review={'path':str(review),'sha256':checkpoint_inventory.sha(review)})
 proof.write_text(json.dumps(value));cfg['independent_gate']['sha256']=checkpoint_inventory.sha(proof)
 assert supervise.verify_gate(cfg,'initial-only')['status']=='PASS'
 cfg['retriever_url']='http://foreign:8000/retrieve'
 with pytest.raises(ValueError,match='complete reviewed'):supervise.verify_gate(cfg,'initial-only')
 cfg['retriever_url']='http://127.0.0.1:8000/retrieve';cfg['training_command'].append('bad_override=1')
 with pytest.raises(ValueError,match='complete reviewed'):supervise.verify_gate(cfg,'initial-only')

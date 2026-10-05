"""Synthetic CPU record-schema counterexamples; never actual GPU acceptance."""
import copy,json
from pathlib import Path
import pytest
from test_recovery_state import fixture_manifest
from search_r1.baselines import recovery as r

def setup_proof(tmp,monkeypatch):
 m,mp,c,train,permit,pp=fixture_manifest(tmp,monkeypatch);root=train.parent.parent;run=root/'runs/bapo-instruct300-hardware-10';run.mkdir();job=10
 def bind(path,value,json_value=True):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)if json_value else value);return dict(path=str(path),bytes=path.stat().st_size,sha256=r.digest(path))
 # Complete native/driver producer identities in the synthetic historical reports.
 for x in m['ranks']:
  rep=r.read(x['report']['path']);rep.update(pid=100+x['rank'],state_digest=str(x['rank'])*64);x['report']=bind(Path(x['report']['path']),rep)
 dr=r.read(m['driver_report']['path']);dr.update(pid=77,state_digest='a'*64);m['driver_report']=bind(Path(m['driver_report']['path']),dr)
 stage=dict(best=.28125,stalled=14,early=False,patience=5);a=r.read(m['algorithm']['path']);a['bapo']=stage;m['algorithm']=bind(Path(m['algorithm']['path']),a)
 oldcfg=r.read(m['resolved_config']['path']);model=Path(oldcfg['actor_rollout_ref']['model']['path']);model_cfg=dict(model_type='qwen2',hidden_size=2048,num_hidden_layers=36,vocab_size=151936,architectures=['Qwen2ForCausalLM']);bind(model/'config.json',model_cfg)
 gm=copy.deepcopy(m);gm.update(target_job_id=10,target_run_root=str(run));gmb=bind(run/'resume-manifest.json',gm)
 # CPU schema fixtures bind concrete module bytes, without executing GPU code.
 native_module=root/'repo/verl/workers/fsdp_workers.py';driver_module=root/'repo/verl/trainer/ppo/ray_trainer.py';bind(native_module,'# synthetic module bytes\n',False);bind(driver_module,'# synthetic module bytes\n',False)
 def process(pid):return dict(pid=pid,start_ticks=900+pid,state='S',actual_env=dict(SLURM_JOB_ID='10',BAPO_RUN_ID=str(run),CUDA_VISIBLE_DEVICES='0'))
 ranks=[]
 for i in range(8):
  rep=r.read(m['ranks'][i]['report']['path']);ranks.append(dict(status='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK',rank=i,world_size=8,step=25,next_step=26,scheduler_last_epoch=25,distributed_backend='nccl',model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=435,model_class='Qwen2ForCausalLM',model_config=model_cfg,old_pid=100+i,new_pid=200+i,old_state_sha256=m['ranks'][i]['state']['sha256'],state_digest=rep['state_digest'],optimizer_state_entries=37,lr=[1e-6*25/114],actual_env=process(200+i)['actual_env'],process=process(200+i),target_source_commit='new',target_run_root=str(run),sampling_continuity='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT',runtime_module_path=str(native_module),runtime_module_sha256=r.digest(native_module)))
 d=dict(status='EXACT_CROSS_JOB_DRIVER_RELOAD_OK',verify_only=True,no_generation_or_optimizer_update=True,step=25,next_step=26,actual_outer_updates=25,epoch=0,sampler_cursor=12800,target_job_id=10,old_pid=77,new_pid=299,loaded_state_digest='a'*64,bapo=stage,saved_sampler_digest='c'*64,restored_sampler_digest='c'*64,saved_rng_digest='d'*64,restored_rng_digest='d'*64,target_source_commit='new',target_run_root=str(run),sampling_continuity='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT',process=process(299),runtime_module_path=str(driver_module),runtime_module_sha256=r.digest(driver_module),rank_witnesses=ranks)
 proof=[bind(run/'recovery-witness'/f'rank_{i}.json',w)for i,w in enumerate(ranks)]+[bind(run/'recovery-witness/driver.json',d)]
 watcher=root/'tools/resource_watch.py';bind(watcher,'# synthetic watcher fixture\n',False);monkeypatch.setattr(r,'RESOURCE_WATCH_SHA',r.digest(watcher))
 summary=dict(status='PASS_ACTUAL_GUARD_JOURNAL_REPLAY',job_id=10,source_commit='new',run_root=str(run),raw_sha256='',samples=1,recovery_samples=0,freeze_events=0,safe_resume_events=0,abort_events=0,unfinished_recovery=False)
 watch=bind(run/'resource-watch.jsonl',{'raw':'CPU schema test not realGPU'});summary['raw_sha256']=watch['sha256']
 helper=root/'tools/guard_journal.py';bind(helper,'def replay(*args):\n    import hashlib\n    from pathlib import Path\n    result='+repr(summary)+'\n    result[\'helper_sha256\']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()\n    result[\'watcher_sha256\']=hashlib.sha256(Path(args[4]).read_bytes()).hexdigest()\n    return result\n',False);monkeypatch.setattr(r,'GUARD_JOURNAL_SHA',r.digest(helper));summary.update(helper_sha256=r.digest(helper),watcher_sha256=r.digest(watcher))
 for i in range(20):bind(root/'tools'/f'tool{i}.py','# synthetic fixture\n',False)
 files={p.name:r.digest(p)for p in (root/'tools').iterdir()if p.is_file()};tools=bind(root/'logs/tools-manifest.json',dict(source_commit='new',files=files))
 cpu_log=bind(root/'logs/cpu-gate.log','327 passed\n',False);cpu=bind(root/'logs/cpu-gate.json',dict(source_commit='new',exit_code=0,failed=0,skipped=0,passed=327,scope='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE',sys_prefix=str(root/'env'),log_sha256=cpu_log['sha256']))
 approval=bind(root/'logs/launch-approval.json',dict(source_commit='new',cpu_expected_passed=327,tools_file_count=len(files),tools_manifest_sha256=tools['sha256'],cpu_gate_record_sha256=cpu['sha256']))
 cfg=copy.deepcopy(oldcfg);cfg['trainer']['resume_verify_only']=True
 raw=dict(slurm_terminal=bind(run/'slurm-terminal.txt','JobId=10 JobState=COMPLETED ExitCode=0:0 DerivedExitCode=0:0 Restarts=0 NodeList=h100-5\n',False),source_start=bind(run/'source-commit.txt','new\n',False),resolved_config=bind(run/'resolved-config.yaml',cfg),tools_manifest=tools,cpu_gate=cpu,cpu_log=cpu_log,cpu_source=bind(root/'logs/cpu-gate-source.txt','new\n',False),launch_approval=approval,guard_summary=bind(run/'guard-summary.json',summary),guard_helper=bind(helper,helper.read_text(),False),resource_watcher=bind(watcher,watcher.read_text(),False),resource_watch=watch,cleanup=bind(run/'own-process-cleanup.json',dict(remaining=[])),train_log=bind(run/'train.log','\n'.join('BAPO_RECOVERY_RANK_JSON='+json.dumps(w)for w in ranks)+'\nBAPO_RECOVERY_DRIVER_JSON='+json.dumps(d)+'\n',False))
 v3=[]
 for i in range(8):
  ptb=bind(run/'recovery-native-v3-roundtrip'/f'rank_{i}.pt','Synthetic CPU bytes not real native Torch state',False)
  vr=dict(status='EXACT_FULL_STATE_RELOAD_OK',native_state_version=3,roundtrip=True,rank=i,world_size=8,step=25,path=ptb['path'],sha256=ptb['sha256'],pid=200+i,scheduler_last_epoch=25,optimizer_state_entries=37,model_state_dict_type='SHARDED_STATE_DICT',model_dtensor_entries=435,state_digest='e'*64,reloaded_state_digest='e'*64,generation_rng_before_digest='f'*64,generation_rng_after_digest='f'*64,generation_rng_topology=dict(version=1,torch_random_states_numel=5056,torch_random_states_dtype='torch.uint8',torch_random_states_shape=[5056],gen_random_states_numel=5056,gen_random_states_dtype='torch.uint8',gen_random_states_shape=[5056],device_mesh=dict(shape=[8,1],names=['dp','infer_tp'],ranks=[[j]for j in range(8)])))
  v3.append(dict(rank=i,state=ptb,report=bind(run/'recovery-native-v3-roundtrip'/f'rank_{i}.json',vr)))
 raw['train_log']=bind(run/'train.log',Path(raw['train_log']['path']).read_text()+''.join('BAPO_RECOVERY_V3_RANK_JSON='+json.dumps(r.read(x['report']['path']))+'\n'for x in v3),False)
 permit.update(status='PASS_INDEPENDENT_BAPO_RESUME25_REVIEW',gpu_witnesses=proof,gpu_validation=dict(job_id=10,run_root=str(run),manifest=gmb,tools_file_count=len(files),raw_artifacts=raw,native_v3_roundtrip=v3))
 return m,permit,oldcfg,root,run,ranks,d,bind

def test_complete_synthetic_schema_only(tmp_path,monkeypatch):
 m,p,c,root,*_=setup_proof(tmp_path,monkeypatch);r.validate_gpu_proof(p,m,c,root)
@pytest.mark.parametrize('case',['dummy_json','non_json','missing_rank','wrong_source','wrong_job','wrong_run','rank_bool','backend_gloo','no_dtensor','wrong_model','old_pid','pid_reused','bad_startticks','wrong_env','wrong_module','changed_module','native_digest','old_state_hash','bad_lr','driver_training','driver_sampler','driver_rng','no_driver_binding','terminal_failed','terminal_restarted','cpu_subset','cpu_source','cleanup','guard_summary','watcher_change','guardhelper_change','raw_training','different_input','duplicate_json','json_nan','bad_process_state','old_pid_bool','old_pid_negative','process_pid_bool'])
def test_reject_forged_gpu_proofs(tmp_path,monkeypatch,case):
 m,p,c,root,run,ranks,d,bind=setup_proof(tmp_path,monkeypatch)
 if case=='duplicate_json':
  path=run/'recovery-witness/rank_0.json';text=json.dumps(ranks[0]);path.write_text(text[:-1]+', \"rank\":0}');p['gpu_witnesses'][0]=dict(path=str(path),bytes=path.stat().st_size,sha256=r.digest(path))
 elif case=='json_nan':
  path=run/'recovery-witness/rank_0.json';text=json.dumps(ranks[0]);path.write_text(text[:-1]+', \"unexpected\":NaN}');p['gpu_witnesses'][0]=dict(path=str(path),bytes=path.stat().st_size,sha256=r.digest(path))
 elif case=='dummy_json':p['gpu_witnesses']=[bind(run/'dummy.json',{'fake':'PASS'})]
 elif case=='non_json':p['gpu_witnesses']=[bind(run/'dummy.txt','not JSON',False)]
 elif case=='missing_rank':p['gpu_witnesses'].pop(0)
 elif case=='wrong_source':ranks[0]['target_source_commit']='old'
 elif case=='wrong_job':ranks[0]['actual_env']['SLURM_JOB_ID']='9'
 elif case=='wrong_run':ranks[0]['target_run_root']=str(root/'runs/foreign')
 elif case=='rank_bool':ranks[0]['rank']=False
 elif case=='backend_gloo':ranks[0]['distributed_backend']='gloo'
 elif case=='no_dtensor':ranks[0]['model_dtensor_entries']=0
 elif case=='wrong_model':ranks[0]['model_config']=dict(model_type='tiny')
 elif case=='old_pid_bool':ranks[0]['old_pid']=True
 elif case=='old_pid_negative':ranks[0]['old_pid']=-1
 elif case=='process_pid_bool':ranks[0]['new_pid']=1;ranks[0]['process']['pid']=True
 elif case=='bad_process_state':ranks[0]['process']['state']='unreadable'
 elif case=='old_pid':ranks[0]['new_pid']=ranks[0]['old_pid']
 elif case=='pid_reused':ranks[0]['new_pid']=ranks[1]['new_pid']
 elif case=='bad_startticks':ranks[0]['process']['start_ticks']=False
 elif case=='wrong_env':ranks[0]['process']['actual_env']['BAPO_RUN_ID']='foreign'
 elif case=='wrong_module':ranks[0]['runtime_module_path']='/tmp/shadow.py'
 elif case=='changed_module':Path(ranks[0]['runtime_module_path']).write_text('changed')
 elif case=='native_digest':ranks[0]['state_digest']='f'*64
 elif case=='old_state_hash':ranks[0]['old_state_sha256']='f'*64
 elif case=='bad_lr':ranks[0]['lr']=[1e-6]
 elif case=='driver_training':d['no_generation_or_optimizer_update']=False
 elif case=='driver_sampler':d['sampler_cursor']=0
 elif case=='driver_rng':d['restored_rng_digest']='e'*64
 elif case=='no_driver_binding':d['rank_witnesses']=[]
 elif case=='different_input':gm=r.read(p['gpu_validation']['manifest']['path']);gm['step']=50;p['gpu_validation']['manifest']=bind(Path(p['gpu_validation']['manifest']['path']),gm)
 else:
  raw=p['gpu_validation']['raw_artifacts'];key={'terminal_failed':'slurm_terminal','terminal_restarted':'slurm_terminal','cpu_subset':'cpu_gate','cpu_source':'cpu_source','cleanup':'cleanup','guard_summary':'guard_summary','watcher_change':'resource_watcher','guardhelper_change':'guard_helper','raw_training':'train_log'}[case];path=Path(raw[key]['path'])
  if case=='terminal_failed':raw[key]=bind(path,'JobId=10 JobState=FAILED ExitCode=1:0 DerivedExitCode=1:0 Restarts=0 NodeList=h100-5',False)
  elif case=='terminal_restarted':raw[key]=bind(path,'JobId=10 JobState=COMPLETED ExitCode=0:0 DerivedExitCode=0:0 Restarts=1 NodeList=h100-5',False)
  elif case=='cpu_subset':v=r.read(path);v['passed']=1;raw[key]=bind(path,v)
  elif case=='cpu_source':raw[key]=bind(path,'old',False)
  elif case=='cleanup':raw[key]=bind(path,dict(remaining=[123]))
  elif case=='guard_summary':v=r.read(path);v['samples']=999;raw[key]=bind(path,v)
  elif case in ('watcher_change','guardhelper_change'):raw[key]=bind(path,'changed',False)
  elif case=='raw_training':raw[key]=bind(path,path.read_text()+'BAPO_TRAIN_METRICS_JSON={}\n',False)
 # Re-bind deliberately tampered GPU JSON to prove semantic validation, not
 # merely refusal of a stale file hash. The raw train log stays original.
 if case not in ('dummy_json','non_json','missing_rank','duplicate_json','json_nan'):
  p['gpu_witnesses']=[bind(run/'recovery-witness'/f'rank_{i}.json',w)for i,w in enumerate(ranks)]+[bind(run/'recovery-witness/driver.json',d)]
 with pytest.raises((ValueError,KeyError,TypeError,json.JSONDecodeError)):r.validate_gpu_proof(p,m,c,root)


def test_readonly_hardware_self_review_same_job_schema(tmp_path,monkeypatch):
 m,p,c,root,run,*_=setup_proof(tmp_path,monkeypatch)
 m.update(target_job_id=10,target_run_root=str(run));p.update(status='PASS_INDEPENDENT_E1_HARDWARE_REVIEW',job_id=10,run_root=str(run),root=str(root),source_commit='new')
 r.validate_gpu_proof(p,m,c,root,hardware_self_review=True)
 # The default training path must still reject the very same accepted self-review.
 with pytest.raises(ValueError,match='Separate actual GPU'):r.validate_gpu_proof(p,m,c,root)

@pytest.mark.parametrize('case',['training_status','wrong_job','wrong_run','wrong_source','wrong_root','nonbool'])
def test_readonly_hardware_mode_cannot_weaken_training_or_identity(tmp_path,monkeypatch,case):
 m,p,c,root,run,*_=setup_proof(tmp_path,monkeypatch);m.update(target_job_id=10,target_run_root=str(run));p.update(status='PASS_INDEPENDENT_E1_HARDWARE_REVIEW',job_id=10,run_root=str(run),root=str(root),source_commit='new');mode=True
 if case=='training_status':p['status']='PASS_INDEPENDENT_BAPO_RESUME25_REVIEW'
 elif case=='wrong_job':p['job_id']=11
 elif case=='wrong_run':p['run_root']=str(root/'runs/foreign')
 elif case=='wrong_source':p['source_commit']='old'
 elif case=='wrong_root':p['root']='/tmp/foreign'
 else:mode=1
 with pytest.raises(ValueError):r.validate_gpu_proof(p,m,c,root,hardware_self_review=mode)

@pytest.mark.parametrize('case',['absent','gen_changed','state_changed','wrong_mesh','wrong_dtype','wrong_pid','wrong_step','missing_marker','fake_directory','mesh_bool','mesh_rank_bool'])
def test_production_v3_roundtrip_proof_required(tmp_path,monkeypatch,case):
 m,p,c,root,run,ranks,d,bind=setup_proof(tmp_path,monkeypatch);v3=p['gpu_validation']['native_v3_roundtrip'];entry=v3[0];vr=r.read(entry['report']['path'])
 if case=='absent':p['gpu_validation'].pop('native_v3_roundtrip')
 elif case=='gen_changed':vr['generation_rng_after_digest']='a'*64
 elif case=='state_changed':vr['reloaded_state_digest']='a'*64
 elif case=='wrong_mesh':vr['generation_rng_topology']['device_mesh']['shape']=[1,8]
 elif case=='wrong_dtype':vr['generation_rng_topology']['gen_random_states_dtype']='torch.float32'
 elif case=='mesh_bool':vr['generation_rng_topology']['device_mesh']['shape']=[8,True]
 elif case=='mesh_rank_bool':vr['generation_rng_topology']['device_mesh']['ranks'][0]=[False]
 elif case=='wrong_pid':vr['pid']=9999
 elif case=='wrong_step':vr['step']=26
 elif case=='fake_directory':entry['state']=bind(run/'fake/rank_0.pt','Synthetic CPU bytes',False);vr['path']=entry['state']['path'];vr['sha256']=entry['state']['sha256']
 entry['report']=bind(Path(entry['report']['path']),vr)
 raw=p['gpu_validation']['raw_artifacts'];text=Path(raw['train_log']['path']).read_text();lines=[line for line in text.splitlines()if not ('BAPO_RECOVERY_V3_RANK_JSON='in line and r.decoder().raw_decode(line.split('BAPO_RECOVERY_V3_RANK_JSON=',1)[1])[0]['rank']==0)]
 if case!='missing_marker':lines.append('BAPO_RECOVERY_V3_RANK_JSON='+json.dumps(vr))
 raw['train_log']=bind(run/'train.log','\n'.join(lines)+'\n',False)
 with pytest.raises(ValueError):r.validate_gpu_proof(p,m,c,root)

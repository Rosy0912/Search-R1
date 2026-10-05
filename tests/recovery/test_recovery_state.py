import copy,json,os
from pathlib import Path
import pytest,torch
from search_r1.baselines import recovery as r
from search_r1.baselines.full_state import state_digest,StatefulSampler,rng_state,restore_rng

def cfg():
 return dict(trainer=dict(total_training_steps=301,e1_gate_enabled=True,acceptance_step=5,resume_full_state=None,resume_verify_only=False),actor_rollout_ref=dict(actor=dict(optim=dict(num_warmup_steps=114))),algorithm=dict(adv_estimator='grpo',kl_ctrl=dict(type='fixed')),baselines=dict(method='bapo',algorithm_source_commit=r.OLD_SOURCE))
def native():
 return dict(version=2,model_state_dict_type='SHARDED_STATE_DICT',rank=0,world_size=8,step=25,model={'p':torch.ones(2)},optimizer=dict(state={0:dict(step=torch.tensor(500.),exp_avg=torch.ones(2),exp_avg_sq=torch.ones(2))},param_groups=[dict(lr=1e-6*25/114)]),scheduler=dict(last_epoch=25,_last_lr=[1e-6*25/114]),rng=rng_state())
def report(p):return dict(state_digest=state_digest(p),optimizer_state_entries=1)
def test_native_actual_content():
 p=native();r.validate_native(p,report(p),0,8)
@pytest.mark.parametrize('field,value',[('version',3),('rank',1),('world_size',7),('step',50),('model_state_dict_type','FULL_STATE_DICT')])
def test_native_identity_reject(field,value):
 p=native();p[field]=value
 with pytest.raises(ValueError):r.validate_native(p,report(p),0,8)
@pytest.mark.parametrize('key,value',[('step',torch.tensor(499.)),('step',torch.tensor(500.5)),('step',torch.tensor([500.])),('step',torch.tensor(float('nan'))),('step',torch.tensor(500,dtype=torch.int64)),('exp_avg',torch.tensor([float('nan')])),('exp_avg_sq',torch.tensor([float('inf')]))])
def test_native_adam_reject(key,value):
 p=native();p['optimizer']['state'][0][key]=value
 with pytest.raises(ValueError):r.validate_native(p,report(p),0,8)
def test_native_digest_and_lr_reject():
 p=native();rep=report(p);p['model']['p'][0]=2
 with pytest.raises(ValueError):r.validate_native(p,rep,0,8)
 p=native();p['optimizer']['param_groups'][0]['lr']=1e-6
 with pytest.raises(ValueError):r.validate_native(p,report(p),0,8)
def test_config_operation_only():
 a=cfg();b=copy.deepcopy(a);b['trainer'].update(resume_full_state='/new',resume_permission='/newpermit',resume_witness_dir='/witness',resume_verify_only=True);b['baselines']['algorithm_source_commit']='new'
 assert r.config_identity(a)==r.config_identity(b)
 b['actor_rollout_ref']['actor']['optim']['num_warmup_steps']=57
 assert r.config_identity(a)!=r.config_identity(b)
def test_sampler_next_batch_and_iterator_rng():
 sampler=StatefulSampler(range(14000),42);iterator=iter(sampler)
 for _ in range(12800):next(iterator)
 state=sampler.state_dict();expected=state['order'][12800:13312];saved=rng_state()
 new=StatefulSampler(range(14000),42);new.load_state_dict(state)
 loader=torch.utils.data.DataLoader(range(14000),batch_size=512,sampler=new,num_workers=0)
 it=iter(loader);restore_rng(saved)
 assert state_digest(rng_state())==state_digest(saved)
 assert next(it).tolist()==expected and new.cursor==13312
 assert state_digest(rng_state())==state_digest(saved)
def fixture_manifest(tmp,monkeypatch):
 old=tmp/'old';old.mkdir();run=old/'runs/original';run.mkdir(parents=True);newrun=tmp/'new/runs/job';newrun.mkdir(parents=True)
 monkeypatch.setattr(r,'OLD_ROOT',old);monkeypatch.setattr(r,'OLD_RUN',run);monkeypatch.setenv('SLURM_JOB_ID','9');monkeypatch.setenv('BAPO_RUN_ID',str(newrun))
 def bind(path,value):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return dict(path=str(path),bytes=path.stat().st_size,sha256=r.digest(path))
 c=cfg();m=dict(version=1,old_source_commit=r.OLD_SOURCE,old_run_root=str(run),step=25,next_step=26,actual_outer_updates=25,world_size=8,target_source_commit='new',target_run_root=str(newrun),target_job_id=9,ranks=[])
 for rank in range(8):
  pt=bind(run/'checkpoints/actor/global_step_25/training_state'/f'rank_{rank}.pt',{'dummy':rank});rp=bind(Path(pt['path']).with_suffix('.json'),dict(rank=rank,world_size=8,step=25,path=pt['path'],sha256=pt['sha256'],status='FULL_STATE_SERIALIZED_OK',scheduler_last_epoch=25,optimizer_state_entries=37,model_dtensor_entries=435,pid=100+rank));m['ranks'].append(dict(rank=rank,state=pt,report=rp))
 model=old/'model';base=bind(model/'model.safetensors',{'weight':'fixed'});mm=bind(model/'VERIFIED-MODEL-MANIFEST.json',dict(repo='Qwen/Qwen2.5-3B-Instruct',revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',files=[dict(file='model.safetensors',bytes=base['bytes'],sha256=base['sha256'])]))
 data=[bind(old/'data'/n,{'data':n})for n in ['train','dev','stage','final','manifest.json']]
 c['data']=dict(train_files=data[0]['path'],val_files=data[1]['path'],final_val_files=data[3]['path'])
 c['baselines'].update(bapo=dict(stage_validation_files=data[2]['path']),holdout=dict(manifest_path=data[4]['path'],manifest_sha256=data[4]['sha256']))
 c['actor_rollout_ref']['model']=dict(path=str(model))
 m['resolved_config']=bind(run/'resolved-config.yaml',c);m['driver']=bind(run/'checkpoints/full_state/global_step_25/driver.pt',{'dummy':1});m['driver_report']=bind(run/'checkpoints/full_state/global_step_25/driver-reload.json',dict(path=m['driver']['path'],sha256=m['driver']['sha256'],status='DRIVER_SERIALIZED_OK',step=25,sampler_cursor=12800))
 m['algorithm']=bind(run/'checkpoints/algorithm/global_step_25/state.json',dict(method='bapo',global_step=25,actual_outer_updates=25,algorithm_source_commit=r.OLD_SOURCE))
 m['shared_inputs']=data+[base,mm];m['hf_files']=[bind(run/'checkpoints/actor/global_step_25'/n,{'file':n})for n in sorted(r.HF_FILES)]
 m['historical_first5_receipt']=bind(run/'oldrelease.json',dict(status='PASS_INDEPENDENT_E1_STEP5_REVIEW',run_root=str(run),source_commit=r.OLD_SOURCE,actual_outer_updates=5))
 prefix=newrun/'prefix.log';prefix.write_text('\n'.join('BAPO_TRAIN_METRICS_JSON='+json.dumps(dict(step=i,metrics={'bapo/actual_outer_updates':i}))for i in range(1,26)));m['train_prefix']=dict(path=str(prefix),bytes=prefix.stat().st_size,sha256=r.digest(prefix));(run/'train.log').write_bytes(prefix.read_bytes());m['train_prefix_live']=dict(path=str(run/'train.log'),bytes=prefix.stat().st_size,sha256=r.digest(prefix))
 mp=newrun/'manifest.json';mp.write_text(json.dumps(m));review=bind(newrun/'review.json',{'independent':True});gpu=bind(newrun/'gpu.json',{'actual':True});permit=dict(status='PASS_INDEPENDENT_BAPO_RESUME25_GPU_VERIFY_PERMISSION',manifest_sha256=r.digest(mp),target_source_commit='new',target_run_root=str(newrun),target_job_id=9,independent_review=review,code_review_evidence=[gpu],sampling_continuity='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT');pp=newrun/'permit.json';pp.write_text(json.dumps(permit));c['trainer']['resume_verify_only']=True;return m,mp,c,newrun,permit,pp

def test_manifest_actual_files_and_verify_only(tmp_path,monkeypatch):
 m,mp,c,run,permit,pp=fixture_manifest(tmp_path,monkeypatch);assert r.validate_manifest(mp,c,'new',run,pp)==m
 c['trainer']['resume_verify_only']=False
 with pytest.raises(ValueError):r.validate_manifest(mp,c,'new',run,pp)
 permit['status']='PASS_INDEPENDENT_BAPO_RESUME25_REVIEW';permit['gpu_witnesses']=permit.pop('code_review_evidence');pp.write_text(json.dumps(permit))
 with pytest.raises(ValueError):r.validate_manifest(mp,c,'new',run,pp)
@pytest.mark.parametrize('kind',['rank_duplicate','step50','config57','wrong_job','missing_gpu','old_receipt','changed_bytes','sampling_claim'])
def test_manifest_counterexamples(tmp_path,monkeypatch,kind):
 m,mp,c,run,permit,pp=fixture_manifest(tmp_path,monkeypatch)
 if kind=='rank_duplicate':m['ranks'][1]=m['ranks'][0]
 if kind=='step50':m['step']=50
 if kind=='config57':c['actor_rollout_ref']['actor']['optim']['num_warmup_steps']=57
 if kind=='wrong_job':monkeypatch.setenv('SLURM_JOB_ID','10')
 if kind=='missing_gpu':permit['code_review_evidence']=[]
 if kind=='old_receipt':permit['status']='PASS_INDEPENDENT_E1_STEP5_REVIEW'
 if kind=='changed_bytes':Path(m['ranks'][0]['state']['path']).write_text('tampered')
 if kind=='sampling_claim':permit['sampling_continuity']='BITWISE_EXACT'
 mp.write_text(json.dumps(m));permit['manifest_sha256']=r.digest(mp);pp.write_text(json.dumps(permit))
 with pytest.raises(ValueError):r.validate_manifest(mp,c,'new',run,pp)

def test_real_fit_verify_only_restores_driver_and_never_trains(tmp_path,monkeypatch):
 from types import SimpleNamespace
 from omegaconf import OmegaConf
 from verl.trainer.ppo.ray_trainer import RayPPOTrainer
 from search_r1.baselines.bapo import BoundaryStage
 from search_r1.baselines.full_state import rng_state
 old=tmp_path/'old';old.mkdir();monkeypatch.setattr(r,'OLD_RUN',old)
 sampler=StatefulSampler(range(14000),None);it=iter(sampler)
 for _ in range(12800):next(it)
 stage=BoundaryStage();stage.load_state_dict(dict(best=.28125,stalled=14,early=False,patience=5))
 payload=dict(version=1,driver_pid=42,global_step=25,next_global_step=26,actual_outer_updates=25,epoch=0,sampler=sampler.state_dict(),rng=rng_state(),bapo=stage.state_dict())
 def binding(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=r.digest(path))
 pt=old/'driver.pt';torch.save(payload,pt);report_path=old/'driver-report.json';report_path.write_text(json.dumps(dict(state_digest=state_digest(payload),pid=42)));algo=old/'algorithm.json';algo.write_text(json.dumps(dict(bapo=stage.state_dict())))
 new=tmp_path/'new';new.mkdir();wdir=new/'witness'
 m=dict(driver=binding(pt),driver_report=binding(report_path),algorithm=binding(algo),target_source_commit='new',target_run_root=str(new),target_job_id=9)
 t=RayPPOTrainer.__new__(RayPPOTrainer);t.resume_manifest=m;t.resume_witness_directory=wdir;t.resume_rank_reports=[]
 t.config=OmegaConf.create(dict(trainer=dict(resume_verify_only=True,total_epochs=1,n_gpus_per_node=8,nnodes=1),max_turns=4,algorithm=dict(no_think_rl=False),retriever=dict(url='http://127.0.0.1:8000',topk=3),data=dict(max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500)))
 t.logger=SimpleNamespace(log=lambda **kwargs:None);t.tokenizer=SimpleNamespace(pad_token_id=0);t.actor_rollout_wg=None;t.val_reward_fn=object()
 t._validate=lambda:(_ for _ in ()).throw(AssertionError('initial validation forbidden'))
 t.e1_sampler=StatefulSampler(range(14000),None);t.train_dataloader=torch.utils.data.DataLoader(range(14000),batch_size=512,sampler=t.e1_sampler,num_workers=0)
 t.baseline_controller=SimpleNamespace(stage=BoundaryStage(),actual_outer_updates=0)
 monkeypatch.setattr(r,'process_identity',lambda:dict(pid=os.getpid(),start_ticks=1,state='S',actual_env={}))
 witness=t.fit()
 assert witness['status']=='EXACT_CROSS_JOB_DRIVER_RELOAD_OK'and witness['verify_only']is True
 assert t.global_steps==26 and t.e1_sampler.cursor==12800 and t.baseline_controller.actual_outer_updates==25
 assert state_digest(rng_state())==state_digest(payload['rng'])and t.baseline_controller.stage.state_dict()==payload['bapo']
 assert (wdir/'driver.json').is_file()

@pytest.mark.parametrize('case',['empty_groups','empty_rates','mismatched_lengths','nan_lr','bool_lr'])
def test_native_nonempty_exact_scheduler_groups(case):
 p=native()
 if case=='empty_groups':p['optimizer']['param_groups']=[]
 if case=='empty_rates':p['scheduler']['_last_lr']=[]
 if case=='mismatched_lengths':p['scheduler']['_last_lr']*=2
 if case=='nan_lr':p['optimizer']['param_groups'][0]['lr']=float('nan')
 if case=='bool_lr':p['optimizer']['param_groups'][0]['lr']=True
 with pytest.raises(ValueError):r.validate_native(p,report(p),0,8)

def test_actual_trainer_verify_only_native_v3_save_call(tmp_path):
 from types import SimpleNamespace
 from verl.trainer.ppo.ray_trainer import RayPPOTrainer
 reports=[dict(status='EXACT_FULL_STATE_RELOAD_OK',native_state_version=3,roundtrip=True,rank=i,state_digest='a'*64,reloaded_state_digest='a'*64,generation_rng_before_digest='b'*64,generation_rng_after_digest='b'*64)for i in range(8)]
 calls=[]
 def save(path,step,roundtrip):
  calls.append((path,step,roundtrip));return reports
 t=RayPPOTrainer.__new__(RayPPOTrainer);t.resume_manifest=dict(target_run_root=str(tmp_path));t.config=SimpleNamespace(trainer=SimpleNamespace(resume_verify_only=True));t.actor_rollout_wg=SimpleNamespace(save_training_state=save)
 assert t._verify_recovery_native_v3()==reports and calls==[(str(tmp_path/'recovery-native-v3-roundtrip'),25,True)]
 # Formal continuation cannot save/reload even via direct invocation.
 t.config.trainer.resume_verify_only=False
 with pytest.raises(ValueError):t._verify_recovery_native_v3()
 assert len(calls)==1
 t.config.trainer.resume_verify_only=True;(tmp_path/'recovery-native-v3-roundtrip').mkdir()
 with pytest.raises(ValueError):t._verify_recovery_native_v3()
 assert len(calls)==1
 # Actual init path contains one save helper call dominated by verify-only.
 import ast
 source=Path(__import__('inspect').getsourcefile(RayPPOTrainer)).read_text();tree=ast.parse(source)
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='RayPPOTrainer');init=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='init_workers')
 guards=[x for x in ast.walk(init)if isinstance(x,ast.If)and ast.unparse(x.test)=='self.config.trainer.resume_verify_only']
 assert len(guards)==1 and [ast.unparse(x)for x in guards[0].body]==['self._verify_recovery_native_v3()']

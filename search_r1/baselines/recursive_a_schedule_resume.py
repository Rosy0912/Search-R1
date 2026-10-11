"""Pinned recursive-A same-world continuation for the human sparse schedule change."""
import hashlib,json,os,socket
from pathlib import Path
from .training_state import (atomic_save,cpu_copy,digest,driver_snapshot,prepare_cold_fsdp_layout,
 restore_driver,restore_worker,verified_state_file,worker_snapshot)
PROTOCOL='recursive_a_sparse_schedule_continuation_v1'
HANDOFF='search_only_and_no_repeated_candidate_proof_v1'

def enabled(environ=None):
 e=os.environ if environ is None else environ
 return 'RECURSIVE_A_CONTINUATION_BINDING' in e

def require(ok,msg):
 if not ok: raise ValueError('Recursive A continuation: '+msg)

def binding(environ=None):
 e=os.environ if environ is None else environ
 raw=Path(e['RECURSIVE_A_CONTINUATION_BINDING']).read_bytes()
 require(hashlib.sha256(raw).hexdigest()==e['RECURSIVE_A_CONTINUATION_SHA256'],'binding bytes changed')
 b=json.loads(raw)
 require(b['protocol']==PROTOCOL and b['parent_completed_step'] in (4,5),'only pinned completed4/5 boundary')
 expected={4:('5045430e67860102c304ef8098e4862ddcdf6e3e','622e3c083cd8e2db5f6ef3ba126cd6d4f71babd85c97feb46f839a20fe593d03'),
           5:('43a209a5ba3f7f1211c18d03c9362d85b3f5cd1b','c5be8a8e679684fbb8bb5ca360d99188a1b88d5a7ab4eff0b08fa6f3bc029553')}
 if b.get('handoff_contract')==HANDOFF and b['parent_completed_step']==5:
  require(b['parent_commit']=='d5542fe5fad12abd672926e67fbfd9df53bb7187' and
   b['parent_config_sha256']=='c5be8a8e679684fbb8bb5ca360d99188a1b88d5a7ab4eff0b08fa6f3bc029553' and
   b['parent_identity']['job_id']=='hkust-direct-3bdfeb78c2a7480980f190a5c1c96bdf','wrong live5 optimization handoff')
  require(type(b.get('parent_optimizer_calls')) is int and 80<=b['parent_optimizer_calls']<=100,'wrong observed5 Adam count')
 else:
  require((b['parent_commit'],b['parent_config_sha256'])==expected[b['parent_completed_step']],'wrong parent chain')
 require(socket.gethostname().split('.')[0]=='xielab1' and os.getuid()==1005,'wrong host/UID')
 return b

def _bind():
 global PARENT_MANIFEST,PARENT_MANIFEST_SHA256,PARENT_OPTIMIZER_CALLS,PARENT_COMPLETED_STEP
 b=binding();PARENT_MANIFEST=Path(b['parent_manifest']);PARENT_MANIFEST_SHA256=b['parent_manifest_sha256']
 PARENT_COMPLETED_STEP=b['parent_completed_step'];PARENT_OPTIMIZER_CALLS=(b['parent_optimizer_calls']
  if b.get('handoff_contract')==HANDOFF and PARENT_COMPLETED_STEP==5 else 20*PARENT_COMPLETED_STEP)
 return b

def validate_entry(config=None,environ=None,initialized=False):
 if not enabled(environ):return False
 b=binding(environ)
 if config is not None:
  from omegaconf import OmegaConf
  old=OmegaConf.to_container(OmegaConf.load(b['parent_config']),resolve=True)
  new=OmegaConf.to_container(config,resolve=True)
  require(hashlib.sha256(Path(b['parent_config']).read_bytes()).hexdigest()==b['parent_config_sha256'],'parent config changed')
  expected=json.loads(json.dumps(old))
  expected['trainer']['evaluation_steps']=[5,50,100,150,200]
  expected['trainer']['checkpoint_save_steps']=[5,50,100,150,200]
  if b.get('handoff_contract')==HANDOFF:
   expected['actor_rollout_ref']['rollout']['tree_action_tag_stop']=True
   expected['actor_rollout_ref']['rollout']['tree_action_stop_mode']='search_only'
   if b.get('eight_gpu_retrieval_gate'):
    gate=Path(b['eight_gpu_retrieval_gate']);require(hashlib.sha256(gate.read_bytes()).hexdigest()==b['eight_gpu_retrieval_gate_sha256'],'eight GPU retrieval gate changed')
    g=json.loads(gate.read_text());require(g.get('accepted') is True and g.get('scope')=='eight_gpu_full_scan_and_http_oracle_equality','eight GPU retrieval not admitted')
    require(b['child_retrieval_url']=='http://10.120.17.101:18006/retrieve','wrong new retrieval endpoint')
    expected['retriever']['url']=b['child_retrieval_url']
  if initialized:
   # Native _create_dataloader injects these scheduler counters before fit.
   expected['actor_rollout_ref']['actor']['optim']['total_training_steps']=200
   expected['critic']['optim']['total_training_steps']=201
  require(new==expected,'change exceeds explicit schedule/search-only performance fields')
 return True

def parent_manifest():
 b=_bind();raw=PARENT_MANIFEST.read_bytes()
 require(hashlib.sha256(raw).hexdigest()==PARENT_MANIFEST_SHA256,'manifest changed')
 m=json.loads(raw);i=m['identity']
 require(m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and i==b['parent_identity'],'parent identity/status changed')
 require(i['completed_step']==PARENT_COMPLETED_STEP and i['actual_update_limit']==200,'parent counters changed')
 g=i['guard_continue_accounting'];counts={str(r):PARENT_OPTIMIZER_CALLS for r in range(8)}
 require(g['completed_optimizer_calls_per_rank']==counts and g['total_optimizer_calls_per_rank']==counts
 and g['completed_outer_updates']==PARENT_COMPLETED_STEP and g['training_attempts']==PARENT_COMPLETED_STEP and g['refused_attempts']==0
 and not g['partial_evidence'],'parent incomplete writes')
 ws=m['worker_states'];require(len(ws)==8 and {r['identity']['rank'] for r in ws}==set(range(8)),'rank coverage')
 for r in ws:
  require(r['actual_total_optimizer_calls']==PARENT_OPTIMIZER_CALLS and r['observed_adam_steps']==[PARENT_OPTIMIZER_CALLS]
   and r['scheduler_last_epoch']==PARENT_COMPLETED_STEP and r['full_worker_state_finite'] is True,'rank counters/finite')
 require(m['driver']['next_step']==PARENT_COMPLETED_STEP+1,'driver cursor boundary')
 return m

def state_record(record,filename):
 r=dict(record);p=PARENT_MANIFEST.parent/filename
 require(Path(r['path'])==p and not p.is_symlink(),'state path changed')
 if 'bytes' in r:require(p.stat().st_size==r['bytes'],'state bytes changed')
 return r

def model_section(state):
    return {key: state[key] for key in ('local_shards', 'parameters', 'buffers')}


def load_worker(worker):
    _bind()
    require(worker._is_actor and worker.config.actor.get('training_state_enabled', False),
            'native actor state is disabled')
    require(worker.world_size == 8 and not worker.actor_optimizer.state
            and worker.actor_lr_scheduler.last_epoch == 0
            and worker.actor._fresh_total_committed_optimizer_calls == 0,
            'cold worker is not untouched official initialization')
    manifest = parent_manifest()
    record = next(r for r in manifest['worker_states'] if r['identity']['rank'] == worker.rank)
    record = state_record(record, f'rank-{worker.rank:02d}.pt')
    saved = verified_state_file(record)
    logical = digest(saved)
    require(logical == record['logical_state_sha256'] and saved['identity'] == record['identity'],
            'actual serialized worker state or identity changed')
    require(saved['optimizer']['state']
            and all(int(value['step']) == PARENT_OPTIMIZER_CALLS
                    for value in saved['optimizer']['state'].values())
            and saved['scheduler']['last_epoch'] == PARENT_COMPLETED_STEP,
            'actual serialized Adam/scheduler counters differ')
    model_hash = digest(model_section(saved))
    model_bytes = sum(t.numel()*t.element_size() for values in model_section(saved).values()
                      for t in values.values())
    prepare_cold_fsdp_layout(worker.actor_module_fsdp, worker._is_offload_param, worker._is_offload_grad)
    restore_worker(worker.actor_module_fsdp, worker.actor_optimizer, worker.actor_lr_scheduler,
                   saved, record['identity'], cold=True)
    del saved
    worker.actor._fresh_total_committed_optimizer_calls = PARENT_OPTIMIZER_CALLS
    actual = worker_snapshot(worker.actor_module_fsdp, worker.actor_optimizer,
                             worker.actor_lr_scheduler, record['identity'], cuda=True)
    live_logical = digest(actual)
    live_model_hash = digest(model_section(actual))
    require(live_logical == logical and live_model_hash == model_hash,
            'actual cold-loaded model/Adam/scheduler/RNG differs from saved parent')
    del actual
    adam_steps = sorted({int(value['step']) for value in worker.actor_optimizer.state.values()})
    require(adam_steps == [PARENT_OPTIMIZER_CALLS]
            and worker.actor_lr_scheduler.last_epoch == PARENT_COMPLETED_STEP,
            'actual cold-loaded Adam/scheduler counters differ')
    return dict(status='COLD_CURRENT_CHAIN_WORKER_STATE_MIGRATED', protocol=PROTOCOL,
        rank=worker.rank, world_size=worker.world_size, parent_identity=record['identity'],
        parent_manifest_sha256=PARENT_MANIFEST_SHA256, state_path=record['path'],
        state_file_sha256=record['sha256'], state_file_bytes=Path(record['path']).stat().st_size,
        actual_file_sha256_verified=True, serialized_logical_state_sha256=logical,
        actual_loaded_logical_state_sha256=live_logical,
        loaded_model_section_sha256=live_model_hash, loaded_model_tensor_bytes=model_bytes,
        model_tensor_byte_scope='Stored local_shards+parameters+buffers; sections may duplicate storage, not unique allocation bytes',
        actual_cold_native_state_equal=True, adam_steps=adam_steps,
        scheduler_step=worker.actor_lr_scheduler.last_epoch,
        actual_total_optimizer_calls=PARENT_OPTIMIZER_CALLS,
        child_code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        child_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        executor_equivalence=False, automatic_resume_supported=False)


def migrate_trainer(trainer):
 require(validate_entry(trainer.config,initialized=True),'missing scope')
 b=_bind();m=parent_manifest();workers=trainer.actor_rollout_wg.load_current_chain_training_state()
 require(len(workers)==8 and {r['rank'] for r in workers}==set(range(8)) and
 all(r['actual_cold_native_state_equal'] and r['adam_steps']==[PARENT_OPTIMIZER_CALLS] and r['scheduler_step']==PARENT_COMPLETED_STEP for r in workers),'cold worker state differs')
 r=state_record(m['driver'],'driver.pt');saved=verified_state_file(r)
 require(saved['identity']==m['identity'] and digest(saved)==r['logical_state_sha256'],'driver bytes/identity')
 require(saved['completed_step']==PARENT_COMPLETED_STEP and saved['actual_outer_updates']==PARENT_COMPLETED_STEP and saved['next_step']==PARENT_COMPLETED_STEP+1
 and saved['sampler']['cursor']==512*PARENT_COMPLETED_STEP and saved['sampler']['epoch']==0,'driver counters/cursor')
 # Validate the old checkpoint with its exact old accounting schema first.
 trainer._guard_round_contract_active=(m['identity']['guard_continue_accounting'].get(
  'round_progress_contract')=='consumed_sampling_rounds_separate_adam_v1')
 restore_driver(trainer,saved,m['identity'])
 require(digest(driver_snapshot(trainer,m['identity']))==r['logical_state_sha256'],'cold driver/RNG differs')
 from .tree_guard_continue import round_progress_enabled
 trainer._guard_round_contract_active=round_progress_enabled(trainer.config.actor_rollout_ref.actor)
 trainer.zero_write_sampling_rounds=getattr(trainer,'zero_write_sampling_rounds',0)
 trainer.normal_guard_rounds=getattr(trainer,'normal_guard_rounds',0)
 provenance=dict(protocol=PROTOCOL,parent_identity=m['identity'],parent_manifest_path=str(PARENT_MANIFEST),
 parent_manifest_sha256=PARENT_MANIFEST_SHA256,parent_completed_step=PARENT_COMPLETED_STEP,parent_cursor=512*PARENT_COMPLETED_STEP,parent_epoch=0,
 new_batch_size=512,child_code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
 child_config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],executor_equivalence=False,
 scope='same-world state restored; vLLM restart not claimed bitwise; sparse recording and unused per-action HF scoring removal bound by child source commit')
 trainer.recovery_provenance=provenance;trainer.acceptance_released=True
 receipt=dict(status='COLD_RECURSIVE_A_STATE_MIGRATED',protocol=PROTOCOL,workers=workers,
 driver_actual_state_equal=True,next_outer=PARENT_COMPLETED_STEP+1,provenance=provenance)
 atomic_save(receipt,Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'sparse-schedule-migrated.json',json_format=True)
 print('RECURSIVE_A_STATE_MIGRATED='+str(PARENT_COMPLETED_STEP),flush=True)
 return receipt

"""Pinned same-world H100 recovery from the actual post-evaluation100 boundary.
Worker/driver restoration below is unchanged from native schedule continuation.
"""
import hashlib,json,os,socket
from pathlib import Path
from .training_state import (atomic_save,cpu_copy,digest,driver_snapshot,prepare_cold_fsdp_layout,
 restore_driver,restore_worker,verified_state_file,worker_snapshot)
PROTOCOL='recursive_a_sparse_schedule_continuation_v1'
HANDOFF='h100_saved100_after_allocation_terminal_v1'
def enabled(environ=None):
 e=os.environ if environ is None else environ
 return 'RECURSIVE_A_CONTINUATION_BINDING' in e
def require(ok,msg):
 if not ok:raise ValueError('H100 saved100 recovery: '+msg)
def binding(environ=None):
 e=os.environ if environ is None else environ
 raw=Path(e['RECURSIVE_A_CONTINUATION_BINDING']).read_bytes()
 require(hashlib.sha256(raw).hexdigest()==e['RECURSIVE_A_CONTINUATION_SHA256'],'binding changed')
 b=json.loads(raw)
 require(b['protocol']==PROTOCOL and b['handoff_contract']==HANDOFF,'wrong recovery scope')
 require(b['parent_completed_step']==100 and b['parent_optimizer_calls']==2000,'wrong completed boundary')
 require(b['parent_manifest']=='/dataset/shaoshitong/recursive-bran-a-h100-20261012/checkpoints/training_state/global_step_100/post-evaluation/manifest.json' and b['parent_manifest_sha256']=='bb0599ced9db88eaa5dfec21ead76d214a8a1f5b4a64aa7baa4b85a41ceebce2','not actual post-evaluation100')
 require(b['parent_config_sha256']=='653193a4eb06bd28667f78ac14a66a2d1c97200ca8199b8521dcc2d3de35b319','parent config')
 require(socket.gethostname().split('.')[0]=='h100-11' and os.getuid()==0,'wrong node or UID')
 return b
def _bind():
 global PARENT_MANIFEST,PARENT_MANIFEST_SHA256,PARENT_OPTIMIZER_CALLS,PARENT_COMPLETED_STEP
 b=binding();PARENT_MANIFEST=Path(b['parent_manifest']);PARENT_MANIFEST_SHA256=b['parent_manifest_sha256']
 PARENT_COMPLETED_STEP=b['parent_completed_step'];PARENT_OPTIMIZER_CALLS=b['parent_optimizer_calls']
 return b
def validate_entry(config=None,environ=None,initialized=False):
 if not enabled(environ):return False
 b=binding(environ)
 if config is not None:
  from omegaconf import OmegaConf
  require(hashlib.sha256(Path(b['parent_config']).read_bytes()).hexdigest()==b['parent_config_sha256'],'parent config changed')
  expected=OmegaConf.to_container(OmegaConf.load(b['parent_config']),resolve=True)
  expected['trainer']['default_local_dir']=b['child_checkpoint_dir']
  expected['retriever']['control_token_file']=b['child_control_token_file']
  if initialized:
   expected['actor_rollout_ref']['actor']['optim']['total_training_steps']=200
   expected['critic']['optim']['total_training_steps']=201
  require(OmegaConf.to_container(config,resolve=True)==expected,'scientific configuration changed')
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

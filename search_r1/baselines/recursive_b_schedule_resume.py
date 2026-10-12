"""Pinned recursive-B same-world continuation for the human sparse schedule change."""
import hashlib,json,os,socket
from pathlib import Path
from .training_state import (atomic_save,cpu_copy,digest,driver_snapshot,prepare_cold_fsdp_layout,
 restore_driver,restore_worker,verified_state_file,worker_snapshot)
PROTOCOL='recursive_b_sparse_schedule_continuation_v1'

def enabled(environ=None):
 e=os.environ if environ is None else environ
 return 'RECURSIVE_B_CONTINUATION_BINDING' in e

def require(ok,msg):
 if not ok: raise ValueError('Recursive B continuation: '+msg)

def binding(environ=None):
 e=os.environ if environ is None else environ
 raw=Path(e['RECURSIVE_B_CONTINUATION_BINDING']).read_bytes()
 require(hashlib.sha256(raw).hexdigest()==e['RECURSIVE_B_CONTINUATION_SHA256'],'binding bytes changed')
 b=json.loads(raw)
 require(b['protocol']==PROTOCOL and b['parent_completed_step'] in (3,4,5),'only explicit completed3/4/5 boundary')
 require((b['parent_commit'],b['parent_config_sha256'])==('3da904591125f117b1eae1ba09546ebe793cb2d4','95d9f8f2195a35f189ff06af8e449d419040ae96c4cde4fc50f9d7062db414f7'),'wrong parent chain')
 require(b['parent_run_id']=='ef09290e69454f20b6a9594ad8e485b6','wrong parent run')
 root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
 require(Path(b['parent_manifest'])==root/'checkpoints/training_state'/('global_step_%d'%b['parent_completed_step'])/'manifest.json','wrong explicit checkpoint path')
 require(Path(b['parent_config'])==root/'receipts/configuration-d79908be70f74dd2a0ce38b5ef880f1e/resolved.yaml','wrong parent config path')
 require(socket.gethostname().split('.')[0]=='xie-lab-4' and os.getuid()==1001,'wrong host/UID')
 return b

def _bind():
 global PARENT_MANIFEST,PARENT_MANIFEST_SHA256,PARENT_OPTIMIZER_CALLS,PARENT_COMPLETED_STEP
 b=binding();PARENT_MANIFEST=Path(b['parent_manifest']);PARENT_MANIFEST_SHA256=b['parent_manifest_sha256']
 PARENT_COMPLETED_STEP=b['parent_completed_step'];PARENT_OPTIMIZER_CALLS=20*PARENT_COMPLETED_STEP
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
  if 'child_checkpoint_dir' in b:
   root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
   run=b['child_run_id']
   require(len(run)==32 and all(c in '0123456789abcdef' for c in run),'invalid child run identity')
   require(Path(b['child_checkpoint_dir'])==root/('output-speed-resume3-'+run)/'checkpoints','child artifact path not isolated')
   expected['trainer']['default_local_dir']=b['child_checkpoint_dir']
  expected['actor_rollout_ref']['rollout']['tree_action_tag_stop']=True
  expected['actor_rollout_ref']['rollout']['tree_action_stop_mode']='search_only'
  if initialized:
   # Native _create_dataloader injects these scheduler counters before fit.
   expected['actor_rollout_ref']['actor']['optim']['total_training_steps']=200
   expected['critic']['optim']['total_training_steps']=201
  require(new==expected,'change exceeds explicit schedule fields')
 return True

def parent_manifest():
 b=_bind();raw=PARENT_MANIFEST.read_bytes()
 require(hashlib.sha256(raw).hexdigest()==PARENT_MANIFEST_SHA256,'manifest changed')
 m=json.loads(raw);i=m['identity']
 require(m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and i==b['parent_identity'],'parent identity/status changed')
 require(i['completed_step']==PARENT_COMPLETED_STEP and i['actual_update_limit']==200,'parent counters changed')
 require(i['code_commit']==b['parent_commit'] and i['config_sha256']==b['parent_config_sha256']
  and i['job_id']=='hkust-direct-'+b['parent_run_id']
  and i['data_identity']=='7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
  and i['model_identity']=='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1','parent scientific identity')
 g=i['guard_continue_accounting'];counts={str(r):PARENT_OPTIMIZER_CALLS for r in range(4)}
 require(g['completed_optimizer_calls_per_rank']==counts and g['total_optimizer_calls_per_rank']==counts
 and g['completed_outer_updates']==PARENT_COMPLETED_STEP and g['training_attempts']==PARENT_COMPLETED_STEP and g['refused_attempts']==0
 and not g['partial_evidence'],'parent incomplete writes')
 ws=m['worker_states'];require(len(ws)==4 and {r['identity']['rank'] for r in ws}==set(range(4)),'rank coverage')
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
    require(worker.world_size == 4 and not worker.actor_optimizer.state
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
    sampling = restore_sampling_stream(worker, binding())
    return dict(sampling_stream=sampling,status='COLD_CURRENT_CHAIN_WORKER_STATE_MIGRATED', protocol=PROTOCOL,
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
 require(len(workers)==4 and {r['rank'] for r in workers}==set(range(4)) and
 all(r['actual_cold_native_state_equal'] and r['adam_steps']==[PARENT_OPTIMIZER_CALLS] and r['scheduler_step']==PARENT_COMPLETED_STEP for r in workers),'cold worker state differs')
 r=state_record(m['driver'],'driver.pt');saved=verified_state_file(r)
 require(saved['identity']==m['identity'] and digest(saved)==r['logical_state_sha256'],'driver bytes/identity')
 require(saved['completed_step']==PARENT_COMPLETED_STEP and saved['actual_outer_updates']==PARENT_COMPLETED_STEP and saved['next_step']==PARENT_COMPLETED_STEP+1
 and saved['sampler']['cursor']==512*PARENT_COMPLETED_STEP and saved['sampler']['epoch']==0,'driver counters/cursor')
 restore_driver(trainer,saved,m['identity'])
 require(digest(driver_snapshot(trainer,m['identity']))==r['logical_state_sha256'],'cold driver/RNG differs')
 provenance=dict(protocol=PROTOCOL,parent_identity=m['identity'],parent_manifest_path=str(PARENT_MANIFEST),
 parent_manifest_sha256=PARENT_MANIFEST_SHA256,parent_completed_step=PARENT_COMPLETED_STEP,parent_cursor=512*PARENT_COMPLETED_STEP,parent_epoch=0,
 new_batch_size=512,child_code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
 child_config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],executor_equivalence=False,
 scope='same-world state restored; vLLM restart not claimed bitwise; sparse recording and unused per-action HF scoring removal bound by child source commit')
 trainer.recovery_provenance=provenance;trainer.acceptance_released=True
 receipt=dict(status='COLD_RECURSIVE_B_STATE_MIGRATED',protocol=PROTOCOL,workers=workers,
 driver_actual_state_equal=True,next_outer=PARENT_COMPLETED_STEP+1,provenance=provenance)
 atomic_save(receipt,Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'sparse-schedule-migrated.json',json_format=True)
 print('RECURSIVE_B_STATE_MIGRATED='+str(PARENT_COMPLETED_STEP),flush=True)
 return receipt


def restore_sampling_stream(worker, b):
    import torch
    records = b['sampling_streams']
    if b.get('sampling_stream_status')=='PARENT_STREAM_NOT_RECORDED_COLD_EXECUTOR_REINITIALIZED':
        require(b['parent_completed_step']==3 and not records, 'missing sampling stream exception only at parent3')
        return dict(rank=worker.rank,actual_separate_sampling_cuda_stream_restored=False,
            parent_completed_step=3,executor_bitwise_equivalence=False,
            status='PARENT_STREAM_NOT_RECORDED_COLD_EXECUTOR_REINITIALIZED',
            scope='Native training RNG restored; sampler stream uses cold engine initialization. Partial outer4 stream is not attached to parent3.')
    require(len(records)==4 and {r['rank'] for r in records}==set(range(4)), 'sampling stream rank coverage')
    r = next(r for r in records if r['rank']==worker.rank)
    p = Path(r['path'])
    require(not p.is_symlink() and p.is_file(), 'sampling stream path')
    require(hashlib.sha256(p.read_bytes()).hexdigest()==r['file_sha256'], 'sampling stream file changed')
    saved = torch.load(p,map_location='cpu',weights_only=False)
    m = saved['metadata']; value = saved['gen_random_states']
    require(m['protocol']=='recursive-b-native-boundary-sampling-rng-v1' and m['completed_step']==PARENT_COMPLETED_STEP
      and m['actual_optimizer_calls']==PARENT_OPTIMIZER_CALLS and m['scheduler_step']==PARENT_COMPLETED_STEP
      and m['rank']==worker.rank and m['world_size']==4 and m['run_id']==b['parent_run_id']
      and m['source_commit']==b['parent_commit'], 'sampling provenance changed')
    require(value.dtype==torch.uint8 and value.device.type=='cpu' and value.ndim==1, 'sampling stream type')
    manager = worker.rollout_sharding_manager
    require(m['gen_dp_rank']==manager.device_mesh['dp'].get_local_rank(), 'sampling DP topology differs')
    raw_sha = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    require(raw_sha==m['state_sha256']==r['state_sha256'], 'sampling stream logical bytes')
    manager.gen_random_states = value.clone()
    require(torch.equal(manager.gen_random_states,value), 'sampling stream actual restore differs')
    return dict(rank=worker.rank,actual_separate_sampling_cuda_stream_restored=True,state_sha256=raw_sha,
        parent_completed_step=PARENT_COMPLETED_STEP,executor_bitwise_equivalence=False)


def save_sampling_stream(worker, local_path, identity):
    import torch
    value = worker.rollout_sharding_manager.gen_random_states
    require(value.dtype==torch.uint8 and value.device.type=='cpu', 'native saved sampling stream dtype')
    metadata = dict(protocol='recursive-b-native-boundary-sampling-rng-v1',
        source_commit=identity['code_commit'],run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        rank=worker.rank,world_size=worker.world_size,completed_step=identity['completed_step'],
        actual_optimizer_calls=worker.actor._fresh_total_committed_optimizer_calls,
        scheduler_step=worker.actor_lr_scheduler.last_epoch,
        gen_dp_rank=worker.rollout_sharding_manager.device_mesh['dp'].get_local_rank(),
        state_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
    path = Path(local_path) / ('sampling-rank-%02d.pt'%worker.rank)
    sha = atomic_save(dict(metadata=metadata,gen_random_states=value.clone()),path)
    return dict(path=str(path),file_sha256=sha,state_sha256=metadata['state_sha256'],bytes=path.stat().st_size,rank=worker.rank)

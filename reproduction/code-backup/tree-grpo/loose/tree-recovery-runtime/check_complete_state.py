"""CPU evidence verifier for actual recovery step105 files, never authorizes release."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from search_r1.baselines.training_state import file_sha,digest,finite,atomic_save
os.environ['CUDA_VISIBLE_DEVICES']=''
p=argparse.ArgumentParser();p.add_argument('--ready',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);started=time.time();ready=json.loads(a.ready.read_bytes())
assert ready['completed_outer_updates']==105 and ready['next_outer_update']==106 and ready['full_state_reload_verified'] is True
manifest_bytes=Path(ready['full_state_manifest']).read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest()==ready['full_state_manifest_sha256']
manifest=json.loads(manifest_bytes);identity=manifest['identity']
assert identity['completed_step']==105 and identity['actual_update_limit']==300 and identity['job_id']==os.environ['SLURM_JOB_ID']
assert len(manifest['worker_states'])==8
records=[]
for rank,record in enumerate(sorted(manifest['worker_states'],key=lambda r:r['identity']['rank'])):
    assert record['identity']['rank']==rank and record['identity']['world_size']==8
    assert all(record['identity'].get(key)==value for key,value in identity.items())
    assert all(ready.get(key)==identity[key] for key in ['job_id','code_commit','config_sha256','data_identity','model_identity','completed_step','actual_update_limit'])
    assert record['exact_same_world_reload_verified'] is True and record['scheduler_last_epoch']==105
    assert file_sha(record['path'])==record['sha256']
    value=torch.load(record['path'],map_location='cpu',weights_only=False);finite(value)
    assert digest(value)==record['logical_state_sha256'] and value['identity']==record['identity']
    assert value['scheduler']['last_epoch']==105
    states=value['optimizer']['state'];assert states
    assert all(state['step'].item()==2450 for state in states.values())
    assert all(abs(group['lr']-1e-6)<1e-15 for group in value['optimizer']['param_groups'])
    assert value['version']==2
    for key in ['parameters','local_shards','buffers','gradients','rng']:assert key in value
    assert value['local_shards'] and value['local_shards'].keys()==value['parameters'].keys()
    for name,shard in value['local_shards'].items():
        assert shard.shape==value['parameters'][name].shape and shard.dtype==value['parameters'][name].dtype
    assert 'cuda' in value['rng']
    records.append(dict(rank=rank,finite=True,logical_sha256=record['logical_state_sha256'],
        parameter_elements=sum(t.numel() for t in value['parameters'].values()),optimizer_state_entries=len(states),optimizer_steps=2450,scheduler_step=105,bytes=record['bytes']))
    del value
record=manifest['driver'];assert file_sha(record['path'])==record['sha256']
driver=torch.load(record['path'],map_location='cpu',weights_only=False)
assert digest(driver)==record['logical_state_sha256'] and driver['identity']==identity
assert driver['completed_step']==driver['actual_outer_updates']==105 and driver['next_step']==106
assert driver['sampler']['cursor']==54400 and driver['sampler']['epoch']==0 and driver['sampler']['rows']==169028
assert driver['algorithm']['method']=='tree_grpo' and driver['algorithm']['tree_rng'] is not None
result=dict(status='COMPLETE_STEP105_STATE_FILES_CPU_VERIFIED',identity=identity,ready_sha256=file_sha(a.ready),
    manifest_sha256=ready['full_state_manifest_sha256'],workers=records,driver_actual=105,sampler_cursor=54400,
    wall_seconds=time.time()-started,scope='Actual serialized shards and same-live-object reload receipts; independent scientific acceptance still required')
atomic_save(result,a.output,json_format=True);print(json.dumps(result,sort_keys=True))

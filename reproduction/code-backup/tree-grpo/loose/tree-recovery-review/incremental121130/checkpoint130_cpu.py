import os,time,json,hashlib,resource,importlib.util,gc
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(4)
started=time.perf_counter();cpu=time.process_time();root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234');gate=None
modulepath=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/recovery-502e9fa-f91c4f17/source/repo/search_r1/baselines/training_state.py')
spec=importlib.util.spec_from_file_location('independent_snapshot_helpers',modulepath);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
proof=[]
def rawjson(p):
 raw=p.read_bytes();proof.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)));return json.loads(raw)
manifestpath=root/'output/checkpoints/tree-recovery300/training_state/global_step_130/manifest.json'
manifest=rawjson(manifestpath);identity=manifest['identity'];ready=None;state_report=None;hf_report=None
assert identity==dict(actual_update_limit=300,code_commit='502e9fafa8e3aba722dbfcb493abc2b9d3ece326',completed_step=130,config_sha256='e54d322ba3635beb8d930c4f8282997f1867b4dbf5a0ca727e54a569bb198f35',data_identity='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',job_id='3234',model_identity='744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0')
assert len(manifest['worker_states'])==8
records=[]
for rank,record in enumerate(sorted(manifest['worker_states'],key=lambda x:x['identity']['rank'])):
 assert record['identity']['rank']==rank and record['identity']['world_size']==8 and all(record['identity'].get(k)==v for k,v in identity.items())
 path=Path(record['path']);actual=helper.file_sha(path);assert actual==record['sha256'];before=path.stat();value=torch.load(path,map_location='cpu',weights_only=False);after=path.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
 helper.finite(value);actualdigest=helper.digest(value);assert actualdigest==record['logical_state_sha256']
 assert value['version']==2 and value['identity']==record['identity'] and value['scheduler']['last_epoch']==130
 optimizer=value['optimizer'];states=optimizer['state'];assert states and all(s['step'].item()==2700 for s in states.values())
 assert all(abs(g['lr']-1e-6)<1e-15 for g in optimizer['param_groups'])
 assert set(states)==set(k for g in optimizer['param_groups'] for k in g['params'])
 assert value['parameters'].keys()==value['local_shards'].keys()
 for name,t in value['parameters'].items():assert value['local_shards'][name].shape==t.shape and value['local_shards'][name].dtype==t.dtype
 assert isinstance(record['exact_same_world_reload_verified'],bool)
 assert 'cuda' in value['rng'] and all(k in value for k in ['buffers','gradients'])
 records.append(dict(rank=rank,sha256=actual,logical_sha256=actualdigest,bytes=path.stat().st_size,all_floating_tensors_finite=True,adam_steps=2700,scheduler_step=130,exact_same_world_reload_verified=record['exact_same_world_reload_verified'],parameter_elements=sum(t.numel() for t in value['parameters'].values())))
 proof.append(dict(path=str(path),sha256=actual,bytes=path.stat().st_size));del value;gc.collect()
record=manifest['driver'];path=Path(record['path']);assert helper.file_sha(path)==record['sha256'];driver=torch.load(path,map_location='cpu',weights_only=False);assert helper.digest(driver)==record['logical_state_sha256'];helper.finite(driver)
assert driver['identity']==identity and driver['completed_step']==driver['actual_outer_updates']==130 and driver['next_step']==131
sampler=driver['sampler'];assert sampler['rows']==169028 and sampler['cursor']==70400 and sampler['epoch']==0 and sampler['batch_size']==640
assert driver['algorithm']['method']=='tree_grpo' and driver['algorithm']['tree_rng'] is not None
proof.append(dict(path=str(path),sha256=record['sha256'],bytes=path.stat().st_size))
print(json.dumps(dict(status='PASS_INDEPENDENT_SERIALIZED_CHECKPOINT130_CPU_ONLY',ready=ready,manifest=manifest,state_report=state_report,hf_report=hf_report,workers=records,driver_actual=130,driver_next=131,sampler_cursor=70400,sampler_order_sha256=helper.digest(sampler['order']),evidence=proof,wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch_threads=4,cuda_initialized=torch.cuda.is_initialized(),limits='Serialized complete state finite/digests verified. Ordinary checkpoint130 reload flag is recorded, not promoted to same-live replay acceptance. No extra live GPU snapshots or independent allparameter HF forward execution in this reader.')),flush=True)

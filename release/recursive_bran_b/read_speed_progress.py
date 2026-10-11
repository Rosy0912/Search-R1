"""CPU-only scalar observation via existing exact B actors; no CUDA/model/RNG writes."""
import argparse,json,os,socket

def read_worker(worker):
 import os
 assert os.getuid()==1001
 assert os.environ['SEARCH_R1_BASELINE_RUN_ID']=='9e3e5f5adc174d3bba2b6c5ccac939da'
 assert os.environ['SEARCH_R1_BASELINE_COMMIT']=='17fa8c1fc24b2399161b0a3179b3d2d300f597a0'
 assert worker.world_size==4 and worker._is_actor
 m=worker.rollout_sharding_manager
 return dict(pid=os.getpid(),rank=worker.rank,uid=os.getuid(),source=os.environ['SEARCH_R1_BASELINE_COMMIT'],
  run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],actual_optimizer_calls=worker.actor._fresh_total_committed_optimizer_calls,
  scheduler=worker.actor_lr_scheduler.last_epoch,cache_generation_calls=m.cache_phase_calls,
  total_weight_sync_calls=m.weight_sync_calls,weights_resident=m.weights_resident,
  cached_policy_version=m.cached_policy_version,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))

p=argparse.ArgumentParser();p.add_argument('--ray-address',required=True);p.add_argument('--pids',required=True);a=p.parse_args()
assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4' and a.ray_address.startswith('10.123.4.12:')
expected={int(x) for x in a.pids.split(',')};assert len(expected)==4
import ray
ray.init(address=a.ray_address,log_to_driver=False)
names={x['name']:x['namespace'] for x in ray.util.list_named_actors(all_namespaces=True)};actors=[]
for r in ray._private.state.actors().values():
 if r.get('Pid') in expected and r.get('State')=='ALIVE':
  assert r['ActorClassName']=='create_colocated_worker_cls.<locals>.WorkerDict'
  actors.append(ray.get_actor(r['Name'],namespace=names[r['Name']]))
assert len(actors)==4
result=ray.get([x.actor_rollout_execute_with_func_generator.remote(func=read_worker) for x in actors]);assert {r['rank'] for r in result}==set(range(4))
print(json.dumps(dict(status='ACTUAL_NATIVE_CACHE_AND_UPDATE_SCALARS_READ',workers=result)))
ray.shutdown()

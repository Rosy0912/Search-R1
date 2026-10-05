"""Capture only the explicitly identified owned run before its replacement."""
import argparse,json,os,time
from pathlib import Path
import ray

p=argparse.ArgumentParser()
p.add_argument('--base',required=True);p.add_argument('--run-id',required=True)
p.add_argument('--address',required=True);a=p.parse_args()
base=Path(a.base);receipt=json.loads((base/'receipts/r1-round-1-grpo.json').read_text())
assert receipt['run_id']==a.run_id and receipt['status']=='RUNNING'
proc=Path('/proc')/str(receipt['pid'])
assert proc.stat().st_uid==os.getuid()
assert ('E1_RUN_ID='+a.run_id).encode() in proc.joinpath('environ').read_bytes().split(b'\0')
root=base/'preserved'/('replacement-state-'+a.run_id)
assert not root.exists();root.mkdir(parents=True)
expected_run=a.run_id;output=str(root)
def capture(w):
 import torch,os,json,time,random,numpy as np
 from pathlib import Path
 assert os.environ['E1_RUN_ID']==expected_run and w.world_size==8
 scheduler=w.actor_lr_scheduler.state_dict();step=int(scheduler['last_epoch'])
 version=torch.tensor([step],device='cuda',dtype=torch.int64)
 versions=[torch.empty_like(version) for _ in range(8)]
 torch.distributed.all_gather(versions,version)
 assert all(int(v.item())==step for v in versions),'Rank state versions differ'
 state=w.actor_optimizer.state_dict()
 optimizer_steps=sorted(set(float(v['step']) for v in state['state'].values() if 'step' in v))
 assert optimizer_steps==[float(step*10)],optimizer_steps
 model={n:p.detach().cpu().clone() for n,p in w.actor_module_fsdp.named_parameters()}
 assert all(torch.isfinite(v).all() for v in model.values())
 target=Path(output)/('full-rank-'+str(w.rank)+'.pt');temp=target.with_suffix('.tmp')
 torch.save(dict(model=model,optimizer=state,scheduler=scheduler,rank=w.rank,world_size=8,global_step=step,parameter_shapes=[(n,list(p.shape),str(p.dtype)) for n,p in w.actor_module_fsdp.named_parameters()],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),run_id=expected_run),temp)
 temp.replace(target)
 result=dict(rank=w.rank,pid=os.getpid(),scheduler_step=step,optimizer_steps=optimizer_steps,path=str(target),bytes=target.stat().st_size,time=time.time())
 (Path(output)/('full-rank-'+str(w.rank)+'.json')).write_text(json.dumps(result,indent=2));return result
ray.init(address=a.address,log_to_driver=False)
names=[n for n in ray.util.list_named_actors(all_namespaces=True) if n['name'].startswith('OhbVrpWorkerDict_0:')]
assert len(names)==8
results=ray.get([ray.get_actor(n['name'],namespace=n['namespace']).actor_rollout_execute_with_func_generator.remote(func=capture) for n in names])
assert sorted(r['rank'] for r in results)==list(range(8))
assert len({r['scheduler_step'] for r in results})==1
(root/'full-complete.json').write_text(json.dumps(dict(run_id=expected_run,world_size=8,global_step=results[0]['scheduler_step'],ranks=results,time=time.time()),indent=2))
print(json.dumps(results),flush=True);ray.shutdown()

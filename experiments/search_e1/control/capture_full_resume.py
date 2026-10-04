import os,json,time,ray
from pathlib import Path
base=Path('/data/shaoshitong/search-e1-config-study-8gpu-20261004')
run='8d708a8556084021ae3069b93335109d'
out=base/'resume-state'/run;out.mkdir(parents=True,exist_ok=True)
def capture(w):
 import os,torch,random,numpy as np,time,json
 from pathlib import Path
 assert os.environ['E1_RUN_ID']=='8d708a8556084021ae3069b93335109d'
 assert w.world_size==8
 scheduler=w.actor_lr_scheduler.state_dict();assert scheduler['last_epoch']==2,scheduler
 root=Path('/data/shaoshitong/search-e1-config-study-8gpu-20261004/resume-state/8d708a8556084021ae3069b93335109d')
 state=w.actor_optimizer.state_dict();steps=sorted(set(float(v['step']) for v in state['state'].values() if 'step' in v))
 assert steps==[20.0],steps
 shapes=[(n,list(p.shape),str(p.dtype)) for n,p in w.actor_module_fsdp.named_parameters()]
 target=root/('full-rank-'+str(w.rank)+'.pt');temp=target.with_suffix('.tmp')
 model={n:p.detach().cpu().clone() for n,p in w.actor_module_fsdp.named_parameters()}
 torch.save(dict(model=model,optimizer=state,scheduler=scheduler,rank=w.rank,world_size=w.world_size,global_step=2,parameter_shapes=shapes,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),run_id=os.environ['E1_RUN_ID']),temp);temp.replace(target)
 r=dict(rank=w.rank,pid=os.getpid(),scheduler=scheduler,optimizer_steps=steps,state_entries=len(state['state']),path=str(target),bytes=target.stat().st_size,time=time.time())
 (root/('full-rank-'+str(w.rank)+'.json')).write_text(json.dumps(r,indent=2));return r
ray.init(address='10.120.17.101:59994',log_to_driver=False)
names=ray.util.list_named_actors(all_namespaces=True);names=[n for n in names if n['name'].startswith('OhbVrpWorkerDict_0:') and n['namespace']=='6d5d5e41-677e-41b8-a810-40e9e2af57ad'];assert len(names)==8
refs=[ray.get_actor(n['name'],namespace=n['namespace']).actor_rollout_execute_with_func_generator.remote(capture) for n in names]
print('EIGHT_OWNED_CAPTURE_CALLS_QUEUED_AFTER_CURRENT_ACTOR_UPDATE',flush=True)
results=ray.get(refs);assert sorted(r['rank'] for r in results)==list(range(8));(out/'full-complete.json').write_text(json.dumps(dict(global_step=2,run_id=run,world_size=8,ranks=results,time=time.time()),indent=2));print(json.dumps(results),flush=True)
ray.shutdown()

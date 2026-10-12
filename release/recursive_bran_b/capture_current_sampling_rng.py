"""Capture this run's sampling stream at an actual completed actor boundary.

Queue only after its update_actor starts. The synchronous actor performs this
before the next queued sampling call. Does not draw or change any RNG state.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def capture(worker, run, source, step, output):
    import hashlib,json,os,time,torch
    from pathlib import Path
    assert os.environ['SEARCH_R1_BASELINE_RUN_ID']==run
    assert os.environ['SEARCH_R1_BASELINE_COMMIT']==source
    assert worker.world_size==4 and worker.actor_lr_scheduler.last_epoch==step
    manager=worker.rollout_sharding_manager
    assert not manager.weights_resident, 'Must be between actor update and new sampling'
    value=manager.gen_random_states.detach().clone()
    assert value.device.type=='cpu' and value.dtype==torch.uint8
    root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
    assert not (root/'checkpoints/training_state'/('global_step_%d'%step)/'manifest.json').exists()
    p=Path(output)/('rank-%02d.pt'%worker.rank);p.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(protocol='recursive-b-native-boundary-sampling-rng-v1',time=time.time(),
        run_id=run,source_commit=source,rank=worker.rank,world_size=4,completed_step=step,
        scheduler_step=step,actual_optimizer_calls=worker.actor._fresh_total_committed_optimizer_calls,
        gen_dp_rank=manager.device_mesh['dp'].get_local_rank(),pid=os.getpid(),
        state_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
    with p.open('xb') as f:torch.save(dict(metadata=metadata,gen_random_states=value),f);f.flush();os.fsync(f.fileno())
    receipt=dict(metadata,path=str(p),file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    p.with_suffix('.json').write_text(json.dumps(receipt,sort_keys=True))
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    p.add_argument('--source',required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
    assert a.address.startswith('10.123.4.12:') and a.step in (4,5)
    assert a.output.startswith('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011/receipts/')
    import ray
    ray.init(address=a.address,log_to_driver=False)
    try:
        names={x['name']:x['namespace'] for x in ray.util.list_named_actors(all_namespaces=True)}
        actors=[ray.get_actor(x['Name'],namespace=names[x['Name']]) for x in ray._private.state.actors().values()
            if x.get('State')=='ALIVE' and x.get('ActorClassName')=='create_colocated_worker_cls.<locals>.WorkerDict']
        assert len(actors)==4
        result=ray.get([x.actor_rollout_execute_with_func_generator.remote(func=functools.partial(capture,
            run=a.run,source=a.source,step=a.step,output=a.output)) for x in actors],timeout=1800)
        assert {x['rank'] for x in result}=={0,1,2,3}
        Path(a.output,'streams.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:ray.shutdown()


if __name__=='__main__':main()

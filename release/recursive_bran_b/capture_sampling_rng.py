"""Read an owned live worker's separate sampling CUDA stream via its existing API.

Queue during the final actor update: the synchronous actor executes this after
update returns, before the driver can enqueue the next sampling RPC. No model,
optimizer, scheduler, or RNG is modified. Full RNG bytes stay on the node.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import socket


def capture(worker, step, output, capture_sha):
    import hashlib, json, os, pathlib, time, torch
    assert os.getuid() == 1001
    assert os.environ['SEARCH_R1_BASELINE_RUN_ID'] == 'ef09290e69454f20b6a9594ad8e485b6'
    assert os.environ['SEARCH_R1_BASELINE_COMMIT'] == '3da904591125f117b1eae1ba09546ebe793cb2d4'
    assert worker.world_size == 4 and worker._is_actor
    assert worker.actor._fresh_total_committed_optimizer_calls == step * 20
    assert worker.actor_lr_scheduler.last_epoch == step
    checkpoint = pathlib.Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011/checkpoints/training_state') / ('global_step_%d' % step) / 'manifest.json'
    assert not checkpoint.exists(), 'Capture must precede this checkpoint commit and next sampling'
    manager = worker.rollout_sharding_manager
    state = manager.gen_random_states
    assert state is not None and state.dtype == torch.uint8 and state.device.type == 'cpu'
    value = state.detach().clone()
    p = pathlib.Path(output) / ('rank-%02d.pt' % worker.rank)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise ValueError('Never overwrite a captured sampling stream')
    metadata = dict(protocol='recursive-b-native-boundary-sampling-rng-v1',
        time=time.time(),source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],rank=worker.rank,world_size=4,
        completed_step=step,actual_optimizer_calls=step*20,scheduler_step=step,
        pid=os.getpid(),capture_source_sha256=capture_sha,
        gen_dp_rank=manager.device_mesh['dp'].get_local_rank(),
        state_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
    with p.open('xb') as f:
        torch.save(dict(metadata=metadata,gen_random_states=value),f)
        f.flush(); os.fsync(f.fileno())
    file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(metadata,path=str(p),file_sha256=file_sha,bytes=p.stat().st_size,
                   status='ACTUAL_SEPARATE_SAMPLING_STREAM_CAPTURED_NO_RNG_DRAW')
    with p.with_suffix('.json').open('x') as f:
        json.dump(receipt,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--ray-address',required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
    assert a.ray_address.startswith('10.123.4.12:') and a.step in (3,4,5)
    assert a.output.startswith('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011/receipts/')
    import ray
    ray.init(address=a.ray_address,log_to_driver=False)
    expected={412273,412683,412684,412685}
    actors=[]
    names = {x['name']:x['namespace'] for x in ray.util.list_named_actors(all_namespaces=True)}
    for record in ray._private.state.actors().values():
        if record.get('Pid') in expected and record.get('State')=='ALIVE':
            assert record.get('Name') and record.get('ActorClassName')=='create_colocated_worker_cls.<locals>.WorkerDict'
            actors.append(ray.get_actor(record['Name'],namespace=names[record['Name']]))
    if len(actors)!=4:
        raise ValueError('Expected all four exact original owned actor PIDs')
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tasks=[x.actor_rollout_execute_with_func_generator.remote(func=functools.partial(capture,step=a.step,output=a.output,capture_sha=digest)) for x in actors]
    result=ray.get(tasks)
    assert {r['rank'] for r in result}==set(range(4))
    print(json.dumps(dict(status='FOUR_SAMPLING_STREAMS_CAPTURED',step=a.step,workers=result)))
    ray.shutdown()


if __name__=='__main__':main()

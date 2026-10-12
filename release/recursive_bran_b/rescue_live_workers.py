"""Preserve actual owned live Adam80/scheduler4 state; no driver-state claim."""
import argparse
import functools
import json
from pathlib import Path
import os
import socket


def capture(worker, run, source, output):
    import os,torch
    from pathlib import Path
    from search_r1.baselines.training_state import snapshot_worker_to_file,atomic_save
    assert os.environ['SEARCH_R1_BASELINE_RUN_ID']==run and os.environ['SEARCH_R1_BASELINE_COMMIT']==source
    assert worker.world_size==4 and worker.actor_lr_scheduler.last_epoch==4
    assert worker.actor._fresh_total_committed_optimizer_calls==80
    assert {int(x['step']) for x in worker.actor_optimizer.state.values()}=={80}
    identity=dict(job_id=os.environ['SLURM_JOB_ID'],code_commit=source,
        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
        model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],completed_step=4,actual_update_limit=200)
    receipt=snapshot_worker_to_file(worker,output,identity,verify=False)
    manager=worker.rollout_sharding_manager
    sampling=manager.gen_random_states.detach().clone()
    path=Path(output)/('sampling-after-incomplete5-rank%02d.pt'%worker.rank)
    sha=atomic_save(sampling,path)
    return dict(worker=receipt,actual_optimizer_calls=80,scheduler_step=4,
        sampling_rng_path=str(path),sampling_rng_sha256=sha,
        sampling_rng_scope='Live stream after incomplete outer5; not exact completed4 boundary',
        complete_driver_state_saved=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
    assert a.address.startswith('10.123.4.12:')
    import ray
    ray.init(address=a.address,log_to_driver=False)
    try:
        names={x['name']:x['namespace'] for x in ray.util.list_named_actors(all_namespaces=True)}
        actors=[ray.get_actor(x['Name'],namespace=names[x['Name']]) for x in ray._private.state.actors().values()
            if x.get('State')=='ALIVE' and x.get('ActorClassName')=='create_colocated_worker_cls.<locals>.WorkerDict']
        assert len(actors)==4
        results=ray.get([x.actor_rollout_execute_with_func_generator.remote(func=functools.partial(capture,
            run=a.run,source=a.source,output=a.output)) for x in actors],timeout=600)
        receipt=dict(run=a.run,source=a.source,workers=results,complete_driver_state_saved=False)
        Path(a.output,'rescue-workers.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
    finally:ray.shutdown()


if __name__=='__main__':main()

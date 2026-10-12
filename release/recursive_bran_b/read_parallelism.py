"""Read actual scheduler scalars from the running B actors; no model/RNG writes."""
import argparse
import functools
import json
import os
import socket
import time


def read_actor(worker, run, source):
    import os
    import torch
    assert os.environ['SEARCH_R1_BASELINE_RUN_ID']==run
    assert os.environ['SEARCH_R1_BASELINE_COMMIT']==source
    assert worker.world_size==4 and worker._is_rollout
    engine=worker.rollout_sharding_manager.inference_engine.llm_engine
    scheduler=engine.scheduler_config
    return dict(rank=worker.rank,pid=os.getpid(),run_id=run,source_commit=source,
        world_size=worker.world_size,tensor_parallel_size=worker.config.rollout.tensor_model_parallel_size,
        actual_max_num_seqs=scheduler.max_num_seqs,
        actual_max_num_batched_tokens=scheduler.max_num_batched_tokens,
        actual_chunked_prefill=getattr(scheduler,'chunked_prefill_enabled',None),
        actual_max_model_len=scheduler.max_model_len,
        actual_kv_gpu_blocks=engine.cache_config.num_gpu_blocks,
        actual_kv_block_size=engine.cache_config.block_size,
        cumulative_kv_preemptions=[s.num_cumulative_preemption for s in engine.scheduler],
        configured_max_num_seqs=worker.config.rollout.get('max_num_seqs'),
        configured_max_num_batched_tokens=worker.config.rollout.get('max_num_batched_tokens'),
        actor_training_microbatch=worker.config.actor.ppo_micro_batch_size,
        baseline_optimizer_calls=worker.actor._fresh_total_committed_optimizer_calls,
        scheduler_step=worker.actor_lr_scheduler.last_epoch,
        cuda_allocated_bytes=torch.cuda.memory_allocated(),
        cuda_reserved_bytes=torch.cuda.memory_reserved(),
        cuda_recorded_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_recorded_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cuda_oom_count=torch.cuda.memory_stats().get('num_ooms'),
        cuda_peak_scope='Existing allocator counters since their last reset; this probe does not reset counters or allocate tensors.',
        detokenizer_length_cache_enabled=os.environ.get('RECURSIVE_B_CACHE_DETOKENIZER_LENGTH')=='1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xie-lab-4'
    assert a.address.startswith('10.123.4.12:')
    import ray
    ray.init(address=a.address,log_to_driver=False)
    try:
        names={x['name']:x['namespace'] for x in ray.util.list_named_actors(all_namespaces=True)}
        actors=[]
        for record in ray._private.state.actors().values():
            if (record.get('State')=='ALIVE' and
                record.get('ActorClassName')=='create_colocated_worker_cls.<locals>.WorkerDict'):
                actors.append(ray.get_actor(record['Name'],namespace=names[record['Name']]))
        assert len(actors)==4
        pending=[x.actor_rollout_execute_with_func_generator.remote(
            func=functools.partial(read_actor,run=a.run,source=a.source)) for x in actors]
        # Observation timeout does not change the running actors or trigger retry.
        result=ray.get(pending,timeout=600)
        assert {x['rank'] for x in result}=={0,1,2,3}
        from pathlib import Path
        receipt=dict(at=time.time(),scope='actual_running_vllm_scheduler_readonly',workers=result)
        Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    finally:ray.shutdown()


if __name__=='__main__':main()

"""Durable CPU-only import/serialization probe. Not GPU or training acceptance."""
import fcntl,json,os,time,traceback
from pathlib import Path
ROOT=Path(os.environ['COSEARCH_RUN_ROOT'])

def write(status,**values):
    value=dict(status=status,epoch=time.time(),pid=os.getpid(),**values)
    (ROOT/'cpu-runtime-status.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)

def main():
    lock=(ROOT/'cpu-runtime.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU probe must mask CUDA')
    write('IMPORTING')
    import torch
    import runtime_fixes
    runtime_fixes.install_worker()
    write('WORKER_FIXES_INSTALLED',cuda_initialized=torch.cuda.is_initialized())
    from verl.experimental.agent_loop.agent_loop import CoSearchAgentLoopManager
    from verl.trainer.ppo.search_r1_reranker_ray_trainer import CoSearchRayTrainer
    manager=CoSearchAgentLoopManager.__new__(CoSearchAgentLoopManager)
    trainer=type('ProbeTrainer',(),{'async_rollout_manager':manager})()
    runtime_fixes.install_controller(trainer,CoSearchRayTrainer)
    import ray.cloudpickle as cloudpickle
    from verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker import CoSearchAgentLoopWorker
    worker=CoSearchAgentLoopWorker.__ray_metadata__.modified_class
    for method in (worker.generate_sequences,worker._postprocess,manager.generate_sequences):
        cloudpickle.loads(cloudpickle.dumps(method))
    assert not torch.cuda.is_initialized()
    write('CPU_RUNTIME_FIXES_SERIALIZATION_PASS',cuda_initialized=False,gpu_accepted=False)
if __name__=='__main__':
    try:main()
    except BaseException:
        write('FAILED',traceback=traceback.format_exc());raise

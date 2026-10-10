"""Retain reviewed native sampling, underfull/empty and optional-metadata handling."""
import hashlib,json,os,time
from pathlib import Path

def record(kind,values):
    import phase_sampling
    root=Path(os.environ['COSEARCH_AUDIT_ROOT']);root.mkdir(parents=True,exist_ok=True)
    with (root/('runtime-'+str(os.getpid())+'.jsonl')).open('a') as f:
        f.write(json.dumps(dict(kind=kind,values=phase_sampling.json_metadata(values),epoch=time.time()),allow_nan=False)+'\n')

def preserve(kind,payload):
    import torch
    root=Path(os.environ['COSEARCH_AUDIT_ROOT'])/'raw';root.mkdir(exist_ok=True)
    path=root/(kind+'-'+str(os.getpid())+'-'+str(time.time_ns())+'.pt')
    torch.save(payload,path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    record('raw_payload',{'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size})
    # Only this new run's prior diagnostic payloads: retain two per stage/process.
    old=sorted(root.glob(kind+'-'+str(os.getpid())+'-*.pt'))
    for p in old[:-2]:p.unlink()
    return path

def install_worker():
    from verl.experimental.agent_loop.search_r1_reranker_agent_loop_worker import CoSearchAgentLoopWorker
    worker=CoSearchAgentLoopWorker.__ray_metadata__.modified_class
    original_init=worker.__init__
    def init_shared_four(instance,*args,**kwargs):
        original_init(instance,*args,**kwargs)
        instance.reranker_n_gpus=4
        record('reranker_alignment_world_size',{'world_size':4,'shared_gpu_pool':True})
    worker.__init__=init_shared_four
    import fallback_alignment,empty_ranker,phase_sampling
    fallback_alignment.install(worker,record,preserve)
    empty_ranker.install_worker(worker,record,preserve)
    phase_sampling.install_sampling(record)

def install_controller(trainer,trainer_class):
    import empty_ranker,phase_sampling,observe_worker
    observe_worker.preserve_raw_payload=preserve
    manager=trainer.async_rollout_manager
    empty_ranker.install_controller(manager,trainer_class,record)
    _,support=phase_sampling.install_aggregation(manager,record)
    original_generate=manager.generate_sequences
    def generate(batch):
        from bounded_sampling import generate_supported
        step=int(batch.meta_info.get('global_steps',-1))
        validation=bool(batch.meta_info.get('validate',False))
        # Native generation errors remain hard failures; no additional support abort.
        return generate_supported(original_generate,batch,support,
            phase_sampling.require_training_support,record,preserve,
            lambda step,validation:None,step,validation)
    manager.generate_sequences=generate

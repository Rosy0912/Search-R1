"""Opt-in Tree memory observer; no kernels, RNG, collectives or counter resets."""
import os
if os.environ.get('TREE_INSTRUCT_TOOLS'):
    # Every inherited preflight snapshot rechecks actual cgroup before querying.
    try:
        from slurm_identity import install_gpu_query_guard
        install_gpu_query_guard()
    except Exception as error:
        raise SystemExit('Required Tree own-cgroup GPU-query guard bootstrap failed') from error
if os.environ.get('TREE_FIXED_ACTOR_PROFILE')=='1':
    import hashlib,json,time
    from pathlib import Path
    import torch
    from search_r1.baselines.tree_rank_audit import TreeRankAudit
    original_init=TreeRankAudit.__init__;original_micro=TreeRankAudit.micro
    original_gradient=TreeRankAudit.gradient;original_finish=TreeRankAudit.finish
    def memory():
        from slurm_identity import verify_runtime_cgroup
        verify_runtime_cgroup()
        if not torch.cuda.is_available():return dict(cuda_available=False)
        return dict(cuda_available=True,device=torch.cuda.current_device(),
            allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            lifetime_peak_allocated=torch.cuda.max_memory_allocated(),lifetime_peak_reserved=torch.cuda.max_memory_reserved())
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self._profile=dict(start=memory(),micro_records=[],optimizer_records=[],observer_cpu_seconds=0.,observer_wall_seconds=0.)
    def micro(self,*args,**kwargs):
        original_micro(self,*args,**kwargs)
        start=time.perf_counter();cpu=time.process_time();unique={};logical=0
        for m in self.pending_microbatches:
            for tensor in [*m['tensors'].values(),*m['loss_parts'].values()]:
                if tensor.device.type=='cuda':
                    logical+=tensor.numel()*tensor.element_size()
                    unique[(str(tensor.device),tensor.untyped_storage().data_ptr())]=tensor.untyped_storage().nbytes()
        self._profile['micro_records'].append(dict(pending_microbatches=len(self.pending_microbatches),unique_retained_cuda_bytes=sum(unique.values()),logical_cuda_tensor_bytes=logical,memory=memory()))
        self._profile['observer_wall_seconds']+=time.perf_counter()-start;self._profile['observer_cpu_seconds']+=time.process_time()-cpu
    def gradient(self,norm):
        before=memory();original_gradient(self,norm)
        self._profile['optimizer_records'].append(dict(before_clear=before,after_clear=memory(),pending_after_clear=len(self.pending_microbatches)))
    def finish(self,*args,**kwargs):
        original_finish(self,*args,**kwargs)
        record=dict(kind='tree_detached_storage_observer',outer_step=self.step,attempt_id=self.attempt_id,rank=self.rank,world_size=self.world,
            code_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
            observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
            **self._profile,end=memory(),scope='Actual actor per-process CUDA allocator lifetime peaks; includes observer, notretriever/wholeNVML; no counter reset or synchronization; cost excludes some gradient/finish publication')
        target=Path(os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR'])/'memory-profile'/f'step-{self.step:06d}{self.attempt_suffix}-rank-{self.rank:02d}.json';target.parent.mkdir(parents=True,exist_ok=True)
        pending=target.with_suffix('.pending')
        with pending.open('x') as stream:json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        os.link(pending,target);pending.unlink()
    TreeRankAudit.__init__=init;TreeRankAudit.micro=micro;TreeRankAudit.gradient=gradient;TreeRankAudit.finish=finish

"""Pinned vLLM 0.6.3 Worker methods; actual synchronous eight-rank KV ACKs."""
import os
from vllm.worker.worker import Worker
class TeacherWorker(Worker):
    def topd_memory(self):
        import torch
        return dict(pid=os.getpid(),uid=os.getuid(),starttime=__import__('pathlib').Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],cgroup=__import__('pathlib').Path('/proc/self/cgroup').read_text(),job=os.environ['SLURM_JOB_ID'],rank=self.rank,local_rank=self.local_rank,
                    org_vocab_size=self.model_runner.model.lm_head.org_vocab_size,model_parameter_dtype=str(next(self.model_runner.model.parameters()).dtype),allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    cache_engine_absent=self.cache_engine is None,gpu_cache_absent=self.gpu_cache is None,
                    model_training=self.model_runner.model.training,
                    requires_grad_parameters=sum(p.requires_grad for p in self.model_runner.model.parameters()))
    def topd_release_kv(self):
        import torch
        self.cache_engine=None;self.gpu_cache=None
        torch.cuda.synchronize();torch.cuda.empty_cache()
        return self.topd_memory()
    def topd_rebuild_kv(self):
        if self.gpu_cache is None:self._init_cache_engine()
        return self.topd_memory()
    def topd_freeze(self):
        self.model_runner.model.eval().requires_grad_(False)
        return self.topd_memory()

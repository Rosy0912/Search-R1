"""Native vLLM0.6.3 custom-executor hook; original MP lifecycle unchanged."""
from vllm.executor.multiproc_gpu_executor import MultiprocGPUExecutor
class TeacherExecutor(MultiprocGPUExecutor):
    def _get_worker_module_and_class(self):
        if self.scheduler_config.is_multi_step or self.speculative_config:
            raise ValueError('Frozen teacher requires original single-step non-speculative MP')
        return ('topd_vllm_teacher.worker','TeacherWorker',None)

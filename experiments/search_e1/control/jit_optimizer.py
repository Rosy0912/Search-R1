"""Stage the existing GPU AdamW states only at an optimizer boundary."""
def patch_actor_optimizer(source):
 source='from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer\n'+source
 needle='        self.actor_optimizer.step()'
 assert source.count(needle)==1
 source=source.replace(needle,'''        jit = os.environ.get('E1_JIT_OPTIMIZER') == '1'
        if jit:
            if not self.config.fsdp_config.get('optimizer_offload', False):
                raise ValueError('JIT states require the original optimizer_offload configuration')
            load_fsdp_optimizer(self.actor_optimizer, torch.cuda.current_device())
        try:
            self.actor_optimizer.step()
        finally:
            if jit:
                offload_fsdp_optimizer(self.actor_optimizer)
                # Upstream uses asynchronous CPU transfers; complete them before reuse.
                torch.cuda.synchronize()
''',1)
 return source

def patch_worker_optimizer(source):
 before,actor=source.split('    def update_actor(',1)
 body,after=actor.split('    def compute_log_prob(',1)
 needle='        if self._is_offload_optimizer:\n            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())'
 assert body.count(needle)==1
 body=body.replace(needle,"        if self._is_offload_optimizer and os.environ.get('E1_JIT_OPTIMIZER') != '1':\n            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())",1)
 return before+'    def update_actor('+body+'    def compute_log_prob('+after

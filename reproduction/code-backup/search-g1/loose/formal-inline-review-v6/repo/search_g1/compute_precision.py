"""Explicit policy compute precision; tensor dtype is not a kernel-math claim."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


def _reject_collectively(error, *, stage, process_group, error_type=RuntimeError):
    """All matching policy ranks decide before any rank enters its next forward."""
    if dist.is_available() and dist.is_initialized():
        backend = str(dist.get_backend(process_group)).lower()
        device = torch.device('cuda', torch.cuda.current_device()) if 'nccl' in backend else torch.device('cpu')
        allowed = torch.tensor(int(error is None), device=device, dtype=torch.int32)
        dist.all_reduce(allowed, op=dist.ReduceOp.MIN, group=process_group)
        if not bool(allowed.item()):
            detail = error if error is not None else 'another policy rank failed the precision invariant'
            raise error_type(f'Policy precision rejected collectively at {stage}: {detail}')
    elif error is not None:
        raise error_type(error)


@dataclass(frozen=True)
class ComputePrecision:
    compute_dtype: str = 'bfloat16'
    allow_tf32: Optional[bool] = None

    def __post_init__(self):
        if self.compute_dtype not in ('bfloat16', 'float32'):
            raise ValueError('compute_dtype must be bfloat16 or float32')
        if self.allow_tf32 is not None and type(self.allow_tf32) is not bool:
            raise ValueError('allow_tf32 must be a bool or null')

    @classmethod
    def from_config(cls, config):
        return cls(config.get('compute_dtype', 'bfloat16'), config.get('allow_tf32', None))

    def validate_module(self, module, process_group=None):
        """Reject FP32 requests backed by lower-precision parameters or FSDP casts."""
        if self.compute_dtype != 'float32':
            return
        error = self._module_error(module)
        _reject_collectively(error, stage='initial_model_dtype', process_group=process_group, error_type=ValueError)

    def _module_error(self, module):
        parameters = list(module.named_parameters())
        if not parameters:
            return 'float32 compute requires actual model parameters'
        for name, value in parameters + list(module.named_buffers()):
            if value.is_floating_point() and value.dtype != torch.float32:
                return f'float32 compute requires float32 model tensor: {name} has {value.dtype}'
        for name, child in module.named_modules():
            mixed = getattr(child, 'mixed_precision', None)
            if mixed is None:
                continue
            for field in ('param_dtype', 'reduce_dtype', 'buffer_dtype'):
                dtype = getattr(mixed, field, None)
                if dtype not in (None, torch.float32):
                    return f'float32 compute conflicts with FSDP {name}.{field}={dtype}'
        return None

    def configure_worker(self):
        """Explicit settings affect this worker process, including its rollout engine."""
        if self.allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32

    def _worker_error(self):
        if self.allow_tf32 is not None and (
                torch.backends.cuda.matmul.allow_tf32 != self.allow_tf32 or
                torch.backends.cudnn.allow_tf32 != self.allow_tf32):
            return 'Worker TF32 flags differ from the explicit policy setting'
        return None

    def validate_worker(self, process_group=None):
        if self.compute_dtype == 'float32' or self.allow_tf32 is not None:
            _reject_collectively(self._worker_error(), stage='before_forward', process_group=process_group)

    def autocast(self, process_group=None):
        # BF16 retains the original context. FP32 overrides even an enclosing
        # CUDA autocast, without replacing torch.autocast anywhere else.
        self.validate_worker(process_group=process_group)
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                              enabled=self.compute_dtype == 'bfloat16')

    def observe_logits(self, logits, process_group=None):
        error = self._worker_error()
        if self.compute_dtype == 'float32' and logits.dtype != torch.float32:
            logits_error = f'float32 compute produced {logits.dtype} logits'
            error = f'{error}; {logits_error}' if error else logits_error
        if self.compute_dtype == 'float32' or self.allow_tf32 is not None:
            _reject_collectively(error, stage='after_forward_logits', process_group=process_group)
        return dict(compute_dtype=self.compute_dtype, logits_dtype=str(logits.dtype),
                    logits_device=str(logits.device), logits_shape=list(logits.shape),
                    autocast_cuda_enabled=torch.is_autocast_enabled(),
                    allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                    allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    tf32_setting_scope='worker process; tensor dtype and flags do not prove internal attention arithmetic')

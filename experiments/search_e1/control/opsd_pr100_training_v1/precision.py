"""Explicit compute contract and synchronous, resumable loss scaling.

No model/GPU/import Torch at import time. Trainer owns durable capsules/RNG.
"""
from contextlib import contextmanager, nullcontext
import math

class PrecisionContract:
    def __init__(self, mode='bf16', device_type='cuda'):
        if mode not in ('bf16', 'fp32_master_fp16'):
            raise ValueError('Unknown precision contract')
        self.mode, self.device_type = mode, device_type
        self.events, self.handles, self.role = [], [], 'unassigned'
        self.dtype_counts={};self.events_dropped=0
        self.stage = 'forward'

    @contextmanager
    def context(self, role=None):
        import torch
        previous = self.role
        if role is not None: self.role = role
        ctx = (torch.autocast(self.device_type, dtype=torch.float16, cache_enabled=False)
               if self.mode == 'fp32_master_fp16' else nullcontext())
        try:
            with ctx: yield
        finally: self.role = previous

    def install(self, model):
        """Cast embedding/norm exits, preserving internal norm FP32 arithmetic.

        Qwen FP32 embedding/RMSNorm weights otherwise promote residuals back
        to FP32 despite autocast. Hooks leave parameters unchanged and also
        run during checkpoint recomputation; head remains under autocast.
        """
        import torch
        if self.handles: raise ValueError('Precision hooks already installed')
        for name, module in model.named_modules():
            cast = isinstance(module, torch.nn.Embedding) or type(module).__name__.endswith(('RMSNorm','LayerNorm'))
            observed = cast or isinstance(module, torch.nn.Linear) or type(module).__name__.endswith(('DecoderLayer','Attention'))
            if not observed: continue
            def hook(mod, args, output, name=name, cast=cast):
                def convert(x):
                    if isinstance(x, torch.Tensor) and x.is_floating_point() and cast and self.mode == 'fp32_master_fp16':
                        return x.to(torch.float16)
                    return x
                result = convert(output)
                if isinstance(output, tuple): result = tuple(convert(x) for x in output)
                tensors = [result] if isinstance(result, torch.Tensor) else [x for x in result if isinstance(x, torch.Tensor)] if isinstance(result, tuple) else []
                event=dict(module=name, role=self.role, stage=self.stage,
                    input_dtypes=[str(x.dtype) for x in args if isinstance(x,torch.Tensor)],
                    output_dtypes=[str(x.dtype) for x in tensors], explicit_activation_cast=bool(cast and self.mode=='fp32_master_fp16'))
                key=(name,self.role,self.stage,tuple(event['input_dtypes']),tuple(event['output_dtypes']))
                self.dtype_counts[key]=self.dtype_counts.get(key,0)+1
                if len(self.events)<4096:self.events.append(event)
                else:self.events_dropped+=1
                return result
            self.handles.append(module.register_forward_hook(hook))
        if self.mode == 'fp32_master_fp16' and any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            self.close(); raise ValueError('P/J require original FP32 parameter storage, not BF16 upcast')
        return self

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles=[]


class SyncedGradScaler:
    """SUM scaled grads, unscale, global finite, clip, then Adam.

    Callbacks are actual distributed operations: reduce_grad(tensor) performs
    in-place SUM, reduce_max(int) returns all-rank MAX; agree_scale(float)
    must verify all-rank exact scale/tracker equality before each step.
    No division by world. Rejected steps leave Adam/cursor to trainer unchanged.
    """
    def __init__(self, enabled=True, scale=65536., growth_factor=2., backoff_factor=.5, growth_interval=2000):
        self.enabled=bool(enabled); self.scale=float(scale if enabled else 1.)
        self.growth_factor=float(growth_factor);self.backoff_factor=float(backoff_factor)
        self.growth_interval=int(growth_interval);self.growth_tracker=0
        self._validate()

    def _validate(self):
        if not math.isfinite(self.scale) or self.scale<=0 or self.growth_factor<=1 or not 0<self.backoff_factor<1 or self.growth_interval<1:
            raise ValueError('Invalid scaler state')

    def state_dict(self):
        return dict(version='synchronous-loss-scaler-v1',enabled=self.enabled,scale=self.scale,
            growth_factor=self.growth_factor,backoff_factor=self.backoff_factor,
            growth_interval=self.growth_interval,growth_tracker=self.growth_tracker)

    def load_state_dict(self, state):
        if state.get('version')!='synchronous-loss-scaler-v1' or state['enabled']!=self.enabled:
            raise ValueError('Scaler version/precision differs')
        for k in ('growth_factor','backoff_factor','growth_interval'):
            if state[k]!=getattr(self,k):raise ValueError('Scaler policy changed on resume')
        self.scale=float(state['scale']);self.growth_tracker=int(state['growth_tracker']);self._validate()
        if not 0<=self.growth_tracker<self.growth_interval:raise ValueError('Invalid growth tracker')

    def scale_loss(self, loss): return loss*self.scale

    def finish_step(self, parameters, optimizer, reduce_grad, reduce_max, agree_scale, local_failed=False):
        import torch
        parameters=list(parameters);before=self.state_dict()
        agree_scale(before)
        initial_failed=bool(reduce_max(int(local_failed)))
        group_parameters=[p for group in optimizer.param_groups for p in group['params']]
        invalid=bool(reduce_max(int(not parameters or any(p.dtype!=torch.float32 or not p.requires_grad for p in parameters)
            or len({id(p) for p in parameters})!=len(parameters)
            or len({id(p) for p in group_parameters})!=len(group_parameters)
            or {id(p) for p in group_parameters}!={id(p) for p in parameters})))
        if invalid:raise ValueError('All-rank optimizer masters must be nonempty FP32')
        failed=initial_failed
        for p in parameters:
            if p.grad is None:p.grad=torch.zeros_like(p)
            reduce_grad(p.grad)
            p.grad.div_(self.scale)
        failed=bool(reduce_max(int(failed or any(not torch.isfinite(p.grad).all().item() for p in parameters))))
        norm=None
        if not failed:
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            failed=bool(reduce_max(int(not torch.isfinite(norm).item() or any(not torch.isfinite(p.grad).all().item() for p in parameters))))
        if failed:
            if self.enabled and not initial_failed:self.scale*=self.backoff_factor
            self.growth_tracker=0;self._validate()
            optimizer.zero_grad(set_to_none=True)
            return dict(accepted=False,overflow=not initial_failed,failure_kind='forward_or_logging_failure' if initial_failed else 'gradient_overflow',adam_advanced=False,cursor_advanced=False,
                scaler_before=before,scaler_after=self.state_dict(),requires_durable_rng_retry_capsule=not initial_failed,requires_durable_failure_evidence=initial_failed)
        optimizer.step();self.growth_tracker+=1
        if self.enabled and self.growth_tracker==self.growth_interval:
            self.scale*=self.growth_factor;self.growth_tracker=0
        elif not self.enabled:self.growth_tracker=0
        self._validate()
        return dict(accepted=True,overflow=False,adam_advanced=True,grad_norm=float(norm),scaler_after=self.state_dict())

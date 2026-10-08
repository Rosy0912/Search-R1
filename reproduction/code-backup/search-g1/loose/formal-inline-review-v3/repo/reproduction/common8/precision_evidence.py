"""Disposable smoke hooks: actual forward precision, never model output changes.

FP32 tensor evidence and disabled Torch TF32 flags do not certify IEEE arithmetic
inside every attention kernel. CUDA backends may still choose FastF32 operators.
"""
import torch


def tensor_dtypes(value):
    if isinstance(value, torch.Tensor):
        return {str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return set().union(*(tensor_dtypes(v) for v in value)) if value else set()
    return set()


class PrecisionObservation:
    def __init__(self, actor, reference):
        self.handles = []
        self.roles = {}
        self.model = actor.rollout.inference_engine.llm_engine.model_executor.worker.model_runner.model
        self.original_logits = self.model.compute_logits
        self.rollout = dict(forward_calls=0, parameter_dtypes=set(), hidden_dtypes=set(),
                            cache_dtypes=set(), cache_tensor_count=0, logit_dtypes=set(),
                            logits_calls=0, autocast_states=set(), attention_backend=None)
        for name, worker in (('actor', actor), ('reference', reference)):
            module = worker.actor_module_fsdp if name == 'actor' else worker.ref_module_fsdp
            row = self.roles[name] = dict(forward_calls=0, logit_dtypes=set(),
                activation_dtypes=set(), autocast_states=set(), parameter_dtypes=set(),
                compute_parameter_dtypes=set(), buffer_dtypes=set(), mixed_parameter_dtype=str(module.mixed_precision.param_dtype),
                mixed_reduce_dtype=str(module.mixed_precision.reduce_dtype),
                mixed_buffer_dtype=str(module.mixed_precision.buffer_dtype))
            def before(mod, args, row=row):
                row['autocast_states'].add(bool(torch.is_autocast_enabled()))
                row['parameter_dtypes'].update(str(p.dtype) for p in mod.parameters())
                row['buffer_dtypes'].update(str(p.dtype) for p in mod.buffers())
            def after(mod, args, output, row=row):
                row['forward_calls'] += 1
                row['logit_dtypes'].update(tensor_dtypes(getattr(output, 'logits', None)))
            def activation(mod, args, output, row=row):
                row['activation_dtypes'].update(tensor_dtypes(output))
            def compute_parameters(mod, args, row=row):
                # Inner Linear hooks run after its enclosing FSDP unshard/cast,
                # unlike the outer storage/pre-FSDP parameter observation above.
                # use_orig_params=False exposes live weights as Tensor views,
                # not registered Parameters during the FSDP computation.
                row['compute_parameter_dtypes'].update(str(p.dtype) for p in
                    (getattr(mod, 'weight', None), getattr(mod, 'bias', None)) if isinstance(p, torch.Tensor))
            self.handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after),
                module.module.get_input_embeddings().register_forward_hook(activation)]
            self.handles += [child.register_forward_pre_hook(compute_parameters)
                             for child in module.modules() if isinstance(child, torch.nn.Linear)]

        def rollout_before(mod, args, kwargs):
            row = self.rollout
            row['autocast_states'].add(bool(torch.is_autocast_enabled()))
            row['parameter_dtypes'].update(str(p.dtype) for p in mod.parameters())
            caches = kwargs.get('kv_caches', args[2] if len(args) > 2 else None)
            row['cache_dtypes'].update(tensor_dtypes(caches))
            row['cache_tensor_count'] += sum(isinstance(v, torch.Tensor) for v in caches) if isinstance(caches, (list, tuple)) else 0
            try:
                impl = mod.model.layers[0].self_attn.attn.impl
                row['attention_backend'] = type(impl).__module__ + '.' + type(impl).__name__
            except (AttributeError, IndexError):
                row['attention_backend'] = None

        def rollout_after(mod, args, output):
            self.rollout['forward_calls'] += 1
            self.rollout['hidden_dtypes'].update(tensor_dtypes(output))

        def logits(*args, **kwargs):
            output = self.original_logits(*args, **kwargs)
            self.rollout['logits_calls'] += 1
            self.rollout['logit_dtypes'].update(tensor_dtypes(output))
            return output

        self.handles += [self.model.register_forward_pre_hook(rollout_before, with_kwargs=True),
                         self.model.register_forward_hook(rollout_after)]
        self.model.compute_logits = logits

    def snapshot(self):
        def serial(row):
            return {k: sorted(v) if isinstance(v, set) else v for k, v in row.items()}
        return dict(roles={k: serial(v) for k, v in self.roles.items()}, rollout=serial(self.rollout),
                    torch_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                    torch_cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                    semantics='Actual observed tensor dtypes and Torch flags; not proof of all kernel internal arithmetic')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model.compute_logits = self.original_logits


def verify_precision(proof):
    assert proof['torch_matmul_allow_tf32'] is False and proof['torch_cudnn_allow_tf32'] is False
    for role in ('actor', 'reference'):
        row = proof['roles'][role]
        assert row['forward_calls'] > 0 and row['autocast_states'] == [False]
        for name in ('parameter_dtypes', 'compute_parameter_dtypes', 'buffer_dtypes', 'activation_dtypes', 'logit_dtypes'):
            assert row[name] == ['torch.float32'], (role, name, row[name])
        if role == 'actor':
            assert all(row[name] == 'torch.float32' for name in
                ('mixed_parameter_dtype', 'mixed_reduce_dtype', 'mixed_buffer_dtype'))
        else:
            assert all(row[name] == 'None' for name in
                ('mixed_parameter_dtype', 'mixed_reduce_dtype', 'mixed_buffer_dtype'))
    row = proof['rollout']
    assert row['forward_calls'] > 0 and row['logits_calls'] > 0 and row['cache_tensor_count'] > 0
    assert row['autocast_states'] == [False]
    for name in ('parameter_dtypes', 'hidden_dtypes', 'cache_dtypes', 'logit_dtypes'):
        assert row[name] == ['torch.float32'], (name, row[name])
    assert isinstance(row['attention_backend'], str) and row['attention_backend']
    return True

"""Actual bound Torch worker recorder backend; no model load or forward.

Per-forward observation is checked with live RNG, tensor aliases, placement and
version counters. Full parameter/internal-buffer/Adam values are additionally
hashed at complete A/B and rejection boundaries by the inherited actual worker
API. A per-forward version check is not mislabeled a full tensor-value proof.
"""
import torch
from calibration_state_v1 import state_digest
from search_r1.baselines.recovery import cpu_snapshot, rng_state


class ExistingForwardTensorBackend:
    def __init__(self, worker):
        self.worker = worker

    def snapshot_micro_tensors(self, micro):
        return {key: cpu_snapshot(value) for key, value in micro.items()}

    def snapshot_probability_tensors(self, entropy, current):
        return dict(entropy=cpu_snapshot(entropy), current_log_prob=cpu_snapshot(current))

    def micro_digest(self, micro):
        return state_digest(dict(micro.items()))

    def capture_observer_read_state(self):
        module, optimizer = self.worker.actor_module_fsdp, self.worker.actor_optimizer
        def identity(tensor):
            return dict(python_id=id(tensor), version=tensor._version,
                        shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                        stride=list(tensor.stride()), storage_offset=tensor.storage_offset())
        names = {id(parameter): name for name, parameter in module.named_parameters()}
        manager = self.worker.rollout_sharding_manager
        observation = dict(
            scope='LIVE_RNG_ALIASES_VERSIONS_LAYOUT_NOT_PER_FORWARD_FULL_VALUE_HASH',
            model={name: identity(value) for name, value in module.named_parameters()},
            internal={name: identity(value._local_shard) for name, value in module.named_parameters()
                      if hasattr(value, '_local_shard')},
            buffers={name: identity(value) for name, value in module.named_buffers()},
            gradients={name: identity(value.grad) for name, value in module.named_parameters()
                       if value.grad is not None},
            optimizer={names[id(parameter)]: {key: identity(value) if isinstance(value, torch.Tensor)
                    else cpu_snapshot(value) for key, value in slot.items()}
                    for parameter, slot in optimizer.state.items()},
            groups=[dict(names=[names[id(parameter)] for parameter in group['params']],
                         controls=cpu_snapshot({key: value for key, value in group.items() if key != 'params'}))
                    for group in optimizer.param_groups],
            modes={name: value.training for name, value in module.named_modules()},
            scheduler=cpu_snapshot(self.worker.actor_lr_scheduler.state_dict()),
            rng=cpu_snapshot(rng_state()),
            sharding_rng={name: cpu_snapshot(getattr(manager, name, None))
                          for name in ('gen_random_states', 'torch_random_states')})
        return state_digest(observation)


def actual_adam_step(worker):
    steps = [float(slot['step'].item()) for slot in worker.actor_optimizer.state.values()
             if 'step' in slot]
    if not steps or len(set(steps)) != 1 or steps[0] != int(steps[0]):
        raise ValueError('Missing/inconsistent actual Adam slots')
    return int(steps[0])

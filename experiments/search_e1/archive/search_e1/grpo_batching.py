"""Batch trajectories while preserving the microbatch=1 GRPO loss reduction.

Upstream averages all tokens inside each microbatch. For microbatch=1 this is
an equal average over trajectories, so batching must retain that reduction.
Only the actor update's scoped reductions are replaced; advantage estimation,
clipping, optimizer steps, masking and reference KL remain upstream code.
"""
from types import FunctionType, SimpleNamespace
from verl.trainer.ppo import core_algos
import verl.utils.torch_functional as functional


def trajectory_mean(values, mask, axis=None):
    if axis is not None:
        return functional.masked_mean(values, mask, axis=axis)
    return functional.masked_mean(values, mask, axis=-1).mean()


def scoped_function(function, **overrides):
    namespace = dict(function.__globals__, **overrides)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


scoped_functional = SimpleNamespace(**dict(vars(functional), masked_mean=trajectory_mean))
trajectory_policy_loss = scoped_function(core_algos.compute_policy_loss, verl_F=scoped_functional)


def install():
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    if getattr(DataParallelPPOActor, '_search_e1_trajectory_reduction', False):
        return
    algorithms = SimpleNamespace(**dict(vars(core_algos), compute_policy_loss=trajectory_policy_loss))
    DataParallelPPOActor.update_policy = scoped_function(
        DataParallelPPOActor.update_policy, core_algos=algorithms,
        verl_F=scoped_functional, masked_mean=trajectory_mean)
    DataParallelPPOActor._search_e1_trajectory_reduction = True

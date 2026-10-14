"""Rank-local Adam transactions; never materialize a full FSDP model."""
import copy
import random

import torch


def _snapshot(value):
    if torch.is_tensor(value):
        return (value.detach().to('cpu', copy=True), value.device)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return copy.deepcopy(value)


def _restore(value):
    if isinstance(value, tuple) and len(value) == 2 and torch.is_tensor(value[0]):
        return value[0].to(value[1], copy=True)
    if isinstance(value, dict):
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return copy.deepcopy(value)


class LocalOptimizerTransaction:
    """Capture only the already-local parameter shards and Adam state.

    Capture after backward, immediately before Adam. No all_gather, disk write,
    or additional model forward is performed. The object is released after
    each accepted write. In particular an empty initial Adam state stays empty
    after rollback of the first rejected write.
    """
    def __init__(self, module, optimizer):
        self.optimizer = optimizer
        parameters = list(dict.fromkeys(module.parameters()))
        self.parameters = [(parameter, parameter.detach().to('cpu', copy=True))
                           for parameter in parameters]
        self.state = {parameter: _snapshot(value) for parameter, value in optimizer.state.items()}
        self.groups = [{key: copy.deepcopy(value) for key, value in group.items() if key != 'params'}
                       for group in optimizer.param_groups]
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.cuda_rng = torch.cuda.get_rng_state(self.cuda_device) if self.cuda_device is not None else None
        self.python_rng = random.getstate()
        try:
            import numpy as np
            self.numpy_rng = np.random.get_state()
        except ImportError:
            self.numpy_rng = None

    @torch.no_grad()
    def rollback(self):
        for parameter, saved in self.parameters:
            if parameter.shape != saved.shape:
                raise RuntimeError('FSDP local parameter shape changed during optimizer transaction')
            parameter.copy_(saved)
        self.optimizer.state.clear()
        self.optimizer.state.update({parameter: _restore(value) for parameter, value in self.state.items()})
        for group, saved in zip(self.optimizer.param_groups, self.groups):
            parameters = group['params']
            group.clear(); group.update(copy.deepcopy(saved)); group['params'] = parameters
        self.optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state(self.cuda_rng, self.cuda_device)
        random.setstate(self.python_rng)
        if self.numpy_rng is not None:
            import numpy as np
            np.random.set_state(self.numpy_rng)

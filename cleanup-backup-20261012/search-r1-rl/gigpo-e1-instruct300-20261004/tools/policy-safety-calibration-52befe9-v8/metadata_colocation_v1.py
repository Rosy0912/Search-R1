"""External compatibility for observational subclasses and frozen Ray colocation.

Frozen factory assumes each user class directly inherits Worker. An observation
subclass has ActorRolloutRefWorker as its immediate base; choosing that base for
outer WorkerDict wrongly calls its(config, role) constructor with no arguments.
Only the outer base choice changes here. Actual inner MRO/constructors, original
DISABLE_WORKER_INIT, method binding, RayClassWithInitArgs and dispatch stay intact.
"""
from contextlib import contextmanager


def metadata_colocated_worker_cls(class_dict):
    import os
    from unittest.mock import patch
    import ray
    from verl.single_controller.base import Worker
    from verl.single_controller.ray.base import (
        _unwrap_ray_remote, _bind_workers_method_to_parent, RayClassWithInitArgs)
    if not class_dict:
        raise ValueError('No actual colocated roles')
    classes, controls, common_base = {}, {}, None
    for role, description in class_dict.items():
        user_class = _unwrap_ray_remote(description.cls)
        if not isinstance(user_class, type) or not issubclass(user_class, Worker):
            raise TypeError('Observational roles must retain the genuine Worker MRO')
        if common_base is None:
            common_base = user_class.__base__
        elif common_base is not user_class.__base__:
            raise ValueError('The original same-base colocation constraint must remain true')
        classes[role] = description.cls
        controls[role] = {'args': description.args, 'kwargs': description.kwargs}

    class WorkerDict(Worker):
        def __init__(self):
            # Retain the genuine outer rank/world/master metadata constructor.
            # Do not substitute a default actor config/role or omit super init.
            super().__init__()
            self.worker_dict = {}
            for role, user_class in classes.items():
                actual_class = _unwrap_ray_remote(user_class)
                with patch.dict(os.environ, {'DISABLE_WORKER_INIT': '1'}):
                    self.worker_dict[role] = actual_class(*controls[role].get('args', ()),
                                                        **controls[role].get('kwargs', {}))

    for role, user_class in classes.items():
        _bind_workers_method_to_parent(WorkerDict, role, _unwrap_ray_remote(user_class))
    return RayClassWithInitArgs(cls=ray.remote(WorkerDict))


@contextmanager
def scoped_metadata_colocation():
    """Patch only this driver's imported factory while creating real workers."""
    import verl.trainer.ppo.ray_trainer as trainer_module
    from verl.single_controller.ray.base import create_colocated_worker_cls as frozen
    if trainer_module.create_colocated_worker_cls is not frozen:
        raise ValueError('Unexpected existing trainer colocation factory; refuse nested/unknown patch')
    trainer_module.create_colocated_worker_cls = metadata_colocated_worker_cls
    try:
        yield
    finally:
        trainer_module.create_colocated_worker_cls = frozen

"""CPU-only probe of actual source/Ray class construction; no Ray cluster/GPU.

Supply the candidate module code through stdin environment in a diagnostic
interpreter. Original source factory and genuine Worker/class-delegation APIs
are exercised. Actual ActorRolloutRefWorker/config/role init is only inspected:
the expected old outer constructor error occurs before NCCL/CUDA/FSDP work.
CPU inner fixtures require(config, role) through the same deeper Worker MRO;
they are not substitutes for future actual eight-GPU CalibrationWorker init.
"""
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types

import ray
import torch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, MAGIC_ATTR, register
from verl.single_controller.ray.base import (
    RayClassWithInitArgs, _unwrap_ray_remote, create_colocated_worker_cls)
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from calibration_worker_v1 import CalibrationWorker

module_bytes = base64.b64decode(os.environ['METADATA_COLOCATION_CANDIDATE_BASE64'])
assert hashlib.sha256(module_bytes).hexdigest() == os.environ['METADATA_COLOCATION_CANDIDATE_SHA256']
module = types.ModuleType('metadata_colocation_cpu_candidate_v7')
exec(compile(module_bytes, 'metadata_colocation_v1.py', 'exec'), module.__dict__)
assert not ray.is_initialized() and not torch.cuda.is_initialized()
assert CalibrationWorker.__base__ is ActorRolloutRefWorker
config_actor, config_ref = {'cpu_identity': 'actor-exact'}, {'cpu_identity': 'ref-exact'}
remote_actual = ray.remote(CalibrationWorker)
actual_descriptions = {
    'actor_rollout': RayClassWithInitArgs(remote_actual, config=config_actor, role='actor_rollout'),
    'ref': RayClassWithInitArgs(remote_actual, config=config_ref, role='ref')}
old = create_colocated_worker_cls(actual_descriptions)
old_outer = _unwrap_ray_remote(old.cls)
assert old_outer.__base__ is ActorRolloutRefWorker
previous = dict(os.environ)
os.environ.update(DISABLE_WORKER_INIT='1', WORLD_SIZE='8', RANK='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='1',
                  CUDA_VISIBLE_DEVICES='')
try:
    try:
        old_outer()
    except TypeError as error:
        original_error = str(error)
        assert 'config' in original_error and 'role' in original_error
    else:
        raise AssertionError('Frozen factory did not reproduce actual3276 constructor error')
    assert not torch.distributed.is_initialized() and not torch.cuda.is_initialized()

    class CPURequiredRole(Worker):
        def __init__(self, config, role):
            super().__init__()
            self.config, self.role = config, role
            self.disable_during_inner_init = os.environ.get('DISABLE_WORKER_INIT')

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def show_role(self):
            return self.config, self.role, self.rank, self.world_size

        @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
        def same_payload(self, data):
            return data

    class CPUObservationalSubclass(CPURequiredRole):
        pass

    remote_cpu = ray.remote(CPUObservationalSubclass)
    descriptions = {
        'actor_rollout': RayClassWithInitArgs(remote_cpu, config_actor, 'actor_rollout'),
        'ref': RayClassWithInitArgs(remote_cpu, config=config_ref, role='ref')}
    actual_fixed = module.metadata_colocated_worker_cls(actual_descriptions)
    assert _unwrap_ray_remote(actual_fixed.cls).__base__ is Worker
    cpu_fixed = module.metadata_colocated_worker_cls(descriptions)
    outer = _unwrap_ray_remote(cpu_fixed.cls)
    assert outer.__base__ is Worker
    os.environ['DISABLE_WORKER_INIT'] = '7'
    instance = outer()
    assert os.environ['DISABLE_WORKER_INIT'] == '7'
    assert instance.rank == 0 and instance.world_size == 8
    assert set(instance.worker_dict) == {'actor_rollout', 'ref'}
    assert type(instance.worker_dict['actor_rollout']) is CPUObservationalSubclass
    assert type(instance.worker_dict['ref']) is CPUObservationalSubclass
    actor = instance.actor_rollout_show_role()
    reference = instance.ref_show_role()
    assert actor[0] is config_actor and reference[0] is config_ref
    assert actor[1:] == ('actor_rollout', 0, 8) and reference[1:] == ('ref', 0, 8)
    for inner in instance.worker_dict.values():
        assert inner.disable_during_inner_init == '1'
    marker = object()
    assert instance.actor_rollout_same_payload(marker) is marker and instance.ref_same_payload(marker) is marker
    for role in ('actor_rollout', 'ref'):
        for name in ('show_role', 'same_payload'):
            assert getattr(getattr(outer, role + '_' + name), MAGIC_ATTR) == getattr(getattr(CPUObservationalSubclass, name), MAGIC_ATTR)
    class CPUFailingRole(CPURequiredRole):
        def __init__(self, config, role):
            super().__init__(config, role)
            assert os.environ['DISABLE_WORKER_INIT'] == '1'
            raise RuntimeError('injected inner constructor failure')

    failing = module.metadata_colocated_worker_cls({'actor_rollout': RayClassWithInitArgs(
        ray.remote(CPUFailingRole), config=config_actor, role='actor_rollout')})
    try:
        _unwrap_ray_remote(failing.cls)()
    except RuntimeError as error:
        assert str(error) == 'injected inner constructor failure'
    else:
        raise AssertionError('Injected inner constructor failure was lost')
    assert os.environ['DISABLE_WORKER_INIT'] == '7'
    try:
        module.metadata_colocated_worker_cls({
            'actor_rollout': descriptions['actor_rollout'],
            'ref': RayClassWithInitArgs(ray.remote(CPURequiredRole), config=config_ref, role='ref')})
    except ValueError:
        pass
    else:
        raise AssertionError('Original unequal immediate-base constraint was weakened')
    import verl.trainer.ppo.ray_trainer as trainer
    original_alias = trainer.create_colocated_worker_cls
    with module.scoped_metadata_colocation():
        assert trainer.create_colocated_worker_cls is module.metadata_colocated_worker_cls
    assert trainer.create_colocated_worker_cls is original_alias
    try:
        with module.scoped_metadata_colocation():
            raise RuntimeError('injected initializer failure')
    except RuntimeError:
        pass
    assert trainer.create_colocated_worker_cls is original_alias
    try:
        with module.scoped_metadata_colocation():
            with module.scoped_metadata_colocation():
                raise AssertionError('Unexpected nested factory admitted')
    except ValueError:
        pass
    assert trainer.create_colocated_worker_cls is original_alias
    assert not ray.is_initialized() and not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    print(json.dumps(dict(status='PASS', scope='ACTUAL_SOURCE_RAY_CPU_CONSTRUCTOR_AND_DISPATCH_ONLY',
        node=socket.gethostname(), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        candidate_sha256=hashlib.sha256(module_bytes).hexdigest(),
        ray_version=ray.__version__, torch_version=torch.__version__,
        frozen_factory_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(create_colocated_worker_cls)).read_bytes()).hexdigest(),
        actual_original_factory_error=original_error,
        actual_old_outer_base=old_outer.__base__.__qualname__, actual_fixed_outer_base=outer.__base__.__qualname__,
        actual_calibration_worker_mro=[row.__qualname__ for row in CalibrationWorker.__mro__],
        original_actor_constructor_signature=str(inspect.signature(ActorRolloutRefWorker.__init__)),
        cpu_roles=['actor_rollout','ref'], constructor_config_identity=True, positional_and_keyword_arguments_preserved=True,
        original_inner_disable_preserved=True, environment_restored_success_and_inner_exception=True,
        original_same_immediate_base_constraint_preserved=True,
        outer_worker_metadata=True, original_role_dispatch_attributes=True, same_payload_identity=True,
        scoped_patch_restored_success_and_exception=True, nested_unknown_factory_rejected=True,
        ray_cluster_created=False, torch_cuda_initialized=False, torch_process_group_initialized=False,
        actual_8gpu_initialization_accepted=False, actual_restore=False, actual_sampling=False, actual_adam_writes=0),
        sort_keys=True),flush=True)
finally:
    os.environ.clear(); os.environ.update(previous)

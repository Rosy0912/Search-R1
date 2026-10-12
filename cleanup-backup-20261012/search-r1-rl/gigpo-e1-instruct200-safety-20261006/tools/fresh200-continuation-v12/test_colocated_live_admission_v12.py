"""UNEXECUTED CPU colocation/config integration, never Ray/FSDP/GPU evidence.

The actual metadata factory, original source normalization if-statements, actual
source admission validator/installer/restorer and checkpoint metadata bodies are
preserved. Only Worker/Ray/GPU placement/model boundaries use CPU substitutes.
Both actual factory descriptions receive the same mutable OmegaConf object.
No per-role fake installer or fake admission validator is used.
"""
import ast
import copy
import hashlib
import json
import math
import os
import random
import re
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf


RUNTIME = Path(__file__).resolve().parent


def source_fixture_root():
    candidates = {Path(entry).resolve() for entry in os.environ.get('PYTHONPATH', '').split(os.pathsep)
                  if entry and (Path(entry) / 'search_r1/baselines/fresh200_live.py').is_file()}
    if len(candidates) == 1:
        return candidates.pop()
    if candidates:
        raise AssertionError('Ambiguous actual source in fixture PYTHONPATH')
    local = RUNTIME.parent.parent / 'gigpo-e1-fresh200-compat11-20261006'
    if (local / 'search_r1/baselines/fresh200_live.py').is_file():
        return local.resolve()
    raise AssertionError('Missing actual source in fixture PYTHONPATH')


class StripImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def selected_functions(path, names, namespace, constants=False, class_name=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = tree.body if class_name is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected, found = [], set()
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            value = copy.deepcopy(node)
            value.decorator_list = []
            selected.append(StripImports().visit(value))
            found.add(node.name)
        elif constants and isinstance(node, ast.Assign):
            selected.append(copy.deepcopy(node))
    if found != names:
        raise AssertionError('Missing exact production fixture bodies: ' + repr(names - found))
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)


def actual_source_namespace(source):
    # CUDA current-device is an explicit placement boundary. No CUDA call occurs.
    cuda_boundary = SimpleNamespace(current_device=lambda: 0, is_initialized=lambda: False)
    torch_boundary = SimpleNamespace(Tensor=torch.Tensor, device=torch.device,
                                    get_rng_state=torch.get_rng_state, cuda=cuda_boundary)
    namespace = dict(copy=copy, hashlib=hashlib, json=json, math=math, os=os,
        stat=stat, Path=Path, OmegaConf=OmegaConf, torch=torch_boundary, re=re, np=np, random=random)
    selected_functions(source / 'search_r1/baselines/fresh200.py',
        {'calibration_initialization', 'validate_attempt_ledger'}, namespace, constants=True)
    selected_functions(source / 'search_r1/baselines/fresh200_live.py', {
        'require', 'digest', 'core', 'read_bound_json', 'existing_record', 'live_initialization',
        'validate_live_initialization', 'validate_live_binding', '_initial_payload',
        'validate_admission', 'validate_config_delta', 'validate_production_review',
        'read_bound_json_lines', 'worker_admission_state', 'install_worker_admission',
        'restore_worker_admission_state'}, namespace, constants=True)
    selected_functions(source / 'search_r1/baselines/recovery.py', {'cpu_snapshot', 'rng_state'}, namespace)
    namespace.update(
        offload_fsdp_param_and_grad=lambda *args, **kwargs: None,
        offload_fsdp_optimizer=lambda *args, **kwargs: None)
    selected_functions(source / 'verl/workers/fsdp_workers.py',
        {'_validate_checkpoint_runtime_placement', '_restore_checkpoint_runtime_metadata'}, namespace)
    return namespace


def synthetic_evidence_class(source, namespace):
    """Reuse only the explicit synthetic evidence builder; no test import."""
    path = source / 'tests/baselines/test_fresh200_live_admission.py'
    fixture_namespace = dict(copy=copy, hashlib=hashlib, json=json, os=os, tempfile=tempfile,
        unittest=unittest, Path=Path, patch=patch, source_namespace=lambda: namespace)
    selected_functions(path, {'fixture_ledger'}, fixture_namespace)
    tree = ast.parse(path.read_text(), filename=str(path))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LiveAdmission')
    names = {'setUp', 'write', 'valid_binding', 'config', 'production_payload'}
    methods = [copy.deepcopy(node) for node in original.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise AssertionError('Incomplete synthetic evidence builder')
    selected = ast.ClassDef(name='SyntheticAdmissionEvidence',
        bases=[ast.Attribute(value=ast.Name(id='unittest', ctx=ast.Load()), attr='TestCase', ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), fixture_namespace)
    original_class = fixture_namespace['SyntheticAdmissionEvidence']

    class EvidenceWithOriginalNormalizerInputs(original_class):
        def config(self):
            value = super().config()
            # Original source multiplies by rollout.n=1; explicit n_agent=5
            # remains the separate question group count used by the trainer.
            value['actor_rollout_ref']['rollout']['n'] = 1
            value['actor_rollout_ref']['actor']['ulysses_sequence_parallel_size'] = 1
            return value

    return EvidenceWithOriginalNormalizerInputs


class WorkerBoundary:
    """CPU substitute for genuine rank/Worker constructor only."""
    def __init__(self):
        self.rank = 0
        self.boundary_init_disable = os.environ.get('DISABLE_WORKER_INIT')


class RayDescriptionBoundary:
    """Ray remote-description boundary; never dispatches a process or RPC."""
    def __init__(self, cls, *args, **kwargs):
        self.cls, self.args, self.kwargs = cls, args, kwargs


def cpu_role_state(worker):
    worker.actor = SimpleNamespace(config=worker.config.actor)
    module = torch.nn.Linear(2, 1, bias=False)
    if worker._is_actor:
        worker.actor_module_fsdp = module
        worker.actor_optimizer = torch.optim.AdamW(module.parameters(), lr=1e-6)
        worker.actor_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(worker.actor_optimizer, lambda step: 1.)
        worker.rollout_sharding_manager = SimpleNamespace(gen_random_states=None, torch_random_states=None)
        worker.fresh200_worker_ledger = dict(schema='FRESH200_WORKER_ACTUAL_ATTEMPTS_V1',
            attempts=0, completed_outer_updates=0, actual_adam_total=0, history=[])
    else:
        worker.ref_module_fsdp = module
    worker._is_offload_param = True
    worker._is_offload_grad = False
    worker._is_offload_optimizer = True


def original_normalization_boundary_class(source, namespace):
    """Keep original role flags and all three normalization if bodies verbatim."""
    path = source / 'verl/workers/fsdp_workers.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    original_class = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name == 'ActorRolloutRefWorker')
    original = next(node for node in original_class.body
                    if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    def self_attribute(node):
        return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == 'self')
    role_names = {'config', 'role', '_is_actor', '_is_rollout', '_is_ref'}
    role_assignments = [copy.deepcopy(node) for node in original.body
        if isinstance(node, ast.Assign) and any(self_attribute(target) and target.attr in role_names
                                               for target in node.targets)]
    normalizers = [copy.deepcopy(node) for node in original.body
        if isinstance(node, ast.If) and self_attribute(node.test)
        and node.test.attr in {'_is_actor', '_is_rollout', '_is_ref'}
        and any(isinstance(child, ast.AugAssign) and isinstance(child.op, ast.FloorDiv)
                for child in ast.walk(node))]
    if len(role_assignments) != 5 or len(normalizers) != 3:
        raise AssertionError('Original role/normalizer AST selection changed')
    boundary_setup = ast.parse('super().__init__()\nself.device_mesh = SimpleNamespace(shape=(8,))\n'
        'self.ulysses_sequence_parallel_size = config.actor.get("ulysses_sequence_parallel_size", 1)\n').body
    body = boundary_setup + role_assignments + normalizers + ast.parse('cpu_role_state(self)').body
    method = ast.FunctionDef(name='__init__',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in ('self', 'config', 'role')],
            kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    metadata = copy.deepcopy(next(node for node in original_class.body
        if isinstance(node, ast.FunctionDef) and node.name == 'checkpoint_runtime_metadata'))
    metadata.decorator_list = []
    selected = ast.ClassDef(name='OriginalNormalizationBoundary',
        bases=[ast.Name(id='WorkerBoundary', ctx=ast.Load())], keywords=[],
        body=[method, StripImports().visit(metadata)], decorator_list=[])
    namespace.update(WorkerBoundary=WorkerBoundary, SimpleNamespace=SimpleNamespace, cpu_role_state=cpu_role_state)
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    original_boundary = namespace['OriginalNormalizationBoundary']

    class ObservationBoundary(original_boundary):
        def __init__(self, config, role, boundary_token=None):
            super().__init__(config, role)
            self.boundary_token = boundary_token

    return ObservationBoundary


def actual_factory_namespace(source):
    namespace = dict(copy=copy, os=os, patch=patch, Worker=WorkerBoundary,
        RayClassWithInitArgs=RayDescriptionBoundary,
        ray=SimpleNamespace(remote=lambda cls: SimpleNamespace(__ray_actor_class__=cls)),
        MAGIC_ATTR='_synthetic_cpu_dispatch_metadata_only')
    selected_functions(source / 'verl/single_controller/ray/base.py',
        {'_unwrap_ray_remote', '_bind_workers_method_to_parent'}, namespace)
    selected_functions(RUNTIME / 'metadata_colocation_v1.py', {'metadata_colocated_worker_cls'}, namespace)
    return namespace


def protected_cpu_state(worker):
    module = worker.actor_module_fsdp if worker._is_actor else worker.ref_module_fsdp
    value = dict(module_id=id(module), model=copy.deepcopy(module.state_dict()),
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state().clone())
    if worker._is_actor:
        value.update(optimizer_id=id(worker.actor_optimizer),
            optimizer=copy.deepcopy(worker.actor_optimizer.state_dict()),
            scheduler_id=id(worker.actor_lr_scheduler),
            scheduler=copy.deepcopy(worker.actor_lr_scheduler.state_dict()),
            ledger=copy.deepcopy(worker.fresh200_worker_ledger))
    return value


class SharedConfigurationAdmission(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture_root()
        self.ns = actual_source_namespace(self.source)
        self.evidence = synthetic_evidence_class(self.source, self.ns)()
        self.evidence.setUp()
        self.addCleanup(self.evidence.doCleanups)
        self.factory = actual_factory_namespace(self.source)
        worker_class = original_normalization_boundary_class(self.source, self.ns)
        self.driver_config = OmegaConf.create(self.evidence.c0_config)
        self.shared = self.driver_config.actor_rollout_ref
        self.driver_before = OmegaConf.to_container(self.driver_config, resolve=True)
        self.token = object()
        ray_class = SimpleNamespace(__ray_actor_class__=worker_class)
        self.descriptions = {
            'actor_rollout': RayDescriptionBoundary(ray_class, config=self.shared, role='actor_rollout', boundary_token=self.token),
            'ref': RayDescriptionBoundary(ray_class, config=self.shared, role='ref', boundary_token=self.token),
        }
        self.assertIs(self.descriptions['actor_rollout'].kwargs['config'], self.descriptions['ref'].kwargs['config'])
        description = self.factory['metadata_colocated_worker_cls'](self.descriptions)
        self.outer = description.cls.__ray_actor_class__()
        self.actor, self.reference = self.outer.worker_dict['actor_rollout'], self.outer.worker_dict['ref']
        self.c0_actor = self.ns['worker_admission_state'](self.actor)
        self.c0_reference = self.ns['worker_admission_state'](self.reference)

    def assert_cpu_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            self.assertTrue(np.array_equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_cpu_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for before, after in zip(left, right):
                self.assert_cpu_equal(before, after)
        else:
            self.assertEqual(left, right)

    def assert_driver_global_config_unchanged(self):
        self.assertEqual(OmegaConf.to_container(self.driver_config, resolve=True), self.driver_before)
        for description in self.descriptions.values():
            self.assertIs(description.kwargs['config'], self.shared)
            self.assertIs(description.kwargs['boundary_token'], self.token)

    def test_real_factory_shared_config_keeps_original_role_micro_normalization(self):
        self.assertIsNot(self.actor.config, self.reference.config)
        self.assertIsNot(self.actor.config, self.shared)
        self.assertIsNot(self.reference.config, self.shared)
        self.assertEqual(self.actor.config.actor.ppo_mini_batch_size, 16)
        self.assertEqual(self.actor.config.actor.ppo_micro_batch_size, 2)
        self.assertEqual(self.actor.config.rollout.log_prob_micro_batch_size, 4)
        self.assertEqual(self.reference.config.ref.log_prob_micro_batch_size, 4)
        self.assertEqual(self.reference.config.actor.ppo_mini_batch_size, 128)
        self.assertEqual(self.reference.config.actor.ppo_micro_batch_size, 16)
        self.assertIs(self.actor.boundary_token, self.token)
        self.assertIs(self.reference.boundary_token, self.token)
        self.assertEqual(self.actor.boundary_init_disable, '1')
        self.assertEqual(self.reference.boundary_init_disable, '1')
        self.assert_driver_global_config_unchanged()

    def test_actual_installer_actor_then_reference_uses_one_shared_process_environment(self):
        before_actor = protected_cpu_state(self.actor)
        before_ref = protected_cpu_state(self.reference)
        actor = self.ns['install_worker_admission'](self.actor, self.evidence.c1)
        # Actor switched this process environment to C1. Reference policy remains
        # genuine C0 until its own actual installer sees the old/new env allowance.
        self.assertEqual(os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'], self.evidence.c1['config_record']['sha256'])
        self.assertEqual(OmegaConf.to_container(self.reference.config.actor.policy_safety, resolve=True), self.evidence.c0['policy'])
        self.assertEqual(self.reference.fresh200_live_admission, self.evidence.c0)
        reference = self.ns['install_worker_admission'](self.reference, self.evidence.c1)
        self.assertEqual(actor['payload'], reference['payload'])
        self.assertEqual(actor['payload']['config_epoch'], 1)
        self.assertEqual(self.ns['worker_admission_state'](self.actor), actor)
        self.assertEqual(self.ns['worker_admission_state'](self.reference), reference)
        self.assert_cpu_equal(protected_cpu_state(self.actor), before_actor)
        self.assert_cpu_equal(protected_cpu_state(self.reference), before_ref)
        self.assert_driver_global_config_unchanged()

    def test_actual_metadata_and_both_role_admission_restore_roundtrip(self):
        actor = self.ns['install_worker_admission'](self.actor, self.evidence.c1)
        reference = self.ns['install_worker_admission'](self.reference, self.evidence.c1)
        metadata = self.actor.checkpoint_runtime_metadata()
        before_actor = protected_cpu_state(self.actor)
        before_ref = protected_cpu_state(self.reference)
        self.ns['restore_worker_admission_state'](self.actor, self.c0_actor)
        self.ns['restore_worker_admission_state'](self.reference, self.c0_reference)
        self.assertEqual(self.ns['worker_admission_state'](self.actor), self.c0_actor)
        self.assertEqual(self.ns['worker_admission_state'](self.reference), self.c0_reference)
        self.ns['_restore_checkpoint_runtime_metadata'](self.actor, metadata)
        self.ns['restore_worker_admission_state'](self.reference, reference)
        self.assertEqual(self.ns['worker_admission_state'](self.actor), actor)
        self.assertEqual(self.ns['worker_admission_state'](self.reference), reference)
        self.assert_cpu_equal(self.actor.checkpoint_runtime_metadata(), metadata)
        self.assert_cpu_equal(protected_cpu_state(self.actor), before_actor)
        self.assert_cpu_equal(protected_cpu_state(self.reference), before_ref)
        self.assert_driver_global_config_unchanged()

    def test_real_factory_refuses_missing_explicit_configuration(self):
        bad = {'actor_rollout': RayDescriptionBoundary(self.descriptions['actor_rollout'].cls, role='actor_rollout')}
        description = self.factory['metadata_colocated_worker_cls'](bad)
        with self.assertRaisesRegex(ValueError, 'explicit config'):
            description.cls.__ray_actor_class__()

"""Affected actual actor/worker AST and real two-rank Gloo/Adam CPU checks.

Synthetic tensors and owned temporary files only: no HF/FSDP/NCCL/Ray proof.
"""
import ast
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import test_tree_behavior_mu_actor_cpu as fixture
from search_r1.baselines import partial_worker_state as source
from search_r1.baselines import policy_safety_runtime as safety_source
from search_r1.baselines.training_state import atomic_save, digest

ROOT = Path(__file__).resolve().parent
# The actual actor refers to __file__ when constructing TreeRankAudit. Only the
# AST execution namespace needs this original filename; production already has it.
fixture.actor_scope['__file__'] = str(ROOT/'verl/workers/actor/dp_actor.py')


def load_scheduler():
    tree = ast.parse((ROOT/'verl/workers/fsdp_workers.py').read_bytes())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == 'fresh200_scheduler_after_actor']
    assert len(nodes) == 1
    scope = dict(torch=torch, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-fresh-scheduler', 'exec'), scope)
    return scope['fresh200_scheduler_after_actor']


SCHEDULE = load_scheduler()


def worker_and_data(factory, run):
    cfg = fixture.config()
    cfg['fresh200_protocol'] = 'tree_fresh200_five_leaf_v1'
    actor = fixture.Actor(cfg)
    data = fixture.make_data(factory, cfg)
    data.meta_info['baseline_outer_step'] = 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(actor.actor_optimizer, lambda step: min(step/57, 1.))
    worker = N(actor=actor, actor_optimizer=actor.actor_optimizer, actor_lr_scheduler=scheduler,
               config=fixture.Config(actor=cfg), _is_offload_param=False, _is_offload_optimizer=False)
    identity = {key: os.environ[value] for key, value in source.ENV_KEYS.items()}
    identity.update(training_protocol='tree_fresh200_five_leaf_v1', attempted_outer=1, completed_outer=0)
    data.meta_info['tree_partial_context'] = dict(protocol=source.PROTOCOL, identity=identity,
        directory=str(Path(run)/'training_state/partial_outer_000001'))
    return worker, data


def read_saved(receipt):
    path = Path(receipt['path'])
    assert path.stat().st_size == receipt['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
    # Only our tiny synthetic CPU state; this is not a restore or real HF reload.
    return torch.load(path, map_location='cpu', weights_only=False)


class Audit:
    def __init__(self, *args): pass
    def epoch(self): pass
    def micro(self, *args, **kwargs): pass
    def gradient(self, *args):
        if dist.get_rank() == 1: raise OSError('synthetic rank1 post-write audit IO failure')
    def finish(self, *args): pass
    def failure(self, *args): pass


def gloo_worker(rank, rendezvous, run, mode):
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    factory = fixture.collector_fixture.BehaviorMuDataTests(); factory.setUp()
    try:
        env = dict(SEARCH_R1_BASELINE_OUTPUT=str(Path(run)/'output'),
                   SEARCH_R1_TREE_RANK_AUDIT_DIR=str(Path(run)/'events'))
        with patch.dict(os.environ, env):
            worker, data = worker_and_data(factory, run)
            source.prepare_partial_worker_context(worker, data)
            original_dump = safety_source.json.dump
            def fail_publish(record, stream, *args, **kwargs):
                if mode == 'publish_io' and rank == 1 and record.get('phase') == 'optimizer_committed':
                    raise OSError('synthetic rank1 publish IO after successful optimizer return')
                return original_dump(record, stream, *args, **kwargs)
            if mode == 'pre_guard' and rank == 1: data.batch['advantages'][0, 0] = float('nan')
            if mode == 'context' and rank == 1: data.meta_info['tree_partial_context']['identity']['run_id'] = 'wrong'
            with patch.object(safety_source.json, 'dump', fail_publish), \
                 patch.object(fixture.rank_audit, 'enabled', return_value=mode == 'post_guard'), \
                 patch.object(fixture.rank_audit, 'TreeRankAudit', Audit):
                try:
                    if mode in ('disk', 'nonfinite', 'context'):
                        worker.actor._actual_committed_optimizer_calls = 0
                        worker.actor.actor_module(torch.ones(1, 1)).sum().backward()
                        worker.actor._optimizer_step()
                        if mode == 'nonfinite' and rank == 1:
                            worker.actor.actor_module.weight.data.fill_(float('nan'))
                        raise safety_source.PolicySafetyRefusal('synthetic matched brake', matched_all_ranks=True)
                    worker.actor.update_policy(data)
                    raise AssertionError('All ranks must refuse this scenario')
                except safety_source.PolicySafetyRefusal as refusal:
                    assert refusal.matched_all_ranks is True
                    before_count = worker.actor._actual_committed_optimizer_calls
                    original_atomic = source.atomic_save
                    def fail_disk(value, path, **kwargs):
                        if mode == 'disk' and rank == 1: raise OSError('synthetic rank1 snapshot disk failure')
                        return original_atomic(value, path, **kwargs)
                    with patch.object(source, 'atomic_save', fail_disk):
                        envelope = source.persist_matched_partial_worker(worker, data, refusal, 0)
                    # Every rank gets byte-identical all-gather metadata before returning.
                    Path(run, f'{rank}.json').write_text(json.dumps(envelope, sort_keys=True, allow_nan=False))
                    assert envelope['outer_completed'] is False and envelope['automatic_rollback'] is False
                    expected = 0 if mode == 'pre_guard' else 1
                    assert before_count == expected
                    assert worker.actor_lr_scheduler.last_epoch == 0
                    own = envelope['worker_states'][rank]
                    if mode not in ('context',) and not (mode == 'disk' and rank == 1):
                        state = read_saved(own)
                        assert state['committed_optimizer_calls'] == expected
                        steps = [int(value['step'].item()) for value in state['optimizer']['state'].values()]
                        assert steps == ([] if expected == 0 else [1])
                        assert state['resume_permitted'] is False
                        assert state['scheduler']['last_epoch'] == 0
                        assert state['gradients']['weight'] is not None if expected else state['gradients']['weight'] is None
                        assert digest(state) == own['logical_state_sha256']
    finally:
        factory.tearDown(); dist.destroy_process_group()


class FakeTimer:
    def __init__(self, *args, **kwargs): self.last = 1.
    def __enter__(self): return self
    def __exit__(self, *args): return False


class WorkerProto:
    def __init__(self, meta_info): self.meta_info = meta_info
    def to(self, device): return self


def load_actual_update_actor(scope):
    tree = ast.parse((ROOT/'verl/workers/fsdp_workers.py').read_bytes())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ActorRolloutRefWorker')
    method = copy.deepcopy(next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'update_actor'))
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'actual-fsdp-worker-update-actor', 'exec'), scope)
    return scope['update_actor']


class Tests(unittest.TestCase):
    def setUp(self):
        self.factory = fixture.collector_fixture.BehaviorMuDataTests(); self.factory.setUp()
        self.tmp = tempfile.TemporaryDirectory(); self.run = Path(self.tmp.name)/'run'; self.run.mkdir()
        self.env = patch.dict(os.environ, dict(SEARCH_R1_BASELINE_OUTPUT=str(self.run/'output'),
            SEARCH_R1_TREE_RANK_AUDIT_DIR=str(self.run/'events'))); self.env.start()
    def tearDown(self): self.env.stop(); self.tmp.cleanup(); self.factory.tearDown()

    def test_two_rank_real_actor_guard_write_publish_disk_and_nonfinite(self):
        modes = ('publish_io', 'post_guard', 'pre_guard', 'disk', 'nonfinite', 'context')
        requested = os.environ.get('TREE_PARTIAL_CPU_FIXTURE_MODE')
        if requested is not None:
            assert requested in modes
            modes = (requested,)
        for mode in modes:
            with self.subTest(mode=mode):
                run = self.run/mode; run.mkdir()
                mp.spawn(gloo_worker, args=(str(run/'gloo-init'), str(run), mode), nprocs=2, join=True)
                first, second = (run/'0.json').read_bytes(), (run/'1.json').read_bytes()
                self.assertEqual(first, second)
                envelope = json.loads(first)
                evidence = os.environ.get('TREE_PARTIAL_CPU_PROOF_DIRECTORY')
                if evidence:
                    proof = Path(evidence); proof.mkdir(parents=True, exist_ok=True)
                    (proof/f'gloo-{mode}.json').write_text(json.dumps(dict(
                        scope='Real two-rank Gloo with tiny CPU Adam; temporary synthetic PT read only and removed after tests',
                        rank_envelopes_identical=(first == second), envelope=envelope), sort_keys=True, allow_nan=False))
                self.assertEqual(envelope['status'], 'PARTIAL_WORKER_STATE_INCOMPLETE' if mode in ('disk', 'context') else 'PARTIAL_WORKER_STATE_SAVED')
                self.assertEqual([r['committed_optimizer_calls'] for r in envelope['worker_states']], [0, 0] if mode == 'pre_guard' else [1, 1])
                if mode == 'nonfinite': self.assertFalse(envelope['worker_states'][1]['state_finite'])
                if mode == 'disk': self.assertEqual(envelope['worker_states'][0]['status'], 'PARTIAL_WORKER_STATE_SAVED')

    def test_actual_counter_precedes_publish_and_does_not_credit_raising_optimizer(self):
        worker, data = worker_and_data(self.factory, self.run)
        worker.actor._actual_committed_optimizer_calls = 0
        # Later nonzero-LR state must also preserve real changed parameters.
        worker.actor_optimizer.param_groups[0]['lr'] = .01
        initial = worker.actor.actor_module.weight.detach().clone()
        worker.actor.actor_module(torch.ones(1, 1)).sum().backward()
        worker.actor._policy_safety = N(committed=0, sync_valid=lambda *a: None,
            publish=lambda *a: (_ for _ in ()).throw(safety_source.PolicySafetyRefusal('after-write', matched_all_ranks=True)))
        with self.assertRaises(safety_source.PolicySafetyRefusal): worker.actor._optimizer_step()
        self.assertEqual(worker.actor._actual_committed_optimizer_calls, 1)
        self.assertEqual(int(next(iter(worker.actor_optimizer.state.values()))['step'].item()), 1)
        self.assertFalse(torch.equal(initial, worker.actor.actor_module.weight))
        result = source.persist_matched_partial_worker(worker, data,
            safety_source.PolicySafetyRefusal('after-write persisted', matched_all_ranks=True), 0)
        self.assertTrue(torch.equal(read_saved(result['worker_states'][0])['parameters']['weight'], worker.actor.actor_module.weight))
        with patch.object(worker.actor_optimizer, 'step', side_effect=RuntimeError('before successful return')):
            with self.assertRaises(RuntimeError): worker.actor._optimizer_step()
        self.assertEqual(worker.actor._actual_committed_optimizer_calls, 1)

    def test_unmatched_refusal_never_enters_any_collective_or_model_capture(self):
        worker, data = worker_and_data(self.factory, self.run)
        for error in (safety_source.PolicySafetyRefusal('manual'), RuntimeError('rank-local')):
            with self.subTest(error=error), patch.object(source, '_gather', side_effect=AssertionError('must not gather')) as gather, \
                 patch.object(source, 'raw_partial_worker_snapshot', side_effect=AssertionError('must not snapshot')) as capture:
                with self.assertRaises(ValueError): source.persist_matched_partial_worker(worker, data, error, 0)
                gather.assert_not_called(); capture.assert_not_called()

    def test_nonfinite_adam_gradient_local_shard_preserved_and_rng_not_changed(self):
        worker, data = worker_and_data(self.factory, self.run)
        worker.actor._actual_committed_optimizer_calls = 0
        parameter = worker.actor.actor_module.weight
        worker.actor.actor_module.register_buffer('synthetic_buffer', torch.tensor([2.]))
        parameter._local_shard = parameter.detach().clone()
        parameter.grad = torch.ones_like(parameter); worker.actor._optimizer_step()
        next(iter(worker.actor_optimizer.state.values()))['exp_avg'].fill_(float('nan'))
        parameter.grad.fill_(float('inf'))
        before = torch.get_rng_state().clone()
        envelope = source.persist_matched_partial_worker(worker, data,
            safety_source.PolicySafetyRefusal('nonfinite actual Adam', matched_all_ranks=True), 0)
        own = envelope['worker_states'][0]; state = read_saved(own)
        self.assertEqual(envelope['status'], 'PARTIAL_WORKER_STATE_SAVED'); self.assertFalse(own['state_finite'])
        self.assertTrue(torch.isnan(next(iter(state['optimizer']['state'].values()))['exp_avg']).all())
        self.assertTrue(torch.isinf(state['gradients']['weight']).all())
        self.assertEqual(set(state['local_shards']), {'weight'}); self.assertFalse(state['resume_permitted'])
        self.assertTrue(torch.equal(state['buffers']['synthetic_buffer'], torch.tensor([2.])))
        self.assertTrue(torch.equal(before, torch.get_rng_state())); self.assertTrue(torch.equal(before, state['rng']['torch']))

    def test_snapshot_noclobber_keeps_original_bytes_and_reports_incomplete(self):
        worker, data = worker_and_data(self.factory, self.run); worker.actor._actual_committed_optimizer_calls = 0
        error = safety_source.PolicySafetyRefusal('before write', matched_all_ranks=True)
        first = source.persist_matched_partial_worker(worker, data, error, 0); own = first['worker_states'][0]
        second = source.persist_matched_partial_worker(worker, data, error, 0)
        self.assertEqual(second['status'], 'PARTIAL_WORKER_STATE_INCOMPLETE')
        self.assertEqual(hashlib.sha256(Path(own['path']).read_bytes()).hexdigest(), own['sha256'])

    def test_preparation_invalid_context_is_matched_zero_write_and_incomplete(self):
        worker, data = worker_and_data(self.factory, self.run); worker.actor._actual_committed_optimizer_calls = 0
        data.meta_info['tree_partial_context']['directory'] = '/unrelated/training_state/partial_outer_000001'
        with self.assertRaises(safety_source.PolicySafetyRefusal) as caught: source.prepare_partial_worker_context(worker, data)
        self.assertTrue(caught.exception.matched_all_ranks)
        result = source.persist_matched_partial_worker(worker, data, caught.exception, 0)
        self.assertEqual(result['status'], 'PARTIAL_WORKER_STATE_INCOMPLETE')
        self.assertEqual(result['worker_states'][0]['committed_optimizer_calls'], 0)
        self.assertEqual(list(self.run.rglob('worker-*.pt')), [])

    def test_real_scheduler_advance_io_refusal_saved_without_outer_credit(self):
        worker, data = worker_and_data(self.factory, self.run); worker.actor._actual_committed_optimizer_calls = 0
        for _ in range(20):
            worker.actor.actor_module(torch.ones(1, 1)).sum().backward(); worker.actor._optimizer_step(); worker.actor_optimizer.zero_grad()
        with patch('search_r1.baselines.training_state.atomic_save', side_effect=OSError('direct record disk failure')):
            with self.assertRaises(safety_source.PolicySafetyRefusal) as caught:
                SCHEDULE(worker, data, {'actor/optimizer_steps': 20, 'actor/skipped_empty_minibatches': 0}, 0, 0.)
        self.assertTrue(caught.exception.matched_all_ranks)
        self.assertEqual(worker.actor_lr_scheduler.last_epoch, 1)
        envelope = source.persist_matched_partial_worker(worker, data, caught.exception, 0)
        own = envelope['worker_states'][0]; state = read_saved(own)
        self.assertEqual(own['committed_optimizer_calls'], 20); self.assertEqual(own['scheduler_step_after_actual'], 1)
        self.assertEqual(state['scheduler']['last_epoch'], 1); self.assertEqual(envelope['scheduler_advanced_actual'], [True])
        self.assertFalse(envelope['outer_completed']); self.assertEqual(int(next(iter(state['optimizer']['state'].values()))['step'].item()), 20)

    def test_counter_disagrees_with_returned_metric_refuses_before_scheduler(self):
        worker, data = worker_and_data(self.factory, self.run)
        worker.actor._actual_committed_optimizer_calls = 19
        with self.assertRaises(safety_source.PolicySafetyRefusal):
            SCHEDULE(worker, data, {'actor/optimizer_steps': 20, 'actor/skipped_empty_minibatches': 0}, 0, 0.)
        self.assertEqual(worker.actor_lr_scheduler.last_epoch, 0)
        record = json.loads((self.run/'output/fresh-state/outer-000001-rank00.json').read_text())
        self.assertEqual(record['actual_optimizer_calls'], 19); self.assertEqual(record['returned_optimizer_calls'], 20)

    def test_actual_worker_catches_only_matched_fresh_and_returns_saved_metadata_before_offload(self):
        worker, data = worker_and_data(self.factory, self.run)
        worker._is_actor = True; worker._is_offload_grad = False; worker.world_size = 1
        worker.ulysses_sharding_manager = N()
        class Sharding:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def preprocess_data(self, data): return data
            def postprocess_data(self, data): return data
        worker.ulysses_sharding_manager = Sharding()
        worker.flops_counter = N(estimate_flops=lambda *args: (1., 1.))
        data.to = lambda *args: data
        scope = dict(torch=torch, os=os, DataProto=WorkerProto, Timer=FakeTimer, logger=None,
            log_gpu_memory_usage=lambda *args, **kwargs: None, fresh200_scheduler_after_actor=SCHEDULE)
        update = load_actual_update_actor(scope)
        original_update = worker.actor.update_policy
        def refuse_after_actual_write(data):
            worker.actor.actor_module(torch.ones(1, 1)).sum().backward(); worker.actor._optimizer_step()
            raise safety_source.PolicySafetyRefusal('matched worker stop', matched_all_ranks=True)
        worker.actor.update_policy = refuse_after_actual_write
        with patch.object(torch.cuda, 'empty_cache', lambda: None): output = update(worker, data)
        self.assertEqual(output.meta_info['tree_partial_state']['status'], 'PARTIAL_WORKER_STATE_SAVED')
        self.assertEqual(output.meta_info['metrics']['actor/optimizer_steps'], 1)
        self.assertEqual(worker.actor_lr_scheduler.last_epoch, 0)
        for profile, error in ((True, safety_source.PolicySafetyRefusal('unmatched')), (True, RuntimeError('generic')),
                               (False, safety_source.PolicySafetyRefusal('matched legacy', matched_all_ranks=True))):
            worker.config.actor['fresh200_protocol'] = 'tree_fresh200_five_leaf_v1' if profile else None
            worker.actor.update_policy = lambda data, error=error: (_ for _ in ()).throw(error)
            with patch.object(source, 'persist_matched_partial_worker', side_effect=AssertionError('must not capture')) as capture:
                with self.assertRaises(type(error)): update(worker, data)
                capture.assert_not_called()
        worker.actor.update_policy = original_update


if __name__ == '__main__':
    unittest.main(verbosity=2)

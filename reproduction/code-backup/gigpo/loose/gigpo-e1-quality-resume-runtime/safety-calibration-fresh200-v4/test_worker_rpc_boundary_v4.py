"""UNEXECUTED CPU fixtures using exact AST bodies of frozen real Ray factory.

No Ray.init, real actor, GPU, model or global ray.get patch. Only fixture-local
factory globals contain the await surrogate. DP splitter/collector/concat and
Ray execute_all/execute_all_async bodies are actual source bodies.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from verl import DataProto
from verl.single_controller.base.worker_group import WorkerGroup
from worker_rpc_boundary_v4 import track_worker_group_rpcs, ACTOR_METHODS, UnsupportedWorkerRPCSchema
from calibration_failure_v2 import preserve_first_refusal


def exact_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    # Defining these original function bodies is a future CPU fixture operation;
    # this draft itself was only read/AST parsed, never imported or executed.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


def make_actual_rpc_fixture(*, role='actor_rollout', output=None, failure=None,
                            failure_stage='await', failure_method='generate_sequences', bad_concat=False, _schema=None):
    import verl
    source = Path(verl.__file__).resolve().parent.parent
    base_path = source / 'verl/single_controller/ray/base.py'
    decorator_path = source / 'verl/single_controller/base/decorator.py'
    calls, events = [], []
    output = output if output is not None else DataProto.from_dict(dict(responses=torch.full((8, 1), 7, dtype=torch.int64)))
    def await_once(refs):
        events.append(('await', refs[0].method))
        if failure is not None and failure_stage == 'await' and refs[0].method.endswith('_' + failure_method):
            raise failure
        return [ref.value for ref in refs]
    base_ns = dict(ray=SimpleNamespace(get=await_once))
    exact_functions(base_path, ['func_generator'], base_ns)
    exact_functions(base_path, ['execute_all', 'execute_all_async'], base_ns, 'RayWorkerGroup')
    decorator_ns = dict(List=list)
    exact_functions(decorator_path, ['dispatch_one_to_all', 'collect_all_to_all',
        '_split_args_kwargs_data_proto', 'dispatch_dp_compute_data_proto',
        'dispatch_dp_compute', 'collect_dp_compute', '_concat_data_proto_or_future',
        'collect_dp_compute_data_proto'], decorator_ns)
    schema = SimpleNamespace(generator=base_ns['func_generator'], namespace=base_ns,
        execute_all=base_ns['execute_all'], execute_all_async=base_ns['execute_all_async'],
        one=(decorator_ns['dispatch_one_to_all'], decorator_ns['collect_all_to_all']),
        dp=(decorator_ns['dispatch_dp_compute_data_proto'], decorator_ns['collect_dp_compute_data_proto']),
        base_path=base_path, decorator_path=decorator_path, data_proto=DataProto)
    if _schema is not None:
        schema = _schema
    class Group(WorkerGroup):
        execute_all = schema.execute_all
        execute_all_async = schema.execute_all_async
    group = object.__new__(Group)
    group._world_size = 8
    class Handle:
        def __init__(self, rank):
            self.rank = rank
        def __getattr__(self, method):
            def remote(*args, **kwargs):
                events.append(('remote', method, self.rank))
                name = method.removeprefix(role + '_')
                if self.rank == 0:
                    calls.append('generate' if name == 'generate_sequences' else 'save-ranks' if name == 'calibration_preserve_zero_adam' else name)
                if failure is not None and failure_stage == 'execute' and name == failure_method and self.rank == 3:
                    raise failure
                if name == 'calibration_preserve_zero_adam':
                    value = dict(rank=self.rank, saved=True, zero_adam_verified=True, scheduler_step=0, errors=[])
                elif name in ('calibration_phase', 'checkpoint_pause_identity', 'calibration_complete_pause_state'):
                    value = dict(rank=self.rank, scheduler_step=0)
                else:
                    value = DataProto.from_dict(dict(responses=output.batch['responses'][self.rank:self.rank+1]),
                        meta_info=output.meta_info.copy(),
                        non_tensors={key: val[self.rank:self.rank+1] for key, val in output.non_tensor_batch.items()})
                    for key, tensor in output.batch.items():
                        if key != 'responses':
                            value.batch[key] = tensor[self.rank:self.rank+1]
                    if bad_concat and name == failure_method and self.rank == 3:
                        value = DataProto.from_dict(dict(responses=torch.ones((1, 2), dtype=torch.int64)))
                return SimpleNamespace(method=method, value=value)
            return SimpleNamespace(remote=remote)
    group._workers = [Handle(rank) for rank in range(8)]
    names = set(ACTOR_METHODS) | {'compute_ref_log_prob', 'calibration_preserve_zero_adam'}
    for name in names:
        dispatch, collect = schema.dp if name in ('generate_sequences','compute_log_prob','compute_ref_log_prob','calibration_raw_current_log_prob','update_actor') else schema.one
        fn = schema.generator(group, role + '_' + name, dispatch, collect, group.execute_all, True)
        setattr(group, name, fn)
        setattr(group, role + '_' + name, fn)
    trainer = SimpleNamespace(actor_rollout_wg=group, recovery_sampler=SimpleNamespace(cursor=512, state_dict=lambda: dict(cursor=512)))
    return trainer, calls, output, schema, events


def save_report(trainer, path, progress, error):
    def torch_save(value, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, target)
        return dict(path=str(target))
    def json_save(target, value):
        import json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    return preserve_first_refusal(trainer, path, progress, error,
        driver_snapshot=lambda *args: dict(actual_cursor=512), atomic_torch_save=torch_save,
        json_save=json_save, cpu_snapshot=lambda value: value, rng_state=lambda: dict(actual_rng_progress=123))


def initial_progress():
    return dict(outer=1, first_adam_possible=False, workers_safe_to_snapshot=True)


@pytest.mark.parametrize('method', ['generate_sequences', 'compute_log_prob', 'calibration_raw_current_log_prob', 'compute_ref_log_prob'])
def test_actual_factory_await_then_actual_dp_concat_failure_retains_rank_outputs(tmp_path, method):
    role = 'ref' if method == 'compute_ref_log_prob' else 'actor_rollout'
    trainer, calls, output, schema, events = make_actual_rpc_fixture(role=role, failure_method=method, bad_concat=True)
    group = trainer.actor_rollout_wg
    progress = initial_progress()
    actor_calls = calls
    if role == 'ref':
        actor, actor_calls, _, _, _ = make_actual_rpc_fixture(_schema=schema)
        trainer.ref_policy_wg = group
        trainer.actor_rollout_wg = actor.actor_rollout_wg
        groups = ((trainer.actor_rollout_wg, 'actor_rollout', ()), (group, role, (method,)))
    else:
        groups = ((group, role, (method,)),)
    originals = dict(vars(group))
    with pytest.raises(RuntimeError) as raised:
        with track_worker_group_rpcs(groups, progress, _schema=schema):
            getattr(group, method)(output)
    assert progress['workers_safe_to_snapshot']
    actual = progress['resolved_worker_rank_outputs'][role + '/' + method]['outputs']
    assert len(actual) == 8 and actual[3].batch['responses'].shape == (1, 2)
    assert sum(event[0] == 'await' for event in events) == 1
    assert all(getattr(group, name) is value for name, value in originals.items())
    report = save_report(trainer, tmp_path, progress, raised.value)
    assert actor_calls[-1] == 'save-ranks' and not report['missing_ranks']
    saved = torch.load(tmp_path / 'probability-rejection-state/driver.pt', weights_only=False)
    assert len(saved['resolved_worker_rank_outputs'][role + '/' + method]['ranks']) == 8



@pytest.mark.parametrize('method', ['generate_sequences', 'calibration_phase'])
@pytest.mark.parametrize('stage', ['execute', 'await'])
def test_unknown_actual_execute_or_phase_await_failure_starts_no_recovery_rpc(tmp_path, method, stage):
    failure = RuntimeError('unknown original Ray/rank failure')
    trainer, calls, output, schema, events = make_actual_rpc_fixture(failure=failure, failure_stage=stage, failure_method=method)
    progress = initial_progress()
    if method == 'calibration_phase':
        trainer.recovery_sampler = SimpleNamespace(cursor=0, state_dict=lambda: dict(cursor=0))
    group = trainer.actor_rollout_wg
    with pytest.raises(RuntimeError) as raised:
        with track_worker_group_rpcs(((group, 'actor_rollout', (method,)),), progress, _schema=schema):
            getattr(group, method)('train' if method == 'calibration_phase' else output)
    assert raised.value is failure and not progress['workers_safe_to_snapshot']
    report = save_report(trainer, tmp_path, progress, failure)
    assert 'save-ranks' not in calls and report['missing_ranks'] == list(range(8))
    assert report['actual_driver_cursor'] == (0 if method == 'calibration_phase' else 512)
    assert not progress.get('resolved_worker_rank_outputs')
    assert sum(event[0] == 'await' for event in events) == (1 if stage == 'await' else 0)


def test_actual_dp_dispatch_failure_before_execute_keeps_quiescent_and_zero_adam(tmp_path, monkeypatch):
    trainer, calls, output, schema, events = make_actual_rpc_fixture()
    group = trainer.actor_rollout_wg
    failure = OSError('injected local chunk/copy failure')
    def bad_chunk(*args, **kwargs):
        raise failure
    monkeypatch.setattr(DataProto, 'chunk', bad_chunk)
    progress = initial_progress()
    with pytest.raises(OSError) as raised:
        with track_worker_group_rpcs(((group, 'actor_rollout', ('update_actor',)),), progress, _schema=schema):
            group.update_actor(output)
    assert raised.value is failure and progress['workers_safe_to_snapshot']
    assert not progress['first_adam_possible'] and not events
    report = save_report(trainer, tmp_path, progress, failure)
    assert calls == ['save-ranks'] and not report['missing_ranks']


@pytest.mark.parametrize('mutation', ['blocking', 'prefix', 'group', 'dispatch', 'collect', 'execute', 'globals'])
def test_unsupported_actual_closure_is_refused_before_any_dispatch(mutation):
    from types import FunctionType
    trainer, calls, output, schema, events = make_actual_rpc_fixture()
    group = trainer.actor_rollout_wg
    original = group.generate_sequences
    closure = dict(zip(original.__code__.co_freevars, (cell.cell_contents for cell in original.__closure__)))
    if mutation == 'blocking': closure['blocking'] = False
    if mutation == 'prefix': closure['method_name'] = 'other_generate_sequences'
    if mutation == 'group': closure['self'] = object()
    if mutation == 'dispatch': closure['dispatch_fn'] = lambda *args: None
    if mutation == 'collect': closure['collect_fn'] = lambda *args: None
    if mutation == 'execute': closure['execute_fn'] = lambda *args: None
    bad = schema.generator(closure['self'], closure['method_name'], closure['dispatch_fn'], closure['collect_fn'], closure['execute_fn'], closure['blocking'])
    if mutation == 'globals':
        bad = FunctionType(bad.__code__, dict(schema.namespace), closure=bad.__closure__)
    group.generate_sequences = group.actor_rollout_generate_sequences = bad
    with pytest.raises(UnsupportedWorkerRPCSchema):
        with track_worker_group_rpcs(((group, 'actor_rollout', ('generate_sequences',)),), initial_progress(), _schema=schema):
            group.generate_sequences(output)
    assert calls == [] and events == [] and group.generate_sequences is bad


def test_successful_call_then_unknown_next_call_cannot_reuse_old_safe_boundary(tmp_path):
    failure = RuntimeError('unknown next phase await')
    trainer, calls, output, schema, events = make_actual_rpc_fixture(failure=failure, failure_method='calibration_phase')
    group = trainer.actor_rollout_wg
    progress = initial_progress()
    with pytest.raises(RuntimeError):
        with track_worker_group_rpcs(((group, 'actor_rollout', ('generate_sequences', 'calibration_phase')),), progress, _schema=schema):
            group.generate_sequences(output)
            assert progress['workers_safe_to_snapshot']
            group.calibration_phase('train')
    report = save_report(trainer, tmp_path, progress, failure)
    assert not progress['workers_safe_to_snapshot'] and 'save-ranks' not in calls
    assert progress['resolved_worker_rank_outputs']['actor_rollout/generate_sequences']['outputs']
    assert report['missing_ranks'] == list(range(8))


def test_combined_actor_reference_callbacks_and_all_aliases_restore_exactly():
    trainer, calls, output, schema, events = make_actual_rpc_fixture()
    ref, ref_calls, _, _, _ = make_actual_rpc_fixture(role='ref', _schema=schema)
    trainer.ref_policy_wg = ref.actor_rollout_wg
    actor_group, ref_group = trainer.actor_rollout_wg, trainer.ref_policy_wg
    original_actor, original_ref = dict(vars(actor_group)), dict(vars(ref_group))
    original_get = schema.namespace['ray'].get
    progress = initial_progress()
    with track_worker_group_rpcs(((actor_group, 'actor_rollout', ACTOR_METHODS),
            (ref_group, 'ref', ('compute_ref_log_prob',))), progress, _schema=schema):
        actor_group.calibration_phase('train')
        actor_group.generate_sequences(output)
        actor_group.compute_log_prob(output)
        ref_group.compute_ref_log_prob(output)
        actor_group.calibration_raw_current_log_prob(output)
        actor_group.calibration_complete_pause_state()
        actor_group.checkpoint_pause_identity()
        assert not progress['first_adam_possible']
        actor_group.update_actor(output)
        assert progress['first_adam_possible'] and progress['workers_safe_to_snapshot']
    assert all(getattr(actor_group, name) is value for name, value in original_actor.items())
    assert all(getattr(ref_group, name) is value for name, value in original_ref.items())
    assert schema.namespace['ray'].get is original_get
    assert len(progress['worker_rpc_boundary_v4']['calls']) == 8
    assert all(call['execute_started'] and call['awaited_all_ranks'] and call['collect_returned']
               for call in progress['worker_rpc_boundary_v4']['calls'])


def test_partial_alias_installation_restores_originals_before_any_dispatch(monkeypatch):
    trainer, calls, output, schema, events = make_actual_rpc_fixture()
    group = trainer.actor_rollout_wg
    original, prefixed = group.generate_sequences, group.actor_rollout_generate_sequences
    ordinary = type(group).__setattr__
    armed = [True]
    def partial_setattr(self, name, value):
        ordinary(self, name, value)
        if armed[0] and name == 'actor_rollout_generate_sequences' and value is not prefixed:
            armed[0] = False
            raise OSError('injected partial alias install after mutation')
    monkeypatch.setattr(type(group), '__setattr__', partial_setattr)
    progress = initial_progress()
    with pytest.raises(UnsupportedWorkerRPCSchema):
        with track_worker_group_rpcs(((group, 'actor_rollout', ('generate_sequences',)),), progress, _schema=schema):
            group.generate_sequences(output)
    assert calls == [] and events == [] and progress['workers_safe_to_snapshot']
    assert group.generate_sequences is original and group.actor_rollout_generate_sequences is prefixed
    assert progress['worker_rpc_boundary_v4']['observer_errors']


def test_alias_restore_error_records_diagnostic_and_never_masks_original_unknown_await(monkeypatch):
    failure = RuntimeError('unknown original await failure')
    trainer, calls, output, schema, events = make_actual_rpc_fixture(failure=failure)
    group = trainer.actor_rollout_wg
    original, prefixed = group.generate_sequences, group.actor_rollout_generate_sequences
    ordinary = type(group).__setattr__
    def fail_restore(self, name, value):
        if name == 'generate_sequences' and value is original:
            raise OSError('injected alias restore failure')
        ordinary(self, name, value)
    monkeypatch.setattr(type(group), '__setattr__', fail_restore)
    progress = initial_progress()
    with pytest.raises(RuntimeError) as raised:
        with track_worker_group_rpcs(((group, 'actor_rollout', ('generate_sequences',)),), progress, _schema=schema):
            group.generate_sequences(output)
    assert raised.value is failure and not progress['workers_safe_to_snapshot']
    assert group.actor_rollout_generate_sequences is prefixed
    assert any('restore/generate_sequences' in error for error in progress['worker_rpc_boundary_v4']['observer_errors'])
    assert calls == ['generate'] and sum(event[0] == 'await' for event in events) == 1
    ordinary(group, 'generate_sequences', original)  # Fixture-only repair of synthetic instance after asserted fault.

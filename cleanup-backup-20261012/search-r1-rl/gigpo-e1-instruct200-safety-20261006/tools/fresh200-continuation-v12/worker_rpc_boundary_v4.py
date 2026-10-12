"""Observe frozen Ray dispatch/execute/await/collect without another RPC or await."""
from contextlib import contextmanager
from pathlib import Path
from types import CodeType, FunctionType, MethodType, SimpleNamespace
import hashlib
from observer_safety_v2 import patches

FROZEN_RAY_BASE_SHA256 = 'b0dca8aad381d4e9a60169df108af2035fd0b5f88ab38dfc833689c842d4caa9'
FROZEN_DECORATOR_SHA256 = 'b6e4cd3c8ad3c4fa2eea07df3ef45344bb5dc2e1e8357c0b94a9d19a6bd4daac'
FREEVARS = {'blocking', 'collect_fn', 'dispatch_fn', 'execute_fn', 'method_name', 'self'}
ACTOR_METHODS = ('calibration_phase', 'generate_sequences', 'compute_log_prob',
    'calibration_raw_current_log_prob', 'calibration_complete_pause_state',
    'update_actor', 'checkpoint_pause_identity', 'calibration_live_admission_identity')
DP_METHODS = {'generate_sequences', 'compute_log_prob', 'compute_ref_log_prob',
              'calibration_raw_current_log_prob', 'update_actor'}


class UnsupportedWorkerRPCSchema(RuntimeError):
    pass


def require_schema(value, message):
    if not value:
        raise UnsupportedWorkerRPCSchema(message + '; refused before dispatch')


def load_frozen_rpc_schema():
    from verl.single_controller.ray import base
    from verl.single_controller.base import decorator
    from verl import DataProto
    return SimpleNamespace(generator=base.func_generator, namespace=vars(base),
        execute_all=base.RayWorkerGroup.execute_all,
        execute_all_async=base.RayWorkerGroup.execute_all_async,
        one=(decorator.dispatch_one_to_all, decorator.collect_all_to_all),
        dp=(decorator.dispatch_dp_compute_data_proto, decorator.collect_dp_compute_data_proto),
        base_path=Path(base.__file__).resolve(), decorator_path=Path(decorator.__file__).resolve(), data_proto=DataProto)


def validate_schema(schema):
    # Files are actual imported implementation origins, not an environment/header claim.
    require_schema(hashlib.sha256(schema.base_path.read_bytes()).hexdigest() == FROZEN_RAY_BASE_SHA256,
                   'Actual Ray factory source bytes differ')
    require_schema(hashlib.sha256(schema.decorator_path.read_bytes()).hexdigest() == FROZEN_DECORATOR_SHA256,
                   'Actual dispatch/collect source bytes differ')
    factory = schema.generator
    require_schema(type(factory) is FunctionType and factory.__name__ == 'func_generator'
        and factory.__globals__ is schema.namespace
        and Path(factory.__code__.co_filename).resolve() == schema.base_path,
        'Actual factory identity/origin differs')
    nested = [code for code in factory.__code__.co_consts if isinstance(code, CodeType) and code.co_name == 'func']
    require_schema(len(nested) == 1 and set(nested[0].co_freevars) == FREEVARS,
                   'Actual blocking factory closure shape differs')
    return nested[0]


def inspect_bound_rpc(group, role, name, schema, code):
    original = getattr(group, name)
    require_schema(type(original) is FunctionType and original.__code__ is code
        and original.__globals__ is schema.namespace and original.__closure__ is not None
        and set(original.__code__.co_freevars) == FREEVARS, 'Actual generated callable identity differs: ' + name)
    captured = dict(zip(original.__code__.co_freevars, (cell.cell_contents for cell in original.__closure__)))
    require_schema(captured['self'] is group and captured['method_name'] == role + '_' + name
        and captured['blocking'] is True, 'Actual group/prefix/blocking differs: ' + name)
    dispatch, collect = schema.dp if name in DP_METHODS else schema.one
    execute = captured['execute_fn']
    async_execute = group.execute_all_async
    require_schema(captured['dispatch_fn'] is dispatch and captured['collect_fn'] is collect,
                   'Actual dispatch/collect identity differs: ' + name)
    require_schema(type(execute) is MethodType and execute.__self__ is group
        and execute.__func__ is schema.execute_all and type(async_execute) is MethodType
        and async_execute.__self__ is group and async_execute.__func__ is schema.execute_all_async,
        'Actual executor identity differs: ' + name)
    require_schema(group.world_size == 8 and len(group._workers) == 8,
                   'Actual full eight-rank group differs: ' + name)
    return original, captured


def _observed_rpc(group, role, name, captured, schema, progress, record):
    # Factory is the original frozen body. Its ray.get remains original and
    # unpatched: the collect callback can only run after that actual await returned.
    dispatch, execute, collect = (captured[key] for key in ('dispatch_fn', 'execute_fn', 'collect_fn'))
    call = None
    def observed_dispatch(worker_group, *args, **kwargs):
        nonlocal call
        call = dict(role=role, method=name, stage='driver_dispatch', execute_started=False,
                    awaited_all_ranks=False, collect_returned=False)
        record['calls'].append(call)
        progress['stage'] = role + '/' + name + '/driver_dispatch'
        if name in DP_METHODS:
            require_schema(all(type(value) is schema.data_proto for value in args)
                and all(type(value) is schema.data_proto for value in kwargs.values()),
                'Nonlocal future/custom DP dispatch argument unsupported: ' + name)
        return dispatch(worker_group, *args, **kwargs)  # Local splitter exactly once; still quiescent.
    def observed_execute(method_name, *args, **kwargs):
        progress['workers_safe_to_snapshot'] = False  # First action before possible partial rank dispatch.
        if name == 'update_actor':
            progress['first_adam_possible'] = True
        call['execute_started'] = True
        call['stage'] = 'execute_or_await'
        progress['stage'] = role + '/' + name + '/execute_or_await'
        return execute(method_name, *args, **kwargs)  # Original executor exactly once; unknown error propagates.
    def observed_collect(worker_group, output):
        progress['workers_safe_to_snapshot'] = True  # First action: genuine original all-worker await succeeded.
        progress.setdefault('resolved_worker_rank_outputs', {})[role + '/' + name] = dict(
            role=role, method=name, call_index=len(record['calls']), outputs=output)
        call['awaited_all_ranks'] = True
        call['stage'] = 'driver_collect'
        progress['stage'] = role + '/' + name + '/driver_collect'
        value = collect(worker_group, output)  # Original local validator/concat exactly once.
        call['collect_returned'] = True
        call['stage'] = 'driver_after_collect'
        if name == 'generate_sequences':
            progress['last_generation_rpc_output'] = value
        return value
    return schema.generator(group, captured['method_name'], observed_dispatch,
                            observed_collect, observed_execute, True)


@contextmanager
def track_worker_group_rpcs(groups, progress, *, _schema=None):
    """Temporary instance aliases only; all original callbacks and awaits stay once.

    groups contains (actual spawned group, exact role prefix, required method names).
    _schema is an internal CPU-fixture seam for AST-extracted frozen function bodies.
    Production always imports the actual module and validates its frozen source origin.
    """
    record = dict(observer_errors=[], calls=[], schema_verified=False)
    progress['worker_rpc_boundary_v4'] = record
    progress['local_rank_preservation_rpc_schema_verified'] = False
    schema = load_frozen_rpc_schema() if _schema is None else _schema
    code = validate_schema(schema)
    planned = []
    # Validate preservation separately before any training RPC may start. It is
    # restored/unwrapped before the first-outer exception handler may dispatch it.
    for group, role, names in groups:
        if role == 'actor_rollout':
            inspect_bound_rpc(group, role, 'calibration_preserve_zero_adam', schema, code)
            progress['local_rank_preservation_rpc_schema_verified'] = True
        for name in names:
            original, captured = inspect_bound_rpc(group, role, name, schema, code)
            observed = _observed_rpc(group, role, name, captured, schema, progress, record)
            aliases = [alias for alias, value in vars(group).items() if value is original]
            require_schema(name in aliases and role + '_' + name in aliases,
                           'Actual short/prefixed alias shape differs: ' + name)
            planned.extend((group, alias, observed) for alias in aliases)
    record['schema_verified'] = True
    with patches(record) as install:
        for group, name, observed in planned:
            installed = install(group, name, observed)
            require_schema(installed and getattr(group, name) is observed, 'Temporary RPC alias installation failed: ' + name)
        yield record
    if record['observer_errors'] or record.get('_observer_recording_failed'):
        raise UnsupportedWorkerRPCSchema('Temporary RPC alias restoration failed: ' + repr(record['observer_errors']))


@contextmanager
def track_first_outer_rpcs(trainer, progress):
    with track_worker_group_rpcs(((trainer.actor_rollout_wg, 'actor_rollout', ACTOR_METHODS),
            (trainer.ref_policy_wg, 'ref', ('compute_ref_log_prob', 'calibration_live_admission_identity'))), progress) as record:
        yield record

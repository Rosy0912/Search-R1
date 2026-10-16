"""Repair only the native partial context's output-root assumption.

The ablation publishes audit output under ``base/output/attempt-*`` and native
states under ``base/checkpoints``. The native validator assumes those locations
share the audit output's parent. All native identity/profile/counter checks are
retained; this hook substitutes only that root with an exact receipt-bound path.
It does not handle exceptions, alter actor math, or turn faults into skips.
"""
import ast
import inspect
import os
from pathlib import Path
import textwrap


_NATIVE_ROOT_EXPRESSION = "Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']).resolve().parent"
_MARKER = '_ablation150_partial_namespace_v1'


def checkpoint_root(record, config, environ):
    """Bind both output namespaces to this exact frozen group receipt."""
    base = Path(record['base'])
    checkpoints = Path(config['trainer']['default_local_dir'])
    output = Path(environ['SEARCH_R1_BASELINE_OUTPUT'])
    expected = base / 'checkpoints'
    alias = record.get('checkpoint_storage_alias')
    if alias:
        if (record.get('execution_backend') != 'pj_pt' or not record.get('resume_manifest')
                or str(checkpoints) != alias['configured'] or str(expected) != alias['physical']
                or checkpoints.resolve() != expected or expected.resolve() != expected):
            raise ValueError('Unbound checkpoint storage alias')
        checkpoints = expected
    if (not base.is_absolute() or not checkpoints.is_absolute() or not output.is_absolute()
            or checkpoints != expected or checkpoints.resolve() != base.resolve() / 'checkpoints'
            or output.parent != base / 'output'
            or output.resolve().parent != base.resolve() / 'output'
            or not output.name.startswith('attempt-')
            or len(output.name.removeprefix('attempt-')) != 32
            or any(value not in '0123456789abcdef' for value in output.name.removeprefix('attempt-'))):
        raise ValueError('Ablation output/checkpoint namespaces differ from this exact group receipt')
    bindings = dict(SEARCH_R1_BASELINE_RUN_ID='run_id', SEARCH_R1_BASELINE_COMMIT='source_commit',
                    SEARCH_R1_BASELINE_CONFIG_SHA256='config_sha256',
                    SEARCH_R1_BASELINE_DATA_IDENTITY='data_identity')
    if any(environ.get(key) != record[field] for key, field in bindings.items()):
        raise ValueError('Ablation partial namespace source/run binding differs from the frozen receipt')
    if record.get('actual_update_limit') != 150 or config['trainer']['actual_update_limit'] != 150:
        raise ValueError('Ablation partial namespace requires the authorized150-round group')
    return checkpoints


def _frozen_namespace():
    from omegaconf import OmegaConf
    from release.recursive_bran_a.ablation150_v1.contracts import read_receipt, validate_config_dict
    record = read_receipt(os.environ['RECURSIVE_ABLATION_RECEIPT'])
    config = OmegaConf.to_container(OmegaConf.load(record['config']), resolve=True)
    validate_config_dict(config)
    return record, checkpoint_root(record, config, os.environ)


def replay_partial_directory(checkpoints, environ, identity, name):
    """The saved first batch keeps attempt1/cursor512 and gets fresh evidence."""
    if (not environ.get('RECURSIVE_ABLATION_ZERO_WRITE_MANIFEST')
            or identity.get('attempted_outer') != 1 or identity.get('completed_outer') != 0
            or identity.get('attempt_id') != 1):
        return None
    return (checkpoints / ('replay-' + Path(environ['SEARCH_R1_BASELINE_OUTPUT']).name)
            / 'training_state' / name)


def bound_partial_root(worker, data):
    """Native validation still runs before this returns the corrected root."""
    _, checkpoints = _frozen_namespace()
    context = data.meta_info['tree_partial_context']
    identity = context['identity']
    name = (f'partial_attempt_{identity["attempt_id"]:06d}_outer_{identity["attempted_outer"]:06d}'
            if worker.config.actor.get('guard_continue_protocol') == 'tree_guard_continue_fresh_rollout_v1'
            else f'partial_outer_{identity["attempted_outer"]:06d}')
    directory = Path(context['directory'])
    expected = checkpoints / 'training_state' / name
    replay = replay_partial_directory(checkpoints, os.environ, identity, name)
    allowed = [expected] + ([replay] if replay is not None else [])
    if not any(directory == path and directory.resolve() == path for path in allowed):
        raise ValueError('Ablation partial state must use this exact group/checkpoint/attempt directory')
    return checkpoints


def install_into(module):
    """Ablation bootstrap callback; safe before or after the native import."""
    original = module._validated_context
    if getattr(original, _MARKER, False):
        return
    # Keep the complete native function, replacing only its incompatible root.
    # Refuse a changed native implementation instead of applying a broad patch.
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'own_run'):
            if ast.dump(node.value, include_attributes=False) != ast.dump(
                    ast.parse(_NATIVE_ROOT_EXPRESSION, mode='eval').body, include_attributes=False):
                raise ValueError('Native partial-state root expression changed; repair needs review')
            node.value = ast.Call(func=ast.Name(id='_ablation_partial_root', ctx=ast.Load()),
                                  args=[ast.Name(id='worker', ctx=ast.Load()),
                                        ast.Name(id='data', ctx=ast.Load())], keywords=[])
            replacements += 1
    if replacements != 1:
        raise ValueError('Expected exactly one native partial-state root binding')
    namespace = dict(original.__globals__, _ablation_partial_root=bound_partial_root)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(original), 'exec'), namespace)
    repaired = namespace[original.__name__]
    setattr(repaired, _MARKER, True)

    def validated_context(worker, data):
        if not os.environ.get('RECURSIVE_ABLATION_RECEIPT'):
            return original(worker, data)
        return repaired(worker, data)

    setattr(validated_context, _MARKER, True)
    validated_context.__name__ = original.__name__
    module._validated_context = validated_context


def install_driver_into(module):
    """Select a fresh evidence directory only when replaying the saved batch1."""
    original = module.partial_context
    if getattr(original, _MARKER, False):
        return

    def partial_context(trainer):
        context = original(trainer)
        if not os.environ.get('RECURSIVE_ABLATION_RECEIPT'):
            return context
        record, checkpoints = _frozen_namespace()
        name = Path(context['directory']).name
        if record.get('checkpoint_storage_alias'):
            configured = Path(record['checkpoint_storage_alias']['configured']) / 'training_state' / name
            expected = checkpoints / 'training_state' / name
            if Path(context['directory']) != configured or configured.resolve() != expected:
                raise ValueError('Partial context escaped the bound checkpoint storage alias')
            context = dict(context, directory=str(expected))
        replay = replay_partial_directory(checkpoints, os.environ, context['identity'], name)
        if replay is not None:
            context = dict(context, directory=str(replay))
        return context

    setattr(partial_context, _MARKER, True)
    module.partial_context = partial_context

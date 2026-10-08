"""Pure-standard-library bindings for actual, unmodified partial raw evidence.

No Torch import, deserialization, model work, or resume occurs in this module.
Large raw files are hashed in bounded blocks on their existing filesystem.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat

PROTOCOL = 'tree_partial_state_v1'
PROFILE = 'tree_fresh200_five_leaf_v1'
IDENTITY_KEYS = {'job_id', 'run_id', 'code_commit', 'config_sha256', 'data_identity',
                 'model_identity', 'training_protocol', 'attempted_outer', 'completed_outer', 'attempt_id'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _signature(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _absolute(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Evidence path must be absolute without traversal')
    return path


@contextmanager
def held_ordinary_file(path):
    """Hold every ancestor with no-follow openat, then bind the ordinary file FD.

    Ancestors remain open until after the caller finishes. Final checks bind
    every pathname component to its held inode and detect mutation of the file.
    """
    path = _absolute(path)
    directories = []
    links = []
    stream = None
    try:
        root = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directories.append(root)
        for name in path.parts[1:-1]:
            parent = directories[-1]
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode), 'Evidence ancestor must be a real directory: ' + str(path))
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            directories.append(child)
            actual = os.fstat(child)
            require((before.st_dev, before.st_ino) == (actual.st_dev, actual.st_ino), 'Ancestor changed during open')
            links.append((parent, name, child))
        parent = directories[-1]
        before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode), 'Evidence must be an ordinary file: ' + str(path))
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        stream = os.fdopen(descriptor, 'rb')
        opened = os.fstat(stream.fileno())
        require(_signature(opened) == _signature(before), 'Evidence changed during open')
        yield stream, opened
        after = os.fstat(stream.fileno())
        named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        require(_signature(opened) == _signature(after) == _signature(named), 'Evidence changed while held: ' + str(path))
        for ancestor_parent, name, child in links:
            named = os.stat(name, dir_fd=ancestor_parent, follow_symlinks=False)
            actual = os.fstat(child)
            require(stat.S_ISDIR(named.st_mode) and (named.st_dev, named.st_ino) ==
                    (actual.st_dev, actual.st_ino), 'Evidence ancestor changed while held')
    finally:
        if stream is not None:
            stream.close()
        for descriptor in reversed(directories):
            os.close(descriptor)


def stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024**2), b''):
        digest.update(block)
    return digest.hexdigest()


def read_json_bound(path, expected=None):
    with held_ordinary_file(path) as (stream, _):
        raw = stream.read()
        observed = hashlib.sha256(raw).hexdigest()
        require(expected is None or observed == expected, 'JSON evidence SHA mismatch')
        value = json.loads(raw)
    return value, observed, raw


def partial_directory(run, identity):
    run = _absolute(run)
    require(type(identity) is dict and set(identity) == IDENTITY_KEYS, 'Complete partial identity required')
    attempt, outer, completed = (identity[key] for key in ('attempt_id', 'attempted_outer', 'completed_outer'))
    require(all(type(value) is int for value in (attempt, outer, completed)) and
            0 <= completed < 200 and outer == completed + 1 and attempt >= outer,
            'Partial actual attempt/outer boundary invalid')
    require(identity['training_protocol'] == PROFILE and identity['run_id'] == run.name and
            identity['job_id'] == run.name.rsplit('-', 1)[-1], 'Partial run/profile identity differs')
    return run / 'checkpoints/training_state' / f'partial_attempt_{attempt:06d}_outer_{outer:06d}'


def raw_file_records(run, partial):
    directory = partial_directory(run, partial['identity'])
    envelope = partial['worker_outcome']
    require(partial['protocol'] == PROTOCOL and envelope['protocol'] == PROTOCOL and
            envelope['identity'] == partial['identity'] and envelope['status'] == 'PARTIAL_WORKER_STATE_SAVED' and
            partial['worker_states_verified'] is True and partial['verification_error'] is None and
            partial['outer_completed'] is False and partial['automatic_rollback'] is False and
            partial['automatic_resume_supported'] is False, 'Partial evidence must be complete and non-resumable')
    workers = envelope['worker_states']
    require(type(workers) is list and len(workers) == 8 and
            all(type(record.get('rank')) is int for record in workers) and
            {record['rank'] for record in workers} == set(range(8)), 'All eight partial workers required')
    records = []
    for record in sorted(workers, key=lambda record: record['rank']):
        rank = record['rank']
        require(record['world_size'] == 8 and type(record['world_size']) is int and
                record['identity'] == dict(partial['identity'], rank=rank, world_size=8) and
                record['status'] == 'PARTIAL_WORKER_STATE_SAVED', 'Partial worker identity/coverage differs')
        records.append((directory / f'worker-{rank:02d}.pt', record, True))
    records.extend(((directory / 'driver.pt', partial['driver'], False),
                    (directory / 'actor-inputs.pt', partial['actor_inputs'], True)))
    for path, record, bytes_required in records:
        require(_absolute(record['path']) == path, 'Raw path must exactly match its own partial attempt directory')
        require(type(record['sha256']) is str and re.fullmatch('[0-9a-f]{64}', record['sha256']), 'Raw SHA pin required')
        if bytes_required or 'bytes' in record:
            require(type(record.get('bytes')) is int and record['bytes'] >= 0, 'Actual raw byte count required')
    return records


def verify_partial_raw_files(run, partial):
    """Return the complete ten-path physical SHA mapping; never deserialize PT.

    Worker and actor-input byte receipts are mandatory. Driver size is checked
    only when its actual receipt includes bytes; no driver size is invented.
    """
    mapping = {}
    for path, record, _ in raw_file_records(run, partial):
        with held_ordinary_file(path) as (stream, before):
            if 'bytes' in record:
                require(before.st_size == record['bytes'], 'Raw file byte count mismatch')
            observed = stream_sha256(stream)
            require(observed == record['sha256'], 'Raw file SHA mismatch: ' + str(path))
        mapping[str(path)] = observed
    require(len(mapping) == 10, 'Partial raw coverage must contain exactly ten files')
    return mapping


def _actual_finite(value, finite):
    try:
        finite(value)
        return True
    except ValueError:
        return False


def _rng(value):
    require(type(value) is dict and {'python', 'numpy', 'torch'} <= set(value) and
            all(value[key] is not None for key in ('python', 'numpy', 'torch')), 'Actual saved RNG payload missing')


def verify_worker_logical(state, partial, record, digest, finite, *, cumulative_calls=None):
    from attempt_accounting import ledger_writes
    identity = partial['identity']
    rank = record['rank']
    require(state['version'] == 'tree_partial_worker_v1' and
            state['identity'] == record['identity'] == dict(identity, rank=rank, world_size=8) and
            digest(state) == record['logical_state_sha256'], 'Actual raw worker logical digest/identity differs')
    for flag in ('resume_permitted', 'automatic_rollback', 'outer_completed'):
        require(state[flag] is False and record[flag] is False, 'Raw partial worker cannot resume/rollback/earn credit')
    calls = state['committed_optimizer_calls']
    ledger = state['optimizer_opportunity_ledger']
    require(type(calls) is int and type(record['committed_optimizer_calls']) is int and
            ledger_writes(ledger) == calls == record['committed_optimizer_calls'] and
            ledger == record['optimizer_opportunity_ledger'], 'Actual saved worker opportunity ledger/count differs')
    before, after = state['scheduler_step_before_actual'], state['scheduler_step_after_actual']
    require(type(before) is int and type(after) is int and
            type(record['scheduler_step_before_actual']) is int and type(record['scheduler_step_after_actual']) is int and
            before == identity['completed_outer'] and
            after == before + int(calls > 0) and before == record['scheduler_step_before_actual'] and
            after == record['scheduler_step_after_actual'] and state['scheduler']['last_epoch'] == after,
            'Actual saved partial scheduler differs from opportunity credit')
    model_fields = ('parameters', 'local_shards', 'buffers', 'optimizer', 'scheduler')
    require(all(type(state[key]) is dict for key in model_fields + ('gradients', 'private_gradient_tensors', 'parameter_layout')) and
            state['parameters'] and set(state['gradients']) == set(state['parameters']) and
            set(state['private_gradient_tensors']) == set(state['parameters']) and
            set(state['parameter_layout']) == set(state['parameters']) and
            set(state['local_shards']) <= set(state['parameters']), 'Actual raw parameter/shard/gradient layout missing')
    finite({key: state[key] for key in model_fields})
    gradients_finite = _actual_finite({key: state[key] for key in ('gradients', 'private_gradient_tensors')}, finite)
    state_finite = _actual_finite(state, finite)
    require(state['model_adam_finite'] is True and record['model_adam_finite'] is True and
            type(state['gradients_finite']) is bool and type(record['gradients_finite']) is bool and
            state['gradients_finite'] == record['gradients_finite'] == gradients_finite and
            type(state['state_finite']) is bool and type(record['state_finite']) is bool and
            state['state_finite'] == record['state_finite'] == state_finite,
            'Actual raw finite classifications differ from persisted state')
    _rng(state['rng'])
    require(type(state['module_training_flags']) is list and state['module_training_flags'] and
            all(type(entry['training']) is bool and type(entry['name']) is str for entry in state['module_training_flags']),
            'Actual module training flags missing')
    if cumulative_calls is not None:
        require(record['total_committed_optimizer_calls'] == cumulative_calls, 'Raw worker cumulative return counter omits history')
        steps = []
        for entry in state['optimizer']['state'].values():
            value = entry['step']
            scalar = value.item() if hasattr(value, 'item') else value
            require(type(scalar) in (int, float) and scalar == cumulative_calls, 'Raw Adam step differs from actual cumulative writes')
            steps.append(scalar)
        require(cumulative_calls == 0 or steps, 'Raw Adam entries missing after actual writes')
    return dict(rank=rank, world_size=8, committed_optimizer_calls=calls, scheduler_before=before,
                scheduler_after=after, logical_sha256=record['logical_state_sha256'], model_adam_finite=True,
                gradients_finite=gradients_finite, state_finite=state_finite, raw_gradients_preserved=True)


def verify_driver_logical(state, partial, digest, finite, *, prior_accounting=None):
    from attempt_accounting import sampler_position
    identity, record = partial['identity'], partial['driver']
    require(state['version'] == 3 and state['snapshot_kind'] == PROTOCOL and state['identity'] == identity and
            digest(state) == record['logical_state_sha256'] and
            state['attempted_outer'] == partial['attempted_outer'] == identity['attempted_outer'] and
            state['completed_outer_updates'] == partial['completed_outer_updates'] == identity['completed_outer'] and
            record['completed_outer_updates'] == identity['completed_outer'], 'Actual partial driver identity/digest/boundary differs')
    require(all(state[flag] is False for flag in ('outer_completed', 'automatic_rollback', 'automatic_resume_supported')) and
            state['recovery_provenance'] is None, 'Partial driver cannot be resumed or inherit recovery')
    finite(state)
    sampler = state['consumed_sampler']
    epoch, cursor = sampler_position(identity['attempt_id'])
    require(sampler['rows'] == 169028 and sampler['batch_size'] == 512 and sampler['identity'] == identity['data_identity'] and
            sampler['shuffle'] is True and type(sampler['cursor']) is int and type(sampler['epoch']) is int and
            sampler['cursor'] == record['consumed_cursor'] == cursor and sampler['epoch'] == record['consumed_epoch'] == epoch and
            len(sampler['order']) == 169028 and all(type(value) is int for value in sampler['order']) and
            set(sampler['order']) == set(range(169028)) and sampler['generator'] is not None,
            'Actual consumed sampler permutation/cursor/generator differs')
    _rng(state['rng'])
    require(state['algorithm']['method'] == 'tree_grpo' and state['algorithm']['tree_rng'] is not None and
            type(state['algorithm']['stage']) is dict, 'Actual partial driver algorithm/Tree RNG missing')
    accounting = state['guard_continue_accounting']
    require(accounting['protocol'] == 'tree_guard_continue_fresh_rollout_v1' and
            accounting['training_attempts'] == identity['attempt_id'] and
            accounting['completed_outer_updates'] == identity['completed_outer'], 'Actual driver pre-result attempt accounting differs')
    if prior_accounting is not None:
        require(accounting == prior_accounting, 'Actual driver accounting is not the history before this result')
    return dict(logical_sha256=record['logical_state_sha256'], consumed_cursor=cursor, consumed_epoch=epoch,
                attempted_outer=identity['attempted_outer'], completed_outer=identity['completed_outer'],
                actual_rng_present=True, resume_permitted=False)


def verify_actor_inputs_logical(state, run, partial):
    identity, record = partial['identity'], partial['actor_inputs']
    require(type(state) is dict and set(state) == {'batch', 'non_tensor_batch', 'meta_info'}, 'Actual raw actor input structure differs')
    batch, metadata = state['batch'], state['meta_info']
    context = metadata['tree_partial_context']
    require(context == dict(protocol=PROTOCOL, identity=identity, directory=str(partial_directory(run, identity))) and
            metadata['baseline_outer_step'] == identity['attempted_outer'] and
            metadata['baseline_attempt_id'] == identity['attempt_id'] and
            metadata['baseline_completed_outer'] == identity['completed_outer'], 'Actual actor tree_partial_context/attempt metadata differs')
    require(type(batch) is dict and batch and set(batch) == set(record['fields']) and
            record['rows'] == 2560 and {'responses', 'attention_mask', 'loss_mask', 'old_log_probs', 'sampled_old_log_probs'} <= set(batch),
            'Actual partial actor input field/row coverage missing')
    def rows(value):
        return int(value.shape[0]) if hasattr(value, 'shape') else len(value)
    require(all(rows(value) == 2560 for value in batch.values()) and
            type(state['non_tensor_batch']) is dict and all(rows(value) == 2560 for value in state['non_tensor_batch'].values()),
            'Actual raw actor input tensor/non-tensor row coverage differs')
    return dict(actual_rows=2560, actual_fields=sorted(batch), actual_partial_context=context,
                behavior_mu_binding_verification='check_behavior_mu_cpu.py owns full mu binding',
                raw_invalid_numeric_values_preserved=True)

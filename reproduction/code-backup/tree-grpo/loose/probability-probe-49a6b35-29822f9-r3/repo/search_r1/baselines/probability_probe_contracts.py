"""CPU metadata contracts for the source-only, zero-optimizer probability probe.

This module neither loads tensors nor launches a job. A structurally valid frozen
selection is a prerequisite, never a GPU/weight/guard acceptance receipt. The
caller must bind every expected SHA to independently frozen original bytes.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import uuid

ASSET_ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
UNFROZEN = 'SOURCE_ONLY_SELECTION_UNFROZEN'
FROZEN = 'SELECTION_FROZEN_FROM_CPU_RESULTS'
IDENTITY_SHA_FIELDS = ('ready_sha256', 'source_sha256', 'config_sha256',
                       'parent_hf_sha256', 'parent_state_manifest_sha256', 'assets_sha256')
FIXED_CONFIG = {
    'data.train_batch_size': 640,
    'actor_rollout_ref.actor.ppo_mini_batch_size': 384,
    'actor_rollout_ref.actor.ppo_micro_batch_size': 16,
    'actor_rollout_ref.actor.ppo_epochs': 1,
    'actor_rollout_ref.actor.use_dynamic_bsz': False,
    'actor_rollout_ref.actor.ulysses_sequence_parallel_size': 1,
    'actor_rollout_ref.actor.fsdp_config.param_offload': True,
    'actor_rollout_ref.actor.fsdp_config.grad_offload': True,
    'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
    'actor_rollout_ref.actor.policy_safety.monitor_rows_per_rank': 32,
    'actor_rollout_ref.model.use_remove_padding': True,
    'actor_rollout_ref.rollout.log_prob_micro_batch_size': 32,
    'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': False,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
    'actor_rollout_ref.rollout.n': 1,
    'actor_rollout_ref.rollout.temperature': 1.0,
    'actor_rollout_ref.rollout.top_p': 1.0,
    'actor_rollout_ref.rollout.top_k': -1,
    'actor_rollout_ref.rollout.dtype': 'bfloat16',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_sha(value, label='SHA256'):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'Missing or invalid ' + label)
    return value


def _finite_json(value):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        require(math.isfinite(value), 'Nonfinite JSON number')
        return
    if type(value) is list:
        for item in value:
            _finite_json(item)
        return
    if type(value) is dict:
        require(all(type(key) is str for key in value), 'JSON object keys must be strings')
        for item in value.values():
            _finite_json(item)
        return
    raise ValueError('Unsupported JSON value: ' + type(value).__name__)


def strict_json_bytes(value):
    _finite_json(value)
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + '\n').encode('utf-8')


def strict_json_load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    result = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=constant)
    _finite_json(result)  # Also catches valid JSON exponent overflow, e.g. 1e999.
    return result


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def require_local_host(hostname=None):
    hostname = socket.gethostname() if hostname is None else hostname
    require(isinstance(hostname, str) and hostname.split('.')[0] == 'h100-9',
            'Probe assets and execution must remain on h100-9')
    return hostname


def local_asset_path(path, root=ASSET_ROOT, must_exist=True):
    """Reject relative/URI/traversal paths and symlinks escaping the asset root.

    The explicit root argument enables synthetic temporary-directory CPU tests;
    production selection metadata must declare exactly ASSET_ROOT.
    """
    root, path = Path(root), Path(path)
    require(root.is_absolute() and '..' not in root.parts, 'Asset root must be an absolute local path')
    require(root.resolve() == root, 'Asset root must not contain a symlink')
    require(path.is_absolute() and '..' not in path.parts, 'Asset path must be absolute without traversal')
    require(path != root and root in path.parents, 'Asset path escaped same-node root')
    resolved = path.resolve(strict=must_exist)
    require(resolved != root and root in resolved.parents, 'Asset path escaped same-node root')
    return resolved


def atomic_write_bytes_no_clobber(path, payload, root=ASSET_ROOT):
    """Publish a complete fsynced inode by an atomic link; never replace a target.

    A hard link, unlike rename/replace, fails atomically when another writer has
    already published the target. All temporary bytes stay in the same directory.
    """
    require(type(payload) is bytes, 'Artifact payload must be bytes')
    raw = Path(path)
    require(not os.path.lexists(raw), 'Artifact target already exists')
    local_asset_path(raw, root, must_exist=False)
    # Keep the original final name. Resolving it after the existence check could
    # redirect publication through a symlink another writer just created.
    target = raw.parent.resolve(strict=True) / raw.name
    require(target.parent == Path(root) or Path(root) in target.parent.parents,
            'Artifact parent escaped same-node root')
    require(target.parent.is_dir(), 'Artifact parent directory must already exist')
    directory = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = '.probability-probe-' + uuid.uuid4().hex + '.tmp'
    created = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        created = True
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target.name, src_dir_fd=directory, dst_dir_fd=directory,
                follow_symlinks=False)
        os.fsync(directory)
    finally:
        if created:
            os.unlink(temporary, dir_fd=directory)
        os.close(directory)
    return dict(path=str(target), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def atomic_write_json_no_clobber(path, value, root=ASSET_ROOT):
    return atomic_write_bytes_no_clobber(path, strict_json_bytes(value), root)


def files_aggregate(hashes):
    """Exactly the healthy-HF model_manifest convention, including JSON spaces."""
    require(type(hashes) is dict and bool(hashes), 'Missing complete file mapping')
    for name, value in hashes.items():
        require(type(name) is str and bool(name), 'Invalid file mapping key')
        require_sha(value)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def bind_exact_files(directory, expected_files, expected_aggregate, asset_root=ASSET_ROOT):
    directory = local_asset_path(directory, asset_root)
    require(directory.is_dir(), 'File bundle is not a directory')
    require(files_aggregate(expected_files) == require_sha(expected_aggregate),
            'Expected complete file aggregate disagrees with expected mapping')
    expected_paths = set()
    for name in expected_files:
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == name,
                'File mapping contains a noncanonical or escaped path')
        expected_paths.add(name)
    actual_paths = set()
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in directories + files:
            entry = Path(parent) / name
            require(not entry.is_symlink(), 'File bundle contains a symlink')
        for name in files:
            entry = Path(parent) / name
            require(stat.S_ISREG(entry.stat().st_mode), 'File bundle contains a nonregular file')
            actual_paths.add(str(entry.relative_to(directory)))
    require(actual_paths == expected_paths, 'Complete file set mismatch')
    actual = {name: file_sha(directory / name) for name in sorted(expected_paths)}
    require(actual == expected_files, 'Full file bytes SHA mismatch')
    require(files_aggregate(actual) == expected_aggregate, 'Actual complete file aggregate mismatch')
    return dict(path=str(directory), files=actual, sha256=expected_aggregate,
                full_files_read=True, acceptance=False)


def bind_sources(entries, asset_root=ASSET_ROOT):
    require(type(entries) is list and bool(entries), 'Missing frozen source files')
    records, seen = [], set()
    for entry in entries:
        require(type(entry) is dict and set(entry) == {'path', 'sha256'}, 'Invalid source file binding')
        path = local_asset_path(entry['path'], asset_root)
        require(path.is_file() and str(path) not in seen, 'Duplicate or non-file source binding')
        expected = require_sha(entry['sha256'])
        require(file_sha(path) == expected, 'Frozen source bytes changed: ' + str(path))
        seen.add(str(path))
        records.append(dict(path=str(path), sha256=expected))
    return records


def validate_fixed_config(config, world_size=8, additional_expected=None):
    require(type(world_size) is int and world_size == 8, 'Eight-rank world required')
    _finite_json(config)
    expected = dict(FIXED_CONFIG)
    for key, value in (additional_expected or {}).items():
        require(key not in expected or (type(value) is type(expected[key]) and value == expected[key]),
                'Additional config cannot override fixed protocol')
        expected[key] = value
    for key, wanted in expected.items():
        value = config
        for part in key.split('.'):
            require(type(value) is dict and part in value, 'Missing fixed config: ' + key)
            value = value[part]
        # Numerical floats may be written as integral YAML values; booleans
        # never pass as 0/1 and batch fields require actual integers.
        compatible = type(value) in (int, float) if type(wanted) is float else type(value) is type(wanted)
        require(compatible and value == wanted, 'Changed fixed config: ' + key)
    return dict(world_size=8, fixed_values=expected, original_local_old_micro=4,
                original_local_actor_micro=2, original_local_mini=48, acceptance=False)


def validate_parameter_inventory(serialized_keys, unique_parameter_names, serialized_to_unique,
                                 expected_serialized_count=435, expected_unique_count=434):
    """Separate serialized coverage from unique model parameters; no alias proof."""
    require(type(expected_serialized_count) is int and expected_serialized_count == 435
            and type(expected_unique_count) is int and expected_unique_count == 434,
            'Fixed HF110 serialized435/unique434 counts cannot change')
    require(type(serialized_keys) is list and type(unique_parameter_names) is list,
            'Full serialized and unique parameter name lists required')
    for names in (serialized_keys, unique_parameter_names):
        require(all(type(name) is str and bool(name) for name in names) and len(set(names)) == len(names),
                'Missing or duplicate parameter names')
    require(len(serialized_keys) == expected_serialized_count, 'Serialized key count must be 435')
    require(len(unique_parameter_names) == expected_unique_count, 'Unique parameter count must be 434')
    require(type(serialized_to_unique) is dict and set(serialized_to_unique) == set(serialized_keys),
            'Exhaustive serialized-to-unique key coverage differs')
    require(set(serialized_to_unique.values()) == set(unique_parameter_names),
            'Unique parameter coverage differs')
    tied = ('model.embed_tokens.weight', 'lm_head.weight')
    require(all(name in serialized_to_unique for name in tied), 'Serialized tied embedding/head keys missing')
    require(serialized_to_unique[tied[0]] == serialized_to_unique[tied[1]],
            'Embedding/head structural alias differs')
    groups = {}
    for name, unique in serialized_to_unique.items():
        groups.setdefault(unique, []).append(name)
    require([set(names) for names in groups.values() if len(names) > 1] == [set(tied)],
            'Unexpected structural aliases')
    return dict(serialized_tensor_keys=len(serialized_keys), unique_parameters=len(unique_parameter_names),
                all_serialized_keys_bound=True, structural_alias_groups=[list(tied)],
                same_gather_window_alias_proven=False, full_tensor_bytes_read=False, acceptance=False)


def _rows(rows, limit, label):
    require(type(rows) is list and bool(rows) and all(type(row) is int and 0 <= row < limit for row in rows),
            'Invalid or out-of-range ' + label)
    require(len(rows) == len(set(rows)), 'Duplicate ' + label)
    return rows


def validate_selection(manifest, expected_identity, inventory, allow_frozen=False):
    """Validate immutable metadata; real CPU-result SHA reads are caller-required.

    inventory keys name actual same-node snapshots; raw row_count describes the
    entire saved snapshot, NOT the <=20 selected rows. original_micro4_groups is
    the full original ordered grouping derived from SHA-bound original metadata.
    Frozen JSON bytes must themselves be SHA-bound by the runtime before calling.
    """
    _finite_json(manifest)
    require(type(allow_frozen) is bool, 'Runtime freeze requirement must be a boolean')
    require(type(manifest) is dict and type(manifest.get('schema_version')) is int
            and manifest['schema_version'] == 1,
            'Unsupported selection schema')
    require(manifest.get('asset_root') == str(ASSET_ROOT), 'Selection asset root differs from h100-9 root')
    status, frozen = manifest.get('status', UNFROZEN), manifest.get('frozen', False)
    require(status in (UNFROZEN, FROZEN) and type(frozen) is bool, 'Invalid selection freeze state')
    require((status == FROZEN) == frozen, 'Selection status and frozen flag disagree')
    identity = manifest.get('identity')
    require(type(identity) is dict and type(expected_identity) is dict and identity == expected_identity,
            'Selection original-byte identities changed')
    require(set(identity) == set(IDENTITY_SHA_FIELDS) | {'core_commit', 'source_files'},
            'Missing or extra selection identity fields')
    for field in IDENTITY_SHA_FIELDS:
        require_sha(identity[field], field)
    require(type(identity['core_commit']) is str and re.fullmatch('[0-9a-f]{40}', identity['core_commit']) is not None,
            'Missing full source core commit')
    sources = identity['source_files']
    require(type(sources) is list and bool(sources), 'Missing source file list')
    source_paths = set()
    for source in sources:
        require(type(source) is dict and set(source) == {'path', 'sha256'}, 'Invalid selection source binding')
        local_asset_path(source['path'], must_exist=False)
        require_sha(source['sha256'])
        require(source['path'] not in source_paths, 'Duplicate selection source binding')
        source_paths.add(source['path'])
    require(type(inventory) is dict and bool(inventory), 'Missing actual asset inventory')
    asset_hashes = {}
    for key, asset in inventory.items():
        require(type(key) is str and type(asset) is dict, 'Invalid inventory item')
        local_asset_path(asset['path'], must_exist=False)
        require_sha(asset['sha256'])
        require(type(asset.get('rank')) is int and 0 <= asset['rank'] < 8, 'Invalid asset rank')
        require(asset.get('kind') in ('raw', 'monitor'), 'Invalid asset population')
        require(type(asset.get('row_count')) is int and asset['row_count'] > 0, 'Missing actual saved row count')
        require(asset['path'] not in asset_hashes, 'Duplicate asset path')
        asset_hashes[asset['path']] = asset['sha256']
        if asset['kind'] == 'raw':
            groups = asset.get('original_micro4_groups')
            expected_groups = [list(range(start, min(start + 4, asset['row_count'])))
                               for start in range(0, asset['row_count'], 4)]
            require(type(groups) is list and all(type(group) is list and
                    all(type(row) is int for row in group) for group in groups) and groups == expected_groups,
                    'Original raw micro4 grouping is incomplete or reordered')
        else:
            require(asset['row_count'] == 32, 'Saved monitor population must be exactly 32/rank')
    require(files_aggregate(asset_hashes) == identity['assets_sha256'], 'Actual asset inventory aggregate differs')
    selections = manifest.get('selections')
    require(type(selections) is list and len(selections) == 8, 'Exactly eight rank selections required')
    ranks, raw_counts = set(), {}
    for selection in selections:
        require(type(selection) is dict and set(selection) == {'rank', 'raw_groups', 'monitor'},
                'Invalid rank selection fields')
        rank = selection['rank']
        require(type(rank) is int and 0 <= rank < 8 and rank not in ranks, 'Duplicate or invalid selection rank')
        ranks.add(rank)
        raw_groups = selection['raw_groups']
        require(type(raw_groups) is list and bool(raw_groups), 'Missing selected raw complete groups')
        seen, count, previous_groups = set(), 0, {}
        for group in raw_groups:
            require(type(group) is dict and set(group) == {'asset', 'source_group', 'row_indices'},
                    'Invalid raw group selection')
            key = group['asset']
            require(key in inventory, 'Selected raw asset missing from inventory')
            asset = inventory[key]
            require(asset['kind'] == 'raw' and asset['rank'] == rank, 'Raw population/rank mismatch')
            number = group['source_group']
            require(type(number) is int and 0 <= number < len(asset['original_micro4_groups']),
                    'Invalid original raw group number')
            require(number > previous_groups.get(key, -1), 'Selected raw original groups reordered')
            previous_groups[key] = number
            rows = _rows(group['row_indices'], asset['row_count'], 'raw row indices')
            require(rows == asset['original_micro4_groups'][number], 'Raw selection cropped or reordered original micro4')
            tokens = {(key, row) for row in rows}
            require(not seen.intersection(tokens), 'Raw row selected more than once')
            seen.update(tokens)
            count += len(rows)
        require(0 < count <= 20, 'Raw selected row count exceeds 20/rank')
        raw_counts[rank] = count
        monitor = selection['monitor']
        require(type(monitor) is dict and set(monitor) == {'asset', 'row_indices'}, 'Invalid monitor selection')
        require(monitor['asset'] in inventory, 'Selected monitor asset missing from inventory')
        asset = inventory[monitor['asset']]
        require(asset['kind'] == 'monitor' and asset['rank'] == rank, 'Monitor population/rank mismatch')
        rows = _rows(monitor['row_indices'], asset['row_count'], 'monitor row indices')
        require(rows == list(range(32)), 'All 32 monitor rows must retain original order')
    require(ranks == set(range(8)), 'Selection rank coverage incomplete')
    cpu_results = manifest.get('cpu_results', {})
    if frozen:
        require(type(cpu_results) is dict and set(cpu_results) == {'position', 'action'},
                'Frozen selection requires actual position/action CPU evidence bindings')
        for entry in cpu_results.values():
            require(type(entry) is dict and set(entry) == {'path', 'sha256'}, 'Invalid CPU evidence binding')
            local_asset_path(entry['path'], must_exist=False)
            require_sha(entry['sha256'])
    else:
        require(cpu_results == {}, 'Unfrozen selection must not imply actual CPU results')
    require(not allow_frozen or frozen, 'SOURCE_ONLY_SELECTION_UNFROZEN: probe execution forbidden')
    return dict(status=status, frozen=frozen, runtime_selection_precondition=bool(allow_frozen and frozen),
                raw_selected_rows_per_rank=[raw_counts[rank] for rank in range(8)], monitor_selected_rows_per_rank=[32] * 8,
                real_cpu_evidence_bytes_verified=False, real_gpu_execution_verified=False, acceptance=False)

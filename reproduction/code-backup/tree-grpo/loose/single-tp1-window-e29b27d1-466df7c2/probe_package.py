"""Pure metadata contract for the isolated, still-unreleased single TP1 window diagnostic.

No Torch import, asset loading, service start, submission, or implicit release.
"""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
SELECTION_SHA = 'bca59a4508a0627a04b3f1f64e632b69488208650fa375aac52ab7d16029f995'
OUTPUT_TOKEN = '@OWN_RUN_OUTPUT@'
DISPATCH_INTENT = str(ROOT/'instruct300/SINGLE_TP1_WINDOW_SCHEDULER_REPAIR3286_ONCE.json')
RELEASE_STATUS = 'ZERO_UPDATE_GPU_PROBE_SOURCE_AND_DISPATCH_INDEPENDENTLY_RELEASED'
BUNDLE_STATUS = 'GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED'
WINDOW_PROFILE = 'original_tp1_vllm_single_window'
RUNNER_MODULE = 'single_window_runner'
WINDOW_READY_STATUS = 'SINGLE_ORIGINAL_TP1_WINDOW_OBSERVED_NO_PROBABILITY_OR_TRAINING_ACCEPTANCE'
WINDOW_CONTRACT = dict(generation_call_id=5, generation_invocations_per_rank=1, world_size=8,
    prompts_per_rank=320, actor_microbatch_rows=4, actor_comparison_calls_per_rank=80,
    emitted_cap_per_row=500, prompts_global=2560, maximum_new_emitted_tokens_global=1280000,
    full_vocab_anchors_global=9)
ORIGINAL_READY_SHA = 'ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468'
ORIGINAL_CONFIG_SHA = 'e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b'
KERNEL_BINARY_PINS = [
    dict(path=str(ROOT/'env/lib/python3.10/site-packages/vllm/_C.abi3.so'), bytes=203177408,
         sha256='bc7bf8aa17c01d6ba2f267f3fdde46a06459aa508257cf2ae208e3e26268bac4'),
    dict(path=str(ROOT/'env/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so'), bytes=603504616,
         sha256='b94d1bdfd8db0caec1ba775219e7cd786d9c521a1541f5d1b7ba2052d2b3ae1e')]

PROTOCOL = dict(node='h100-9', nodes=1, tasks=1, gpu_count=8, exclusive=True,
                requested_cpus=32, memory_request='256G', time_limit_minutes=30,
                no_requeue=True, optimizer_writes=0, backward_calls=0)
PROTECTED = {1680782:1502352181,1680783:1502352181,1680784:1502352181,1680785:1502352181,
             1696443:1502398762,1696444:1502398762,1696445:1502398762,1696446:1502398762}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def duplicate_free(pairs):
    output = {}
    for key, value in pairs:
        require(key not in output, 'Duplicate JSON key: '+key)
        output[key] = value
    return output


def decode(payload):
    def reject_constant(value):
        raise ValueError('Non-finite JSON: '+value)
    value = json.loads(payload.decode('utf-8'), object_pairs_hook=duplicate_free,
                       parse_constant=reject_constant)
    # The JSON parser also admits overflowing finite literals such as 1e999.
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'Non-finite parsed JSON number')
        elif isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
    finite(value)
    return value


def stat_identity(record):
    return (record.st_dev, record.st_ino, record.st_size, record.st_mtime_ns,
            record.st_ctime_ns)


def read_bound(path, expected):
    """Parse the very bytes hashed on one held FD; reject path/stat mutation."""
    require(re.fullmatch('[0-9a-f]{64}', expected or '') is not None, 'Invalid SHA')
    path = Path(path)
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        require(hashlib.sha256(payload).hexdigest() == expected, 'Changed bytes: '+str(path))
        value = decode(payload)
        after = os.fstat(stream.fileno())
        require(stat_identity(before) == stat_identity(after) == stat_identity(path.stat()),
                'Metadata FD/path identity changed: '+str(path))
    return value


def file_sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for part in iter(lambda: stream.read(8*1024**2), b''): h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    """Publish new evidence once, with durable file and directory metadata."""
    path = Path(path)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())
    finally:
        directory = os.open(str(path.parent), os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    return dict(path=str(path), sha256=file_sha(path))


def same_node_path(value, root=ROOT):
    path = Path(value)
    require(path.is_absolute(), 'Absolute h100-9 task path required')
    resolved = path.resolve()
    require(resolved == root or root in resolved.parents, 'Path escaped own task root')
    return resolved


def runtime_output(job):
    require(type(job) is str and re.fullmatch('[1-9][0-9]*', job) is not None, 'Invalid Slurm job')
    return str(ROOT/'runs'/('tree-probability-probe-'+job)/'gpu-records')


def bundle_payload(bundle, job=None):
    """One dynamic field and one excluded envelope reference, both explicit."""
    value = copy.deepcopy(bundle)
    value.pop('release', None)
    expected = OUTPUT_TOKEN if job is None else runtime_output(job)
    require(value.get('output') == expected, 'Output differs from frozen job rule')
    require(canonical({k:v for k,v in value.items() if k!='output'}).count(
            OUTPUT_TOKEN.encode()) == 0, 'Output token appears outside sole output field')
    value['output'] = OUTPUT_TOKEN
    return value


def window_anchors():
    values = [dict(id='rank%02d-first'%rank, rank=rank, input_row=0, response_offset=0)
              for rank in range(8)]
    values.append(dict(id='rank04-row160-offset17', rank=4, input_row=160, response_offset=17))
    return values


def validate_window_inputs(value):
    """Validate only the signed metadata plan; rank CPU preparation is separate.

    This routine does not assert that a PT has been loaded or prompts prepared.
    All actual tensor-byte checks must finish through the all-rank CPU gate.
    """
    require(type(value) is dict and value.get('status') ==
            'ORIGINAL_CALL5_METADATA_PLAN_FROZEN_PENDING_TENSOR_READ' and
            value.get('actual_tensor_rows_verified') is False and value.get('generation_call') == 5,
            'Only the complete original call5 window is allowed')
    require(value.get('original_selection_sha256') == SELECTION_SHA and
            value.get('original_ready_sha256') == ORIGINAL_READY_SHA and
            value.get('original_config_sha256') == ORIGINAL_CONFIG_SHA,
            'Original call5 provenance changed')
    require(canonical(value.get('anchors')) == canonical(window_anchors()), 'Fixed global9 anchor plan changed')
    require(type(value.get('maximum_new_emitted_tokens')) is int and
            value['maximum_new_emitted_tokens'] == 1280000, 'Full window token budget changed')
    position = value.get('source_position_report')
    require(type(position) is dict and set(position) == {'path', 'sha256'},
            'Prior accepted CPU position report pin missing')
    same_node_path(position['path'])
    require(re.fullmatch('[0-9a-f]{64}', position['sha256'] or ''), 'Invalid position report SHA')
    rows = value.get('ranks')
    require(type(rows) is list and len(rows) == 8 and
            [r.get('rank') for r in rows] == list(range(8)), 'Eight canonical complete ranks required')
    for rank, row in enumerate(rows):
        require(type(row.get('rank')) is int and row['rank'] == rank and
                type(row.get('local_rows')) is int and row['local_rows'] == 320 and
                type(row.get('expected_actor_micro4_calls')) is int and
                row['expected_actor_micro4_calls'] == 80 and
                row.get('original_physical_rows') == list(range(320)) and
                all(type(x) is int for x in row['original_physical_rows']),
                'Original rank320 prompts/80micro4/full row order changed')
        require(type(row.get('prompt_width')) is int and 0 < row['prompt_width'] <= 4096,
                'Original prompt width missing or exceeds cap')
        identities = row.get('source_full_input_identity')
        require(type(identities) is dict and set(identities) == {'input_ids','attention_mask','position_ids'},
                'Complete original fullinput identities required')
        for identity in identities.values():
            require(type(identity) is dict and identity.get('dtype') == 'torch.int64' and
                    identity.get('shape') == [320,row['prompt_width']+500] and
                    re.fullmatch('[0-9a-f]{64}', identity.get('shape_and_bytes_sha256','')) is not None,
                    'Original padded fullinput dtype/shape/bytes identity changed')
        for field in ('source_snapshot', 'source_record'):
            entry = row.get(field)
            require(type(entry) is dict and set(entry) == {'path', 'sha256'},
                    'Original complete call5 source pin missing: '+field)
            same_node_path(entry['path'])
            require(re.fullmatch('[0-9a-f]{64}', entry['sha256'] or ''), 'Invalid input source SHA')
    return value


def validate_window_envelope(package, template, release):
    for value in (package, template, release):
        require(value.get('diagnostic_profile') == WINDOW_PROFILE,
                'Old native release cannot authorize a vLLM generation window')
        require(canonical(value.get('window_contract')) == canonical(WINDOW_CONTRACT),
                'Single-window budget/observation contract changed')
        require(value.get('runner_module') == RUNNER_MODULE, 'Wrong single-window entry module')
    require(template.get('window_profile') == WINDOW_PROFILE, 'Runtime profile changed')
    inputs = validate_window_inputs(template.get('window_inputs'))
    wanted = digest(inputs)
    for value in (package, template, release):
        require(value.get('window_inputs_sha256') == wanted, 'Complete input metadata payload changed')
    require(type(template.get('vllm_installed_sources')) is dict and template['vllm_installed_sources'],
            'Explicit installed vLLM source set missing')
    for value in (package, template, release):
        require(value.get('installed_kernel_binaries') == KERNEL_BINARY_PINS,
                'Native kernel binary metadata pins changed')
    require(package.get('vllm_installed_sources') == template['vllm_installed_sources'],
            'Installed vLLM source mappings differ')
    return inputs


def validate_release(package, package_sha, template, release):
    """A status string never substitutes a separately reviewed envelope."""
    require(release.get('status') == RELEASE_STATUS, 'Independent GPU release missing')
    validate_window_envelope(package, template, release)
    require(package.get('schema_version') == 1 and canonical(package.get('protocol')) == canonical(PROTOCOL),
            'Package protocol changed')
    require(release.get('package_manifest_sha256') == package_sha, 'Release/package mismatch')
    wanted = digest(bundle_payload(template))
    require(package.get('bundle_payload_sha256') == wanted == release.get('bundle_payload_sha256'),
            'Release/full bundle content mismatch')
    require(package.get('selection_sha256') == SELECTION_SHA == release.get('selection_sha256') ==
            template['selection_file']['sha256'], 'Actual CPU3277 selection changed')
    commit = package.get('runner_commit', '')
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None and commit ==
            release.get('runner_commit') == template.get('runner_commit'), 'Runner commit differs')
    require(canonical(release.get('protocol')) == canonical(PROTOCOL),'Release allocation differs')
    require(type(release.get('actual_optimizer_writes')) is int and release['actual_optimizer_writes'] == 0,
            'Probe zero-write scope differs')
    require(str(same_node_path(release['dispatch_intent_path']))==DISPATCH_INTENT,
            'Single window must use its new fixed intent; old or arbitrary intent paths are refused')
    require(template.get('status') == BUNDLE_STATUS, 'Wrong runner bundle label')
    require(package.get('reviewed_resource_adapter') == template.get('reviewed_resource_adapter') and
            release.get('adapter_sha256') == template['reviewed_resource_adapter']['sha256'],
            'Adapter source binding differs')
    # Full bundle payload SHA signs every source/installed entry without repeating
    # large arrays or introducing a second recursive approval envelope.
    for field in ('probe_source_files','installed_sources','installed_versions'):
        require(package.get(field) == template.get(field),'Complete package/bundle mapping differs: '+field)
    return True


def validate_package_files(package_root, package):
    root = Path(package_root).resolve()
    files = package.get('files')
    require(type(files) is dict and files, 'Full immutable package file manifest missing')
    required = {'start.sh', 'start.py', 'supervisor.py', 'probe_package.py',
                'dispatch_once.py', 'bundle-template.json',
                'repo/search_r1/baselines/preflight.py', 'repo/single_window_runner.py',
                'repo/window_runtime.py', 'repo/window_observer.py', 'repo/window_contracts.py',
                'repo/vllm_window_weights.py', 'resource_adapter_window.py'}
    require(required <= set(files), 'Incomplete launcher/package source manifest')
    actual=set()
    for path in root.rglob('*'):
        parts=path.relative_to(root).parts
        if '.git' in parts:continue
        require('__pycache__' not in parts and path.suffix not in ('.pyc','.pyo'),
                'Unbound Python bytecode in immutable source package')
        require(not path.is_symlink(),'Alias in immutable package')
        if path.is_file() and path.name!='package-manifest.json':actual.add(path.relative_to(root).as_posix())
    require(actual==set(files),'Unlisted/missing immutable package files')
    for name, wanted in files.items():
        path = root/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Package manifest relative path escaped')
        require(path.resolve() == path and path.is_file(), 'Package file alias or absent: '+name)
        require(file_sha(path) == wanted, 'Frozen package file changed: '+name)
    for path, wanted in package.get('external_files', {}).items():
        require(file_sha(same_node_path(path)) == wanted, 'External runtime helper changed')
    require(set(package.get('external_files', {})) >=
            {str(ROOT/'tools/tree_shared_owned_runtime_fixed.py'),
             str(ROOT/'tools/resource_guard_identity.py'), str(ROOT/'tools/expected-gpu-uuids.json')},
            'Original owned-runtime/guard/UUID helper binding incomplete')


def allocation(text, job):
    for token in ('JobId='+job, 'JobState=RUNNING', 'NodeList=h100-9', 'OverSubscribe=NO'):
        require(re.search(r'(?:^|\s)'+re.escape(token)+r'(?:\s|$)', text), 'Allocation token missing: '+token)
    require(re.search(r'TresPerNode=gres:gpu(?::h100)?:8(?:\s|$)',text), '8GPU allocation missing')
    require(re.search(r'CPUs/Task=32(?:\s|$)',text), 'Requested CPU32 changed')
    require(re.search(r'TimeLimit=00:30:00(?:\s|$)',text), '30 minute bound changed')
    require(re.search(r'Requeue=0(?:\s|$)',text), 'Actual no-requeue proof missing')
    # Slurm normalizes --mem=256G to MiB; exact request, actual CPU allocation separate.
    require(re.search(r'MinMemoryNode=(?:256G|262144M)(?:\s|$)',text), '256G request changed')
    match = re.search(r'NumCPUs=(\d+)', text)
    require(match is not None and int(match.group(1)) >= 32, 'Actual CPU allocation missing')
    return int(match.group(1))


def guard_record(record, job, uuids, now):
    require(record.get('job_id') == job and not record.get('event'), 'Guard job or STOP')
    age = now-record.get('epoch', -math.inf)
    require(math.isfinite(age) and 0 <= age <= 15, 'Guard stale/future/nonfinite')
    gpus = record.get('gpus', [])
    require(len(gpus) == 8 and {x.get('uuid') for x in gpus} == set(uuids), 'Guard UUIDs changed')
    for gpu in gpus:
        memory, sm = gpu.get('foreign_memory_fraction'), gpu.get('foreign_sm_upper')
        require(type(memory) in (int,float) and math.isfinite(memory) and 0 <= memory <= .15 and
                type(sm) in (int,float) and math.isfinite(sm) and 0 <= sm <= 15,
                'Foreign GPU use exceeds dual15 or unknown')
    return record


def validate_owned_lease_fds(records, uuids, lease_root=None):
    """Live same-OFD portion of accepted preflight; own GPU load is excluded.

    The original complete preflight performs total occupancy checks at startup.
    During FSDP forwards own memory/SM may exceed15%; the identity-aware guard
    owns ongoing foreign dual15. This routine proves the parent still owns each
    original inherited lease, rather than merely observing some lock holder.
    """
    require(len(records)==8 and len(uuids)==8 and len(set(uuids))==8 and
            {row['uuid'] for row in records}==set(uuids),'Live lease inventory changed')
    root=Path(lease_root) if lease_root is not None else Path('/tmp')/('search-r1-gpu-leases-'+str(os.getuid()))
    require(len({row['fd'] for row in records})==8,'Duplicate live lease descriptors')
    for row in records:
        fd=row['fd']; require(type(fd) is int and fd>=0,'Invalid inherited lease FD')
        path=root/(row['uuid']+'.lock');before=os.fstat(fd);disk=path.stat()
        require((before.st_dev,before.st_ino)==(disk.st_dev,disk.st_ino),'Lease FD/inode differs')
        probe=os.open(str(path),os.O_RDWR|getattr(os,'O_NOFOLLOW',0))
        try:
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise ValueError('Original GPU lease no longer locked')
            # Idempotent only if this inherited open-file-description owns it.
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            after=os.fstat(fd);disk=path.stat()
            require((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino)==
                    (disk.st_dev,disk.st_ino),'Lease identity changed during live check')
        finally:os.close(probe)
    return True


def validate_window_ready(row, rank, bundle, bundle_sha256):
    """Completion contract only; finite numeric disagreement is a valid result."""
    require(row.get('status') == WINDOW_READY_STATUS and type(row.get('rank')) is int and
            row['rank'] == rank and row.get('world_size') == 8,
            'Old native or wrong-rank ready is not a new window measurement')
    require(row.get('source_and_input_bundle_identity') == bundle['identity'] and
            row.get('window_profile') == WINDOW_PROFILE and row.get('bundle_sha256') == bundle_sha256 and
            row.get('window_inputs_sha256') == bundle['window_inputs_sha256'], 'Rank window input/source identity changed')
    expected = dict(original_generation_call=5, original_prompt_rows=320, generation_invocations=1,
                    actor_scoring_native_forwards=80, warmup_native_forwards=1, actor_native_forwards=81,
                    maximum_new_emitted_tokens=160000, optimizer_committed=0)
    for field, wanted in expected.items():
        require(type(row.get(field)) is int and row[field] == wanted, 'Actual window count differs: '+field)
    for field in ('optimizer_instantiated','scheduler_instantiated','scheduler_advanced','backward_called',
                  'probability_accepted','training_accepted','GPU_capacity_proven'):
        require(row.get(field) is False, 'Unexpected update or acceptance: '+field)
    require(row.get('original_master_state_unchanged') is True and row.get('original_modes_restored') is True,
            'Actual actor master/gradients/buffers/mode invariance missing')
    require(type(row.get('actual_new_emitted_tokens')) is int and 0 <= row['actual_new_emitted_tokens'] <= 160000,
            'Actual emitted count missing or exceeds frozen budget')
    require(row.get('actual_loaded_binary_identity_proven') is False,
            'Path hash metadata must not claim actual loaded binary identity')
    cpu = row.get('cpu_preparation',{})
    local = bundle['window_inputs']['ranks'][rank]
    require(cpu.get('status') == 'ACTUAL_CALL5_CPU_INPUTS_VERIFIED_BEFORE_CUDA' and
            cpu.get('actual_tensor_rows_verified') is True and cpu.get('cuda_initialized') is False and
            cpu.get('rank') == rank and cpu.get('local_rows') == 320 and cpu.get('original_generation_call') == 5 and
            cpu.get('expected_actor_micro4_calls') == 80 and
            cpu.get('source_snapshot') == local['source_snapshot'] and cpu.get('source_record') == local['source_record'] and
            cpu.get('source_full_input_identity') == local['source_full_input_identity'] and
            cpu.get('window_inputs_sha256') == bundle['window_inputs_sha256'] and
            type(cpu.get('model_forwards')) is int and cpu['model_forwards'] == 0 and
            type(cpu.get('optimizer_writes')) is int and cpu['optimizer_writes'] == 0,
            'Actual all-rank pre-CUDA tensor preparation/source identity not completed')
    require(row.get('services',{}).get('ray_initialized') is False and
            row['services'].get('vllm_engine_required') is True and
            row['services'].get('optimizer_instantiated') is False, 'Unexpected native services')
    parallel = row.get('actual_runtime',{}).get('configs',{}).get('parallel_config',{})
    require(type(parallel.get('tensor_parallel_size')) is int and parallel['tensor_parallel_size'] == 1,
            'Actual native TP1 engine proof missing')
    wrappers = row.get('wrappers', {})
    counts = wrappers.get('counts', {})
    require(type(counts.get('sync')) is int and type(counts.get('output')) is int and
            counts['sync'] == counts['output'] == 1 and type(counts.get('execute')) is int and
            counts['execute'] > 0 and wrappers.get('registered_requests') == 320,
            'Actual single native window/request count proof missing')
    observer = row.get('observer', {})
    require(observer.get('status') == 'PASSIVE_WINDOW_OBSERVATION_COMPLETE' and
            observer.get('observer_evidence_complete') is True and observer.get('hooks_restored') is True and
            observer.get('errors') == [] and observer.get('probability_accepted') is False and
            observer.get('training_accepted') is False and observer.get('observer_replaced_native_outputs') is False and
            observer.get('observer_changed_sampling_parameters') is False, 'Passive observer refused or changed native behavior')
    generated = observer.get('generated')
    require(type(generated) is list and len(generated) == 320 and
            all(type(x.get('input_row')) is int for x in generated) and
            sorted(x.get('input_row') for x in generated) == list(range(320)), 'Native final outputs omit/duplicate original rows')
    lengths = [x.get('true_response_length') for x in generated]
    require(all(type(n) is int and 0 <= n <= 500 for n in lengths) and
            sum(lengths) == row['actual_new_emitted_tokens'], 'True emitted metadata count differs')
    expected_anchors = [x for x in window_anchors() if x['rank'] == rank]
    anchors = observer.get('anchors')
    require(type(anchors) is list and len(anchors) == len(expected_anchors) and
            len({x.get('id') for x in anchors}) == len(expected_anchors), 'Missing/duplicate planned anchors')
    by_id = {x['id']:x for x in anchors}
    for wanted in expected_anchors:
        actual = by_id.get(wanted['id'], {})
        require(canonical({key:actual.get(key) for key in wanted}) == canonical(wanted), 'Wrong original anchor geometry')
        require(actual.get('status') in ('observed','absent_native_early_finish'), 'Unexplained absent anchor')
        if actual['status'] == 'absent_native_early_finish':
            require(type(actual.get('true_response_length')) is int and
                    actual['true_response_length'] <= wanted['response_offset'] and
                    actual.get('finish_reason') == 'stop', 'Anchor absent without actual early finish')
    paired = row.get('anchor_observations')
    require(type(paired) is list and len(paired) == len(expected_anchors) and
            len({x.get('anchor_id') for x in paired}) == len(expected_anchors),
            'Missing/duplicate actual cross-backend anchor observations')
    pair_by_id = {x['anchor_id']:x for x in paired}
    for wanted in expected_anchors:
        actual, pair = by_id[wanted['id']], pair_by_id.get(wanted['id'],{})
        require(pair.get('input_row') == wanted['input_row'] and pair.get('response_offset') == wanted['response_offset'] and
                pair.get('probability_accepted') is False, 'Cross-backend anchor identity/scope changed')
        if actual['status'] == 'absent_native_early_finish':
            require(pair.get('status') == 'absent_native_early_finish' and
                    pair.get('true_response_length') == actual['true_response_length'] and
                    pair.get('finish_reason') == 'stop', 'Paired anchor absence lacks actual early finish')
            continue
        require(pair.get('status') == 'SAME_CAUSAL_CONTEXT_LOGITS_OBSERVED' and
                pair.get('training_accepted') is False and
                type(pair.get('vllm_context_token_ids')) is list and pair['vllm_context_token_ids'] and
                pair['vllm_context_token_ids'] == pair.get('actor_context_token_ids') and
                type(pair.get('vllm_context_positions')) is list and
                pair['vllm_context_positions'] == pair.get('actor_context_positions') and
                len(pair['vllm_context_positions']) == len(pair['vllm_context_token_ids']),
                'Actual backend anchors have unmatched causal token/position keys')
        shape = pair.get('vocab_shape')
        target = pair.get('vllm_target_token_id')
        require(type(shape) is list and len(shape) == 1 and type(shape[0]) is int and shape[0] > 0 and
                type(target) is int and target == pair.get('actor_target_token_id') and 0 <= target < shape[0],
                'Actual anchor target/vocabulary identity changed')
        vectors = pair.get('tensor_identities',{})
        require(set(vectors) == {'head','sampler_input','actor'} and
                all(x.get('shape') == shape for x in vectors.values()), 'Full anchor vector identity incomplete')
        for field in ('full_vocab_actor_minus_vllm_head','full_vocab_actor_minus_vllm_head_after_constant',
                      'full_vocab_sampler_input_minus_head'):
            require(pair.get(field,{}).get('count') == shape[0], 'Incomplete full vocabulary observation count')
    weights = row.get('live_window_weights', {})
    require(weights.get('status') == 'LIVE_SYNC_AND_PRE_OFFLOAD_WEIGHTS_OBSERVED_NO_PROBABILITY_ACCEPTANCE' and
            weights.get('buffers_compared') is True and weights.get('probability_accepted') is False and
            weights.get('training_accepted') is False and weights.get('historical_3268_identity_proven') is False,
            'Live window weight observation scope changed')
    pre, post = weights.get('pre', {}), weights.get('post', {})
    require(pre.get('serialized_source_keys') == 435 and pre.get('serialized_target_keys') == 255 and
            pre.get('tensor_parallel_size') == 1 and pre.get('actual_device_is_cuda') is True and
            type(pre.get('all_parameter_values_match')) is bool and
            post.get('status') == 'LIVE_PRE_OFFLOAD_BYTES_AND_ALIAS_UNCHANGED' and
            post.get('same_window_tie') is True and post.get('CPU_offload_snapshot_used') is False,
            'Incomplete live original435-to-packed255 or alias evidence')
    # all_parameter_values_match may be False: preserve that measurement.
    require(type(row.get('raw_probability_observations')) is dict, 'No finite raw probability measurement metadata')
    return row


def rank_records(output, bundle, bundle_sha256):
    records = []
    for rank in range(8):
        path = Path(output)/('rank-%02d'%rank)/'ready.json'
        wanted = file_sha(path)
        row = read_bound(path, wanted)
        validate_window_ready(row, rank, bundle, bundle_sha256)
        evidence = row.get('tensor_evidence', {})
        require(type(evidence) is dict and set(evidence) == {'path','sha256','bytes'} and
                type(evidence['bytes']) is int and evidence['bytes'] > 0 and
                Path(evidence['path']).resolve() == Path(evidence['path']) and
                Path(evidence['path']).resolve() == (Path(output)/('rank-%02d'%rank)/'window.pt').resolve(),
                'Tensor evidence must stay in this own rank directory')
        require(Path(evidence['path']).stat().st_size == evidence['bytes'] and
                file_sha(evidence['path']) == evidence['sha256'], 'Actual own rank evidence bytes changed')
        records.append(dict(rank=rank, path=str(path), sha256=wanted, record=row))
    return records

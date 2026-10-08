"""Pure metadata contract for the isolated, still-unreleased GPU diagnostic.

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
DISPATCH_INTENT = str(ROOT/'instruct300/PROBABILITY_PROBE_HEADER_REPAIR_ONCE-3280.json')
RELEASE_STATUS = 'ZERO_UPDATE_GPU_PROBE_SOURCE_AND_DISPATCH_INDEPENDENTLY_RELEASED'
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


def validate_release(package, package_sha, template, release):
    """A status string never substitutes a separately reviewed envelope."""
    require(release.get('status') == RELEASE_STATUS, 'Independent GPU release missing')
    require(package.get('schema_version') == 1 and package.get('protocol') == PROTOCOL,
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
    require(release.get('protocol') == PROTOCOL,'Release allocation differs')
    require(type(release.get('actual_optimizer_writes')) is int and release['actual_optimizer_writes'] == 0,
            'Probe zero-write scope differs')
    require(str(same_node_path(release['dispatch_intent_path']))==DISPATCH_INTENT,
            'Header repair must use its new fixed intent; old or arbitrary intent paths are refused')
    require(template.get('status') == 'GPU_PROBE_BUNDLE_INDEPENDENTLY_RELEASED', 'Wrong runner bundle label')
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
                'repo/search_r1/baselines/preflight.py'}
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


def rank_records(output, identity):
    records = []
    for rank in range(8):
        path = Path(output)/('rank-%02d'%rank)/'ready.json'
        row = decode(path.read_bytes())
        require(row.get('rank') == rank and row.get('world_size') == 8 and row.get('status') ==
                'SPECIFIED_INPUT_NATIVE_FSDP_DIAGNOSTICS_COMPLETE_NO_TRAINING_ACCEPTANCE',
                'Missing actual rank diagnostic completion')
        require(row.get('source_and_input_bundle_identity') == identity, 'Rank identity changed')
        for field in ('optimizer_instantiated','scheduler_instantiated','scheduler_advanced','backward_called',
                      'training_acceptance','probability_consistency_acceptance','gpu_training_capacity_proven'):
            require(row.get(field) is False, 'Unexpected rank update/acceptance: '+field)
        require(row.get('optimizer_committed') == 0 and row.get('full_state_unchanged_since_postwarm_baseline') is True
                and row.get('original_module_modes_restored') is True, 'Rank state invariance failed')
        require(type(row.get('original_forward_calls')) is int and row['original_forward_calls'] > 1,
                'No actual original forward evidence')
        records.append(dict(rank=rank, path=str(path), sha256=file_sha(path), record=row))
    return records

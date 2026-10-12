"""Read-only target code/proof SHA and inherited asset metadata; no CUDA."""
import datetime
import inspect
import json
import socket
import subprocess
import sys
from pathlib import Path
from deployment_contract_v1 import (SOURCE, REPO, ENV_PYTHON, HELPERS, CONTROLS, require,
    sha, read_bound, validate_candidate, validate_physical, write_once, closure)


def metadata(path, digest):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, sha256=digest, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def verify(path, expected, output):
    candidate = read_bound(dict(path=str(path), sha256=expected))
    validate_candidate(candidate)
    require(socket.gethostname().split('.')[0] == 'h100-8' and sys.executable == str(ENV_PYTHON), 'Wrong node/interpreter')
    inherited = read_bound(candidate['inherited_physical'])
    inherited_files = closure(inherited['metadata'], 'inherited physical metadata')
    actual, hashed = [], []
    for entry in candidate['files']:
        item = metadata(entry['path'], entry['sha256'])
        old = inherited_files.get(entry['path'])
        if old is not None and not entry['path'].startswith(str(REPO) + '/'):
            require({key: item[key] for key in ('path', 'bytes', 'sha256', 'mtime_ns', 'device', 'inode')} ==
                    {key: old[key] for key in ('path', 'bytes', 'sha256', 'mtime_ns', 'device', 'inode')},
                    'Inherited asset/environment identity changed: ' + entry['path'])
        else:
            require(sha(entry['path']) == entry['sha256'], 'New source/helper/control changed')
            hashed.append(entry['path'])
        require(item['bytes'] == entry['bytes'], 'Actual file size differs')
        actual.append(item)
    proofs = []
    for entry in candidate['evidence']:
        require(sha(entry['path']) == entry['sha256'] and Path(entry['path']).stat().st_size == entry['bytes'], 'Actual proof changed')
        proofs.append(dict(entry))
    clone = read_bound(candidate['source_actual_clone'])
    require(clone['clean'] is True and clone['source_commit'] == SOURCE and clone['repo'] == str(REPO), 'Actual source clone differs')
    require({x['path']:x for x in clone['files']} == {x['path']:x for x in candidate['files'] if x['path'].startswith(str(REPO) + '/')},
            'Actual clone file closure differs')
    require(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
            and not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip(), 'Source moved/dirty')
    import torch
    import search_r1.baselines.action_prefix_forward as prefix
    import search_r1.baselines.recovery as recovery
    import search_r1.baselines.policy_safety as safety
    import verl.workers.actor.dp_actor as actor
    imports = dict(action_prefix_module=inspect.getfile(prefix), actor_module=inspect.getfile(actor),
                   recovery_module=inspect.getfile(recovery), policy_safety_module=inspect.getfile(safety))
    record = dict(status='PHYSICAL_NATIVE_SOURCE_HELPER_SHA_INHERITED_ASSET_METADATA_ONLY_NOT_GPU',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node='h100-8', code_commit=SOURCE,
        candidate_sha256=expected, metadata=actual, evidence=proofs, source_clean=True, source_file_count=480,
        newly_hashed_paths=hashed, executable=sys.executable, torch=torch.__version__, imports=imports,
        cuda_initialized=torch.cuda.is_initialized(), runtime_all_file_sha_before_gpu_required=True,
        inherited_physical_sha256=candidate['inherited_physical']['sha256'])
    validate_physical(record, candidate, expected)
    write_once(output, record)
    return dict(physical=str(output), physical_sha256=sha(output), cuda_initialized=False, submitted=False)


if __name__ == '__main__':
    require(len(sys.argv) == 4, 'Need candidate/pathSHA/new physical output')
    print(json.dumps(verify(*sys.argv[1:])))

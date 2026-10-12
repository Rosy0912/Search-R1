"""Read-only deployment closure: hash code/proofs, preserve inherited asset metadata."""
import datetime
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path


def sha(p):
    with Path(p).open('rb') as f:
        h = hashlib.sha256()
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def metadata(p):
    s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns, device=s.st_dev, inode=s.st_ino)


def main():
    candidate_path, expected_sha, output = map(str, sys.argv[1:])
    assert sha(candidate_path) == expected_sha
    c = json.loads(Path(candidate_path).read_text())
    assert c['status'] == 'CANDIDATE_NOT_READY'
    assert c['validation_scope'] == 'GIGPO_POLICY_SAFETY_NATIVE_ONLY'
    assert c['allowed_safety_modes'] == ['native_fixture']
    assert c['native_submitted'] is False and c['training_submitted'] is False
    parent = c['inherited_physical']
    assert sha(parent['path']) == parent['sha256']
    inherited = {x['path']: x for x in json.loads(Path(parent['path']).read_text())['metadata']}
    actual = []
    hashed = []
    for item in c['files']:
        p = Path(item['path'])
        m = metadata(p)
        if str(p).startswith(c['repo'] + '/') or str(p).startswith(c['helpers'] + '/'):
            assert sha(p) == item['sha256'], str(p)
            hashed.append(str(p))
        else:
            assert m == inherited[str(p)], 'Inherited metadata changed: ' + str(p)
        actual.append(m)
    for item in c['evidence']:
        assert sha(item['path']) == item['sha256'], item['path']
    assert subprocess.check_output(['git', '-C', c['repo'], 'rev-parse', 'HEAD'], text=True).strip() == c['code_commit']
    assert not subprocess.check_output(['git', '-C', c['repo'], 'status', '--porcelain'], text=True).strip()
    assert sys.executable == str(Path(c['repo']).parent / 'env/bin/python')
    import torch
    import search_r1.baselines.policy_safety as policy_safety
    import search_r1.baselines.recovery as recovery
    assert inspect.getfile(policy_safety) == c['repo'] + '/search_r1/baselines/policy_safety.py'
    assert inspect.getfile(recovery) == c['repo'] + '/search_r1/baselines/recovery.py'
    assert torch.__version__ == '2.4.0+cu121' and not torch.cuda.is_initialized()
    record = dict(status='NEW_CODE_HELPER_AND_PROOF_SHA_INHERITED_ASSET_METADATA_NOT_GPU', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), candidate_sha256=expected_sha, code_commit=c['code_commit'], new_files_sha_checked=len(hashed), inherited_evidence_sha_checked=len(c['evidence']), metadata_count=len(actual), metadata=actual, torch=torch.__version__, executable=sys.executable, policy_safety_import=inspect.getfile(policy_safety), recovery_import=inspect.getfile(recovery), cuda_initialized=torch.cuda.is_initialized(), inherited_physical_sha256=parent['sha256'], resource_guard=c['resource_guard'], resource_ownership=c['resource_ownership'], runtime_all_file_sha_required=True)
    with Path(output).open('x') as f:
        f.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'metadata'}))
    print('physical_sha256=' + sha(output))


if __name__ == '__main__':
    main()

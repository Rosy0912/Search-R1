"""Collect original terminal evidence; no GPU/model/data changes or tensor transfer."""
import base64
import datetime
import gzip
import hashlib
import json
import os
import errno
import subprocess
import sys
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def raw(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), content_base64=base64.b64encode(data).decode())


def cgroup_release(job, result_cgroup):
    base = Path('/sys/fs/cgroup/freezer/slurm_h100-8')
    job_path = base / 'uid_0' / ('job_' + job)
    scope = Path(result_cgroup)
    assert str(scope) == os.path.normpath(str(scope))
    assert scope == job_path or job_path in scope.parents, 'Foreign cgroup scope'
    for ancestor in (base.parent, base):
        assert ancestor.is_dir()
        os.listdir(ancestor)  # A missing/unreadable hierarchy must not appear released.
    for absent in (job_path, scope):
        try:
            os.stat(absent)
        except FileNotFoundError as error:
            assert error.errno == errno.ENOENT
        else:
            raise AssertionError('Own job cgroup still exists: ' + str(absent))
    return dict(job_id=job, path=str(job_path), result_cgroup=str(scope),
                ancestor_path=str(base), ancestor_readable=True,
                stat_outcome='ENOENT', scope_stat_outcome='ENOENT', removed=True)


def main():
    job = sys.argv[1]
    assert job.isdigit()
    run = ROOT / ('runs/gigpo-e1-policy-safety-native-job' + job)
    target = ROOT / ('logs/policy-safety-native' + job + '-terminal-raw-v3.json.gz')
    assert not target.exists(), 'Existing collector: preserve and inspect'
    terminal_path = ROOT / ('logs/quality10-native-terminal-' + job + '-raw-v1.json')
    terminal = json.loads(terminal_path.read_text())
    assert terminal['fields']['JobId'] == job and terminal['fields']['NodeList'] == 'h100-8'
    assert terminal['fields']['JobState'] == 'COMPLETED' and terminal['fields']['ExitCode'] == '0:0'
    assert terminal['fields']['Requeue'] == '0' and terminal['fields']['Restarts'] == '0'
    result = json.loads((run / 'result.json').read_text())
    assert result['job'] == job and result['status'] == 'PASS' and result['actual_outer_updates'] == 0
    assert result['fixtures_passed'] is True and result['cpu_exit'] == 0 and result['gpu_exit'] == 0
    assert result['restore_manifest_sha256'] == '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
    assert result['validation_scope'] == 'GIGPO_POLICY_SAFETY_NATIVE_ONLY'
    assert result['cleanup_residual_states'] == {} and result['final_gpu']['processes'] == []
    assert result['code_commit'] == '52befe968c856162d1bc79cd0171bb38309f4f9e'
    assert result['restore_step'] == 25 and result['safety_scope'] == 'native_fixture_not_production_calibration'
    binding_path = ROOT / 'logs/policy-safety-native-52befe9-reviewed-binding-v2.json'
    binding = json.loads(binding_path.read_text())
    assert result['binding_sha256'] == sha(binding_path)
    assert result['guard_identity'] == {k: binding[k] for k in ('resource_guard', 'resource_ownership')}
    release = cgroup_release(job, result['cgroup'])
    cgroups = []  # Explicitly proven absent by cgroup_release, never inferred from glob.
    queue = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%i %T'], text=True)
    assert not queue.strip()
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True)
    assert gpu.returncode == 0
    node_queue = subprocess.check_output(['squeue', '-h', '-w', 'h100-8', '-o', '%i %j %T'], text=True)
    originals = []
    physical = []
    for p in sorted(run.rglob('*')):
        if p.is_file():
            physical.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
            if p.suffix in ('.json', '.jsonl', '.log'):
                originals.append(raw(p))
    for p in (terminal_path, binding_path, ROOT / ('logs/policy-safety-native-' + job + '.log'), ROOT / 'logs/policy-safety-native-52befe9-submit-receipt-v2.json'):
        originals.append(raw(p))
    payload = dict(status='ORIGINAL_NATIVE_TERMINAL_AND_PHYSICAL_EVIDENCE_UNREVIEWED', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), job=job, run=str(run), result=result, terminal=terminal, physical=physical, originals=originals, current_compute=dict(returncode=gpu.returncode, stdout=gpu.stdout, stderr=gpu.stderr), current_node_queue=node_queue, own_job_queue=queue, own_cgroups=cgroups, cgroup_release=release, scope='Native actual exit and saved historical exclusive-allocation finalGPUempty, not production calibration or training outer. Current other jobs are recorded separately.')
    data = json.dumps(payload, separators=(',', ':')).encode()
    with target.open('xb') as f:
        f.write(gzip.compress(data))
    print(json.dumps(dict(path=str(target), sha256=sha(target), uncompressed_bytes=len(data), physical_files=len(physical), raw_files=len(originals), actual_outer_updates=0)))


if __name__ == '__main__':
    main()

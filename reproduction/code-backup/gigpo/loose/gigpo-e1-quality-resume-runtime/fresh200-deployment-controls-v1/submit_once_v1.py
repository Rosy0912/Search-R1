"""One exact five-argument diagnostic sbatch. A durable intent is never retried."""
import copy
import datetime
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from deployment_contract_v1 import (CONTROLS, READY, BINDING, INTENT, RECEIPT, SCOPE,
    require, read_bound, record, sha, validate_ready, verify_installed_controls, write_once)


def submit(ready_record, review_record, run=subprocess.run, hostname=socket.gethostname):
    require(hostname().split('.')[0] == 'h100-8', 'Submission only on authorized h100-8')
    require(Path(__file__).resolve() == CONTROLS / 'submit_once_v1.py', 'Different installed submission helper')
    require(Path(ready_record['path']) == READY and not any(os.path.lexists(p) for p in (BINDING, INTENT, RECEIPT)),
            'Existing claim/uncertainty: inspect originals/scheduler, never resubmit')
    require(not any(key.startswith('GIGPO_QUALITY_') for key in os.environ), 'Inherited old quality authorization')
    ready = read_bound(ready_record); candidate = validate_ready(ready)
    verify_installed_controls(candidate)
    review = read_bound(review_record)
    require(review['status'] == 'PASS' and review['scope'] == 'FRESH200_CALIBRATION2_SINGLE_SUBMIT_REVIEW'
            and review['independent_review'] is True and review['ready_sha256'] == ready_record['sha256']
            and review['submit_helper_sha256'] == sha(Path(__file__))
            and review['controls_manifest_sha256'] == candidate['proofs']['controls_manifest']['sha256'],
            'No exact independent single-submit acceptance')
    # Rehash all new code and proof bytes. Preserve accepted immutable asset SHA
    # plus current dev/inode/mtime/size; the runtime still hashes every asset
    # before GPU, without this control redundantly scanning the huge index.
    physical = read_bound(ready['physical_record'])
    physical_rows = {r['path']: r for r in physical['metadata']}
    assets = {r['path'] for r in candidate['asset_files']}
    for item in ready['files'] + ready['evidence']:
        if item['path'] in assets:
            stat = Path(item['path']).stat(); prior = physical_rows[item['path']]
            require((stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns) ==
                    tuple(prior[key] for key in ('bytes', 'device', 'inode', 'mtime_ns')), 'Asset identity changed after READY')
        else:
            require(record(item['path']) == {k: item[k] for k in ('path', 'bytes', 'sha256')}, 'Input changed after READY')
    observations = {}
    for name, command in (('queue', ['squeue', '-h', '-w', 'h100-8', '-o', '%i %j %T']),
        ('gpu_compute', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])):
        done = run(command, text=True, capture_output=True, timeout=15)
        require(done.returncode == 0 and isinstance(done.stdout, str) and not done.stdout.strip(), 'Node occupied/unreadable: no claim')
        observations[name] = dict(command=command, returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)
    write_once(INTENT, dict(status='CLAIMED_NO_RESUBMISSION', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ready=ready_record, single_submit_review=review_record, source_commit=candidate['source_commit'],
        submit_helper_sha256=sha(Path(__file__)), scope=SCOPE, phase='probability_probe_then_wait',
        retained_training_updates_contributed=0, observations=observations))
    binding = copy.deepcopy(ready)
    # A later audit follows READY and submit review explicitly. No source/native
    # acceptance is inferred; first-Adam probability tolerances remain null.
    write_once(BINDING, binding)
    command = ['sbatch', '--parsable', str(Path(candidate['helpers']) / 'calibration_v1.sbatch'),
               str(BINDING), sha(BINDING), candidate['repo'], candidate['helpers'], candidate['environment_root']]
    try:
        done = run(command, text=True, capture_output=True, timeout=60)
        match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', done.stdout.strip())
        result = dict(status='UNCERTAIN_NO_RESUBMISSION', command=command, stdout=done.stdout, stderr=done.stderr,
            returncode=done.returncode, binding=record(BINDING), ready=ready_record, review=review_record,
            scope=SCOPE, retained_training_updates_contributed=0, actual_gpu_acceptance=False,
            actual_probability_acceptance=False)
        if done.returncode == 0 and match:
            result.update(status='SUBMITTED_EXACTLY_ONCE', job_id=match.group(1))
    except BaseException as error:
        write_once(RECEIPT, dict(status='UNCERTAIN_NO_RESUBMISSION', command=command,
            error_type=type(error).__name__, error=str(error), binding=record(BINDING), ready=ready_record,
            review=review_record, scope=SCOPE, retained_training_updates_contributed=0))
        raise
    write_once(RECEIPT, result)
    require(result['status'] == 'SUBMITTED_EXACTLY_ONCE', 'Unknown sbatch outcome: preserve intent/receipt and never retry')
    return result


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'Need SHA-bound JSON spec with READY and single-submit review records')
    spec = read_bound(json.loads(sys.argv[1]))
    require(set(spec) == {'ready', 'single_submit_review'}, 'Wrong single-submit input roles')
    print(json.dumps(submit(spec['ready'], spec['single_submit_review'])))

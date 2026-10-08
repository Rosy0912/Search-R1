"""One exact owned live200 sbatch. A durable intent is never retried."""
import copy
import datetime
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from deployment_contract_v7 import (CONTROLS, READY, BINDING, INTENT, RECEIPT, SCOPE,
    require, read_bound, record, sha, validate_ready, verify_installed_controls, write_once)

SAFE_ENVIRONMENT = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1')
SBATCH_OPTIONS = ('--parsable', '--partition=gpu', '--nodelist=h100-8', '--nodes=1', '--ntasks=1',
                  '--gres=gpu:h100:8', '--cpus-per-task=128', '--mem=512G', '--exclusive', '--no-requeue',
                  '--time=1-00:00:00', '--export=ALL')


def scheduler_environment(inherited):
    forbidden = sorted(key for key in inherited if key.startswith(('SBATCH_', 'SLURM_', 'SALLOC_', 'SRUN_', 'GIGPO_QUALITY_')))
    require(not forbidden, 'Inherited scheduler/old authorization inputs forbidden: ' + ','.join(forbidden))
    # No inherited environment values are passed to Slurm or its observations.
    # Fixed PATH excludes caller scripts and removes loader/config/plugin inputs.
    return dict(SAFE_ENVIRONMENT)


def submit(ready_record, review_record, run=subprocess.run, hostname=socket.gethostname):
    require(hostname().split('.')[0] == 'h100-8', 'Submission only on authorized h100-8')
    require(Path(__file__).resolve() == CONTROLS / 'submit_once_v7.py', 'Different installed submission helper')
    require(Path(ready_record['path']) == READY and not any(os.path.lexists(p) for p in (BINDING, INTENT, RECEIPT)),
            'Existing claim/uncertainty: inspect originals/scheduler, never resubmit')
    environment = scheduler_environment(os.environ)
    ready = read_bound(ready_record); candidate = validate_ready(ready)
    verify_installed_controls(candidate)
    review = read_bound(review_record)
    require(review['status'] == 'PASS' and review['scope'] == 'FRESH200_LIVE200_SINGLE_SUBMIT_REVIEW'
            and review['independent_review'] is True and review['ready_sha256'] == ready_record['sha256']
            and review['submit_helper_sha256'] == sha(Path(__file__))
            and review['controls_manifest_sha256'] == candidate['proofs']['controls_manifest']['sha256']
            and review['effective_submission_environment'] == environment
            and review['sbatch_options'] == list(SBATCH_OPTIONS)
            and type(review['live_goal_updated_at']) is int and type(review['live_authorization_goal_updated_at']) is int
            and review['live_goal_updated_at'] == candidate['live_goal_updated_at']
            and review['live_authorization_goal_updated_at'] == candidate['live_authorization_goal_updated_at']
            and review['first5_count_toward_goal'] is True and review['same_process_continuation_required'] is True
            and review['extra_formal200_submission_allowed'] is False,
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
        done = run(command, text=True, capture_output=True, timeout=15, env=environment)
        require(done.returncode == 0 and isinstance(done.stdout, str) and not done.stdout.strip(), 'Node occupied/unreadable: no claim')
        observations[name] = dict(command=command, returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)
    # Audit the READY and separate submit review again as the same verified
    # buffers, after the final observations and before the durable claim.
    require(read_bound(ready_record) == ready and read_bound(review_record) == review,
            'READY/single-submit review changed before intent')
    write_once(INTENT, dict(status='CLAIMED_NO_RESUBMISSION', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ready=ready_record, single_submit_review=review_record, source_commit=candidate['source_commit'],
        submit_helper_sha256=sha(Path(__file__)), scope=SCOPE, phase='owned_live200_before_first_Adam_probability_gate',
        initial_retained_training_updates=0, first5_count_toward_goal=True,
        official_training_outer_target=200, live_goal_updated_at=candidate['live_goal_updated_at'],
        live_authorization_goal_updated_at=candidate['live_authorization_goal_updated_at'], observations=observations,
        effective_submission_environment=environment, sbatch_options=list(SBATCH_OPTIONS),
        scheduler_policy='ONE_NONARRAY_JOB_DEFAULT_LOCAL_CONFIG_FULL8_EXCLUSIVE_NO_REQUEUE'))
    binding = copy.deepcopy(ready)
    # A later audit follows READY and submit review explicitly. No source/native
    # acceptance is inferred; first-Adam probability tolerances remain null.
    write_once(BINDING, binding)
    command = ['sbatch', *SBATCH_OPTIONS, str(Path(candidate['helpers']) / 'calibration_v1.sbatch'),
               str(BINDING), sha(BINDING), candidate['repo'], candidate['helpers'], candidate['environment_root']]
    try:
        done = run(command, text=True, capture_output=True, timeout=60, env=environment)
        match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?', done.stdout.strip())
        result = dict(status='UNCERTAIN_NO_RESUBMISSION', command=command, stdout=done.stdout, stderr=done.stderr,
            returncode=done.returncode, binding=record(BINDING), ready=ready_record, review=review_record,
            effective_submission_environment=environment, sbatch_options=list(SBATCH_OPTIONS),
            scope=SCOPE, actual_counted_outer_updates=None, first5_count_toward_goal=True,
            official_training_outer_target=200, actual_gpu_acceptance=False,
            actual_probability_acceptance=False)
        if done.returncode == 0 and match:
            result.update(status='SUBMITTED_EXACTLY_ONCE', job_id=match.group(1))
    except BaseException as error:
        write_once(RECEIPT, dict(status='UNCERTAIN_NO_RESUBMISSION', command=command,
            error_type=type(error).__name__, error=str(error), binding=record(BINDING), ready=ready_record,
            effective_submission_environment=environment, sbatch_options=list(SBATCH_OPTIONS),
            review=review_record, scope=SCOPE, actual_counted_outer_updates=None,
            first5_count_toward_goal=True, official_training_outer_target=200))
        raise
    write_once(RECEIPT, result)
    require(result['status'] == 'SUBMITTED_EXACTLY_ONCE', 'Unknown sbatch outcome: preserve intent/receipt and never retry')
    return result


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'Need SHA-bound JSON spec with READY and single-submit review records')
    spec = read_bound(json.loads(sys.argv[1]))
    require(set(spec) == {'ready', 'single_submit_review'}, 'Wrong single-submit input roles')
    print(json.dumps(submit(spec['ready'], spec['single_submit_review'])))

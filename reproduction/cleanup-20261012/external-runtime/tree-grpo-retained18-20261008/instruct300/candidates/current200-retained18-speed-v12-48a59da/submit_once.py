"""One human-authorized native5 speed-repair continuation; unknown never retries."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time
from production_contract import ROOT, SOURCE, CHAIN_SCOPE, DIRECT_MODE, require, read_bound, production_inputs
from slurm_identity import CLUSTER, NODE, local_environment, cluster_identity, query, create_journal, finalize_journal, verify_management_scope, timeout_output
from build_package_draft import read_regular_bound


def stopped_predecessor_observation(manifest, run, environment):
    pin = manifest['stopped_predecessor3335']
    evidence, _, _ = read_bound(pin['path'], pin['sha256'])
    require(evidence['job_id'] == 3335 and evidence['scheduler_state'] == 'CANCELLED'
            and evidence['manual_human_stop'] is True and evidence['last_completed_outer'] == 5
            and evidence['own_cgroup_boundary_released_verified'] is True
            and evidence['stopped_outer6_partial_adam_writes'] is None,
            'Exact human-stopped3335 terminal evidence required')
    previous = ROOT/'instruct300/SUBMISSION_ONCE-retained18-current200-v11-6efb604.json'
    journal, _, _ = read_bound(previous)
    require(journal['status'] == 'SUBMITTED' and journal['job_id'] == 3335
            and journal['manifest_sha256'] == '1a2a7ea8dea6a599d7d170671fb0ff6f2a6076b2dc5c8b61653e0381642ebb6b',
            'Original3335 once identity changed')
    verify_management_scope(environment)
    result = run(['scontrol', '--local', 'show', 'job', '3335'], env=environment,
                 text=True, capture_output=True, timeout=30)
    if result.returncode == 0:
        require(result.stderr == '' and re.search(r'\bJobId=3335\b', result.stdout)
                and re.search(r'\bJobState=CANCELLED\b', result.stdout),
                '3335 is not a proven cancelled terminal job')
        terminal_scope = 'ACTUAL_CURRENT_CONTROLLER_CANCELLED'
    else:
        require(not result.stdout.strip()
                and result.stderr.strip() == 'slurm_load_jobs error: Invalid job id specified',
                '3335 controller observation unknown; no submission')
        require(evidence['scheduler_exit_code'] == '0:15'
                and evidence['actual_end_time_beijing'] == '2026-10-08T14:05:51+08:00',
                'Absent handle lacks bound actual cancelled terminal evidence')
        terminal_scope = 'PRIOR_ACTUAL_CANCELLED_RECEIPT_HANDLE_EXPIRED'
    cgroup = Path('/sys/fs/cgroup/freezer/slurm_h100-10/uid_0/job_3335')
    if cgroup.exists():
        require((cgroup/'tasks').is_file(), 'Existing3335 cgroup has no readable task inventory')
        tasks = [cgroup/'tasks'] + sorted(cgroup.glob('*/tasks'))
        require(all(not path.read_text().strip() for path in tasks if path.exists()),
                'Own3335 cgroup still contains tasks; do not submit')
    queue = query(['squeue', '--local', '-h', '-o', '%i|%j|%T|%b'], run,
                  environment, management=True)
    for line in queue.splitlines():
        fields = line.split('|')
        require(len(fields) == 4 and fields[0] != '3335', 'Ambiguous queue or old3335 still active')
        if fields[1].startswith('tree-'):
            require(fields[3] in ('gres/gpu:0', 'gres/gpu=0', 'gpu:0'),
                    'Active/pending Tree GPU job exists; do not submit another')
    return dict(status=terminal_scope, evidence=evidence, observed_epoch=time.time(),
                scontrol_stdout=result.stdout, scontrol_stderr=result.stderr,
                scontrol_returncode=result.returncode, own_cgroup_exists=cgroup.exists(),
                own_cgroup_tasks_empty=True, queue_stdout=queue,
                other_jobs_modified=False, partial_outer6_writes_unknown=True)


def submit(args, run=subprocess.run):
    verify_management_scope(os.environ, socket.gethostname())
    environment = local_environment(os.environ)
    identity = cluster_identity(run, environment, management=True)
    package = Path(args.package)
    require(package.is_absolute() and package.resolve().is_relative_to((ROOT/'instruct300/candidates').resolve()),
            'Own retained18 package required')
    manifest, _, _ = read_bound(package/'manifest.json', args.manifest_sha)
    require(manifest['packet_revision'] == 12 and manifest['code_commit'] == SOURCE
            and manifest['dispatch']['nodes'] == [NODE] and manifest['dispatch']['gpus'] == 8,
            'Exact V12 speed packet identity required')
    _, pins = production_inputs(manifest, manifest_sha256=args.manifest_sha,
        dispatch_review_descriptor=dict(path=args.release_path, sha256=args.release_sha))
    require(Path(__file__).resolve() == (package/'submit_once.py').resolve(), 'Different dispatcher source')
    for name, expected in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe package member')
        read_regular_bound(package/relative, expected)
    from current_chain_runtime import parent_metadata
    parent_metadata(manifest)
    storage, storage_sha, _ = read_bound(ROOT/'.node-migration/output-storage-verified.json',
                                        manifest['retained18_migration']['storage_receipt_sha256'])
    require(storage['shared_read_verified'] is True and storage['server_ip'] == '10.33.0.18'
            and storage['server_retained'] is True, 'Own retained18 shared storage not proven')
    from render_dispatch import render
    script = read_regular_bound(args.script, args.script_sha)
    wall = manifest['dispatch']['train_wall_hours']
    require(wall == 168 and script == render(package, args.manifest_sha,
            args.release_path, args.release_sha, wall).encode(), 'Exact168h dispatch bytes required')
    path = Path(manifest['dispatch']['once_intent'])
    require(path.parent == ROOT/'instruct300'
            and re.fullmatch(r'SUBMISSION_ONCE-retained18-current200-speed-v12-[0-9a-f]{7}\.json', path.name),
            'New exclusive speed intent namespace required')
    fd = os.open(ROOT/'instruct300/FRESH200_TRAIN_DISPATCH.lock', os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), 'Dispatch lock not ordinary')
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not path.exists() and not path.is_symlink(), 'Existing intent: no retry')
        require(not list(path.parent.glob('SUBMISSION_ONCE-retained18-current200-speed-v12-*.json')),
                'Another speed intent exists; unknown/submitted is never retried')
        observed = stopped_predecessor_observation(manifest, run, environment)
        initial = dict(status='SUBMITTING', execution_mode=DIRECT_MODE, scope=CHAIN_SCOPE,
            current_chain_resume=manifest['current_chain_resume'], code_commit=SOURCE,
            independent_acceptance=False, training_accepted=False, production_thresholds_accepted=False,
            original_sampler_actor_gate_accepted=False, cluster_name=CLUSTER, node_list=NODE,
            local_cluster_identity=identity, manifest_sha256=args.manifest_sha, review_sha256=args.release_sha,
            dispatch_intent_path=str(path), train_wall_hours=wall, dispatch_script_sha256=args.script_sha,
            submit_argv=['sbatch', '--parsable'], submitted_input_sha256=hashlib.sha256(script).hexdigest(),
            submitted_input_bytes=len(script), submitted_input_base64=base64.b64encode(script).decode(),
            submit_mode='verified_bytes_via_stdin', production_input_pins=pins,
            shared_storage_receipt_sha256=storage_sha, stopped_predecessor_observation=observed,
            foreign_jobs_modified=False, start_epoch=time.time(), stdout=None, stderr=None, job_id=None)
        create_journal(path, initial)
        result = None
        try:
            result = run(['sbatch', '--parsable'], input=script, env=environment, capture_output=True, timeout=60)
            output = result.stdout.decode('utf8'); error = result.stderr.decode('utf8')
            require(result.returncode == 0 and error == ''
                    and re.fullmatch(r'[1-9][0-9]*(;wuhan-h100)?\n?', output), 'Unknown/ambiguous submission')
            parsable = output.rstrip('\n'); job = int(parsable.split(';')[0])
            final = dict(initial, status='SUBMITTED', job_id=job, run_id='tree-current200-continue-'+str(job),
                         stdout=output, stderr=error, returncode=0, parsable_stdout=parsable, end_epoch=time.time())
        except BaseException as error:
            final = dict(initial, status='UNKNOWN', error_type=type(error).__name__, error=str(error),
                end_epoch=time.time(), submitted_returncode=None if result is None else result.returncode,
                submitted_stdout=None if result is None else timeout_output(result.stdout),
                submitted_stderr=None if result is None else timeout_output(result.stderr))
            finalize_journal(path, final)
            raise
        finalize_journal(path, final)
        return final
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('package', 'manifest-sha', 'release-path', 'release-sha', 'script', 'script-sha'):
        parser.add_argument('--'+key, required=True)
    print(json.dumps(submit(parser.parse_args()), sort_keys=True))

"""One fresh BranPO200 submission; unknown never retries."""
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
from datetime import datetime
from production_contract import ROOT, SOURCE, CHAIN_SCOPE, DIRECT_MODE, require, read_bound, production_inputs
from slurm_identity import CLUSTER, NODE, local_environment, cluster_identity, query, create_journal, finalize_journal, verify_management_scope, timeout_output
from build_package_draft import read_regular_bound


def fresh_admission(manifest,run,environment):
    verify_management_scope(environment)
    from storage_guard import observe_storage
    storage=observe_storage(ROOT/'runs',manifest['storage_guard'])
    queue=query(['squeue','--local','-h','-o','%i|%j|%T|%b'],run,environment,management=True)
    for line in queue.splitlines():
        fields=line.split('|');require(len(fields)==4,'Ambiguous queue')
        require(not (fields[1].startswith('tree-') and fields[3] not in ('gres/gpu:0','gres/gpu=0','gpu:0')),'Another Tree GPU job active/pending')
    return dict(epoch=time.time(),queue_stdout=queue,storage=storage,foreign_jobs_modified=False)


def submit(args, run=subprocess.run):
    verify_management_scope(os.environ, socket.gethostname())
    environment = local_environment(os.environ)
    identity = cluster_identity(run, environment, management=True)
    package = Path(args.package)
    require(package.is_absolute() and package.resolve().is_relative_to((ROOT/'instruct300/candidates').resolve()),
            'Own retained18 package required')
    manifest, _, _ = read_bound(package/'manifest.json', args.manifest_sha)
    require(manifest['packet_revision'] == 1 and manifest['code_commit'] == SOURCE
            and manifest['dispatch']['nodes'] == [NODE] and manifest['dispatch']['gpus'] == 8,
            'Exact V14 speed packet identity required')
    _, pins = production_inputs(manifest, manifest_sha256=args.manifest_sha,
        dispatch_review_descriptor=dict(path=args.release_path, sha256=args.release_sha))
    require(Path(__file__).resolve() == (package/'submit_once.py').resolve(), 'Different dispatcher source')
    for name, expected in manifest['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe package member')
        read_regular_bound(package/relative, expected)
    from render_dispatch import render
    script = read_regular_bound(args.script, args.script_sha)
    wall = manifest['dispatch']['train_wall_hours']
    require(wall == 168 and script == render(package, args.manifest_sha,
            args.release_path, args.release_sha, wall).encode(), 'Exact168h dispatch bytes required')
    path = Path(manifest['dispatch']['once_intent'])
    require(path.parent == ROOT/'instruct300'
            and re.fullmatch(r'SUBMISSION_ONCE-branpo-fresh200-[0-9a-f]{40}\.json', path.name),
            'New exclusive speed intent namespace required')
    fd = os.open(ROOT/'instruct300/FRESH200_TRAIN_DISPATCH.lock', os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), 'Dispatch lock not ordinary')
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not path.exists() and not path.is_symlink(), 'Existing intent: no retry')
        require(not list(path.parent.glob('SUBMISSION_ONCE-branpo-fresh200-*.json')),
                'Another speed intent exists; unknown/submitted is never retried')
        observed = fresh_admission(manifest, run, environment)
        initial = dict(status='SUBMITTING', execution_mode=DIRECT_MODE, scope=CHAIN_SCOPE,
            code_commit=SOURCE,
            independent_acceptance=False, training_accepted=False, production_thresholds_accepted=False,
            original_sampler_actor_gate_accepted=False, cluster_name=CLUSTER, node_list=NODE,
            local_cluster_identity=identity, manifest_sha256=args.manifest_sha, review_sha256=args.release_sha,
            dispatch_intent_path=str(path), train_wall_hours=wall, dispatch_script_sha256=args.script_sha,
            submit_argv=['sbatch', '--parsable'], submitted_input_sha256=hashlib.sha256(script).hexdigest(),
            submitted_input_bytes=len(script), submitted_input_base64=base64.b64encode(script).decode(),
            submit_mode='verified_bytes_via_stdin', production_input_pins=pins,
            fresh_admission=observed,
            foreign_jobs_modified=False, start_epoch=time.time(), stdout=None, stderr=None, job_id=None)
        create_journal(path, initial)
        result = None
        try:
            result = run(['sbatch', '--parsable'], input=script, env=environment, capture_output=True, timeout=60)
            output = result.stdout.decode('utf8'); error = result.stderr.decode('utf8')
            require(result.returncode == 0 and error == ''
                    and re.fullmatch(r'[1-9][0-9]*(;wuhan-h100)?\n?', output), 'Unknown/ambiguous submission')
            parsable = output.rstrip('\n'); job = int(parsable.split(';')[0])
            final = dict(initial, status='SUBMITTED', job_id=job, run_id='tree-branpo-fresh200-'+str(job),
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

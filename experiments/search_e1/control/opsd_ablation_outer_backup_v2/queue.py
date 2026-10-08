"""Independent Mac serial backup queue; discover is read-only HKUST1 CPU only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import types
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
b = types.ModuleType('outer_backup_source'); b.__file__ = str(HERE / 'backup.py')
exec(compile((HERE / 'backup.py').read_bytes(), b.__file__, 'exec'), b.__dict__)


def discover(goal):
    """Only completed immutable paths are returned; pending/mutable latest ignored."""
    jobs = []
    for group in b.validate_goal(goal):
        config, cref = b.checked_json(group['config']['path'], group['config']['sha256'])
        b.require(config['run_id'] == group['run_id'] and config['variant'] == group['variant']
                  and config['study_id'] == goal['study_id']
                  and config['scientific']['outer_iterations'] == 10
                  and config['paths']['output_root'] == group['output_root'], 'Discovery config differs')
        for outer in range(10):
            path = Path(group['output_root']) / 'outers' / ('outer-%05d' % outer) / 'training/outer-complete.json'
            if not path.exists():
                continue
            value, completion = b.checked_json(path, b.fileproof(path)['sha256'])
            b.require(value.get('accepted') is True and value['schema'] == 'opsd-online-outer-complete-v1'
                      and value['run_id'] == group['run_id'] and value['outer_index'] == outer,
                      'Existing completion is not accepted for this exact group')
            updates = value['optimizer_updates']; cpref = value['checkpoint']
            b.target_scope(group['run_id'], updates, outer, cpref['sha256'], goal)
            expected = Path(group['output_root']) / 'checkpoints' / ('update-%08d-outer-%05d' % (updates, outer)) / 'manifest.json'
            b.require(Path(cpref['path']) == expected, 'K0 or other checkpoint cannot be silently relabeled')
            cp, actual = b.checked_json(cpref['path'], cpref['sha256'])
            b.require(cp.get('accepted') is True and cp['run_id'] == group['run_id']
                      and cp['config_sha256'] == cref['sha256'] and cp['cursor']['outer_index'] == outer
                      and cp['optimizer_updates'] == updates and cp['cursor']['next_pair_cursor'] == value['K'],
                      'Discovery checkpoint state differs')
            jobs.append(dict(variant=group['variant'], run_id=group['run_id'], outer_index=outer,
                             optimizer_updates=updates, config=group['config'],
                             checkpoint=dict(path=actual['path'], sha256=actual['sha256']),
                             outer_complete=dict(path=completion['path'], sha256=completion['sha256']),
                             empty_outer_no_new_adam=value['K'] == 0))
    return jobs


def job_key(job):
    return hashlib.sha256(b.canonical(job)).hexdigest()


def merge_jobs(previous, current):
    """A previously pinned outer can never be replaced by a newer path/SHA."""
    result = dict(previous)
    pinned = {(x['job']['run_id'], x['job']['outer_index']): x['job'] for x in result.values()}
    for job in current:
        slot = (job['run_id'], job['outer_index'])
        b.require(slot not in pinned or pinned[slot] == job, 'Previously pinned outer identity changed')
        key = job_key(job)
        result.setdefault(key, dict(job=job, state='PENDING', attempts=[], retry_after_unix=0))
        pinned[slot] = job
    return result


def atomic_status(path, value):
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    b.write_json(temp, value); os.replace(temp, path); b.fsync_dir(path.parent)


def verified_complete(item):
    """Persisted rc/SHA proof plus fresh metadata continuity, not a new tensor audit."""
    ref = item['complete_receipt']; value, _ = b.checked_json(ref['path'], ref['sha256']); job = item['job']
    b.require(value.get('accepted') is True and value['actual_transfer_returncode'] == 0
              and value['independent_local_disk_readback'] is True and value['run_id'] == job['run_id']
              and value['optimizer_updates'] == job['optimizer_updates'] and value['outer_index'] == job['outer_index']
              and value['checkpoint']['path'] == job['checkpoint']['path']
              and value['checkpoint']['sha256'] == job['checkpoint']['sha256']
              and value['outer_complete']['path'] == job['outer_complete']['path']
              and value['outer_complete']['sha256'] == job['outer_complete']['sha256'], 'Prior queue completion is not this pinned job')
    inventory, _ = b.checked_json(value['node_inventory']['path'], value['node_inventory']['sha256'])
    complete = inventory['outer_complete']
    bound = [r for r in inventory['files'] if r['relative'] == 'closure/outer-complete.json']
    b.require(len(bound) == 1 and complete['path'] == job['outer_complete']['path']
              and complete['sha256'] == job['outer_complete']['sha256']
              and bound[0]['path'] == complete['path'] and bound[0]['sha256'] == complete['sha256'],
              'Restart saved inventory completion differs from queued immutable proof')
    b.stable(value['archive'])
    for proof in value['files'].values(): b.stable(proof)


def discovery_command(args):
    b.require(args.host == 'hkust1', 'Only HKUST1 is authorized')
    remote = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
              'E1_RUN_ID=backup-' + args.goal_identity_sha256[:16], 'python3', '-B',
              str(Path(args.remote_helper_root) / 'queue.py'), 'discover',
              '--goal-identity', args.remote_goal_identity, '--goal-identity-sha256', args.goal_identity_sha256,
              '--source-manifest-sha256', args.source_manifest_sha256, '--publication', args.remote_publication,
              '--publication-sha256', args.publication_sha256, '--source-commit', args.source_commit]
    return ['ssh', '-oConnectTimeout=15', args.host, shlex.join(remote)]


def validate_discovery(value, goal):
    b.require(value.get('schema') == 'opsd-ablation-outer-backup-discovery-v1'
              and value.get('accepted') is True, 'Discovery did not succeed')
    groups = {g['run_id']: g for g in b.validate_goal(goal)}
    jobs = value['jobs']; seen = set()
    for job in jobs:
        g = b.target_scope(job['run_id'], job['optimizer_updates'], job['outer_index'], job['checkpoint']['sha256'], goal)
        b.require(job['variant'] == g['variant'] and job['config'] == g['config'], 'Discovered group differs')
        slot = (job['run_id'], job['outer_index']); b.require(slot not in seen, 'Duplicate discovered outer'); seen.add(slot)
        cp = Path(g['output_root']) / 'checkpoints' / ('update-%08d-outer-%05d' % (job['optimizer_updates'], job['outer_index'])) / 'manifest.json'
        complete = Path(g['output_root']) / 'outers' / ('outer-%05d' % job['outer_index']) / 'training/outer-complete.json'
        b.require(job['checkpoint']['path'] == str(cp) and job['outer_complete']['path'] == str(complete)
                  and b.SHA.fullmatch(job['outer_complete']['sha256']), 'Discovered pinned paths differ')
    b.require(len(jobs) <= len(groups) * 10, 'Discovery exceeds seventy targets')
    return jobs


def queue(args, goal, publication):
    root = Path(args.destination).absolute(); b.checked_path(root.parent, directory=True)
    if not root.exists():
        root.mkdir(mode=0o700); b.fsync_dir(root.parent)
    b.checked_path(root, directory=True)
    identity = dict(goal_sha256=args.goal_identity_sha256, source_manifest_sha256=args.source_manifest_sha256,
                    publication_sha256=args.publication_sha256, source_commit=args.source_commit)
    status_path = root / 'queue-status.json'
    state = dict(schema='opsd-ablation-outer-backup-queue-v1', identity=identity, jobs={}, training_not_blocked=True)
    with (root / '.backup-queue.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Mac queue only; no GPU/experiment lease.
        if status_path.exists():
            state = json.loads(status_path.read_bytes()); b.require(state['identity'] == identity, 'Existing queue belongs to another goal/source')
        while True:
            try:
                result = subprocess.run(discovery_command(args), capture_output=True, check=False)
                observation = dict(actual_returncode=result.returncode, stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
                                   stderr_sha256=hashlib.sha256(result.stderr).hexdigest(), observed_unix=time.time())
                b.require(result.returncode == 0, 'Read-only discovery transport failed')
                current = json.loads(result.stdout)
                b.require(current['goal_sha256'] == args.goal_identity_sha256
                          and current['source_manifest_sha256'] == args.source_manifest_sha256
                          and current['source_commit'] == args.source_commit, 'Discovery source/goal differs')
                state['jobs'] = merge_jobs(state['jobs'], validate_discovery(current, goal))
                state['last_discovery'] = observation
                for key, item in sorted(state['jobs'].items(), key=lambda kv: (kv[1]['job']['outer_index'], kv[1]['job']['variant'])):
                    if item['state'] == 'COMPLETE':
                        verified_complete(item)
                        continue
                    if time.time() < item['retry_after_unix']:
                        continue
                    job = item['job']; attempt = root / (key + '-attempt-' + uuid.uuid4().hex)
                    item['attempts'].append(str(attempt)); item['state'] = 'TRANSFERRING'; atomic_status(status_path, state)
                    options = types.SimpleNamespace(**vars(args))
                    options.config = job['config']['path']; options.config_sha256 = job['config']['sha256']
                    options.checkpoint = job['checkpoint']['path']; options.checkpoint_sha256 = job['checkpoint']['sha256']
                    options.expected_run = job['run_id']; options.expected_updates = job['optimizer_updates']; options.expected_outer = job['outer_index']
                    options.expected_completion = job['outer_complete']['path']; options.expected_completion_sha256 = job['outer_complete']['sha256']
                    options.destination = str(attempt)
                    try:
                        b.backup(options, publication)
                        cp = b.fileproof(attempt / 'complete-sha-verified.json')
                        item['complete_receipt'] = cp
                        verified_complete(item)
                        item.update(state='COMPLETE', completed_unix=time.time())
                    except Exception as error:
                        item.update(state='BACKLOG', last_error=type(error).__name__ + ': ' + str(error), retry_after_unix=time.time()+300)
                    atomic_status(status_path, state)
                state['updated_unix'] = time.time(); atomic_status(status_path, state)
            except Exception as error:
                state['last_discovery_error'] = dict(reason=type(error).__name__ + ': ' + str(error), observed_unix=time.time())
                atomic_status(status_path, state)
            if args.once or len(state['jobs']) == 70 and all(x['state'] == 'COMPLETE' for x in state['jobs'].values()):
                return state
            time.sleep(args.interval_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('discover','queue'))
    for name in ('goal-identity','goal-identity-sha256','source-manifest-sha256','publication','publication-sha256','source-commit'):
        p.add_argument('--'+name, required=True)
    for name in ('remote-helper-root','remote-publication','remote-goal-identity','destination'): p.add_argument('--'+name)
    p.add_argument('--host', default='hkust1'); p.add_argument('--margin-gib', type=int, default=5)
    p.add_argument('--once', action='store_true'); p.add_argument('--interval-seconds', type=int, default=60)
    args = p.parse_args(); b.require(args.interval_seconds >= 30, 'Discovery must not busy-poll')
    goal, _ = b.checked_json(args.goal_identity, args.goal_identity_sha256); groups = b.validate_goal(goal)
    pub = b.source_publication(HERE/'source-manifest.json', args.source_manifest_sha256,
                              args.publication,args.publication_sha256,args.source_commit,groups[0]['run_id'])
    if args.mode == 'discover':
        b.require(os.getuid() == os.geteuid() == 1005, 'Discovery requires actual HKUST1 UID1005')
        result = dict(schema='opsd-ablation-outer-backup-discovery-v1',accepted=True,jobs=discover(goal),
                      goal_sha256=args.goal_identity_sha256,source_manifest_sha256=args.source_manifest_sha256,source_commit=args.source_commit)
        b.source_publication(HERE/'source-manifest.json',args.source_manifest_sha256,args.publication,args.publication_sha256,args.source_commit,groups[0]['run_id'])
    else:
        b.require(args.destination and args.remote_helper_root and args.remote_publication and args.remote_goal_identity, 'Missing Mac queue/remote paths')
        result = queue(args,goal,pub)
    print(b.canonical(result).decode())


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error),file=sys.stderr); raise SystemExit(1)

"""Independent Mac serial compact queue for the same seventy immutable CPs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent


def load_source(expected):
    raw = (HERE / 'source-manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected: raise RuntimeError('Compact manifest differs')
    manifest = json.loads(raw); path = HERE / 'compact.py'; source = path.read_bytes()
    row = manifest['files']['compact.py']
    if len(source) != row['bytes'] or hashlib.sha256(source).hexdigest() != row['sha256']:
        raise RuntimeError('Compact source differs')
    module = types.ModuleType('compact_source'); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def legacy_status(b, q, args, identity, goal):
    path = Path(args.legacy_queue_root).absolute() / 'queue-status.json'
    value, _ = b.checked_json(path, b.fileproof(path)['sha256'])
    b.require(value['schema'] == 'opsd-ablation-outer-backup-queue-v1' and value['identity'] == identity,
              'Old queue belongs to another immutable goal/source')
    jobs = q.validate_discovery(dict(schema='opsd-ablation-outer-backup-discovery-v1',
        accepted=True, jobs=[x['job'] for x in value['jobs'].values()]), goal)
    b.require(set(value['jobs']) == {q.job_key(j) for j in jobs}, 'Old queue job keys differ')
    return value['jobs']


def queue(c, b, q, args, goal, publications):
    old_identity = dict(goal_sha256=args.goal_identity_sha256, source_manifest_sha256=args.source_manifest_sha256,
        publication_sha256=args.publication_sha256, source_commit=args.source_commit)
    identity = dict(legacy=old_identity, compact_manifest_sha256=args.compact_manifest_sha256,
        compact_publication_sha256=args.compact_publication_sha256, compact_source_commit=args.compact_source_commit,
        legacy_queue_root=str(Path(args.legacy_queue_root).absolute()))
    # Acquire the unchanged old lock BEFORE creating or mutating new output.
    with c.QueueLease(b, args.legacy_queue_root):
        root = Path(args.destination).absolute(); b.checked_path(root.parent, directory=True)
        legacy = Path(args.legacy_queue_root).absolute()
        b.require(root != legacy and legacy not in root.parents and root not in legacy.parents, 'Independent compact root required')
        if not root.exists(): root.mkdir(mode=0o700); b.fsync_dir(root.parent)
        b.checked_path(root, directory=True)
        status = root / 'queue-status.json'
        state = dict(schema='opsd-ablation-compact-queue-v1', identity=identity, jobs={}, training_not_blocked=True)
        if status.exists():
            state, _ = b.checked_json(status, b.fileproof(status)['sha256'])
            b.require(state['schema'] == 'opsd-ablation-compact-queue-v1' and state['identity'] == identity,
                      'Compact queue identity differs')
        old_jobs = legacy_status(b, q, args, old_identity, goal)
        state['jobs'] = q.merge_jobs(state['jobs'], [x['job'] for x in old_jobs.values()])
        q.atomic_status(status, state)
        while True:
            try:
                result = subprocess.run(q.discovery_command(args), capture_output=True, check=False)
                observation = root / ('discovery-' + uuid.uuid4().hex)
                observation.mkdir(mode=0o700)
                c.write_private(observation / 'stderr.private', result.stderr)
                proof = b.write_json(observation / 'execution.json', dict(actual_returncode=result.returncode,
                    stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr=b.fileproof(observation / 'stderr.private'),
                    observed_unix=time.time()))
                b.require(result.returncode == 0, 'Read-only discovery failed; prior pins retained')
                current = json.loads(result.stdout)
                b.require(current['goal_sha256'] == args.goal_identity_sha256 and
                    current['source_manifest_sha256'] == args.source_manifest_sha256 and current['source_commit'] == args.source_commit,
                    'Discovery source/goal differs')
                state['jobs'] = q.merge_jobs(state['jobs'], q.validate_discovery(current, goal))
                state['last_discovery'] = proof; state.pop('last_discovery_error', None)
            except Exception as error:
                state['last_discovery_error'] = dict(reason=type(error).__name__ + ': ' + str(error), observed_unix=time.time())
            q.atomic_status(status, state)
            for key, item in sorted(state['jobs'].items(), key=lambda kv: (kv[1]['job']['outer_index'], kv[1]['job']['variant'])):
                job = item['job']
                try:
                    if item['state'] == 'HOLD': continue
                    if item['state'] != 'COMPLETE':
                        # A crash after receipt fsync but before queue-status fsync
                        # reuses that exact durable attempt, never an unverified partial.
                        for name in item['attempts']:
                            path = Path(name); b.require(path.parent == root and path.name.startswith(key + '-attempt-'), 'Attempt escapes queue')
                            complete = path / 'compact-complete.json'
                            if complete.exists():
                                ref = b.fileproof(complete); c.verified_compact(b, ref, job, args, full=True)
                                item.update(state='COMPLETE', complete_receipt=ref, completed_unix=time.time()); break
                    if item['state'] != 'COMPLETE':
                        if time.time() < item['retry_after_unix']: continue
                        attempt = root / (key + '-attempt-' + uuid.uuid4().hex)
                        item['attempts'].append(str(attempt)); item['state'] = 'TRANSFERRING'; q.atomic_status(status, state)
                        old = old_jobs.get(key)
                        if old and old['state'] == 'COMPLETE':
                            ref = c.import_v2(b, q, old, args, publications, attempt)
                        else: ref = c.transfer(b, job, args, publications, attempt)
                        item.update(state='COMPLETE', complete_receipt=ref, completed_unix=time.time())
                    c.verified_compact(b, item['complete_receipt'], job, args)
                    # Commit COMPLETE before explicit local reclamation. Its input
                    # is the independent full readback receipt, never a boolean.
                    q.atomic_status(status, state)
                    if args.reclaim_legacy:
                        value = c.verified_compact(b, item['complete_receipt'], job, args)
                        if value['historical_v2'] and 'legacy_reclaimed' not in item:
                            try:
                                item['legacy_reclaimed'] = c.reclaim_legacy(b, item['complete_receipt'], job, args)
                                item.pop('reclaim_error', None)
                            except Exception as error:
                                item['reclaim_error'] = dict(reason=type(error).__name__ + ': ' + str(error), observed_unix=time.time())
                except Exception as error:
                    # A completed package failing continuity is HOLD, never silently
                    # redownloaded/deleted. Failed new attempts remain BACKLOG.
                    item.update(state='HOLD' if item['state'] == 'COMPLETE' else 'BACKLOG',
                        last_error=type(error).__name__ + ': ' + str(error), retry_after_unix=time.time()+300)
                q.atomic_status(status, state)
            state['updated_unix'] = time.time(); q.atomic_status(status, state)
            if args.once: return state
            if 'last_discovery_error' not in state and len(state['jobs']) == 70 and all(x['state'] == 'COMPLETE' for x in state['jobs'].values()):
                for item in state['jobs'].values(): c.verified_compact(b, item['complete_receipt'], item['job'], args)
                if not args.reclaim_legacy or all('reclaim_error' not in x for x in state['jobs'].values()): return state
            time.sleep(args.interval_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('queue', 'verify'))
    for name in ('legacy-helper-root','legacy-queue-root','goal-identity','goal-identity-sha256',
        'source-manifest-sha256','publication','publication-sha256','source-commit','compact-manifest-sha256',
        'compact-publication','compact-publication-sha256','compact-source-commit', 'remote-helper-root',
        'remote-publication','remote-goal-identity'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--destination'); p.add_argument('--complete-receipt'); p.add_argument('--complete-receipt-sha256')
    p.add_argument('--host', default='hkust1'); p.add_argument('--margin-gib', type=int, default=5)
    p.add_argument('--interval-seconds', type=int, default=60); p.add_argument('--once', action='store_true')
    p.add_argument('--reclaim-legacy', action='store_true')
    args = p.parse_args()
    if sys.platform != 'darwin': raise RuntimeError('Production entry is Mac-only; never deploy it as a Node queue')
    c = load_source(args.compact_manifest_sha256)
    b, q = c.legacy_modules(args.legacy_helper_root, args.source_manifest_sha256)
    b.require(args.host == 'hkust1' and args.interval_seconds >= 30, 'Only low-rate HKUST1 readonly discovery allowed')
    goal, _ = b.checked_json(args.goal_identity, args.goal_identity_sha256); groups = b.validate_goal(goal)
    old, own = c.publication(b, args, groups[0]['run_id']); pubs = dict(legacy=old, compact=own)
    if args.mode == 'verify':
        b.require(args.complete_receipt and args.complete_receipt_sha256 and not args.reclaim_legacy, 'Verify requires exact reference; no reclamation')
        with c.QueueLease(b, args.legacy_queue_root):
            ref = dict(path=args.complete_receipt, sha256=args.complete_receipt_sha256)
            value, _ = b.checked_json(ref['path'], ref['sha256'])
            q.validate_discovery(dict(schema='opsd-ablation-outer-backup-discovery-v1', accepted=True, jobs=[value['job']]), goal)
            result = c.verified_compact(b, ref, value['job'], args, full=True)
    else:
        b.require(args.destination, 'Independent destination required')
        result = queue(c, b, q, args, goal, pubs)
    print(b.canonical(result).decode())


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr); raise SystemExit(1)

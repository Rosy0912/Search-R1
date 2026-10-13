"""Pinned, local-only native manifest metadata audit; never controls training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SOURCE = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
UNIT = 'sst-tree-nested-branch-fresh200-' + SOURCE[:8] + '.service'
RUN = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + SOURCE)
SCHEMA = 'native_manifest_history_watch_v1'
PROPERTIES = ('Id', 'LoadState', 'ActiveState', 'SubState', 'MainPID', 'ExecMainPID',
              'ExecMainCode', 'ExecMainStatus', 'ExecMainExitTimestampMonotonic',
              'InvocationID', 'Result')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_small(path, maximum):
    require(not path.is_symlink() and path.is_file(), 'Pinned ordinary file required: ' + str(path))
    before = path.stat()
    require(before.st_size <= maximum, 'Small metadata bound exceeded: ' + str(path))
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns) and len(raw) == after.st_size,
            'Metadata changed during read: ' + str(path))
    return raw


def atomic_state(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    require(len(raw) <= 100000, 'Watcher state exceeds 100KB')
    require(not path.is_symlink(), 'Watcher state symlink forbidden')
    temporary = path.with_name('.' + path.name + '.pending-' + str(os.getpid()) + '-' + str(time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def training_observation():
    # Read only. No signal, kill, stop/start/restart, cancel or training subprocess.
    result = subprocess.run(['/bin/systemctl', 'show', UNIT,
                             '--property=' + ','.join(PROPERTIES)],
                            capture_output=True, text=True, timeout=10, check=False)
    require(result.returncode == 0, 'Training unit observation failed: ' + result.stderr[-1000:])
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    require(fields.get('Id') == UNIT and fields.get('LoadState') == 'loaded',
            'Authorized training unit missing or identity differs')
    return fields


def is_terminal(fields):
    # An unknown/missing unit, a live PID, or an exit-less idle unit is not completion.
    return (fields.get('ActiveState') in ('inactive', 'failed') and
            fields.get('SubState') in ('dead', 'failed', 'exited') and
            fields.get('MainPID') == '0' and int(fields.get('ExecMainPID', '0')) > 0 and
            int(fields.get('ExecMainCode', '0')) > 0 and
            int(fields.get('ExecMainExitTimestampMonotonic', '0')) > 0)


def pinned_collector(release):
    directory = Path(__file__).resolve().parent
    require(release['schema'] == SCHEMA and release['training_source_commit'] == SOURCE and
            release['training_unit'] == UNIT and release['run_directory'] == str(RUN) and
            release['interval_seconds'] == 60, 'Watcher release scope differs')
    for name in ('watch_native_history.py', 'native_manifest_history_collector.py'):
        raw = read_small(directory / name, 200000)
        require(digest(raw) == release['files'][name], 'Pinned code SHA differs: ' + name)
    raw = read_small(directory / 'native_manifest_history_collector.py', 200000)
    module = {'__name__': 'pinned_native_history_collector', '__file__': str(directory / 'native_manifest_history_collector.py')}
    exec(compile(raw, module['__file__'], 'exec'), module)
    require(module['CHILD_SOURCE'] == SOURCE and module['RUN_ROOT'] == str(RUN.parent),
            'Collector source/run binding differs')
    # Reuse the byte-identical collector validators and atomic immutable publisher.
    # Do not execute its default two-run collector or its SSH CLI.
    marker = '\ntry:\n native_manifest_history_receipt='
    require(module['REMOTE'].count(marker) == 1, 'Collector definition boundary differs')
    namespace = {}
    exec(compile(module['REMOTE'].split(marker)[0], module['__file__'] + ':local_validators', 'exec'), namespace)
    return namespace


def collect(collector, history):
    results, errors, skipped = [], [], []
    rolling = RUN / 'checkpoints/training_state/rolling-completed'
    for path in sorted(rolling.glob('global_step_*/manifest.json')):
        suffix = path.parent.name.removeprefix('global_step_')
        if not suffix.isdecimal() or int(suffix) < 2:
            continue
        try:
            _, value, _ = collector['_native_read'](path)
            if value.get('status') != 'COMPLETE_NATIVE_ROLLING_STATE_SAVED':
                skipped.append({'step': int(suffix), 'reason': 'incomplete snapshot'})
                continue
            results.append(collector['_native_capture'](path, SOURCE, 'child_native1_continuation', history))
        except FileNotFoundError:
            skipped.append({'step': int(suffix), 'reason': 'pruned/saving race'})
        except Exception as exc:
            errors.append({'path': str(path), 'error': repr(exc)[:2000]})
    steps = []
    for path in sorted(history.glob('source-' + SOURCE + '-outer-*.receipt.json')):
        try:
            _, receipt, _ = collector['_native_read'](path)
            step = receipt.get('completed_outer')
            require(receipt.get('source_commit') == SOURCE and type(step) is int and step >= 2,
                    'History receipt scope differs')
            steps.append(step)
        except Exception as exc:
            errors.append({'path': str(path), 'error': repr(exc)[:2000]})
    return results, sorted(set(steps)), skipped, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    raw = read_small(args.release, 16000)
    require(digest(raw) == args.release_sha256, 'Release SHA differs')
    release = json.loads(raw)
    collector = pinned_collector(release)
    history = RUN / 'output/native-manifest-history'
    history.mkdir(parents=True, exist_ok=True)
    require(not history.is_symlink() and history.resolve() == RUN.resolve() / 'output/native-manifest-history',
            'History destination escapes owned run')
    state_dir = history / 'watcher-ba9f6627'
    state_dir.mkdir(exist_ok=True)
    require(not state_dir.is_symlink(), 'Watcher state directory symlink forbidden')
    # Audit-only singleton; compatible with the Mac collector's per-manifest locks.
    import fcntl
    with (state_dir / '.watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(schema=SCHEMA, status='WATCHING', training_source_commit=SOURCE,
                     training_unit=UNIT, run_directory=str(RUN), release_sha256=args.release_sha256,
                     collector_sha256=release['files']['native_manifest_history_collector.py'],
                     watcher_sha256=release['files']['watch_native_history.py'],
                     started_epoch=time.time(), polls=0, manifest_new_count=0, manifest_json_bytes_checked=0, audit_error_count=0,
                     cpu_seconds=0., collection_wall_seconds=0., unit_query_wall_seconds=0.,
                     indexed_completed_outers=[], missing_outers_since2=[], last_errors=[],
                     tensor_contents_read=False, PT_file_sha256_recomputed=False,
                     training_signals_sent=False, frozen_source_modified=False,
                     independent_cold_reload_performed=False)
        invocation = None
        while True:
            tick = time.monotonic()
            cpu = time.process_time()
            fields, errors = {}, []
            state['polls'] += 1
            try:
                query = time.monotonic()
                fields = training_observation()
                state['unit_query_wall_seconds'] += time.monotonic() - query
                current = fields.get('InvocationID')
                if current:
                    require(invocation is None or current == invocation, 'Training unit invocation changed; audit scope frozen')
                    invocation = current
                state['training_unit_observation'] = fields
                begun = time.monotonic()
                results, steps, skipped, errors = collect(collector, history)
                state['collection_wall_seconds'] += time.monotonic() - begun
                state['manifest_new_count'] += sum(r['created'] for r in results)
                state['manifest_json_bytes_checked'] += sum(r['source_manifest_bytes'] for r in results)
                state['last_collected_outers'] = [r['completed_outer'] for r in results]
                state['indexed_completed_outers'] = steps
                state['missing_outers_since2'] = sorted(set(range(2, max(steps, default=1) + 1)) - set(steps))
                state['last_skipped'] = skipped[-10:]
            except Exception as exc:
                errors.append({'error': repr(exc)[:2000]})
            state['cpu_seconds'] += time.process_time() - cpu
            state['audit_error_count'] += len(errors)
            state['last_errors'] = (state['last_errors'] + errors)[-20:]
            state['observed_epoch'] = time.time()
            state['last_poll_wall_seconds'] = time.monotonic() - tick
            terminal = bool(fields and is_terminal(fields))
            state['status'] = ('TRAINING_TERMINAL_OBSERVED_METADATA_ONLY' if terminal else
                               'WATCHING_WITH_AUDIT_ERROR' if errors else 'WATCHING')
            try:
                atomic_state(state_dir / 'state.json', state)
            except Exception as exc:
                print(json.dumps({'status': 'AUDIT_STATE_WRITE_ERROR', 'error': repr(exc)[:2000],
                                  'training_signals_sent': False}), flush=True)
            if terminal:
                # This poll already did the final collection, including a final audit-error record.
                try:
                    collector['_native_atomic_new'](state_dir / 'final.json',
                        (json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
                except Exception as exc:
                    print(json.dumps({'status': 'AUDIT_FINAL_WRITE_ERROR', 'error': repr(exc)[:2000],
                                      'training_signals_sent': False}), flush=True)
                print(json.dumps({'status': state['status'], 'training_unit': UNIT,
                                  'training_result_observed': fields.get('Result'),
                                  'archived_outers': state['indexed_completed_outers'],
                                  'audit_error_count': state['audit_error_count'],
                                  'training_acceptance_claimed': False}), flush=True)
                return
            time.sleep(max(0., 60. - (time.monotonic() - tick)))


if __name__ == '__main__':
    main()

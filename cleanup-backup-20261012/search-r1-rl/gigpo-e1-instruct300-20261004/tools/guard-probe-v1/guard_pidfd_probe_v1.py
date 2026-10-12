"""CPU-only Linux pidfd/Z/reap/cgroup ownership observation of our own children.

Never imports NVML, runs a GPU API, signals others, or changes the live guard.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
import pathlib
import platform
import select
import signal
import time


def snapshot(pid, fd=None):
    result = {'pid': pid, 'epoch_ns': time.time_ns()}
    for name in ('stat', 'cgroup', 'status'):
        try:
            result[name + '_raw'] = pathlib.Path('/proc', str(pid), name).read_text()
        except FileNotFoundError:
            result[name + '_raw'] = None
    if result['stat_raw']:
        fields = result['stat_raw'].rsplit(')', 1)[1].split()
        result['state'] = fields[0]
        result['start_ticks'] = int(fields[19])
    else:
        result['state'] = None
        result['start_ticks'] = None
    if fd is not None:
        try:
            result['pidfd_fdinfo_raw'] = pathlib.Path('/proc/self/fdinfo', str(fd)).read_text()
            values = dict(line.split(':', 1) for line in result['pidfd_fdinfo_raw'].splitlines() if ':' in line)
            result['pidfd_pid'] = int(values['Pid'].strip())
            result['pidfd_nspid'] = values.get('NSpid', '').strip()
            result['pidfd_readable'] = bool(select.select([fd], [], [], 0)[0])
        except FileNotFoundError:
            result['pidfd_fdinfo_raw'] = None
    return result


def wait_state(pid, state, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        item = snapshot(pid)
        if item['state'] == state:
            return item
        time.sleep(.01)
    raise RuntimeError(f'Own child {pid} did not enter {state}: {snapshot(pid)}')


def fork_waiter():
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(writer)
            os.read(reader, 1)
        finally:
            os._exit(0)
    os.close(reader)
    return pid, writer


def extract_guard(raw, fake_job):
    tree = ast.parse(raw)
    keep = {'has_job', 'process_identity', 'fd_pid', 'own'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in keep]
    assert {node.name for node in functions} == keep
    context = {'os': os, 'pathlib': pathlib, 'job': fake_job, 'known_own': {}, 'zombie_proofs': {}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<read-only-guard-functions>', 'exec'), context)
    return context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resource-guard', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert platform.system() == 'Linux'
    assert not args.output.exists()
    raw = args.resource_guard.read_bytes()
    job = '999' + str(os.getpid()) + str(time.time_ns())
    guard = extract_guard(raw, job)
    record = {'scope': 'CPU_ONLY_SELF_CHILD_IDENTITY_NOT_NATIVE3259_LIVE_EVIDENCE',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'host': platform.node(), 'kernel': platform.release(), 'python': platform.python_version(),
              'self': snapshot(os.getpid()), 'resource_guard': str(args.resource_guard),
              'resource_guard_sha256': hashlib.sha256(raw).hexdigest(),
              'synthetic_job_id': job, 'pid_namespace_self': os.readlink('/proc/self/ns/pid'),
              'pid_namespace_init': os.readlink('/proc/1/ns/pid'),
              'gpu_used': False, 'live_guard_modified': False, 'other_processes_signalled': False,
              'experiments': [], 'errors': []}
    parent_group = pathlib.Path('/sys/fs/cgroup/freezer') / ('codex_guard_probe_' + str(os.getpid()) + '_' + str(time.time_ns()))
    own_group = parent_group / ('job_' + job)
    foreign_group = parent_group / 'outside_owned_job'
    cgroups = []
    children = []
    fds = []
    try:
        try:
            for group in (parent_group, own_group, foreign_group):
                group.mkdir()
                cgroups.append(group)
            record['isolated_freezer_cgroups_created'] = True
        except OSError as exc:
            record['isolated_freezer_cgroups_created'] = False
            record['cgroup_creation_error'] = repr(exc)
        pid, exit_writer = fork_waiter()
        children.append((pid, exit_writer))
        fd = os.pidfd_open(pid)
        fds.append(fd)
        wait_state(pid, 'S')
        rows = [{'phase': 'live_before_group_assignment', **snapshot(pid, fd)}]
        if record['isolated_freezer_cgroups_created']:
            (own_group / 'tasks').write_text(str(pid))
            rows.append({'phase': 'live_in_own_group', 'guard_own': guard['own'](pid), **snapshot(pid, fd)})
            cached = guard['known_own'][pid]
            rows[-1]['cached_start_ticks'] = cached['start']
            rows[-1]['cached_pidfd_fdinfo'] = pathlib.Path('/proc/self/fdinfo', str(cached['fd'])).read_text()
            (foreign_group / 'tasks').write_text(str(pid))
            rows.append({'phase': 'same_live_identity_moved_outside_own_group', 'guard_own': guard['own'](pid), **snapshot(pid, fd)})
            (own_group / 'tasks').write_text(str(pid))
            rows.append({'phase': 'same_live_identity_returned_to_own_group', 'guard_own': guard['own'](pid), **snapshot(pid, fd)})
        os.write(exit_writer, b'x')
        os.close(exit_writer)
        children[-1] = (pid, None)
        wait_state(pid, 'Z')
        rows.append({'phase': 'exited_zombie_before_reap', 'guard_own': guard['own'](pid), 'zombie_proofs': dict(guard['zombie_proofs']), **snapshot(pid, fd)})
        waited_pid, status = os.waitpid(pid, 0)
        children.pop()
        assert waited_pid == pid
        rows.append({'phase': 'fully_reaped_pidfd_still_held', 'guard_own': guard['own'](pid), 'wait_status': status, **snapshot(pid, fd)})
        record['experiments'].append({'label': 'observed_live_then_zombie_then_reaped', 'rows': rows})
        if record['isolated_freezer_cgroups_created']:
            pid, exit_writer = fork_waiter()
            children.append((pid, exit_writer))
            fd = os.pidfd_open(pid)
            fds.append(fd)
            wait_state(pid, 'S')
            (own_group / 'tasks').write_text(str(pid))
            first = snapshot(pid, fd)
            os.write(exit_writer, b'x')
            os.close(exit_writer)
            children[-1] = (pid, None)
            wait_state(pid, 'Z')
            rows = [{'phase': 'live_not_seen_by_guard_cache', **first},
                    {'phase': 'first_guard_observation_only_after_zombie', 'guard_own': guard['own'](pid), **snapshot(pid, fd)}]
            waited_pid, status = os.waitpid(pid, 0)
            children.pop()
            rows.append({'phase': 'unknown_child_fully_reaped', 'guard_own': guard['own'](pid), 'wait_status': status, **snapshot(pid, fd)})
            record['experiments'].append({'label': 'not_previously_cached_zombie', 'rows': rows})
        record['status'] = 'OBSERVATIONS_COMPLETE'
    except BaseException as exc:
        record['status'] = 'PROBE_FAILED'
        record['errors'].append(repr(exc))
        raise
    finally:
        # Reap only exact children we created; pidfd signal avoids PID reuse.
        for pid, writer in children:
            if writer is not None:
                os.close(writer)
            try:
                fd = os.pidfd_open(pid)
                signal.pidfd_send_signal(fd, signal.SIGKILL)
                os.close(fd)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        for fd in fds:
            os.close(fd)
        for cache in guard['known_own'].values():
            os.close(cache['fd'])
        removed = []
        for group in reversed(cgroups):
            try:
                group.rmdir()
                removed.append(str(group))
            except OSError as exc:
                record['errors'].append('Own temporary cgroup cleanup: ' + repr(exc))
        record['temporary_cgroups_removed'] = removed
        record['temporary_cgroups_remaining'] = [str(g) for g in cgroups if g.exists()]
        record['owned_children_remaining'] = [pid for pid, _ in children if pathlib.Path('/proc', str(pid)).exists()]
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({'status': record['status'], 'output': str(args.output), 'gpu_used': False,
                          'remaining': record['temporary_cgroups_remaining'], 'errors': record['errors']}))

if __name__ == '__main__':
    main()

"""Own CPU children only: controlled proc-read exit race and leader-Z semantics."""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import signal
import threading
import time
from guard_pidfd_probe_v1 import extract_guard, fork_waiter, snapshot, wait_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resource-guard', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert platform.system() == 'Linux' and not args.output.exists()
    raw = args.resource_guard.read_bytes()
    job = '999' + str(os.getpid()) + str(time.time_ns())
    guard = extract_guard(raw, job)
    parent = Path('/sys/fs/cgroup/freezer') / ('codex_guard_race_' + str(os.getpid()) + '_' + str(time.time_ns()))
    owned = parent / ('job_' + job)
    outside = parent / 'outside_owned_job'
    groups = []
    children = []
    handles = []
    out = {'scope': 'CPU_ONLY_CONTROLLED_EXIT_READ_RACE_NOT_NATIVE3259_LIVE_PROOF',
           'host': platform.node(), 'kernel': platform.release(), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'resource_guard_sha256': hashlib.sha256(raw).hexdigest(), 'gpu_used': False,
           'guard_source_not_modified': True, 'experiments': [], 'errors': []}
    try:
        for group in (parent, owned, outside):
            group.mkdir(); groups.append(group)
        pid, writer = fork_waiter()
        children.append((pid, [writer]))
        fd = os.pidfd_open(pid); handles.append(fd)
        wait_state(pid, 'S')
        (owned / 'cgroup.procs').write_text(str(pid))
        assert guard['own'](pid) is True
        before = snapshot(pid, fd)
        real_identity = guard['process_identity']
        first_stat = []
        def exit_between_proc_reads(target):
            value = real_identity(target)
            if target == pid and not first_stat:
                first_stat.append(value)
                os.write(writer, b'x')
                wait_state(pid, 'Z')
            return value
        guard['process_identity'] = exit_between_proc_reads
        decision = guard['own'](pid)
        after = snapshot(pid, fd)
        guard['process_identity'] = real_identity
        assert first_stat[0][0] == 'S' and after['state'] == 'Z' and after['pidfd_pid'] == pid
        assert decision is False
        stable_z_decision = guard['own'](pid)
        assert stable_z_decision is True
        out['experiments'].append({'label': 'exit_exactly_between_original_stat_and_cgroup_reads',
             'before': before, 'stale_stat_tuple_used': first_stat[0], 'original_guard_decision': decision,
             'after': after, 'subsequent_stable_z_guard_decision': stable_z_decision,
             'known_cache_still_present': pid in guard['known_own']})
        os.close(writer); children[-1] = (pid, [])
        os.waitpid(pid, 0); children.pop()
        # A process leader may be a Z while another thread is still alive.
        lead_r, lead_w = os.pipe(); worker_r, worker_w = os.pipe(); ready_r, ready_w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(lead_w); os.close(worker_w); os.close(ready_r)
            def worker():
                os.write(ready_w, (str(threading.get_native_id()) + '\n').encode())
                os.read(worker_r, 1)
                os._exit(0)
            threading.Thread(target=worker, daemon=True).start()
            os.read(lead_r, 1)
            libc = ctypes.CDLL('libc.so.6')
            libc.pthread_exit.argtypes = [ctypes.c_void_p]
            libc.pthread_exit.restype = None
            libc.pthread_exit(None)
            os._exit(99)
        os.close(lead_r); os.close(worker_r); os.close(ready_w)
        children.append((pid, [lead_w, worker_w, ready_r]))
        fd = os.pidfd_open(pid); handles.append(fd)
        assert select.select([ready_r], [], [], 3)[0]
        worker_tid = int(os.read(ready_r, 100).strip())
        os.close(ready_r); children[-1] = (pid, [lead_w, worker_w])
        (owned / 'cgroup.procs').write_text(str(pid))
        assert guard['own'](pid) is True
        rows = [{'phase': 'live_multithread_all_in_own_group', 'leader': snapshot(pid, fd), 'worker': snapshot(worker_tid), 'guard_own': True}]
        os.write(lead_w, b'x'); os.close(lead_w); children[-1] = (pid, [worker_w])
        wait_state(pid, 'Z')
        rows.append({'phase': 'leader_Z_other_thread_alive_in_own_group', 'leader': snapshot(pid, fd), 'worker': snapshot(worker_tid), 'guard_own': guard['own'](pid)})
        assert rows[-1]['worker']['state'] is not None
        (outside / 'tasks').write_text(str(worker_tid))
        rows.append({'phase': 'leader_Z_live_owned_child_thread_moved_outside_group', 'leader': snapshot(pid, fd), 'worker': snapshot(worker_tid), 'guard_own': guard['own'](pid)})
        os.write(worker_w, b'x'); os.close(worker_w); children[-1] = (pid, [])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not select.select([fd], [], [], 0)[0]:
            time.sleep(.01)
        rows.append({'phase': 'all_threads_exited_before_reap', 'leader': snapshot(pid, fd), 'worker': snapshot(worker_tid), 'guard_own': guard['own'](pid)})
        os.waitpid(pid, 0); children.pop()
        rows.append({'phase': 'all_threads_reaped', 'leader': snapshot(pid, fd), 'guard_own': guard['own'](pid)})
        out['experiments'].append({'label': 'leader_Z_is_not_necessarily_full_threadgroup_exit', 'rows': rows})
        out['status'] = 'OBSERVATIONS_COMPLETE'
    except BaseException as exc:
        out['status'] = 'PROBE_FAILED'; out['errors'].append(repr(exc)); raise
    finally:
        for pid, writers in children:
            for writer in writers:
                try: os.close(writer)
                except OSError: pass
            try:
                fd = os.pidfd_open(pid); signal.pidfd_send_signal(fd, signal.SIGKILL); os.close(fd)
            except ProcessLookupError: pass
            try: os.waitpid(pid, 0)
            except ChildProcessError: pass
        for fd in handles: os.close(fd)
        for cached in guard['known_own'].values(): os.close(cached['fd'])
        for group in reversed(groups):
            try: group.rmdir()
            except OSError as exc: out['errors'].append('temporary own cgroup cleanup: ' + repr(exc))
        out['remaining_cgroups'] = [str(group) for group in groups if group.exists()]
        out['owned_children_remaining'] = [pid for pid, _ in children if Path('/proc', str(pid)).exists()]
        out['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with args.output.open('x') as stream:
            json.dump(out, stream, indent=2, sort_keys=True, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps({'status': out['status'], 'output': str(args.output), 'remaining': out['remaining_cgroups'], 'errors': out['errors']}))

if __name__ == '__main__': main()

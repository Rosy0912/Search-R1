"""Real Linux, self-owned CPU child: live scheduling transitions are not reuse."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import signal
import sys
import time
from guard_pidfd_probe_v1 import snapshot, wait_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ownership-module-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert platform.system() == 'Linux' and not args.output.exists()
    sys.path.insert(0, str(args.ownership_module_dir))
    from resource_ownership_v2 import OwnershipObserver
    job = '999' + str(os.getpid()) + str(time.time_ns())
    events = []
    observer = OwnershipObserver(job, events.append)
    parent = Path('/sys/fs/cgroup/freezer') / ('codex_guard_schedule_' + str(os.getpid()) + '_' + str(time.time_ns()))
    owned = parent / ('job_' + job)
    groups = []
    pid = writer = fd = None
    out = dict(scope='REAL_LINUX_CPU_ONLY_LIVE_SCHEDULING_NOT_NATIVE3259_PID_FORENSICS',
               host=platform.node(), kernel=platform.release(), gpu_used=False,
               utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               candidate_module_sha256=hashlib.sha256((args.ownership_module_dir / 'resource_ownership_v2.py').read_bytes()).hexdigest(),
               experiments=[], ownership_events=events, errors=[])
    try:
        for group in (parent, owned):
            group.mkdir(); groups.append(group)
        reader, writer = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(writer)
            while os.read(reader, 1):
                until = time.monotonic() + .5
                while time.monotonic() < until:
                    pass
            os._exit(0)
        os.close(reader)
        fd = os.pidfd_open(pid)
        wait_state(pid, 'S')
        (owned / 'cgroup.procs').write_text(str(pid))
        assert observer.own(pid) is True
        identity = observer.backend.identity
        for before_state, after_state in (('S', 'R'), ('R', 'S')):
            if before_state == 'R':
                os.write(writer, b'r'); wait_state(pid, 'R')
            else:
                wait_state(pid, 'S')
            reads = []
            def scheduling_transition(target):
                value = identity(target); reads.append(value)
                if len(reads) == 1:
                    assert value['state'] == before_state, value
                    if before_state == 'S': os.write(writer, b'r')
                    wait_state(pid, after_state)
                return value
            observer.backend.identity = scheduling_transition
            decision = observer.own(pid)
            observer.backend.identity = identity
            assert len(reads) == 2 and reads[0]['state'] == before_state and reads[1]['state'] == after_state, reads
            assert reads[0]['start'] == reads[1]['start'] and decision is True
            out['experiments'].append(dict(label=before_state + '_to_' + after_state,
                                          actual_raw_identity_reads=reads, guard_own=decision,
                                          identity_after=snapshot(pid, fd), held_cache=observer._cached(pid)))
        out['status'] = 'REAL_LIVE_SCHEDULING_CLASSIFICATION_PASS'
    except BaseException as exc:
        out['status'] = 'PROBE_FAILED'; out['errors'].append(repr(exc)); raise
    finally:
        observer.backend.identity = identity if 'identity' in locals() else observer.backend.identity
        if writer is not None: os.close(writer)
        if pid is not None:
            try:
                if fd is not None and not select.select([fd], [], [], 0)[0]:
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError: pass
            try: os.waitpid(pid, 0)
            except ChildProcessError: pass
        if fd is not None: os.close(fd)
        observer.close()
        for group in reversed(groups):
            try: group.rmdir()
            except OSError as exc: out['errors'].append('own cgroup cleanup: ' + repr(exc))
        out['remaining_cgroups'] = [str(group) for group in groups if group.exists()]
        out['owned_child_remaining'] = bool(pid is not None and Path('/proc', str(pid)).exists())
        out['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with args.output.open('x') as stream:
            json.dump(out, stream, indent=2, sort_keys=True, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(dict(status=out['status'], output=str(args.output), remaining=out['remaining_cgroups'], errors=out['errors'])))


if __name__ == '__main__': main()

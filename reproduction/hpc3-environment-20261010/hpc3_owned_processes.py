"""Own-child cleanup for HPC3 kernels without pidfd; never enumerate foreign jobs.

The controller becomes a child subreaper before spawning. During cleanup it stops
parents before children, so stopped parents cannot reap/reuse child PIDs while
signals are sent. Direct/adopted children remain unreaped by this controller.
"""
import ctypes
import os
import pathlib
import signal
import subprocess
import threading
import time

REAP_LOCK = threading.RLock()
_installed = False

def protect_controller_lifecycle():
    """Serialize Python child creation/reaping against identity-based signalling.

    Popen._internal_poll captures waitpid in a default argument, so guarding only
    os.waitpid is insufficient. Guard its complete poll/wait paths as well.
    This controller runs no Ray/multiprocessing or foreign SIGCHLD handler.
    """
    global _installed
    if _installed:
        return
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError('Unexpected controller SIGCHLD handler')
    def guarded(function):
        def call(*args, **kwargs):
            with REAP_LOCK:
                return function(*args, **kwargs)
        return call
    for name in ('__init__', '_internal_poll'):
        setattr(subprocess.Popen, name, guarded(getattr(subprocess.Popen, name)))
    # Blocking waits must release the lifecycle lock between polls, otherwise a
    # monitor waiting for a child could prevent cleanup from terminating it.
    def popen_wait(child, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while child.returncode is None:
            child._internal_poll()
            if child.returncode is not None:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(child.args, timeout)
            time.sleep(.01)
        return child.returncode
    subprocess.Popen._wait = popen_wait
    original_waitpid = os.waitpid
    def polling_wait(function, args, options):
        while True:
            with REAP_LOCK:
                result = function(*args, options | os.WNOHANG)
            if result[0] or options & os.WNOHANG:
                return result
            time.sleep(.01)
    os.waitpid = lambda pid, options: polling_wait(original_waitpid, (pid,), options)
    os.wait = lambda: os.waitpid(-1, 0)
    for name in ('wait3', 'wait4'):
        if hasattr(os, name):
            original = getattr(os, name)
            if name == 'wait3':
                setattr(os, name, lambda options, fn=original: polling_wait(fn, (), options))
            else:
                setattr(os, name, lambda pid, options, fn=original: polling_wait(fn, (pid,), options))
    _installed = True


def identity(pid):
    p = pathlib.Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    uid = next(x for x in (p / 'status').read_text().splitlines() if x.startswith('Uid:')).split()[1]
    return {'pid': pid, 'ppid': int(fields[1]), 'starttime_ticks': int(fields[19]),
            'state': fields[0], 'uid': int(uid), 'cgroup': (p / 'cgroup').read_text().splitlines()}


def members(scope):
    return {int(x) for p in pathlib.Path(scope).rglob('cgroup.procs') for x in p.read_text().splitlines()}


class OwnedProcesses:
    def __init__(self, scope, relative, cwd=None):
        protect_controller_lifecycle()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
        self.scope, self.relative, self.cwd = pathlib.Path(scope), relative, cwd
        self.baseline = members(scope)
        self.children, self.streams = [], []
        self.revoked = False

    def spawn(self, command, log, env=None, pass_fds=()):
        with REAP_LOCK:
            if self.revoked:
                raise RuntimeError('Owned scope revoked; no further spawn')
            stream = pathlib.Path(log).open('w')
            self.streams.append(stream)
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env=env, cwd=self.cwd, pass_fds=tuple(pass_fds))
            self.children.append((child, None))
            return child

    def _owned(self, row):
        return row['uid'] == os.getuid() and any(
            x.split(':', 2)[-1] == self.relative or
            x.split(':', 2)[-1].startswith(self.relative + '/') for x in row['cgroup'])

    def _freeze(self, pending):
        frozen, todo = {}, set(pending)
        self._frozen = frozen
        deadline = time.monotonic() + 5
        while todo:
            progressed = False
            for pid in sorted(todo):
                try:
                    row = identity(pid)
                except FileNotFoundError:
                    todo.remove(pid)
                    progressed = True
                    continue
                if row['ppid'] != os.getpid() and row['ppid'] not in frozen:
                    continue
                if not self._owned(row):
                    raise RuntimeError('Refuse to signal a process outside own Slurm step')
                # Controller Python reaping is locked, or parent is stopped.
                # Neither can reap the target while its identity is checked.
                again = identity(pid)
                if (again['ppid'], again['starttime_ticks']) != (row['ppid'], row['starttime_ticks']):
                    raise RuntimeError('Child identity changed before freeze')
                frozen[pid] = row
                if again['state'] != 'Z':
                    os.kill(pid, signal.SIGSTOP)
                    while identity(pid)['state'] not in ('T', 't', 'Z'):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Own child did not stop')
                        time.sleep(.01)
                todo.remove(pid)
                progressed = True
            if not progressed:
                raise RuntimeError('Refuse cleanup of an unproven process ancestry')
        return frozen

    def cleanup(self):
        with REAP_LOCK:
            return self._cleanup_locked()

    def _cleanup_locked(self):
        self.revoked = True
        began = time.monotonic()
        self.cleanup_record = {'backend': 'SUBREAPER_REAP_LOCK_STOPPED_PARENT_IDENTITY',
                               'signals': [], 'remaining': [], 'actual_cgroup_empty': False}
        frozen = {}
        self._frozen = frozen
        sent = set()
        try:
            while time.monotonic() - began < 30:
                pending = members(self.scope) - self.baseline
                if not pending:
                    break
                frozen = self._freeze(pending)
                sig = signal.SIGTERM if time.monotonic() - began < 5 else signal.SIGKILL
                # Descendants before parents; this subreaper does not reap yet.
                for pid, row in reversed(list(frozen.items())):
                    now = identity(pid)
                    if now['starttime_ticks'] != row['starttime_ticks'] or not self._owned(now):
                        raise RuntimeError('Frozen child identity changed')
                    key = (pid, row['starttime_ticks'], int(sig))
                    if now['state'] != 'Z' and key not in sent:
                        os.kill(pid, sig)
                        sent.add(key)
                        self.cleanup_record['signals'].append({'pid': pid, 'signal': int(sig),
                                                               'starttime_ticks': row['starttime_ticks']})
                for pid in frozen:
                    try:
                        os.kill(pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                frozen = {}
                self._frozen = frozen
                time.sleep(.1)
                for child, _ in self.children:
                    child.poll()
                # Reap adopted orphans only after all signals/resumes above.
                for pid in pending:
                    if pid in {child.pid for child, _ in self.children}:
                        continue
                    try:
                        if identity(pid)['ppid'] == os.getpid():
                            os.waitpid(pid, os.WNOHANG)
                    except (FileNotFoundError, ChildProcessError):
                        pass
            remaining = members(self.scope) - self.baseline
            self.cleanup_record['remaining'] = sorted(remaining)
            self.cleanup_record['actual_cgroup_empty'] = not remaining
            if remaining:
                raise RuntimeError('Own descendants remain; scheduler teardown required')
            for child, _ in self.children:
                child.wait(timeout=10)
            direct = {child.pid for child, _ in self.children}
            for entry in pathlib.Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                if pid in self.baseline or pid in direct:
                    continue
                try:
                    row = identity(pid)
                    if row['ppid'] == os.getpid() and row['state'] == 'Z' and self._owned(row):
                        os.waitpid(pid, os.WNOHANG)
                except (FileNotFoundError, ChildProcessError, ProcessLookupError):
                    pass
        finally:
            for pid, expected in self._frozen.items():
                try:
                    row = identity(pid)
                    if row['starttime_ticks'] == expected['starttime_ticks'] and self._owned(row):
                        os.kill(pid, signal.SIGCONT)
                except FileNotFoundError:
                    pass
            for stream in self.streams:
                stream.close()

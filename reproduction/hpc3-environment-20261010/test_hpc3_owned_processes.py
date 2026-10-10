"""CPU integration check on the actual Slurm cgroup, including protected baseline."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import threading
from hpc3_owned_processes import OwnedProcesses, members, identity

row = next(x.split(':', 2) for x in pathlib.Path('/proc/self/cgroup').read_text().splitlines()
           if 'freezer' in x.split(':', 2)[1].split(','))
scope = pathlib.Path('/sys/fs/cgroup/freezer') / row[2].lstrip('/')
baseline = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
try:
    owner = OwnedProcesses(scope, row[2])
    log = pathlib.Path(tempfile.mkdtemp(dir='/data/user/jxu957/experiments/topd-20261008/outputs')) / 'child.log'
    # The root exits; its live grandchild must be adopted and cleaned up.
    owner.spawn([sys.executable, '-c',
                 'import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])'], log)
    # A second root ignores TERM, exercising escalation while its PID is held.
    resistant = owner.spawn([sys.executable, '-c',
                             'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'], log.with_name('resistant.log'))
    time.sleep(1)
    stop = threading.Event()
    errors = []
    def poll_concurrently():
        while not stop.is_set():
            resistant.poll()
            stop.wait(.01)
    def guardian_cleanup():
        try: owner.cleanup()
        except BaseException as e: errors.append(repr(e))
    poller = threading.Thread(target=poll_concurrently)
    guardian = threading.Thread(target=guardian_cleanup)
    poller.start(); guardian.start()
    guardian.join(45)
    stop.set(); poller.join(10)
    assert not guardian.is_alive() and not poller.is_alive() and not errors, errors
    assert baseline.poll() is None, 'Protected baseline was signalled'
    assert resistant.returncode is not None
    assert not (members(scope) - owner.baseline)
    # A protected baseline parent later forks: refuse its new child and resume
    # already-stopped owned children instead of leaving them frozen on failure.
    trigger = log.with_name('fork-trigger')
    pidfile = log.with_name('protected-child.pid')
    code = ('import pathlib,subprocess,sys,time;trigger=pathlib.Path(sys.argv[1]);'
            'exec("while not trigger.exists(): time.sleep(.02)");'
            'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
            'pathlib.Path(sys.argv[2]).write_text(str(p.pid));time.sleep(60)')
    protected = subprocess.Popen([sys.executable, '-c', code, str(trigger), str(pidfile)])
    second = OwnedProcesses(scope, row[2])
    own = second.spawn([sys.executable, '-c', 'import time;time.sleep(60)'], log.with_name('resume.log'))
    protected_child = None
    try:
        trigger.touch()
        deadline = time.monotonic() + 5
        while not pidfile.exists():
            assert time.monotonic() < deadline
            time.sleep(.02)
        protected_child = int(pidfile.read_text())
        try:
            second.cleanup()
        except RuntimeError as e:
            assert 'unproven process ancestry' in str(e)
        else:
            raise AssertionError('Unproven baseline descendant was accepted')
        time.sleep(.1)
        assert identity(own.pid)['state'] not in ('T', 't')
        assert protected.poll() is None
        assert identity(protected_child)['state'] not in ('T', 't', 'Z')
    finally:
        own.terminate()
        own.wait(timeout=10)
        protected.terminate()
        protected.wait(timeout=10)
        if protected_child is not None:
            # Our fixture is now an unreaped direct child of this subreaper.
            assert identity(protected_child)['ppid'] == os.getpid()
            os.kill(protected_child, 15)
            os.waitpid(protected_child, 0)
    print(json.dumps({'status': 'OWN_CHILD_TREE_CLEANUP_PASS_BASELINE_PROTECTED',
                      'job': os.environ['SLURM_JOB_ID'], 'unknown_ancestry_refused_and_owned_resumed': True,
                      'guardian_thread_cleanup_with_concurrent_poll_pass': True,
                      'cleanup': owner.cleanup_record}))
finally:
    baseline.terminate()
    baseline.wait(timeout=10)

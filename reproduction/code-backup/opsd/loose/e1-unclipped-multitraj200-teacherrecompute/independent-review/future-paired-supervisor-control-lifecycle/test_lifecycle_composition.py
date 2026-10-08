"""Two independent local compositions. No production main, GPU or scheduler."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


args = argparse.ArgumentParser()
args.add_argument('--source', required=True)
args.add_argument('--source-sha256', required=True)
a, remainder = args.parse_known_args()
path = Path(a.source).resolve(strict=True); raw = path.read_bytes()
if hashlib.sha256(raw).hexdigest() != a.source_sha256:
    raise ValueError('Sealed input full SHA differs')
s = types.ModuleType('independent_paired_control_lifecycle')
s.__file__ = str(path)
exec(compile(raw, str(path), 'exec', dont_inherit=True, optimize=0), s.__dict__)


class LifecycleCompositionTests(unittest.TestCase):
    def test_real_run_roles_failure_composition_cleans_direct_eval_when_head_already_gone(self):
        # CPU-only controlled Popen/Slurm/Ray fixture. This executes the actual
        # run_roles finally branch; it does not claim real GPU process cleanup.
        for unknown_cleanup in (False, True):
            with self.subTest(unknown_cleanup=unknown_cleanup), tempfile.TemporaryDirectory() as name:
                run = Path(name).resolve(); events = []; direct = []
                class Child:
                    def __init__(self, pid): self.pid = pid; self.returncode = None
                    def poll(self): return self.returncode
                head, child = Child(111), Child(222)
                class Tracker:
                    def __init__(self, *args): self.cleanup_invoked = False
                    def scan(self, roots): events.append(('scan', tuple(roots)))
                    def cleanup(self, children, sampler, service):
                        self.cleanup_invoked = True; direct.extend(children)
                        events.append(('cleanup', tuple(p.pid for p in children), head.poll()))
                        if unknown_cleanup: raise RuntimeError('Fixture cleanup remains unknown')
                        return {'status': 'CPU_SYNTHETIC_NOT_GPU_CLEANUP'}
                def folder_reserve(folder):
                    if Path(folder).parent == Path('/tmp'): return
                    Path(folder).mkdir()
                def command(argv, folder, label):
                    (folder / (label + '.stdout')).write_bytes(b'')
                def role_env(role, *args):
                    events.append(('role', role)); return {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}
                def failing_eval(*args, **kwargs):
                    kwargs['on_popen'](child); head.returncode = 0
                    raise TimeoutError('Fixture eval is alive while head already exited')
                lease = type('FixtureLease', (), {'fds': [], 'check': lambda self: None})()
                sampler = type('FixtureSampler', (), {'sample': lambda self: {'gpu_processes': []}})()
                sock = type('FixtureSocket', (), {'close': lambda self: None})()
                q = {'role_timeout_seconds': 1, 'control_tools': {'paired_cpu_prepare.py': 'a' * 64,
                                                                 'paired_supervisor.py': 'b' * 64}}
                with patch.object(s, 'checkpoint_guard', return_value={'fixture': 'unchanged'}), \
                     patch.object(s, 'reserve', side_effect=folder_reserve), \
                     patch.object(s, 'role_env', side_effect=role_env), patch.object(s, 'command', side_effect=command), \
                     patch.object(s.subprocess, 'Popen', return_value=head) as popen, \
                     patch.object(s, 'proc_identity', return_value={'pid': head.pid}), \
                     patch.object(s, 'RoleProcesses', Tracker), patch.object(s, 'wait_child', side_effect=failing_eval), \
                     patch.object(s.socket, 'create_connection', return_value=sock):
                    error = RuntimeError if unknown_cleanup else TimeoutError
                    with self.assertRaises(error):
                        s.run_roles(run, q, {'request': {'path': '/CPU_FIXTURE', 'sha256': 'c' * 64}}, {},
                                    {'job_id': 'CPU_FIXTURE', 'step_id': '0'}, lease, sampler, {'pid': 333})
                self.assertEqual(direct, [head, child])
                self.assertEqual([e for e in events if e[0] == 'role'], [('role', 'official_base')])
                self.assertEqual([e for e in events if e[0] == 'cleanup'], [('cleanup', (111, 222), 0)])
                self.assertEqual(popen.call_count, 1)
                self.assertFalse((run / 'actual_hf200').exists())

    def test_real_nine_fd_lease_inheritance_retains_exclusive_locks_until_actual_cpu_child_exit(self):
        # Temporary ordinary files exercise actual POSIX FD/flock inheritance;
        # these are not production global/GPU lease paths or an allocation.
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name).resolve(); lease = s.LeaseSet(['GPU-CPU-FIXTURE-' + str(i) for i in range(8)])
            child = None; probes = []
            try:
                for i in range(9):
                    file = folder / ('lease-' + str(i)); fd = os.open(file, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); os.set_inheritable(fd, True); st = os.fstat(fd)
                    lease.fds.append(fd); lease.records.append({'path': str(file), 'fd': fd, 'device': st.st_dev,
                                                                'inode': st.st_ino, 'uid': st.st_uid})
                lease.check(); env = lease.environment()
                self.assertEqual(len(s.unique(env['SEARCH_R1_GPU_LEASES'])), 8)
                program = 'import os,sys,json;fds=json.loads(sys.argv[1]);[os.fstat(fd) for fd in fds];print("READY",flush=True);sys.stdin.readline()'
                child = subprocess.Popen([sys.executable, '-B', '-c', program, s.canonical(lease.fds).decode()],
                                         pass_fds=lease.fds, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(child.stdout.readline(), b'READY\n')
                lease.close()
                for rec in lease.records:
                    fd = os.open(rec['path'], os.O_RDWR | os.O_NOFOLLOW); probes.append(fd)
                    with self.assertRaises(BlockingIOError): fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                child.stdin.write(b'finish\n'); child.stdin.flush(); child.wait(timeout=5)
                self.assertEqual(child.returncode, 0)
                for fd in probes:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                if child is not None:
                    if child.poll() is None: child.kill(); child.wait()
                    for stream in (child.stdin, child.stdout, child.stderr): stream.close()
                lease.close()
                for fd in probes: os.close(fd)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]] + remainder, verbosity=2)

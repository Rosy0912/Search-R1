"""Owned Slurm cleanup copied exactly from reviewed latest launcher; no GPU allocation."""
import os,re,signal,subprocess,time
from pathlib import Path
ROOT=Path(os.environ.get("AGENT_ROOT","/dataset/shaoshitong/search-r1-rl/topd"))
def require(condition,message):
    if not condition:raise RuntimeError(message)

def cgroup_scope(job):
    """Find ONLY the current Slurm step cgroup; never walk another job."""
    rows = [line.split(':', 2) for line in Path('/proc/self/cgroup').read_text().splitlines()]
    for _, controllers, relative in rows:
        if controllers == '' or 'freezer' in controllers.split(','):
            require(re.search(r'(^|/)job_' + re.escape(job) + r'(/|$)', relative), 'Current cgroup does not identify this Slurm job')
            root = Path('/sys/fs/cgroup') if controllers == '' else Path('/sys/fs/cgroup/freezer')
            path = root / relative.lstrip('/')
            require((path / 'cgroup.procs').is_file(), 'Slurm cgroup process list unavailable')
            return path, relative
    raise RuntimeError('Cannot verify own Slurm cgroup; refusing service launch')


def cgroup_pids(scope):
    return {int(line) for file in scope.rglob('cgroup.procs') for line in file.read_text().splitlines()}


class OwnedProcesses:
    def __init__(self, scope, relative):
        self.scope, self.relative = scope, relative
        self.baseline = cgroup_pids(scope)
        self.children, self.streams = [], []

    def spawn(self, command, log, env=None):
        stream = Path(log).open('w')
        self.streams.append(stream)
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=ROOT/'repo')
        self.children.append((child, os.pidfd_open(child.pid)))
        return child

    def cleanup(self):
        # Re-scan after TERM: Ray descendants may daemonize/spawn during shutdown.
        # pidfd protects identity; current membership must still be this job step.
        for attempt in range(12):
            for child,_ in self.children:
                child.poll()  # Reap direct zombies while retaining their original pidfd.
            pending=cgroup_pids(self.scope)-self.baseline
            if not pending:
                break
            for pid in pending:
                try:
                    fd=os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                try:
                    membership=Path(f'/proc/{pid}/cgroup').read_text().splitlines()
                    if not any(line.split(':',2)[-1]==self.relative or line.split(':',2)[-1].startswith(self.relative+'/') for line in membership):
                        continue  # Never signal a process that left our own step.
                    signal.pidfd_send_signal(fd,signal.SIGTERM if attempt==0 else signal.SIGKILL)
                except (ProcessLookupError,FileNotFoundError):
                    pass
                finally:
                    os.close(fd)
            time.sleep(2 if attempt==0 else .25)
        require(not (cgroup_pids(self.scope)-self.baseline),'Own Slurm step descendants remain; scheduler teardown still required')
        for child,fd in self.children:
            child.wait(timeout=10)
            os.close(fd)
        for stream in self.streams:
            stream.close()


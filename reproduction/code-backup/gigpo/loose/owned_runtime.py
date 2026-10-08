"""Own Slurm process cleanup and detached resource sampling; adapted from reviewed TOP-D launcher."""
import hashlib,os,re,signal,socket,subprocess,time
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo')
def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

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

    def spawn(self, command, log, env=None, pass_fds=()):
        stream = Path(log).open('w')
        self.streams.append(stream)
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=ROOT/'repo', pass_fds=pass_fds)
        self.children.append((child, os.pidfd_open(child.pid)))
        return child

    def cleanup(self):
        # Re-scan after TERM: Ray descendants may daemonize/spawn during shutdown.
        # pidfd protects identity; current membership must still be this job step.
        deadline=time.monotonic()+60
        attempt=0
        while True:
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
            if time.monotonic()>=deadline:
                break
            time.sleep(min(2 if attempt==0 else .25, max(0,deadline-time.monotonic())))
            attempt+=1
        remaining={}
        for pid in cgroup_pids(self.scope)-self.baseline:
            try:
                # A zombie has exited and cannot execute or retain CUDA resources.
                # Only the scheduler/PID1 can reap orphan zombies; never treat a
                # living or unreadable process as cleaned up.
                remaining[str(pid)]=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
            except FileNotFoundError:
                continue
        self.cleanup_residual=remaining
        require(all(state=='Z' for state in remaining.values()),
                'Live/unrecognized own Slurm descendants remain: '+repr(remaining))
        for child,fd in self.children:
            child.wait(timeout=10)
            os.close(fd)
        for stream in self.streams:
            stream.close()

def process_sample(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid, user_cpu_ticks=int(fields[11]), system_cpu_ticks=int(fields[12]),
                    hz=os.sysconf('SC_CLK_TCK'),
                    memory=[line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:'))])
    except (OSError, IndexError):
        return dict(pid=pid, exited=True)

def gpu_sample():
    query = ['nvidia-smi','--format=csv,noheader,nounits']
    return dict(gpus=subprocess.check_output(query+['--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu'],text=True).splitlines(),
                processes=subprocess.check_output(query+['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'],text=True).splitlines())

def available(port):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',port))

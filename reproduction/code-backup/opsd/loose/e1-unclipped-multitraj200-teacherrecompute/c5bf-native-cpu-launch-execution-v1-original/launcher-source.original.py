"""One CPU-only overlapping step in existing own3304; no model or evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
TRAIN = ROOT / 'repo-fcecbea5d939'
EVAL = ROOT / 'repo-final200-eval-c5bf4c8db335'
TOOLS = ROOT / 'tools/final200-eval-native-cpu-ee31da655703'
MANIFEST_SHA = 'ee31da65570318227ad6b037187cffa14e4dc489bac2a504d2bc54010c95da05'
CONTROLLER_SHA = '176ade36207d1ddf13ffdcf1faf859d0f1604464909a5336e8ba751993ccd507'
CHECKER_SHA = '74ce068cbfcad64b9776a5e4c3941cf1d4ce008ad5e0ebc8e6d6af6c4edfb67f'
INVENTORY = ROOT / 'tools/final200-eval-source-publication-v1/source-inventory.actual.json'
INVENTORY_SHA = 'fd8fa25d8321a4cd11b9faa5d8fdbd5f1692172a9e51ea5eba447c3565b8cc0d'
PYTHON = '/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python'
JOB = '3304'
DRIVER = 2548319
DRIVER_START = '1834784786'
BASE = ROOT / 'runs/formal-3304/evidence'
LAUNCH = BASE / 'c5bf-native-cpu-launch-once-v1'
OUTPUT = BASE / 'c5bf-native-cpu-preparation-once-v1'
SCOPE = 'CPU_ONLY_C5BF_IMPORT_PARTIAL_CONFIG_PREPARATION_NOT_MODEL_EVALUATION'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stable(path, expected=None):
    need(path.is_absolute() and path.resolve(strict=True) == path, 'Physical absolute file required')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_uid == 0 and before.st_nlink == 1, 'Owned regular single-link file required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        chunks = []
        while True:
            chunk = os.read(fd, 1024**2)
            if not chunk:
                break
            chunks.append(chunk)
        ended = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_uid', 'st_mode', 'st_nlink', 'st_mtime_ns', 'st_ctime_ns')
    need(all(getattr(before, k) == getattr(s, k) for s in (opened, ended, after) for k in keys), 'File changed during read')
    raw = b''.join(chunks)
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        need(sha == expected, 'Pinned file differs: ' + str(path))
    return raw, dict(path=str(path), sha256=sha, bytes=len(raw), inode=before.st_ino,
                    device=before.st_dev, uid=before.st_uid, nlink=before.st_nlink,
                    mode=stat.S_IMODE(before.st_mode))


def write_once(path, value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b'\n'
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def process(pid):
    root = Path('/proc') / str(pid)
    first = (root / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    status = (root / 'status').read_text()
    cgroup = (root / 'cgroup').read_text()
    last = (root / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    need(first == last, 'Process start changed')
    uids = [int(x) for x in re.search(r'^Uid:\s+(.*)', status, re.M).group(1).split()]
    need(uids == [0] * 4, 'Process UID differs')
    return dict(pid=pid, start_ticks=first, uids=uids, cgroup=cgroup, status_original=status)


def context():
    need(os.getuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10', 'Own h100-10 UID0 required')
    driver = process(DRIVER)
    need(driver['start_ticks'] == DRIVER_START and re.search(r'(?:^|/)job_3304/step_0(?:/|$)', driver['cgroup'], re.M), 'Original training driver differs')
    def read_command(argv):
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        need(p.returncode == 0, 'Read-only context command failed')
        return dict(argv=argv, waited=True, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    head = read_command(['git', '--no-optional-locks', '-C', str(TRAIN), 'rev-parse', 'HEAD'])
    clean = read_command(['git', '--no-optional-locks', '-C', str(TRAIN), 'status', '--porcelain', '--untracked-files=all'])
    need(head['stdout'].strip() == 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6' and not clean['stdout'].strip(), 'Frozen training source differs')
    slurm = read_command(['scontrol', 'show', 'job', JOB, '-o'])
    fields = dict(x.split('=', 1) for x in slurm['stdout'].split() if '=' in x)
    for key, wanted in dict(JobId=JOB, JobState='RUNNING', Restarts='0', NodeList='h100-10', UserId='root(0)', TresPerNode='gres:gpu:h100:8').items():
        need(fields.get(key) == wanted, 'Current own allocation differs: ' + key)
    raw, manifest_record = stable(TOOLS / 'manifest.json', MANIFEST_SHA)
    manifest = json.loads(raw)
    need(manifest['source_commit'] == 'c5bf4c8db33562a259fa227a940dc69543901b98' and len(manifest['tools']) == 22, 'Published preparation bundle differs')
    records = {name: stable(TOOLS / name, sha)[1] for name, sha in manifest['tools'].items()}
    need(set(p.name for p in TOOLS.iterdir()) == set(records) | {'manifest.json'}, 'Unexpected tool bundle members')
    need(all(r['mode'] == 0o444 for r in records.values()) and stat.S_IMODE(TOOLS.stat().st_mode) == 0o555, 'Immutable bundle modes differ')
    inventory = stable(INVENTORY, INVENTORY_SHA)[1]
    return dict(driver=driver, source_head=head, source_clean=clean, slurm=slurm,
                tool_manifest=manifest_record, tool_records=records, source_inventory=inventory)


CPU_BOOTSTRAP = r'''
import hashlib, os, pathlib, re, stat, sys
job=os.environ.get('SLURM_JOB_ID'); step=os.environ.get('SLURM_STEP_ID')
if job != '3304' or not isinstance(step,str) or re.fullmatch('[0-9]+',step) is None: raise ValueError('Actual CPU step identity missing')
if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not sys.dont_write_bytecode: raise ValueError('CPU-only -B required')
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
if not p.is_absolute() or p.resolve(strict=True)!=p: raise ValueError('Physical controller required')
a=p.lstat(); fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
try:
 b=os.fstat(fd); chunks=[]
 while True:
  chunk=os.read(fd,1024**2)
  if not chunk: break
  chunks.append(chunk)
 c=os.fstat(fd)
finally:os.close(fd)
d=p.lstat();keys=('st_dev','st_ino','st_size','st_uid','st_mode','st_nlink','st_mtime_ns','st_ctime_ns')
if not stat.S_ISREG(a.st_mode) or a.st_uid!=0 or a.st_nlink!=1 or any(getattr(a,k)!=getattr(s,k) for s in (b,c,d) for k in keys):raise ValueError('Controller source changed')
raw=b''.join(chunks)
if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Controller execution SHA differs')
sys.argv=[str(p)]+sys.argv[3:]+['--execution-job-id',job,'--execution-step-id',step]
exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),{'__name__':'__main__','__file__':str(p),'__package__':None,'__builtins__':__builtins__})
'''


def command():
    return ['srun', '--jobid=' + JOB, '--overlap', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=4', '--gres=none', '--nodelist=h100-10',
            'env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
            'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2',
            'PYTHONPATH=' + str(EVAL), PYTHON, '-B', '-c', CPU_BOOTSTRAP,
            str(TOOLS / 'run_eval_native_cpu_once.py'), CONTROLLER_SHA,
            '--source-inventory', str(INVENTORY), '--source-inventory-sha256', INVENTORY_SHA,
            '--self-sha256', CONTROLLER_SHA, '--checker-sha256', CHECKER_SHA,
            '--output-directory', str(OUTPUT)]


def main():
    before = context()
    need(BASE.resolve(strict=True) == BASE and BASE.stat().st_uid == 0, 'Actual existing owned evidence parent required')
    need(not os.path.lexists(OUTPUT), 'Previous output may not be retried')
    LAUNCH.mkdir(mode=0o700)
    sync_dir(BASE)
    sync_dir(LAUNCH)
    argv = command()
    environment = dict(os.environ)
    removed = [n for n in environment if n.startswith(('SBATCH_', 'SRUN_', 'SLURM_', 'OPSD_FINAL200_')) or n == 'OPSD_E1_RETRIEVER_URL']
    for name in removed:
        environment.pop(name)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    write_once(LAUNCH / 'intent.original.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        before=before, argv=argv, removed_environment_variable_names=sorted(removed),
        scope=SCOPE, original_training_job_id=JOB, actual_eval_job_id=None,
        expected_future_model_directory=None, expected_future_stage_manifest=None,
        supervisor=process(os.getpid()), retry_policy='NO_AUTOMATIC_RETRY_UNKNOWN_OUTCOME_REQUIRES_REVIEW'))
    sampling_error = None
    with (LAUNCH / 'srun.stdout.original').open('xb') as stdout, (LAUNCH / 'srun.stderr.original').open('xb') as stderr:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=environment, cwd=str(BASE))
        try:
            write_once(LAUNCH / 'srun-popen.original.json', dict(pid=child.pid, parent_pid=os.getpid(), argv=argv))
            write_once(LAUNCH / 'srun-start.original.json', process(child.pid))
        except BaseException as error:
            sampling_error = dict(type=type(error).__name__, message=str(error))
        code = child.wait()
        stdout.flush(); stderr.flush(); os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    write_once(LAUNCH / 'srun-wait.original.json', dict(returncode=code, waited=True, pid=child.pid,
        parent_pid=os.getpid(), argv=argv, start_record_error=sampling_error))
    after = context()
    write_once(LAUNCH / 'after.original.json', after)
    result = dict(scope=SCOPE, srun_waited=True, srun_returncode=code,
        start_record_error=sampling_error, launch_directory=str(LAUNCH), output_directory=str(OUTPUT),
        stored_training_job_id=JOB, actual_evaluation_job_id=None,
        training_original_driver_unchanged=True, training_source_unchanged=True)
    write_once(LAUNCH / 'result.original.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if sampling_error is not None:
        raise ValueError('Actual srun process sampling failed; wait preserved')
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Candidate paired-final200 control path. Import is inert; main is production.

This software does not qualify training200 or authenticate any future allocation
by being tested locally. Only a root-signed request and a separate actual ready
approval allow its production control path. It never submits or retries a job.
"""
import argparse
import datetime
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
import types

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006')
REPO = ROOT / 'repo-final200-eval-c5bf4c8db335'
TRAIN_REPO = ROOT / 'repo-fcecbea5d939'
TRAIN = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
EVAL = 'c5bf4c8db33562a259fa227a940dc69543901b98'
PYTHON = '/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python'
RAY = '/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/ray'
PUBLIC = ROOT / 'runs/formal-3304/bindings/public.json'
PUBLIC_SHA = 'ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
CONTRACT_SHA = 'a12edf342425c4f0279142f9366f4aaa98219313a434de687acc8a3b4407c496'
CHECKER_SHA = '74ce068cbfcad64b9776a5e4c3941cf1d4ce008ad5e0ebc8e6d6af6c4edfb67f'
COMMON_SHA = 'e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581'
GATE_SHA = '09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'
STRICT = ROOT / 'tools/final200-prerequisites-85f014e167c8/verify_actual200_prerequisites.py'
STRICT_SHA = '7dff811b01b302f45c729ee12ae9ee22bb561b359f24fcd0d37352192db2d075'
STRICT_MANIFEST_SHA = '85f014e167c8ca0fbb0ea4cf74bf2065e012d8efa1ec89c38b1f7240b09a62eb'
INVENTORY_SHA = 'fd8fa25d8321a4cd11b9faa5d8fdbd5f1692172a9e51ea5eba447c3565b8cc0d'
CLOSURE_SHA = '378277254a4b7d92236dfc4a8618a9de5fc49cf3ed16664ac41f21b918d1b20e'
BASE = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/assets/qwen2.5-3b-instruct-aa8e725')
HF200 = ROOT / 'runs/formal-3304/checkpoints/actor/global_step_200'
MODELS = {'official_base': BASE, 'actual_hf200': HF200}
DATA_ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/common-data')
ASSETS = Path('/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets')
GPU_LIB = Path('/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib')
FAISS_MODULE_BINDINGS = {'faiss': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/__init__.py', 'sha256': '2c5ea56bcb1c4da095b5f31467a4c05a2ecbcf5966ee9b39fdbf16da2f77006a'}, 'faiss._swigfaiss': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/_swigfaiss.cpython-310-x86_64-linux-gnu.so', 'sha256': '1992d0aa5d8c2ac443a03d1e8728a2c783ed00032aaf0778113001044dd7ae45'}, 'faiss.array_conversions': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/array_conversions.py', 'sha256': '0f89c2577f6dd37351e3acff0eeb43527ccedc9662effe30d03e45e872951610'}, 'faiss.class_wrappers': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/class_wrappers.py', 'sha256': 'fc795b327a9b406239cf6e1b46212009d1396806a14d77bdb8b4ccafe9e31ca2'}, 'faiss.extra_wrappers': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/extra_wrappers.py', 'sha256': '60f1074650da6273f3004613d59dca29482d2482d8ec081da056df297350fe4a'}, 'faiss.gpu_wrappers': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/gpu_wrappers.py', 'sha256': '3bfe002017723622d693c552241d4382fe830f0fb8a3b3eb72dafc90e5966c63'}, 'faiss.loader': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/loader.py', 'sha256': '225956ad1a2ee379e3fd0845f094c1424ecec59c26ea372f3b89857f3f2de921'}, 'faiss.shared_library_preload_trigger': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/shared_library_preload_trigger.py', 'sha256': 'f2ec69268e54ce0aa45a1763e8522a8934b0f8c124d6acad0d5772f96180c01c'}, 'faiss.shared_library_preloader': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/shared_library_preloader.py', 'sha256': '7811046f03e79f7efbd3f5e766492c5f962f32838fc8981497b01732450c6781'}, 'faiss.swigfaiss': {'path': '/mnt/data/shaoshitong/search-r1-rl/opsd/retriever-gpu-lib/faiss/swigfaiss.py', 'sha256': 'c12ab4624aadd9a9935110580efda9008541cb774e520be6c622233c37f3fabe'}}
TOOL_ASSET_SHA = '87b5e85df12cbf4635cb787b08161950ddc23e2e5b850e68fb7249668573cc87'
ROLES = ('official_base', 'actual_hf200')
FIXED_TOOLS = {'paired_eval_contract.py': CONTRACT_SHA, 'inspect_eval_native_cpu.py': CHECKER_SHA, 'common.py': COMMON_SHA}
TOOL_NAMES = set(FIXED_TOOLS) | {'paired_supervisor.py', 'paired_cpu_prepare.py', 'paired_service_entry.py'}
FIELDS = ('st_dev', 'st_ino', 'st_uid', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def unique(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key'); out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Exact fullSHA required')


def same(actual, expected, label):
    if isinstance(actual, Path) and isinstance(expected, Path):
        require(str(actual) == str(expected), label); return
    if type(actual) is set and type(expected) is set:
        require(actual == expected, label); return
    require(canonical(actual) == canonical(expected), label)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def physical(path, root=None):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.resolve(strict=True) == path, 'Physical absolute path required')
    if root is not None: require(path.is_relative_to(Path(root)), 'Path outside fixed scope')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_nlink == 1, 'Owned single-link regular file required')
    return path, s


def capture(path, expected=None, keep=False, root=None, limit=None):
    """Hash, parse/compile and identify the same opened bytes, never sha then read."""
    path, before = physical(path, root)
    if expected is not None: sha(expected)
    if limit is not None: require(before.st_size <= limit, 'Input too large')
    h = hashlib.sha256(); parts = []; count = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        while True:
            part = os.read(fd, 1024**2)
            if not part: break
            count += len(part); h.update(part)
            if keep: parts.append(part)
        ended = os.fstat(fd)
    finally: os.close(fd)
    after = path.lstat()
    require(all(getattr(before, k) == getattr(s, k) for s in (opened, ended, after) for k in FIELDS), 'Input changed while read')
    require(count == before.st_size and (expected is None or h.hexdigest() == expected), 'Full bytes/SHA differ')
    rec = dict(path=str(path), sha256=h.hexdigest(), bytes=count, uid=before.st_uid, device=before.st_dev,
               inode=before.st_ino, mode=stat.S_IMODE(before.st_mode), nlink=before.st_nlink,
               mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)
    return (b''.join(parts) if keep else None), rec


def reference(value, root=ROOT, expected=None):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact original reference required')
    raw, record = capture(value['path'], expected or value['sha256'], keep=True, root=root, limit=32*1024**2)
    if expected is not None: same(value['sha256'], expected, 'Reference pin differs')
    return unique(raw), record


def write_once(path, value, raw=False):
    path = Path(path); data = value if raw else canonical(value) + b'\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    finally: sync_dir(path.parent)
    return capture(path)[1]


def reserve(path):
    path = Path(path); path.mkdir(mode=0o700); sync_dir(path.parent); sync_dir(path)


def exact_module(path, expected, name):
    raw, record = capture(path, expected, True, limit=32*1024**2)
    module = types.ModuleType(name); module.__file__ = record['path']; sys.modules[name] = module
    exec(compile(raw, record['path'], 'exec', dont_inherit=True, optimize=0), module.__dict__)
    return module, record


def tools(request):
    require(type(request['control_tools']) is dict and set(request['control_tools']) == TOOL_NAMES, 'Control tool closure differs')
    folder = Path(__file__).resolve().parent; records = {}
    for name, pin in request['control_tools'].items():
        if name in FIXED_TOOLS: same(pin, FIXED_TOOLS[name], 'Neutral helper changed')
        records[name] = capture(folder / name, pin)[1]
    return records


def proc_identity(pid, job, step=None):
    require(type(pid) is int and pid > 0, 'PID must be typed integer')
    base = Path('/proc') / str(pid)
    raw = (base / 'stat').read_text(); tail = raw[raw.rfind(')')+2:].split()
    require(len(tail) >= 22, 'Malformed proc stat')
    status = (base / 'status').read_text(); match = re.search(r'^Uid:\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$', status, re.M)
    require(match is not None and [int(v) for v in match.groups()] == [os.getuid()]*4, 'Process four-UID ownership differs')
    cg = (base / 'cgroup').read_text()
    step_pattern = re.escape(step) if step is not None else '(?:[0-9]+|batch|extern)'
    pat = r'(?:^|/)job[_-]' + re.escape(job) + r'(?:\.scope)?/step[_-]' + step_pattern + r'(?:\.scope)?(?:/|$)'
    require(any(re.search(pat, line.split(':', 2)[-1]) for line in cg.splitlines()), 'Process is not in the same actual job/step path')
    # A second stat prevents combining a reused PID's UID/cgroup with its prior stat.
    end=(base/'stat').read_text();end_tail=end[end.rfind(')')+2:].split()
    require(len(end_tail)>=22 and (end_tail[19],end_tail[1])==(tail[19],tail[1]),'Process PID/start/parent changed during identity capture')
    return dict(pid=pid, start_ticks=int(tail[19]), parent_pid=int(tail[1]), uid=os.getuid(),
                four_uids=[os.getuid()]*4, cgroup=cg, state=tail[0], job_id=job, step_id=step)


def same_process(record):
    try:
        current = proc_identity(record['pid'], record['job_id'], record.get('step_id'))
    except FileNotFoundError: return False
    same(current['start_ticks'], record['start_ticks'], 'PID reused')
    return True


def allocation():
    job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
    require(sys.platform == 'linux' and os.getuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10', 'Actual UID0 h100-10 Linux only')
    require(type(job) is str and re.fullmatch('[1-9][0-9]*', job) and job != '3304', 'Actual separate evaluation allocation required')
    require(type(step) is str and re.fullmatch('(?:[0-9]+|batch|extern)', step), 'Actual scheduler step required')
    return proc_identity(os.getpid(), job, step)


def slurm_allocation(raw,job):
    """Installed21.08 literal format: preserves TRES and typed TresPerNode."""
    require(type(raw) is str and raw.count('JobId=')==1,'Exactly one literal job record required')
    fields={}
    for token in raw.split():
        if '=' not in token:continue
        k,v=token.split('=',1);require(k not in fields,'Duplicate Slurm field');fields[k]=v
    for k,v in dict(JobId=job,UserId='root(0)',JobState='RUNNING',Restarts='0',NodeList='h100-10',BatchHost='h100-10',
                    NumNodes='1',NumCPUs='192',NumTasks='1',OverSubscribe='NO',TresPerNode='gres:gpu:h100:8').items():
        same(fields.get(k),v,'Actual new Slurm allocation differs: '+k)
    require('cpu=192' in fields.get('TRES','').split(',') and 'mem=1T' in fields.get('TRES','').split(','),'Actual CPU/memory TRES differs')
    return fields


def wait_child(argv, folder, label, env=None, pass_fds=(), timeout=None, sample=None, metadata=proc_identity, job=None, step=None,on_popen=None):
    """Original true wait survives metadata/sampling failures. Unknown is not rc0."""
    write_once(folder/(label+'-command.json'), dict(argv=argv, environment=env, pass_fds=list(pass_fds), timeout=timeout))
    proc = None; result = dict(waited=False, returncode=None, pid=None, errors=[]); raised = None
    out = open(folder/(label+'.stdout'), 'xb'); err = open(folder/(label+'.stderr'), 'xb')
    try:
        proc = subprocess.Popen(argv, stdout=out, stderr=err, env=env, pass_fds=pass_fds)
        if on_popen is not None:on_popen(proc)
        result['pid'] = proc.pid
        write_once(folder/(label+'-popen.json'), dict(pid=proc.pid, actual_popen=True))
        start = metadata(proc.pid, job, step) if job is not None else dict(pid=proc.pid)
        write_once(folder/(label+'-start.json'), start)
        deadline = None if timeout is None else time.monotonic()+timeout
        while True:
            try:
                proc.wait(timeout=2); break
            except subprocess.TimeoutExpired:
                if sample is not None: sample()
                if deadline is not None and time.monotonic() >= deadline: raise TimeoutError('Child wait deadline exceeded')
    except BaseException as exc:
        raised = exc; result['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
    finally:
        if proc is not None:
            try:
                # Even after controller failure, retain the actually waited result.
                proc.wait(timeout=10 if raised is not None else timeout)
                result.update(waited=True, returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                result['errors'].append(dict(type='TimeoutExpired', message='Child still running; no automatic retry'))
        for f in (out, err): f.flush(); os.fsync(f.fileno()); f.close()
        sync_dir(folder); write_once(folder/(label+'-exit.json'), result)
    if raised is not None: raise raised
    require(result['waited'] is True and type(result['returncode']) is int and result['returncode'] == 0, 'Child did not truly finish0: '+label)
    return result


def command(argv, folder, label, timeout=60):
    return wait_child(argv, folder, label, timeout=timeout)


def git_clean(folder, repo, expected, label):
    command(['git', '-C', str(repo), 'rev-parse', 'HEAD'], folder, label+'-head')
    command(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], folder, label+'-status')
    same((folder/(label+'-head.stdout')).read_text().strip(), expected, 'Frozen Git HEAD differs')
    require(not (folder/(label+'-status.stdout')).read_bytes().strip(), 'Frozen Git is dirty')


def load_request(path, pin):
    envelope, record = reference({'path':str(path), 'sha256':pin})
    require(type(envelope) is dict and set(envelope) == {'payload', 'signature_hex'}, 'Root signed request required')
    public, _ = reference({'path':str(PUBLIC), 'sha256':PUBLIC_SHA})
    gate, _ = exact_module(REPO/'search_r1/baselines/checkpoint_gate.py', GATE_SHA, 'paired_exact_public_gate')
    gate.verify_rsa_signature(envelope['payload'], envelope['signature_hex'], public)
    q = envelope['payload']
    require(type(q) is dict and set(q) == {'version','kind','training_commit','eval_commit','stored_training_job_id',
        'layered_receipt','root_eligibility_review','peer_eligibility_review','source_inventory','native_import_closure',
        'common_manifest','tool_asset_manifest','checkpoint31','control_tools','approval_wait_seconds','role_timeout_seconds'}, 'Execution request schema differs')
    same({k:q[k] for k in ('version','kind','training_commit','eval_commit','stored_training_job_id')},
         dict(version=1,kind='ROOT_AUTHORIZED_PAIRED_FINAL200_EXECUTION_REQUEST',training_commit=TRAIN,eval_commit=EVAL,stored_training_job_id='3304'), 'Request identity differs')
    receipt, _ = reference(q['layered_receipt'])
    require(type(receipt) is dict and set(receipt) == {'payload','signature_hex'}, 'Actual layered receipt required')
    gate.verify_rsa_signature(receipt['payload'], receipt['signature_hex'], public)
    require(receipt['payload'].get('kind') == 'LAYERED_TRAINING3304_ACTUAL200_HF200_EVIDENCE_RECEIPT_NOT_COMPLETION', 'First5/legacy receipt cannot qualify200')
    for role, key in (('ROOT_REVIEWER','root_eligibility_review'),('INDEPENDENT_REVIEWER','peer_eligibility_review')):
        review, _ = reference(q[key])
        same({k:review.get(k) for k in ('version','kind','role','status','receipt_sha256','training_commit','eval_commit')},
             dict(version=1,kind='PAIRED_FINAL200_SUBMISSION_ELIGIBILITY_REVIEW',role=role,
                  status='PASS_ACTUAL200_FOR_SEPARATE_PAIRED_EVAL_ONLY',receipt_sha256=q['layered_receipt']['sha256'],
                  training_commit=TRAIN,eval_commit=EVAL), 'Actual root/peer qualification required')
    reference(q['source_inventory'], expected=INVENTORY_SHA); reference(q['native_import_closure'], expected=CLOSURE_SHA)
    contract, _ = exact_module(Path(__file__).parent/'paired_eval_contract.py', CONTRACT_SHA, 'paired_exact_contract')
    reference(q['common_manifest'], root=Path('/mnt/data/shaoshitong/search-r1-rl'), expected=contract.COMMON)
    reference(q['tool_asset_manifest'], expected=TOOL_ASSET_SHA)
    require(type(q['checkpoint31']) is dict and len(q['checkpoint31']) == 31, 'Original31 checkpoint map required')
    neutral, _ = exact_module(Path(__file__).parent/'common.py', COMMON_SHA, 'paired_exact_neutral_common')
    expected = set(str(p) for p in neutral.checkpoint_paths().values())
    same(set(q['checkpoint31']), expected, 'Original31 member paths differ')
    for p, h in q['checkpoint31'].items(): sha(h)
    for key in ('approval_wait_seconds','role_timeout_seconds'):
        require(type(q[key]) is int and 1 <= q[key] <= 7*24*3600, 'Bounded actual control timeout required')
    tools(q)
    return q, record


def checkpoint_guard(q):
    return {p:capture(p,h,root=ROOT/'runs/formal-3304')[1] for p,h in q['checkpoint31'].items()}


def clean_env():
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(('BASELINE_', 'SEARCH_R1_', 'OPSD_', 'E1_', 'RAY_', 'SBATCH_', 'SRUN_')) or name.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'):
            env.pop(name)
    env.update(PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='offline', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
    return env


def submission_preview(request_reference, control_reference):
    """Pure fixed argv/script preview. Root owns future one-shot submission."""
    for ref in (request_reference, control_reference):
        require(type(ref) is dict and set(ref) == {'path','sha256'}, 'Preview exact references required'); sha(ref['sha256'])
        require(type(ref['path']) is str and Path(ref['path']).is_absolute() and Path(ref['path']).is_relative_to(ROOT) and '..' not in Path(ref['path']).parts, 'Preview path scope differs')
    # Fixed JSON rendering avoids shell/user-code injection; the shell has no free input.
    boot = bootstrap(control_reference['path'],control_reference['sha256'],['--request',request_reference['path'],'--request-sha256',request_reference['sha256']])
    import shlex
    # The supervisor must observe the real step created by Slurm, rather than
    # invent SLURM_STEP_ID for a batch-shell process.
    script = '#!/bin/bash\nset -euo pipefail\nexec '+shlex.join(['srun','--nodes=1','--ntasks=1','--cpus-per-task=192','--gres=gpu:h100:8',PYTHON,'-B','-c',boot])+'\n'
    return dict(scope='SUBMISSION_PREVIEW_ONLY_NOT_SUBMITTED_OR_TEST_ONLY',actual_job_id=None,actual_wait=None,
        argv=['env'] + sum((['-u',n] for n in sorted(os.environ) if n.startswith(('SBATCH_','SRUN_','SLURM_'))),[]) +
        ['sbatch','--parsable','--no-requeue','--nodelist=h100-10','--nodes=1','--ntasks=1','--cpus-per-task=192','--mem=1T','--gres=gpu:h100:8','--exclusive','--time=7-00:00:00'],
        batch_script=script,request=request_reference,control=control_reference,unknown_submission_policy='KEEP_ORIGINALS_DO_NOT_RESUBMIT')


def bootstrap(path, pin, args):
    """Native -B stdin-free same-FD compile bootstrap, no project pyc dispatch."""
    sha(pin)
    return ('import os,stat,hashlib,sys\np='+repr(str(path))+'\nh='+repr(pin)+'\n'
        'from pathlib import Path\nassert Path(p).resolve(strict=True)==Path(p)\n'
        'a=os.lstat(p);assert stat.S_ISREG(a.st_mode) and a.st_uid==os.getuid() and a.st_nlink==1\n'
        'f=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);b=os.fstat(f);r=b""\n'
        'while True:\n x=os.read(f,1048576)\n if not x:break\n r+=x\n'
        'c=os.fstat(f);os.close(f);d=os.lstat(p)\n'
        'ks='+repr(FIELDS)+'\nassert all(getattr(a,k)==getattr(z,k) for z in (b,c,d) for k in ks)\n'
        'assert hashlib.sha256(r).hexdigest()==h\nsys.argv='+repr([str(path)]+list(args))+'\n'
        'exec(compile(r,p,"exec",dont_inherit=True,optimize=0),{"__name__":"__main__","__file__":p})\n')


class LeaseSet:
    def __init__(self, uuids): self.uuids=list(uuids); self.fds=[]; self.records=[]
    def acquire(self):
        folder = Path('/tmp')/('search-r1-gpu-leases-'+str(os.getuid()))
        if not folder.exists(): folder.mkdir(mode=0o700); sync_dir(folder.parent)
        require(folder.resolve(strict=True)==folder and folder.stat().st_uid==os.getuid(), 'Lease root differs')
        paths=[Path('/mnt/data/shaoshitong/search-r1-rl/opsd/gpu-job.lock')]+[folder/(u+'.lock') for u in self.uuids]
        try:
            for p in paths:
                require(p.parent.resolve(strict=True)==p.parent, 'Lease parent symlink')
                fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600); s=os.fstat(fd)
                require(stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1 and (s.st_dev,s.st_ino)==(p.lstat().st_dev,p.lstat().st_ino), 'Lease physical identity differs')
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); os.set_inheritable(fd,True)
                self.fds.append(fd); self.records.append(dict(path=str(p),fd=fd,device=s.st_dev,inode=s.st_ino,uid=s.st_uid))
                os.fsync(fd); sync_dir(p.parent)
        except BaseException:
            self.close(); raise
        return self.records
    def check(self):
        require(len(self.fds)==9, 'All leases must remain held')
        for fd,rec in zip(self.fds,self.records):
            s=os.fstat(fd); same((s.st_dev,s.st_ino,s.st_uid),(rec['device'],rec['inode'],rec['uid']),'Lease changed')
            same((Path(rec['path']).stat().st_dev,Path(rec['path']).stat().st_ino),(s.st_dev,s.st_ino),'Lease name replaced')
            probe=os.open(rec['path'],os.O_RDWR|os.O_NOFOLLOW)
            try:
                try: fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: pass
                else: raise ValueError('Lease not exclusively held by parent')
            finally: os.close(probe)
    def environment(self):
        self.check(); return {'SEARCH_R1_GPU_PREFLIGHT':'1','SEARCH_R1_GPU_LEASES':canonical([
            dict(uuid=u,fd=f) for u,f in zip(self.uuids,self.fds[1:])]).decode()}
    def close(self):
        for fd in reversed(self.fds): os.close(fd)
        self.fds=[]


def parse_gpu_inventory(raw):
    out=[]
    for row in raw.decode().splitlines():
        parts=[v.strip() for v in row.split(',')]; require(len(parts)==6,'Malformed GPU inventory')
        index,u,name,total,used,util=parts
        require(re.fullmatch('[0-9]+',index) and re.fullmatch('GPU-[0-9a-fA-F-]+',u) and 'H100' in name,'Unexpected GPU inventory identity')
        nums=[float(v) for v in (total,used,util)]; require(all(math.isfinite(v) and v>=0 for v in nums) and nums[0]>0,'Malformed GPU quantities')
        out.append(dict(index=int(index),uuid=u,name=name,total_mib=nums[0],used_mib=nums[1],utilization=nums[2]))
    require(len(out)==8 and len({v['uuid'] for v in out})==8 and {v['index'] for v in out}==set(range(8)),'Exactly eight allocated H100s required')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    require(set(visible)=={str(v['index']) for v in out} or set(visible)=={v['uuid'] for v in out},'Actual CUDA allocation inventory differs')
    return out


def parse_gpu_apps(raw):
    out=[]
    for row in raw.decode().splitlines():
        if not row.strip():continue
        p,u,m=[v.strip() for v in row.split(',')]
        require(re.fullmatch('[1-9][0-9]*',p) and re.fullmatch('GPU-[0-9a-fA-F-]+',u),'Malformed NVML application')
        require(m=='[N/A]' or re.fullmatch('[0-9]+',m),'Malformed NVML memory')
        out.append(dict(pid=int(p),uuid=u,used_mib=None if m=='[N/A]' else int(m)))
    return out


def classify_app(app, known, recheck_apps, stat_gone, identity_reader, job, step):
    """Only known own + NVML gone + explicit /proc/stat ENOENT is an exit race."""
    try:
        current=identity_reader(app['pid'],job,step)
        if app['pid'] in known: same(current['start_ticks'],known[app['pid']]['start_ticks'],'NVML PID reused')
        return dict(classification='OWNED_LIVE',process=current)
    except OSError as exc:
        require(exc.errno==errno.ENOENT and app['pid'] in known,'Unreadable/unknown NVML PID cannot be classified exited')
        prior=known[app['pid']]
        same(prior['job_id'],job,'Prior owner belongs to another job')
        require(not any(v['pid']==app['pid'] for v in recheck_apps()),'NVML PID remains live')
        require(stat_gone(app['pid']) is True,'Explicit proc stat ENOENT required')
        return dict(classification='EXITED_BETWEEN_OBSERVATIONS',prior_process=prior,proc_stat_errno=errno.ENOENT)


def explicit_gone(pid):
    try: (Path('/proc')/str(pid)/'stat').read_bytes()
    except OSError as exc:
        if exc.errno==errno.ENOENT:return True
        raise
    return False


class Sampler:
    def __init__(self, run, owner, uuids): self.run=run; self.owner=owner; self.uuids=set(uuids); self.known={}; self.count=0
    def query(self,kind):
        self.count+=1; label='nvml-%06d-%s'%(self.count,kind)
        fields='index,uuid,name,memory.total,memory.used,utilization.gpu' if kind=='inventory' else 'pid,gpu_uuid,used_gpu_memory'
        flag='--query-gpu=' if kind=='inventory' else '--query-compute-apps='
        command(['nvidia-smi',flag+fields,'--format=csv,noheader,nounits'],self.run,label)
        raw=(self.run/(label+'.stdout')).read_bytes()
        return parse_gpu_inventory(raw) if kind=='inventory' else parse_gpu_apps(raw)
    def sample(self):
        inventory=self.query('inventory'); apps=self.query('apps')
        require({v['uuid'] for v in inventory}==self.uuids,'GPU allocation changed')
        row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=self.owner['job_id'],gpu_inventory=inventory,gpu_processes=apps,ownership=[])
        try:
            for app in apps:
                require(app['uuid'] in self.uuids,'Foreign GPU outside allocation')
                item=classify_app(app,self.known,lambda:self.query('apps'),explicit_gone,proc_identity,self.owner['job_id'],self.owner['step_id'])
                row['ownership'].append(dict(application=app,**item))
                if item['classification']=='OWNED_LIVE':self.known[app['pid']]=item['process']
        except BaseException as exc:
            row['error']=dict(type=type(exc).__name__,message=str(exc)); raise
        finally:
            with open(self.run/'resources.ndjson','ab') as f:f.write(canonical(row)+b'\n');f.flush();os.fsync(f.fileno())
        return row


def idle(row):
    require(not row['gpu_processes'],'Pre-GPU allocation is not empty')
    require(all(v['utilization']<=15 and v['used_mib']<=.15*v['total_mib'] for v in row['gpu_inventory']),'Pre-GPU allocation not idle')


def bundle_check(path, pin):
    data,_=reference({'path':str(Path(path)/'manifest.json'),'sha256':pin})
    require(type(data.get('tools')) is dict,'Pinned bundle tools missing')
    same(set(p.name for p in Path(path).iterdir()),set(data['tools'])|{'manifest.json'},'Pinned bundle membership differs')
    records={n:capture(Path(path)/n,h)[1] for n,h in data['tools'].items()}
    require(Path(path).stat().st_uid==os.getuid() and stat.S_IMODE(Path(path).stat().st_mode)==0o555,'Pinned bundle directory ownership/mode differs')
    require(all(v['mode']==0o444 for v in records.values()),'Pinned bundle member modes differ')
    return records


def proposed_stage(q, contract, owner, role, model_files, input_record, source_files):
    """New builder: no call to a12.proposed_role_stage (its review ref is circular).

    Root/peer later approve these unchanged bytes in a signed sidecar. The null
    optional review field is not passed off as an already existing review.
    """
    budget=contract.BUDGET
    return dict(kind='REVIEWED_FINAL200_PAIRED_EVAL_STAGE',model_role=role,training_commit=TRAIN,
        training_job_id='3304',actual_outer_updates=200,eval_commit=EVAL,eval_job_id=owner['job_id'],
        common_manifest_sha256=contract.COMMON,official_base_revision=contract.REVISION,
        official_base_manifest_sha256=contract.MODEL_MANIFEST,model_directory=str(MODELS[role]),model_files=model_files,
        source_files=source_files,input_manifest_sha256=input_record['sha256'],generation_budget=budget,
        generation_budget_sha256=hashlib.sha256(canonical(budget)).hexdigest(),training_hf_receipt=q['layered_receipt'],
        strict_training_hf_verifier=dict(path=str(STRICT),sha256=STRICT_SHA),root_pre_gpu_review=None,
        stage_construction_scope='PROPOSED_CONTENT_REQUIRES_EXTERNAL_SIGNED_READY_APPROVAL',actual_allocation=owner,
        actual_c5bf_source_inventory=q['source_inventory'],actual_c5bf_native_import_closure=q['native_import_closure'])


def await_approval(run,ready,ready_record,gate,timeout,lease,sampler):
    path=run/'pre-gpu-approval.json';deadline=time.monotonic()+timeout
    while not path.exists():
        require(time.monotonic()<deadline,'Pre-GPU approval absent; unknown wait is not permission')
        lease.check();sampler.sample();time.sleep(10)
    raw,record=capture(path,keep=True,root=run,limit=32*1024**2);envelope=unique(raw)
    require(type(envelope) is dict and set(envelope)=={'payload','signature_hex'},'Signed actual pre-GPU approval required')
    public,_=reference({'path':str(PUBLIC),'sha256':PUBLIC_SHA});gate.verify_rsa_signature(envelope['payload'],envelope['signature_hex'],public)
    payload=envelope['payload']
    expected=dict(version=1,kind='ROOT_SIGNED_PAIRED_FINAL200_PRE_GPU_APPROVAL_INITIALIZATION_ONLY',
        ready_sha256=ready_record['sha256'],request_sha256=ready['request']['sha256'],nonce=ready['nonce'],
        parent_identity=ready['parent_identity'],gpu_uuids=ready['gpu_uuids'],leases=ready['leases'],
        stage_sha256={r:ready['stages'][r]['sha256'] for r in ROLES},
        configuration_sha256={r:ready['configurations'][r]['sha256'] for r in ROLES})
    same({k:payload.get(k) for k in expected},expected,'Actual approval does not bind exact ready/holder/stages/config')
    require(set(payload)==set(expected)|{'root_review','peer_review'},'Approval schema differs')
    for role,key in (('ROOT_REVIEWER','root_review'),('INDEPENDENT_REVIEWER','peer_review')):
        review,_=reference(payload[key])
        same({k:review.get(k) for k in ('version','role','status','ready_sha256','request_sha256')},
             dict(version=1,role=role,status='PASS_ACTUAL_PAIRED_FINAL200_PRE_GPU_INITIALIZATION_ONLY',
                  ready_sha256=ready_record['sha256'],request_sha256=ready['request']['sha256']),'Actual root/peer pre-GPU review differs')
    lease.check();idle(sampler.sample());require(same_process(ready['parent_identity']),'Parent no longer owns approval identity')
    return record


def role_env(role,run,prepared,stages,leases,token):
    env=clean_env();env.update(leases.environment())
    env.update(SEARCH_R1_OPSD_LOCAL_HOST='h100-10',SEARCH_R1_REPO=str(REPO),PAIRED_EVAL_ROLE_TOKEN=token,
        OPSD_E1_OFFICIAL_MODEL=str(BASE),OPSD_E1_COMMON_DATA_ROOT=str(DATA_ROOT),SEARCH_R1_BASELINE_RUN_ROOT=str(run/role),
        OPSD_E1_COMMON_MANIFEST=prepared['common_manifest']['path'],OPSD_E1_RETRIEVER_URL='http://127.0.0.1:18084/retrieve',
        OPSD_FINAL200_ACTUAL_MODEL_DIRECTORY=str(MODELS[role]),OPSD_FINAL200_ACTUAL_REVIEWED_STAGE=stages[role]['path'],
        OPSD_FINAL200_INPUT_MANIFEST=prepared['input_manifest']['path'],OPSD_FINAL200_INPUT_MANIFEST_SHA256=prepared['input_manifest']['sha256'],
        OPSD_FINAL200_ACTUAL_FRESH_OUTPUT_DIRECTORY=str(run/role/'raw'),OPSD_FINAL200_EVAL_STAGE_SHA256=stages[role]['sha256'],
        OPSD_CORRECTED_EVAL_STAGE_SHA256=stages[role]['sha256'],RAY_ADDRESS='127.0.0.1:16384',
        RAY_NAMESPACE='paired-'+run.name+'-'+role,PYTHONPATH=str(REPO))
    return env


class RoleProcesses:
    """Track real descendants while alive; never broad-kill a job or a cluster."""
    def __init__(self,owner,token,temp,folder):self.owner=owner;self.token=token;self.temp=Path(temp);self.folder=folder;self.known={};self.logs={};self.cleanup_invoked=False
    def scan(self,root_pids):
        infos={}
        for p in Path('/proc').iterdir():
            if not p.name.isdecimal():continue
            pid=int(p.name)
            try:
                text=(p/'stat').read_text();tail=text[text.rfind(')')+2:].split();parent=int(tail[1])
                env=(p/'environ').read_bytes().split(b'\0')
            except FileNotFoundError:continue
            except PermissionError:continue  # Cannot confer ownership; GPU resolution independently fails closed.
            infos[pid]=(parent,b'PAIRED_EVAL_ROLE_TOKEN='+self.token.encode() in env)
        selected=set(root_pids)|set(self.known)
        changed=True
        while changed:
            old=len(selected);selected|={pid for pid,(ppid,tag) in infos.items() if ppid in selected or tag};changed=len(selected)!=old
        for pid in selected:
            try:rec=proc_identity(pid,self.owner['job_id'],self.owner['step_id'])
            except FileNotFoundError:continue
            if pid in self.known:same(rec['start_ticks'],self.known[pid]['start_ticks'],'Role PID reused')
            self.known[pid]=rec
            for stream in (1,2):
                try:target=Path(os.readlink('/proc/%d/fd/%d'%(pid,stream)))
                except FileNotFoundError:continue
                if not target.is_absolute():continue
                # Own direct redirected streams plus physical Ray files, no arbitrary fd paths.
                if not (target.is_relative_to(self.temp) or target.is_relative_to(self.folder)):continue
                try:p,s=physical(target)
                except FileNotFoundError:continue
                key=str(p);item=dict(path=key,device=s.st_dev,inode=s.st_ino,uid=s.st_uid,process=rec,stream=stream)
                if key in self.logs:same({k:item[k] for k in ('device','inode','uid')},{k:self.logs[key][k] for k in ('device','inode','uid')},'Role log path replaced')
                self.logs[key]=item
        write_once(self.folder/('process-observation-%06d.json'%len(list(self.folder.glob('process-observation-*.json')))),dict(known=self.known,logs=self.logs))
    def cleanup(self,direct,sampler,service):
        require(self.cleanup_invoked is False,'Cleanup result unknown; do not automatically repeat');self.cleanup_invoked=True
        actions=[];exits=[];error=None
        try:
            self.scan([p.pid for p in direct if p.poll() is None])
            # Signal only captured, still same-start own processes; service is excluded.
            for sig,delay in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
                for pid,rec in sorted(self.known.items(),reverse=True):
                    require(pid!=service['pid'] and pid!=os.getpid(),'Cleanup target escapes role scope')
                    if same_process(rec):
                        os.kill(pid,sig);actions.append(dict(process=rec,signal=int(sig)))
                deadline=time.monotonic()+delay
                while time.monotonic()<deadline:
                    if all(explicit_gone(pid) for pid in self.known):break
                    time.sleep(.5)
                if all(explicit_gone(pid) for pid in self.known):break
            for p in direct:
                try:p.wait(timeout=10);exits.append(dict(pid=p.pid,waited=True,returncode=p.returncode))
                except subprocess.TimeoutExpired:exits.append(dict(pid=p.pid,waited=False,returncode=None))
            require(all(explicit_gone(pid) for pid in self.known),'Role process remains alive; block role2')
            first=sampler.sample();time.sleep(.2);second=sampler.sample()
            require(all(app['pid']==service['pid'] for row in (first,second) for app in row['gpu_processes']),'Non-service GPU application remains; block role2')
            require(all(v['waited'] is True and type(v['returncode']) is int for v in exits),'Cleanup direct child wait unknown')
            record=dict(actions=actions,direct_waits=exits,known_processes=self.known,first_nvml=first,second_nvml=second,
                status='OWN_ROLE_PROCESSES_AND_GPU_APPS_GONE_IN_TWO_OBSERVATIONS_NOT_SLURM_RELEASE')
            write_once(self.folder/'cleanup.json',record)
            return record
        except BaseException as exc:
            error=dict(type=type(exc).__name__,message=str(exc));raise
        finally:
            if not exits:
                for p in direct:
                    try:p.wait(timeout=10);exits.append(dict(pid=p.pid,waited=True,returncode=p.returncode))
                    except subprocess.TimeoutExpired:exits.append(dict(pid=p.pid,waited=False,returncode=None))
            write_once(self.folder/'cleanup-waits.original.json',dict(direct_waits=exits,error=error))
            write_once(self.folder/'cleanup-actions.original.json',actions)


def full_logs(group):
    """Copy known physical streams only after two global no-writer scans."""
    keys={(v['device'],v['inode']) for v in group.logs.values()}
    for attempt in range(2):
        writers=[]
        for p in Path('/proc').iterdir():
            if not p.name.isdecimal():continue
            try:fds=list((p/'fd').iterdir())
            except FileNotFoundError:continue
            for fd in fds:
                try:s=fd.stat();flags=(p/'fdinfo'/fd.name).read_text()
                except FileNotFoundError:continue
                if (s.st_dev,s.st_ino) in keys:
                    match=re.search(r'^flags:\s+([0-7]+)$',flags,re.M);require(match is not None,'Unknown file writer flags')
                    if int(match.group(1),8)&os.O_ACCMODE in (os.O_WRONLY,os.O_RDWR):writers.append(dict(pid=int(p.name),fd=int(fd.name)))
        require(not writers,'Role logs still have actual write FDs')
        if attempt==0:time.sleep(.1)
    target=group.folder/'closed-original-streams';reserve(target);out={}
    for n,(path,origin) in enumerate(sorted(group.logs.items())):
        _,before=capture(path);same((before['device'],before['inode'],before['uid']),(origin['device'],origin['inode'],origin['uid']),'Bound log origin changed')
        output=target/('%03d-%s'%(n,Path(path).name));src=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);dst=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            while True:
                data=os.read(src,1024**2)
                if not data:break
                with_data=memoryview(data)
                while with_data:with_data=with_data[os.write(dst,with_data):]
            os.fsync(dst)
        finally:os.close(src);os.close(dst)
        _,after=capture(path);same(before,after,'Closed log changed during copy');record=capture(output,before['sha256'])[1];out[path]=dict(origin=origin,source=after,copy=record)
    sync_dir(target);return write_once(group.folder/'closed-streams-manifest.json',out)


def role_output_check(folder,prepared,role):
    value,rec=reference(dict(path=str(folder/'raw/manifest.json'),sha256=capture(folder/'raw/manifest.json')[1]['sha256']),root=folder)
    same(value.get('scope'),'FULL_FINAL200_TOOL_TURNS_AND_NATIVE_SCORING_VIEW','Full raw output incomplete')
    same(value.get('count'),51713,'Final raw cardinality differs');same(value.get('input_manifest_sha256'),prepared['input_manifest']['sha256'],'Final input identity differs')
    # This validates the source's zero-update completion report, not raw replay/scientific acceptance.
    audit=value.get('actual_eval_worker_zero_optimizer');require(type(audit) is dict and set(audit)=={'start','finish'},'Zero worker audit absent')
    require(type(value.get('zero_update_guard')) is dict and type(value['zero_update_guard'].get('attempted_update_dispatches')) is int
            and value['zero_update_guard']['attempted_update_dispatches']==0,'Driver zero-step guard absent/nonzero')
    for key in ('files','full_turn_files'):
        require(type(value.get(key)) is list and len(value[key])==809,'Full raw/scoring records absent')
    worker_ids={};source_pin=prepared['source_files']['search_r1/baselines/final200_evaluation.py']
    for phase in ('start','finish'):
        rows=audit[phase];require(type(rows) is list and len(rows)==8,'Eight actual worker zero-step records required')
        ranks=[]
        for row in rows:
            require(type(row['rank']) is int and type(row['world_size']) is int and row['world_size']==8,'Typed actual zero-step rank/world required');ranks.append(row['rank'])
            for k in ('attempted_optimizer_steps','attempted_scheduler_steps'):require(type(row[k]) is int and row[k]==0,'Actual worker update attempted')
            same(row['source_sha256'],source_pin,'Zero guard source differs');same(row['before'],row['after'],'Worker Adam/scheduler clocks changed')
            expected='install' if phase=='start' else 'finish';same(row['phase'],expected,'Worker zero-step phase differs')
            p=row['process'];require(type(p['pid']) is int and type(p['process_start_ticks']) is str and re.fullmatch('[1-9][0-9]*',p['process_start_ticks']),'Actual frozen writer PID/start types differ')
            if phase=='start':worker_ids[row['rank']]=p
            else:
                prior=worker_ids[row['rank']]
                for k in ('pid','process_start_ticks','uid','uids','cgroup'):same(p[k],prior[k],'Eval worker identity changed')
        same(set(ranks),set(range(8)),'Worker zero-step ranks differ')
    members={};cursor=0
    for scoring,full in zip(value['files'],value['full_turn_files']):
        expected=min(64,51713-cursor)
        for entry in (scoring,full):
            require(type(entry['start']) is int and type(entry['count']) is int and entry['start']==cursor and entry['count']==expected,'Raw batch order/count differs')
        for name,pin in ((scoring['tensors'],scoring['tensors_sha256']),(scoring['rows'],scoring['rows_sha256']),
                         (full['path'],full['sha256']),(full['tool_batches'],full['tool_batches_sha256'])):
            require(type(name) is str and Path(name).name==name and name not in members,'Unsafe/duplicate raw member')
            members[name]=capture(folder/'raw'/name,pin,root=folder/'raw')[1]
        cursor+=expected
    same(set(p.name for p in (folder/'raw').iterdir()),set(members)|{'manifest.json'},'Full raw directory membership differs')
    same(value['provenance']['model_role'],role,'Output role differs')
    return dict(manifest=rec,members=members,workers=worker_ids,scope='OUTPUT_COMPLETENESS_REQUIRES_INDEPENDENT_WHOLE_RAW_REPLAY',model_role=role)


def run_roles(run,q,prepared,stages,owner,lease,sampler,service):
    outcomes={}
    for role in ROLES:
        if outcomes:
            require(outcomes['official_base']['cleanup']['status'].startswith('OWN_ROLE_PROCESSES'),'Role1 cleanup not proven')
        checkpoint_before=checkpoint_guard(q);write_once(run/(role+'-checkpoint31-before.json'),checkpoint_before)
        folder=run/role;reserve(folder);token=secrets.token_hex(32);tmp=Path('/tmp')/('pfe-'+owner['job_id']+'-'+role)
        reserve(tmp);env=role_env(role,run,prepared,stages,lease,token)
        require(not any(v.startswith('http') for k,v in env.items() if k.lower().endswith('proxy')),'Proxy environment must be empty')
        command(['ss','-H','-lntp'],folder,'listeners-before');require(':16384' not in (folder/'listeners-before.stdout').read_text(),'Ray port not free')
        head_argv=[RAY,'start','--head','--block','--node-ip-address=127.0.0.1','--port=16384','--num-gpus=8','--num-cpus=80',
            '--object-store-memory=8589934592','--include-dashboard=false','--min-worker-port=30000','--max-worker-port=30999','--temp-dir='+str(tmp)]
        write_once(folder/'head-command.json',dict(argv=head_argv,env=env,pass_fds=lease.fds))
        head_out=open(folder/'head.stdout','xb');head_err=open(folder/'head.stderr','xb');head=None;direct=[];tracker=RoleProcesses(owner,token,tmp,folder)
        try:
            head=subprocess.Popen(head_argv,stdout=head_out,stderr=head_err,env=env,pass_fds=lease.fds);direct.append(head);write_once(folder/'head-start.json',proc_identity(head.pid,owner['job_id'],owner['step_id']))
            deadline=time.monotonic()+180
            while True:
                require(head.poll() is None,'Own Ray head exited before role launch');tracker.scan([head.pid]);sampler.sample()
                try:
                    sock=socket.create_connection(('127.0.0.1',16384),timeout=1);sock.close();break
                except (ConnectionRefusedError,socket.timeout):require(time.monotonic()<deadline,'Own Ray head not ready');time.sleep(2)
            # Explicit c5bf bootstrap verifies actual94 imports in the driver, then the frozen Hydra main.
            args=['--mode','eval','--request',prepared['request']['path'],'--request-sha256',prepared['request']['sha256'],
                '--run',str(run),'--supervisor-sha256',q['control_tools']['paired_supervisor.py'],'--role',role]
            cpu=Path(__file__).parent/'paired_cpu_prepare.py'
            cpu_env=dict(env,CUDA_VISIBLE_DEVICES='',PAIRED_EVAL_ASSIGNED_CUDA_DEVICES=env['CUDA_VISIBLE_DEVICES'])
            wait_child([PYTHON,'-B','-c',bootstrap(cpu,q['control_tools']['paired_cpu_prepare.py'],args)],folder,'eval',cpu_env,
                lease.fds,q['role_timeout_seconds'],lambda:(lease.check(),tracker.scan([p.pid for p in direct]),sampler.sample()),job=owner['job_id'],step=owner['step_id'],on_popen=direct.append)
            # Capture descendants and both stdout/stderr paths before stopping their head.
            tracker.scan([head.pid]);output=role_output_check(folder,prepared,role)
            for worker in output['workers'].values():
                for stream in (1,2):
                    require(any(v['process']['pid']==worker['pid'] and str(v['process']['start_ticks'])==worker['process_start_ticks'] and v['stream']==stream for v in tracker.logs.values()),'Actual worker stdout/stderr path was not captured alive')
            after=checkpoint_guard(q);same(checkpoint_before,after,'Original31 training checkpoint changed during eval')
            write_once(run/(role+'-checkpoint31-after.json'),after)
            cleanup=tracker.cleanup(direct,sampler,service)
            for f in (head_out,head_err):f.flush();os.fsync(f.fileno());f.close()
            closed=full_logs(tracker)
            outcomes[role]=dict(output=output,cleanup=cleanup,closed_streams=closed)
            write_once(folder/'role-output-review-required.json',outcomes[role])
        finally:
            if direct and not tracker.cleanup_invoked:
                # Failure cleanup is scoped and retained. It never authorizes role2.
                tracker.cleanup(direct,sampler,service)
            for f in (head_out,head_err):
                if not f.closed:f.flush();os.fsync(f.fileno());f.close()
    return outcomes


def service_start(run,q,prepared,owner,lease,sampler,on_popen):
    asset,rec=reference(q['tool_asset_manifest'],expected=TOOL_ASSET_SHA)
    same(asset.get('status'),'FULL_TOOL_ASSET_CONTENT_VERIFIED','Tool assets not fully pinned')
    files=asset['results']['e5']+[asset['results'][k] for k in ('wiki_gzip','index','corpus')]
    verified=[]
    for item in files:verified.append(capture(item['path'],item['sha256'],root=ASSETS)[1])
    entry=Path(__file__).parent/'paired_service_entry.py';pin=q['control_tools']['paired_service_entry.py']
    capture(entry,pin)
    env=clean_env();env.update(lease.environment());env.update(SEARCH_R1_REPO=str(REPO),E1_FORMAL_RUN=str(run),
        PYTHONPATH=str(REPO)+':'+str(GPU_LIB),SEARCH_R1_OPSD_LOCAL_HOST='h100-10')
    argv=[PYTHON,'-u','-B','-c',bootstrap(entry,pin,['--supervisor-sha256',q['control_tools']['paired_supervisor.py'],
        '--adapter-sha256',q['control_tools']['paired_cpu_prepare.py'],'--run',str(run),
        '--','--index_path',asset['results']['index']['path'],
        '--corpus_path',asset['results']['corpus']['path'],'--retriever_model',str(Path(asset['results']['e5'][0]['path']).parent),
        '--device','cuda','--faiss_gpu','--host','127.0.0.1','--port','18084','--threads','8','--topk','3'])]
    command(['ss','-H','-lntp'],run,'service-port-before');require(':18084' not in (run/'service-port-before.stdout').read_text(),'Retriever port already has a listener')
    write_once(run/'service-command.json',dict(argv=argv,env=env,pass_fds=lease.fds,assets=verified,asset_manifest=rec))
    out=open(run/'retriever.stdout','xb');err=open(run/'retriever.stderr','xb')
    try:proc=subprocess.Popen(argv,stdout=out,stderr=err,env=env,pass_fds=lease.fds)
    except BaseException as exc:
        write_once(run/'service-popen-error.json',dict(type=type(exc).__name__,message=str(exc)))
        for f in (out,err):f.flush();os.fsync(f.fileno());f.close()
        raise
    # Register all three handles in the owning finally before any fallible
    # exclusive ledger write or metadata read. Unknown identity is never killed.
    on_popen(proc,out,err)
    write_once(run/'service-popen.json',dict(pid=proc.pid,actual_popen=True))
    return proc,out,err


def service_ready(run,owner,lease,sampler,proc,identity):
    command_record=unique((run/'service-command.json').read_bytes());verified=command_record['assets']
    deadline=time.monotonic()+600
    import urllib.request,urllib.error
    queries=['What is the capital of France?','Who wrote Hamlet?','What is water made of?','Where is Tokyo?','When was Apollo 11?','Who invented the telephone?','What is Mount Everest?','What is the Moon?']*3
    while True:
        require(proc.poll() is None and same_process(identity),'Own retriever exited/reused')
        try:
            req=urllib.request.Request('http://127.0.0.1:18084/retrieve',data=canonical(dict(queries=queries,topk=3,return_scores=True)),headers={'Content-Type':'application/json'})
            started=time.monotonic()
            with urllib.request.urlopen(req,timeout=120) as r:raw=r.read();status=r.status
            result=unique(raw);require(status==200 and type(result['result']) is list and len(result['result'])==24 and all(len(v)==3 for v in result['result']),'Actual top3 response differs')
            write_once(run/'http-top3-24.json',dict(request=dict(queries=queries,topk=3),response=result,elapsed_seconds=time.monotonic()-started,
                timing_scope='CLIENT_WALL_INCLUDES_MODEL_RETRIEVAL_AND_TRANSPORT_NOT_NETWORK_ALONE'));break
        except (ConnectionRefusedError,urllib.error.URLError):require(time.monotonic()<deadline,'Retriever readiness absent');sampler.sample();time.sleep(3)
    command(['ss','-H','-lntp'],run,'service-listener');listener=(run/'service-listener.stdout').read_text()
    require(re.search(r'127\.0\.0\.1:18084\s.*pid='+str(proc.pid)+r'\b',listener),'Service listener is not own loopback PID')
    raw,faiss=capture(run/'actual-faiss-runtime.json',keep=True);origin=unique(raw)
    same(origin['pid'],proc.pid,'FAISS origin PID differs');same(origin['runtime_root'],str(GPU_LIB),'FAISS library root differs')
    same(origin['modules'],FAISS_MODULE_BINDINGS,'Loaded GPU FAISS module paths/fullSHA differ')
    for item in origin['modules'].values():capture(item['path'],item['sha256'],root=GPU_LIB)
    loaded,loaded_record=reference(dict(path=str(run/'actual-loaded-retriever.json'),sha256=capture(run/'actual-loaded-retriever.json')[1]['sha256']),root=run)
    for k in ('pid','start_ticks','uid','four_uids','cgroup','job_id','step_id'):same(loaded['process'][k],identity[k],'Loaded retriever process differs')
    same(loaded['gpu_index']['devices'],list(range(8)),'Actual GPU FAISS shard device set differs')
    same(loaded['e5']['parameter_devices'],['cuda:0'],'Actual E5 parameter device differs')
    sample=sampler.sample()
    require({v['uuid'] for v in sample['gpu_processes'] if v['pid']==proc.pid}==set(lease.uuids),'Retriever not visible on all allocated GPUs')
    write_once(run/'service-ready.json',dict(identity=identity,faiss_origin=faiss,loaded_retriever=loaded_record,assets=verified,http=capture(run/'http-top3-24.json')[1],nvml=sample,
        scope='ACTUAL_SAME_NODE_SERVICE_OBSERVATION_REQUIRES_INDEPENDENT_LOADED_FAISS_AND_GPU_E5_PROOF'))


def supervise(request_path,request_pin):
    owner=allocation();q,request_record=load_request(request_path,request_pin)
    run=ROOT/'paired-final200-runs'/('eval-'+owner['job_id']);run.parent.mkdir(exist_ok=True)
    require(run.parent.resolve(strict=True)==run.parent and run.parent.stat().st_uid==os.getuid(),'Run parent differs');sync_dir(run.parent.parent)
    reserve(run);write_once(run/'intent.json',dict(request=request_record,parent_identity=owner,status='ONCE_KEEP_UNKNOWN_NO_RETRY'))
    lease=None;service=None;service_identity=None;streams=[];terminal=dict(status='CONTROL_NOT_COMPLETED',slurm_terminal=None,resource_release=None)
    try:
        git_clean(run,TRAIN_REPO,TRAIN,'training-before');git_clean(run,REPO,EVAL,'eval-source-before')
        command(['scontrol','show','job',owner['job_id'],'-o'],run,'slurm-before')
        slurm=slurm_allocation((run/'slurm-before.stdout').read_text(),owner['job_id'])
        nvml_folder=run/'observations';reserve(nvml_folder)
        command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],run,'initial-inventory')
        gpus=parse_gpu_inventory((run/'initial-inventory.stdout').read_bytes());uuids=[v['uuid'] for v in sorted(gpus,key=lambda x:x['index'])]
        lease=LeaseSet(uuids);lease.acquire();lease.check();sampler=Sampler(nvml_folder,owner,uuids);idle(sampler.sample());idle(sampler.sample())
        strict_members=bundle_check(STRICT.parent,STRICT_MANIFEST_SHA)
        write_once(run/'strict-current-content.json',dict(records=strict_members,original_publisher_exit='UNKNOWN_NOT_RETRIED'))
        proofs={}
        for role in ROLES:
            args=['--receipt',q['layered_receipt']['path'],'--receipt-sha256',q['layered_receipt']['sha256'],'--model-role',role,'--model-directory',str(MODELS[role])]
            wait_child([PYTHON,'-B','-c',bootstrap(STRICT,STRICT_SHA,args)],run,'strict-'+role,dict(clean_env(),CUDA_VISIBLE_DEVICES=''),timeout=3600,job=owner['job_id'],step=owner['step_id'])
            raw=(run/('strict-'+role+'.stdout')).read_bytes();proof=unique(raw)
            same(proof,dict(kind='STRICT_ACTUAL_TRAINING3304_HF200_PREREQUISITES',training_commit=TRAIN,training_job_id='3304',actual_outer_updates=200,hf_step=200,final_questions=51713,
                receipt_sha256=q['layered_receipt']['sha256'],model_role=role,model_directory=str(MODELS[role])),'Exact nine strict fields differ')
            proofs[role]=proof
        before=checkpoint_guard(q);write_once(run/'checkpoint31-before.json',before)
        args=['--mode','prepare','--request',str(request_path),'--request-sha256',request_pin,'--run',str(run),'--supervisor-sha256',q['control_tools']['paired_supervisor.py']]
        cpu=Path(__file__).parent/'paired_cpu_prepare.py'
        wait_child([PYTHON,'-B','-c',bootstrap(cpu,q['control_tools']['paired_cpu_prepare.py'],args)],run,'cpu-prepare',dict(clean_env(),CUDA_VISIBLE_DEVICES=''),timeout=3600,job=owner['job_id'],step=owner['step_id'])
        prepared,_=reference(dict(path=str(run/'cpu-prepared.json'),sha256=capture(run/'cpu-prepared.json')[1]['sha256']),root=run)
        same(prepared['request'],dict(path=str(request_path),sha256=request_pin),'CPU prepared request differs')
        contract,_=exact_module(Path(__file__).parent/'paired_eval_contract.py',CONTRACT_SHA,'paired_stage_contract')
        stages={}
        for role in ROLES:
            stages[role]=write_once(run/(role+'-stage.json'),proposed_stage(q,contract,dict(owner,gpu_uuids=uuids,node='h100-10'),role,
                prepared['model_files'][role],prepared['input_manifest'],prepared['source_files']))
        ready=dict(version=1,kind='ACTUAL_PAIRED_FINAL200_PRE_GPU_READY_NOT_CAPACITY_OR_EVAL_PASS',request=dict(path=str(request_path),sha256=request_pin),
            parent_identity=owner,nonce=secrets.token_hex(32),gpu_uuids=uuids,leases=lease.records,strict_proofs=proofs,stages=stages,
            configurations=prepared['configurations'],cpu_prepare=capture(run/'cpu-prepared.json')[1],checkpoint31=before,
            source_inventory=q['source_inventory'],native_import_closure=q['native_import_closure'],public_key_sha256=PUBLIC_SHA,slurm_literal=slurm)
        ready_record=write_once(run/'pre-gpu-ready.json',ready);gate,_=exact_module(REPO/'search_r1/baselines/checkpoint_gate.py',GATE_SHA,'paired_ready_gate')
        approval=await_approval(run,ready,ready_record,gate,q['approval_wait_seconds'],lease,sampler)
        # Long external review waits must not admit changed model/source/input
        # bytes. Preserve the actual fresh readback, not just a prior pin label.
        fresh=prepared_recheck(q,ready,prepared);write_once(run/'post-review-full-input-readback.json',fresh)
        write_once(run/'actual-approval-consumed.json',dict(approval=approval,ready=ready_record))
        def register_service(proc,out,err):
            nonlocal service,streams
            service=proc;streams=[out,err]
        service,out,err=service_start(run,q,prepared,owner,lease,sampler,register_service)
        service_identity=proc_identity(service.pid,owner['job_id'],owner['step_id']);write_once(run/'service-start.json',service_identity)
        service_ready(run,owner,lease,sampler,service,service_identity)
        outcomes=run_roles(run,q,prepared,stages,owner,lease,sampler,service_identity)
        same(before,checkpoint_guard(q),'Original31 changed across the pair');git_clean(run,TRAIN_REPO,TRAIN,'training-after');git_clean(run,REPO,EVAL,'eval-source-after')
        terminal.update(status='BOTH_ROLE_OUTPUTS_COMPLETE_REVIEW_REQUIRED_NOT_SLURM_OR_RELEASE_PASS',outcomes=outcomes)
    except BaseException as exc:
        terminal['error']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        try:
            if service is not None:
                if service.poll() is None and service_identity is not None:
                    require(same_process(service_identity),'Do not terminate stale/foreign retriever');service.terminate()
                try:service.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if service_identity is not None:
                        require(same_process(service_identity),'Do not kill stale/foreign retriever');service.kill()
                        try:service.wait(timeout=20)
                        except subprocess.TimeoutExpired:pass
                waited=service.poll() is not None
                write_once(run/'service-exit.json',dict(waited=waited,returncode=service.returncode if waited else None,identity=service_identity))
                require(waited,'Retriever exit remains unknown; no resource-release claim')
            for f in streams:f.flush();os.fsync(f.fileno());f.close()
        except BaseException as exc:
            terminal['cleanup_error']=dict(type=type(exc).__name__,message=str(exc));raise
        finally:
            if lease is not None:lease.close()
            write_once(run/'supervisor-result-review-required.json',terminal)


def prepared_recheck(q,ready,prepared):
    records={}
    entries=list(prepared['source_members'].values())+list(prepared['configurations'].values())+list(ready['stages'].values())
    entries+=list(prepared['tool_asset_records'])+list(prepared['faiss_gpu_library_records'])+[prepared['input_manifest']]
    for role in ROLES:entries+=list(prepared['model_physical_records'][role].values())
    for rec in entries:
        path=rec.get('physical_path',rec.get('path'));require(type(path) is str,'Exact prepared physical path missing')
        actual=capture(path,rec['sha256'])[1]
        for key in ('bytes','uid','device','inode','mtime_ns','ctime_ns'):same(actual[key],rec[key],'Prepared physical identity changed: '+key)
        records[path]=actual
    same(checkpoint_guard(q),ready['checkpoint31'],'Original31 changed during external gate')
    tools(q);bundle_check(STRICT.parent,STRICT_MANIFEST_SHA)
    for key in ('source_inventory','native_import_closure','common_manifest','layered_receipt'):reference(q[key],root=Path('/mnt/data/shaoshitong/search-r1-rl'))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    a=p.parse_args();supervise(Path(a.request),a.request_sha256)


if __name__=='__main__':main()

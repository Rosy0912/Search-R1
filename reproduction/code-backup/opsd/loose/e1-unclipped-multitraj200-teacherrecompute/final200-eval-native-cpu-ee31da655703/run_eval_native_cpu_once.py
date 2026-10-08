#!/usr/bin/env python3
"""Exclusive CPU-step-only c5bf preparation controller. No scheduler or launch."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

COMMON_SHA = 'e00be844fde1b3a6a6102f4a8d4d1afafbc1b7141764942a24d2ee848ff32581'
SOURCE = 'c5bf4c8db33562a259fa227a940dc69543901b98'
SCOPE = 'NATIVE_CPU_C5BF_IMPORT_AND_PARTIAL_CONFIG_ONLY_NOT_EVALUATION'

# The child executes the very bytes it hashes, not a path reread by Python
# after the controller's earlier source check.
CHILD_BOOTSTRAP = r'''
import hashlib, os, pathlib, re, stat, sys
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
if not p.is_absolute() or p.resolve(strict=True)!=p: raise ValueError('Physical child source required')
before=p.lstat()
if not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or before.st_size>32*1024**2: raise ValueError('Owned child source required')
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
try:
    opened=os.fstat(fd); chunks=[]
    while True:
        chunk=os.read(fd,1024**2)
        if not chunk: break
        chunks.append(chunk)
    ended=os.fstat(fd)
finally: os.close(fd)
after=p.lstat(); keys=('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
if any(getattr(before,k)!=getattr(s,k) for s in (opened,ended,after) for k in keys): raise ValueError('Child source changed during read')
raw=b''.join(chunks)
if re.fullmatch('[0-9a-f]{64}',expected) is None or hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('Child bytes differ before execution')
sys.argv=[str(p)]+sys.argv[3:]
scope={'__name__':'__main__','__file__':str(p),'__package__':None,'__builtins__':__builtins__}
exec(compile(raw,str(p),'exec',dont_inherit=True,optimize=0),scope)
'''


def exact_common():
    path = Path(__file__).resolve().parent / 'common.py'
    before = path.lstat()
    if path.resolve(strict=True) != path or not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
        raise ValueError('Physical exact common helper required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); raw = b''
        while True:
            chunk = os.read(fd, 1024**2)
            if not chunk: break
            raw += chunk
        ended = os.fstat(fd)
    finally: os.close(fd)
    after = path.lstat()
    keys = ('st_dev','st_ino','st_size','st_uid','st_mode','st_mtime_ns','st_ctime_ns')
    if any(getattr(before,k) != getattr(s,k) for s in (opened,ended,after) for k in keys) or hashlib.sha256(raw).hexdigest() != COMMON_SHA:
        raise ValueError('Common helper changed or differs before execution')
    module = types.ModuleType('once_exact_common'); module.__file__ = str(path)
    exec(compile(raw,str(path),'exec',dont_inherit=True,optimize=0),module.__dict__)
    return module


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write_once(c, path, value):
    raw = c.canonical(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    sync_dir(Path(path).parent)
    return c.read_file(path, hashlib.sha256(raw).hexdigest())[1]


def reserve_once(c, out, intent):
    out = Path(out)
    c.require(out.is_absolute() and out.parent.resolve(strict=True) == out.parent and out.parent.stat().st_uid == os.getuid(), 'Physical owned existing output parent required')
    # mkdir is exclusive and precedes any child launch. A partial or failed
    # previous attempt remains reserved; never delete or reuse it.
    out.mkdir(mode=0o700)
    sync_dir(out.parent); sync_dir(out)
    return write_once(c, out / 'intent.original.json', intent)


def child_identity(pid, execution):
    root = Path('/proc') / str(pid)
    try:
        status = (root / 'status').read_text()
        match = re.search(r'^Uid:\s+(.*)$', status, re.M)
        if match is None: raise ValueError('Child UID missing')
        uids = [int(x) for x in match.group(1).split()]
        stat_raw = (root / 'stat').read_text()
        fields = stat_raw[stat_raw.rfind(')')+2:].split()
        start = int(fields[19]); parent = int(fields[1])
        cgroup = (root / 'cgroup').read_text()
    except FileNotFoundError:
        return dict(pid=pid, process_sample='ALREADY_EXITED_BEFORE_PROC_SAMPLE', parent_pid=None, start_ticks=None, uids=None, cgroup=None)
    job = execution['execution_job_id']; step = execution['execution_step_id']
    pattern = r'(?:^|/)job[_-]'+re.escape(job)+r'(?:\.scope)?/step[_-]'+re.escape(step)+r'(?:\.scope)?(?:/|$)'
    if parent != os.getpid() or uids != [os.getuid()]*4 or start <= 0 or not re.search(pattern,cgroup,re.M):
        raise ValueError('Actual CPU child parent/UID/cgroup differs')
    return dict(pid=pid,parent_pid=parent,start_ticks=start,uids=uids,cgroup=cgroup,process_sample='OWN_CPU_CHILD_PROC_OBSERVED')


def execute_child(c, command, out, environment, execution):
    out = Path(out)
    with (out / 'child.stdout.original').open('xb') as stdout, (out / 'child.stderr.original').open('xb') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, cwd=str(out))
        # Persist the Popen identity even if the subsequent /proc sample fails.
        sampling_error = None
        try:
            write_once(c, out / 'child-popen.original.json', dict(pid=process.pid, parent_pid=os.getpid(), argv=command))
            observed = child_identity(process.pid, execution)
            write_once(c, out / 'child-start.original.json', observed)
        except BaseException as error:
            sampling_error = dict(type=type(error).__name__, message=str(error))
            # A failure recording start cannot bypass wait. The durable wait
            # below records this original exception after the child terminates.
        # Actual wait is retained for every launched child. No retry, fallback,
        # signal, training, scheduler or GPU operation is performed here.
        code = process.wait()
        stdout.flush(); stderr.flush(); os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    wait = dict(pid=process.pid, parent_pid=os.getpid(), argv=command, waited=True, returncode=code,
                process_identity_error=sampling_error)
    write_once(c, out / 'child-wait.original.json', wait)
    if sampling_error is not None: raise ValueError('Child identity sample failed; real wait retained')
    return wait


def qualify(c, report, execution, checker_record, inventory_record):
    c.require(type(report) is dict, 'Child report missing')
    for key, target in dict(version=1,kind='ACTUAL_C5BF_NATIVE_CPU_IMPORT_PARTIAL_CONFIG_PREPARATION',
            scope=SCOPE,passed=True,eval_source_commit=SOURCE,stored_training_job_id='3304',
            actual_eval_job_id=None,actual_model_directory=None,actual_stage_manifest=None,
            actual_input_manifest=None,actual_output_directory=None,model_loaded=False,
            gpu_execution_verified=False,evaluation_started=False,
            evaluation_eligibility='BLOCKED_ACTUAL200_AND_REVIEWED_STAGE_REQUIRED').items():
        c.typed(report.get(key),target,'Native CPU report qualification differs: '+key)
    c.typed(report['tool_source'],checker_record,'Executed checker source record differs')
    c.typed(report['source_inventory'],inventory_record,'Executed source inventory differs')
    c.verify_record(report['tool_source']); c.verify_record(report['source_inventory'])
    for key in ('execution_job_id','execution_step_id','uid','uids','node','stored_training_job_id'):
        c.typed(report['execution'].get(key),execution[key],'Child actual execution differs: '+key)
    c.require(type(report['execution']['pid']) is int and report['execution']['pid'] > 0 and report['execution']['pid'] != execution['pid'], 'Child cannot impersonate controller')
    c.require(type(report['execution']['uids']) is list and len(report['execution']['uids']) == 4 and all(type(x) is int for x in report['execution']['uids']), 'Child UID vector type differs')
    c.typed(report['runtime']['cuda_is_initialized'],False,'Child initialized CUDA')
    c.require(type(report['module_count']) is int and report['module_count'] > 0 and type(report['verified_live_function_code_count']) is int and report['verified_live_function_code_count'] > 0,'Real import/code counts missing')
    for key in ('import_closure','partial_config','source_physical_records'): c.verify_record(report[key])
    c.typed(report['source_tree_members'],857,'Complete actual c5bf source tree missing')
    c.typed(report['configuration']['runtime_resolved_config'],False,'Partial config misreported as runtime resolved')
    c.typed(report['configuration']['retrieval_service_verified'],False,'Authored default is not live retrieval proof')


def run(args):
    c = exact_common()
    _, self_record = c.read_file(Path(__file__).resolve(),args.self_sha256)
    checker_path = Path(__file__).resolve().parent / 'inspect_eval_native_cpu.py'
    _, checker_record = c.read_file(checker_path,args.checker_sha256)
    _, inventory_record = c.read_json(args.source_inventory,args.source_inventory_sha256)
    c.typed(sys.platform,'linux','Native CPU controller requires Linux /proc identity')
    execution = c.execution_identity(args.execution_job_id,args.execution_step_id)
    c.typed(execution['uid'],0,'CPU controller requires actual own UID0')
    c.require(sys.version_info[:2] == (3,10) and sys.dont_write_bytecode and sys.flags.optimize == 0,'Native Python3.10 -B required')
    out = Path(args.output_directory)
    intent = dict(version=1,kind='EXCLUSIVE_C5BF_NATIVE_CPU_PREPARATION_INTENT',scope=SCOPE,
        execution=execution,argv=list(sys.argv),controller_source=self_record,
        checker_source=checker_record,source_inventory=inventory_record,
        stored_training_job_id='3304',eval_source_commit=SOURCE,actual_eval_job_id=None,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        retry_policy='NEVER_REUSE_THIS_OUTPUT_DIRECTORY_UNKNOWN_OUTCOME_REQUIRES_REVIEW')
    reserve_once(c,out,intent)
    command = [sys.executable,'-B','-c',CHILD_BOOTSTRAP,str(checker_path),checker_record['sha256'],
        '--source-inventory',inventory_record['physical_path'],
        '--source-inventory-sha256',inventory_record['sha256'],'--self-sha256',checker_record['sha256'],
        '--execution-job-id',args.execution_job_id,'--execution-step-id',args.execution_step_id,
        '--output',str(out/'native-cpu-preparation.json')]
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES']=''; environment['PYTHONDONTWRITEBYTECODE']='1'
    removed = [n for n in environment if n.startswith('OPSD_FINAL200_') or n == 'OPSD_E1_RETRIEVER_URL']
    for name in removed: environment.pop(name)
    write_once(c,out/'child-invocation.original.json',dict(argv=command,cwd=str(out),
        removed_environment_variable_names=sorted(removed),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    try:
        wait = execute_child(c,command,out,environment,execution)
        c.require(type(wait['returncode']) is int and wait['returncode'] == 0,'Actual CPU child failed; no content qualification')
        report, report_record = c.read_json(out/'native-cpu-preparation.json')
        qualify(c,report,execution,checker_record,inventory_record)
        c.typed(report['execution']['pid'],wait['pid'],'Report child PID differs from actual Popen')
        observed, _ = c.read_json(out/'child-start.original.json')
        if observed['process_sample'] == 'OWN_CPU_CHILD_PROC_OBSERVED':
            for key in ('pid','start_ticks','uids','cgroup'):
                c.typed(report['execution'][key],observed[key],'Report differs from observed child process: '+key)
        c.verify_record(self_record);c.verify_record(checker_record);c.verify_record(inventory_record)
        complete = dict(version=1,kind='C5BF_NATIVE_CPU_PREPARATION_CONTROLLER_COMPLETE',scope=SCOPE,
            cpu_preparation_qualified=True,execution=execution,child_wait=wait,report=report_record,
            controller_source=self_record,checker_source=checker_record,
            actual_eval_job_id=None,evaluation_eligibility='BLOCKED_ACTUAL200_AND_REVIEWED_STAGE_REQUIRED')
        record = write_once(c,out/'controller-complete.original.json',complete)
        print(c.canonical(record).decode(),flush=True)
        return complete
    except BaseException as error:
        write_once(c,out/'controller-failure.original.json',dict(type=type(error).__name__,message=str(error),
            scope=SCOPE,CPU_preparation_qualified=False,actual_eval_job_id=None,
            output_directory_remains_reserved=True))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-inventory','source-inventory-sha256','self-sha256','checker-sha256','execution-job-id','execution-step-id','output-directory'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()

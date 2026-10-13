"""One reviewed own shared-unit entry with durable real startup accounting.

Early failures publish actual CPU/elapsed/exception and own fd cleanup evidence.
No model kernel, child service, training restore, foreign signal, journal rewrite
or retry occurs in this entry before its successful supervisor exec.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from production_contract import ROOT,SOURCE,production_inputs,read_bound,require
from shared_identity import verify_cgroup,establish_leases,observe_host_admission,unit_name,job_id


def publish_once(path,value):
    require(not path.exists() and not path.is_symlink(),'Existing startup audit: refuse overwrite/repeat')
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    started=time.time();clock=time.perf_counter();cpu=time.process_time();usage=resource.getrusage(resource.RUSAGE_SELF)
    audit=ROOT/'instruct300/startup-audits';audit.mkdir(mode=0o700,exist_ok=True)
    prefix=audit/('nested-branch-fresh200-'+SOURCE)
    base=dict(schema_version='tree_shared_startup_v1',unit=unit_name(),job_id=job_id(),code_commit=SOURCE,
      entry_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=os.environ.get('TREE_INSTRUCT_PACKAGE_SHA256'),release_path=os.environ.get('TREE_PRODUCTION_RELEASE_PATH'),
      release_sha256=os.environ.get('TREE_INSTRUCT_REVIEW_SHA256'),pid=os.getpid(),start_epoch=started,
      start_process_cpu_seconds=cpu,start_user_cpu_seconds=usage.ru_utime,start_system_cpu_seconds=usage.ru_stime,
      physical_gpus=8,slurm_reserved_gpus=0,training_started=False,training_or_optimizer_restored=False,
      own_children_started=False,foreign_processes_signaled=False,automatic_retry=False)
    publish_once(prefix.with_suffix('.started.json'),dict(base,status='ACTUAL_OWN_STARTUP_ENTERED'))
    opened=[];phase='own_identity';identity=None;host=None
    try:
        require(sys.flags.optimize==0,'Shared runtime requires unoptimized Python')
        identity=verify_cgroup()
        for name in ('TREE_DIRECT_OWNED_MODELS_LOADED','TREE_DIRECT_ALLOCATION_IDENTITY_PATH','TREE_DIRECT_ALLOCATION_IDENTITY_SHA256'):
            os.environ.pop(name,None)
        phase='package_source_and_release'
        tools=Path(os.environ['TREE_INSTRUCT_TOOLS'])
        package,manifest_sha,_=read_bound(tools/'manifest.json',os.environ['TREE_INSTRUCT_PACKAGE_SHA256'])
        for relative,expected in package['files'].items():
            from build_package_draft import read_regular_bound
            read_regular_bound(tools/relative,expected)
        production_inputs(package,manifest_sha256=manifest_sha,dispatch_review_descriptor=dict(path=os.environ['TREE_PRODUCTION_RELEASE_PATH'],sha256=os.environ['TREE_INSTRUCT_REVIEW_SHA256']))
        phase='truthful_predecessor'
        from predecessor_gate import verify_predecessor
        verify_predecessor(package)
        phase='actual_host_admission';host=observe_host_admission()
        phase='physical_uuid_lease_and_memory_admission';establish_leases(opened)
        phase='supervisor_exec'
        publish_once(prefix.with_suffix('.admission.json'),dict(base,status='ACTUAL_PRELAUNCH_ADMISSION_COMPLETE_PENDING_EXEC',
          end_epoch=time.time(),wall_seconds=time.perf_counter()-clock,end_process_cpu_seconds=time.process_time(),
          process_cpu_seconds=time.process_time()-cpu,own_cgroup=identity,host_admission=host,
          opened_lease_count=len(opened),supervisor_source_sha256=hashlib.sha256((tools/'supervisor.py').read_bytes()).hexdigest()))
        os.chdir(tools/'source/repo')
        os.execv(sys.executable,[sys.executable,str(tools/'supervisor.py')])
    except BaseException as error:
        cleanup=[]
        # Every entry is an fd opened in this process by establish_leases. Closing
        # our new description cannot release a foreign process's OFD lock.
        for record in opened:
            row=dict(record,close_succeeded=False)
            try:
                require(record['opened_by_current_pid']==os.getpid(),'Unknown fd owner; do not close')
                held=os.fstat(record['fd'])
                require((held.st_dev,held.st_ino)==(record['st_dev'],record['st_ino']),'Own opened fd identity changed; do not close')
                os.close(record['fd']);row['close_succeeded']=True
            except BaseException as close_error:row['close_error']=repr(close_error)
            cleanup.append(row)
        end_usage=resource.getrusage(resource.RUSAGE_SELF)
        publish_once(prefix.with_suffix('.failure.json'),dict(base,status='ACTUAL_OWN_PRELAUNCH_FAILED',phase=phase,
          end_epoch=time.time(),wall_seconds=time.perf_counter()-clock,end_process_cpu_seconds=time.process_time(),
          process_cpu_seconds=time.process_time()-cpu,user_cpu_seconds=end_usage.ru_utime-usage.ru_utime,
          system_cpu_seconds=end_usage.ru_stime-usage.ru_stime,own_cgroup=identity,host_admission=host,
          exception_type=type(error).__name__,exception=repr(error),own_opened_fd_cleanup=cleanup,
          own_opened_fd_cleanup_complete=all(row['close_succeeded'] for row in cleanup),
          scope='Actual pre-supervisor entry CPU/wall; no GPU model kernel or child training/service was started; journal SUBMITTED/UNKNOWN retained unchanged'))
        raise
if __name__=='__main__':main()

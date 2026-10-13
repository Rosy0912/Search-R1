"""One fresh shared-systemd submission; unknown never automatically retries."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time
from production_contract import ROOT,SOURCE,require,read_bound,production_inputs
from slurm_identity import verify_management_scope,local_environment,cluster_identity,create_journal,finalize_journal,timeout_output
from shared_identity import PROTOCOL,job_id,unit_name,observe_host_admission
from render_shared_dispatch import payload,render
from build_package_draft import read_regular_bound

def admission(manifest,run,environment):
    verify_management_scope(environment)
    from predecessor_gate import verify_predecessor
    from storage_guard import observe_storage
    predecessor=verify_predecessor(manifest);host=observe_host_admission();storage=observe_storage(ROOT/'runs',manifest['storage_guard'])
    queue=run(['squeue','--local','-h','-o','%i|%j|%T'],env=environment,text=True,capture_output=True,timeout=30)
    require(queue.returncode==0 and queue.stderr=='','Read-only old/task queue lookup ambiguous')
    for line in queue.stdout.splitlines():
        fields=line.split('|');require(len(fields)==3,'Ambiguous same-node queue line')
        require(fields[0] not in ('3360','3392') and not fields[1].startswith(('tree-nested-branch-fresh200','tree-branpo-fresh200')),
                'Own predecessor/new duplicate still active or queued; never cancel for fit')
    state=run(['systemctl','show',unit_name(),'--property=LoadState,ActiveState'],env=environment,text=True,capture_output=True,timeout=15)
    require(state.returncode==0 and 'LoadState=not-found' in state.stdout,'Unit already exists; never duplicate')
    physical=run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],env=environment,text=True,capture_output=True,timeout=30)
    require(physical.returncode==0 and physical.stderr=='','Physical memory admission failed')
    rows=[]
    for line in physical.stdout.splitlines():
        uid,name,total,free=[item.strip() for item in line.split(',')]
        rows.append(dict(uuid=uid,name=name,total_mib=int(total),free_mib=int(free)))
    require(len(rows)==8 and {row['uuid'] for row in rows}==set(manifest['dispatch']['expected_gpu_uuids'])
            and all('H100' in row['name'] and row['free_mib']>=68*1024 for row in rows),
            'All exact8 H100 GPUs need68GiB initial free; no foreign process modifications')
    return dict(epoch=time.time(),predecessor=predecessor,storage=storage,queue_stdout=queue.stdout,
                actual_physical_gpus=rows,host_admission=host,resource_protocol=PROTOCOL,slurm_reserved_gpus=0,foreign_jobs_modified=False)

def submit(args,run=subprocess.run):
    verify_management_scope(os.environ,socket.gethostname());environment=local_environment(os.environ)
    for key in list(environment):
        if key.startswith(('SLURM_','SBATCH_','SQUEUE_','SCANCEL_','SACCT_','SCONTROL_')):environment.pop(key,None)
    identity=cluster_identity(run,environment,management=True)
    package=Path(args.package);manifest,_,_=read_bound(package/'manifest.json',args.manifest_sha)
    require(package.is_absolute() and package.resolve().is_relative_to((ROOT/'instruct300/candidates').resolve())
            and manifest['code_commit']==SOURCE and manifest['dispatch']['kind']=='shared_systemd', 'Exact own shared source package required')
    _,pins=production_inputs(manifest,manifest_sha256=args.manifest_sha,dispatch_review_descriptor=dict(path=args.release_path,sha256=args.release_sha))
    require(Path(__file__).resolve()==(package/'submit_shared_once.py').resolve(),'Different shared submitter source')
    for name,expected in manifest['files'].items():read_regular_bound(package/name,expected)
    dispatch=read_regular_bound(args.script,args.script_sha)
    require(dispatch==payload(package,args.manifest_sha,args.release_path,args.release_sha),'Exact reviewed systemd argv bytes required')
    path=Path(manifest['dispatch']['once_intent'])
    require(path.parent==ROOT/'instruct300' and re.fullmatch(r'SUBMISSION_ONCE-nested-branch-fresh200-[0-9a-f]{40}\.json',path.name),'Same global fresh intent namespace required')
    fd=os.open(ROOT/'instruct300/FRESH200_TRAIN_DISPATCH.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Dispatch lock not ordinary');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not path.exists() and not path.is_symlink()
                and not list(path.parent.glob('SUBMISSION_ONCE-nested-branch-fresh200-*.json')),
                'Existing/unknown Slurm or systemd nested intent forbids any repeat')
        observed=admission(manifest,run,environment);argv=render(package,args.manifest_sha,args.release_path,args.release_sha)
        initial=dict(status='SUBMITTING',dispatch_kind='shared_systemd',cluster_name='wuhan-h100',node_list='h100-10',
          local_cluster_identity=identity,code_commit=SOURCE,manifest_sha256=args.manifest_sha,review_sha256=args.release_sha,
          dispatch_intent_path=str(path),dispatch_script_sha256=args.script_sha,submitted_input_sha256=hashlib.sha256(dispatch).hexdigest(),
          submitted_input_base64=base64.b64encode(dispatch).decode(),submit_argv=argv,unit=unit_name(),job_id=job_id(),
          run_id='tree-nested-branch-fresh200-'+job_id(),resource_protocol=PROTOCOL,physical_gpus=8,slurm_reserved_gpus=0,
          production_input_pins=pins,fresh_admission=observed,start_epoch=time.time(),foreign_jobs_modified=False,
          independent_acceptance=False,training_accepted=False)
        initial['fresh_admission']['host_admission']=observe_host_admission()
        create_journal(path,initial);result=None
        try:
            result=run(argv,env=environment,text=True,capture_output=True,timeout=30)
            require(result.returncode==0,'Shared unit request failed/ambiguous: preserve UNKNOWN and inspect exact unit')
            final=dict(initial,status='SUBMITTED',systemd_start_returncode=0,systemd_start_stdout=result.stdout,
                       systemd_start_stderr=result.stderr,end_epoch=time.time())
        except BaseException as error:
            final=dict(initial,status='UNKNOWN',error_type=type(error).__name__,error=str(error),end_epoch=time.time(),
                       systemd_start_returncode=None if result is None else result.returncode,
                       systemd_start_stdout=None if result is None else timeout_output(result.stdout),
                       systemd_start_stderr=None if result is None else timeout_output(result.stderr))
            finalize_journal(path,final);raise
        finalize_journal(path,final);return final
    finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('package','manifest-sha','release-path','release-sha','script','script-sha'):p.add_argument('--'+key,required=True)
    print(json.dumps(submit(p.parse_args()),sort_keys=True))

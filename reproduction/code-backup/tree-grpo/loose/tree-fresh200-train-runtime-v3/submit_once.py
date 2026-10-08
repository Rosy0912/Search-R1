"""Cluster-bound production candidate dispatcher. Not executed by preparation.

Existence/unknown is irreversible for this intent; no removal, resubmit, restart,
or cancellation path. All tests inject an in-memory fake scheduler.
"""
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
from production_contract import ROOT,require,read_bound,production_inputs
from slurm_identity import (CLUSTER,NODE,local_environment,cluster_identity,query,
    create_journal,finalize_journal,timeout_output)

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
INTENT_RE=r'TREE_FRESH200_TRAIN_ONCE-[A-Za-z0-9_-]+\.json'

def queue_clear(run,environment):
    raw=query(['squeue','--local','-h','-w',NODE,'-o','%i|%j|%T|%b'],run,environment)
    for line in raw.splitlines():
        if not line.strip():continue
        parts=line.split('|');require(len(parts)==4,'Ambiguous local queue snapshot')
        if parts[1].startswith('tree-'):
            # Only an explicitly zero GPU request can coexist for this task.
            zero=parts[3] in ('gres/gpu:0','gres/gpu=0','gpu:0')
            require(zero,'Another Tree GPU allocation/pending lineage exists; never duplicate')
    return raw

def prior_journals_clear(path,run,environment):
    require(not path.exists() and not path.is_symlink(),'Intent already exists; no retry even UNKNOWN')
    observations=[]
    for prior in sorted(path.parent.glob('TREE_FRESH200_TRAIN_ONCE-*.json')):
        value,sha,_=read_bound(prior)
        require(value.get('status')=='SUBMITTED' and value.get('cluster_name')==CLUSTER
                and type(value.get('job_id')) is int and value['job_id']>0,'Previous unknown/legacy/foreign intent blocks dispatch')
        job=str(value['job_id'])
        output=value.get('parsable_stdout');require(output in (job,job+';'+CLUSTER),'Foreign prior submit suffix')
        raw=query(['sacct','--local','-X','-n','-P','-j',job,'-o','JobIDRaw,Cluster,State,ExitCode'],run,environment)
        rows=[line.split('|') for line in raw.splitlines() if line.strip()]
        require(len(rows)==1 and len(rows[0])==4 and rows[0][0]==job and rows[0][1]==CLUSTER
                and rows[0][2] in TERMINAL and re.fullmatch(r'\d+:\d+',rows[0][3]),'Previous own terminal lineage unknown; accounting failure never means stopped')
        observations.append(dict(path=str(prior),sha256=sha,accounting_stdout=raw))
    return observations

def submission_output(value):
    if type(value) is bytes:return value.decode('utf-8',errors='strict')
    require(type(value) is str,'Unexpected native submit output type')
    return value

def parse_submission(result):
    stdout,stderr=submission_output(result.stdout),submission_output(result.stderr)
    require(result.returncode==0 and stderr=='','Submit not exactly confirmed')
    require(re.fullmatch(r'[1-9][0-9]*(;wuhan-h100)?\n?',stdout),'Foreign/ambiguous submit suffix')
    parsable=stdout.rstrip('\n');return int(parsable.split(';')[0]),parsable

def submit(package_path,manifest_sha,review_path,review_sha,script_path,script_sha,*,run=subprocess.run,hostname=None):
    require((socket.gethostname() if hostname is None else hostname).split('.')[0]==NODE,'Dispatcher only runs on own h100-9')
    from calibration_contract import forbid_recovery_environment
    forbid_recovery_environment(os.environ)
    require(not any(k in os.environ for k in ('TREE_PPO_CALIBRATION_ONLY','TREE_FRESH_INITIALIZATION_ONLY')),'Production dispatch forbids inherited zero-only flags')
    environment=local_environment(os.environ);identity=cluster_identity(run,environment)
    package_path=Path(package_path)
    require(package_path.is_absolute() and package_path.is_relative_to(ROOT/'instruct300/candidates'),'Own candidate directory required')
    manifest,_,_=read_bound(package_path/'manifest.json',manifest_sha)
    require(manifest['scope']=='FROZEN_PENDING_INDEPENDENT_FRESH200_TRAIN200_PRE_DISPATCH_REVIEW'
            and manifest['stage']=='fresh_mu_train200' and manifest['bootstrap_thresholds_accepted'] is True
            and manifest['production_thresholds_accepted'] is False,'Only independently frozen first-five bootstrap package may dispatch')
    _,pins=production_inputs(manifest,manifest_sha256=manifest_sha,dispatch_review_descriptor=dict(path=str(review_path),sha256=review_sha))
    release,_,_=read_bound(review_path,review_sha)
    path=Path(manifest['dispatch']['once_intent'])
    require(path.parent==ROOT/'instruct300' and re.fullmatch(INTENT_RE,path.name)
            and manifest['dispatch']['cluster_name']==CLUSTER and release['dispatch_intent_path']==str(path),'Fixed reviewed NEW production intent/cluster required')
    from build_package_draft import read_regular_bound
    for name,expected in manifest['files'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe production package member')
        read_regular_bound(package_path/relative,expected)
    require(Path(__file__).resolve()==(package_path/'submit_once.py').resolve(),'Dispatcher is not the reviewed package source')
    script=read_regular_bound(script_path,script_sha)
    require(release['dispatch_renderer_sha256']==manifest['files']['render_dispatch.py']
            and release['dispatch_submitter_sha256']==manifest['files']['submit_once.py'],'Exact dispatcher/renderer source not independently bound')
    # Verify rendered actual argv/script exactly, not substring directives.
    from render_dispatch import render
    require(script==render(package_path,manifest_sha,review_path,review_sha,72).encode(),'Submitted script differs from exact reviewed production renderer')
    lock_path=ROOT/'instruct300/FRESH200_TRAIN_DISPATCH.lock'
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Dispatch lock must be ordinary file')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=prior_journals_clear(path,run,environment);queue=queue_clear(run,environment)
        # The immutable bytes below, not the filename, are the submitted program.
        submit_argv=['sbatch','--parsable']
        initial=dict(status='SUBMITTING',cluster_name=CLUSTER,local_cluster_identity=identity,node_list=NODE,
            manifest_sha256=manifest_sha,review_sha256=review_sha,dispatch_intent_path=str(path),
            dispatch_script_sha256=script_sha,submit_argv=submit_argv,
            submitted_input_sha256=hashlib.sha256(script).hexdigest(),submitted_input_bytes=len(script),
            submitted_input_base64=base64.b64encode(script).decode('ascii'),submit_mode='verified_bytes_via_stdin',
            selected_script_path=str(script_path),production_input_pins=pins,prior_accounting=prior,
            queue_stdout=queue,start_epoch=time.time(),stdout=None,stderr=None,job_id=None)
        create_journal(path,initial)
        result=None
        try:
            result=run(submit_argv,input=script,env=environment,capture_output=True,timeout=60)
            job,parsable=parse_submission(result)
            final=dict(initial,status='SUBMITTED',job_id=job,run_id='tree-fresh200-train-'+str(job),
                stdout=submission_output(result.stdout),stderr=submission_output(result.stderr),returncode=result.returncode,parsable_stdout=parsable,
                submitted_result_stdout_raw=timeout_output(result.stdout),submitted_result_stderr_raw=timeout_output(result.stderr),end_epoch=time.time())
        except BaseException as error:
            final=dict(initial,status='UNKNOWN',error_type=type(error).__name__,error=str(error),end_epoch=time.time(),
                timeout_stdout=timeout_output(getattr(error,'stdout',None)),timeout_stderr=timeout_output(getattr(error,'stderr',None)),
                submitted_returncode=None if result is None else result.returncode,
                submitted_result_stdout_raw=timeout_output(None if result is None else result.stdout),
                submitted_result_stderr_raw=timeout_output(None if result is None else result.stderr))
            finalize_journal(path,final)
            raise
        finalize_journal(path,final);return final
    finally:os.close(fd)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('package','manifest-sha','release-path','release-sha','script','script-sha'):p.add_argument('--'+key,required=True)
    args=p.parse_args();print(json.dumps(submit(args.package,args.manifest_sha,args.release_path,args.release_sha,args.script,args.script_sha),sort_keys=True))

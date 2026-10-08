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
from production_contract import ROOT,require,read_bound,production_inputs,DIRECT_MODE,descriptor,SOURCE
from slurm_identity import (CLUSTER,NODE,local_environment,cluster_identity,query,
    create_journal,finalize_journal,timeout_output,verify_management_scope)

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
INTENT_RE=r'SUBMISSION_ONCE-fresh200-direct-[A-Za-z0-9_-]+\.json'

def queue_clear(run,environment):
    raw=query(['squeue','--local','-h','-w',NODE,'-o','%i|%j|%T|%b'],run,environment,management=True)
    for line in raw.splitlines():
        if not line.strip():continue
        parts=line.split('|');require(len(parts)==4,'Ambiguous local queue snapshot')
        if parts[1].startswith('tree-'):
            # Only an explicitly zero GPU request can coexist for this task.
            zero=parts[3] in ('gres/gpu:0','gres/gpu=0','gpu:0')
            require(zero,'Another Tree GPU allocation/pending lineage exists; never duplicate')
    return raw

def prior_journals_clear(path,run,environment,predecessor):
    require(not path.exists() and not path.is_symlink(),'Intent already exists; no retry even UNKNOWN')
    observations=[]
    prior_paths=set(path.parent.glob('TREE_FRESH200_TRAIN_ONCE-*.json'))|set(path.parent.glob('SUBMISSION_ONCE-fresh200-direct-*.json'))
    for prior in sorted(prior_paths):
        value,sha,_=read_bound(prior)
        require(value.get('status')=='SUBMITTED' and value.get('cluster_name')==CLUSTER
                and type(value.get('job_id')) is int and value['job_id']>0,'Previous unknown/legacy/foreign intent blocks dispatch')
        job=str(value['job_id'])
        output=value.get('parsable_stdout');require(output in (job,job+';'+CLUSTER),'Foreign prior submit suffix')
        if str(prior)==predecessor['once_journal']['path']:
            require(sha==predecessor['once_journal']['sha256'] and job=='3321'
                    and value['manifest_sha256']=='33fd8b5351a0a28e0a7a1e5742ce5c49db604d950985f35f234ba36e2a7de504',
                    'Exact retained FAILED3321 journal/source binding differs')
            observations.append(dict(path=str(prior),sha256=sha,positive_terminal_cleanup_proof=predecessor,
                accounting_disabled_not_interpreted_as_terminal=True))
            continue
        raw=query(['sacct','--local','-X','-n','-P','-j',job,'-o','JobIDRaw,Cluster,State,ExitCode'],run,environment,management=True)
        rows=[line.split('|') for line in raw.splitlines() if line.strip()]
        require(len(rows)==1 and len(rows[0])==4 and rows[0][0]==job and rows[0][1]==CLUSTER
                and rows[0][2] in TERMINAL and re.fullmatch(r'\d+:\d+',rows[0][3]),'Previous own terminal lineage unknown; accounting failure never means stopped')
        observations.append(dict(path=str(prior),sha256=sha,accounting_stdout=raw))
    return observations

def submission_output(value):
    if type(value) is bytes:return value.decode('utf-8',errors='strict')
    require(type(value) is str,'Unexpected native submit output type')
    return value

def read_regular_bound(path,expected):
    path=Path(path);require(stat.S_ISREG(path.lstat().st_mode),'Pinned package/script must be regular file')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno());raw=stream.read();after=os.fstat(stream.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
            and hashlib.sha256(raw).hexdigest()==expected,'Pinned package/script bytes changed: '+str(path))
    return raw

def parse_submission(result):
    stdout,stderr=submission_output(result.stdout),submission_output(result.stderr)
    require(result.returncode==0 and stderr=='','Submit not exactly confirmed')
    require(re.fullmatch(r'[1-9][0-9]*(;wuhan-h100)?\n?',stdout),'Foreign/ambiguous submit suffix')
    parsable=stdout.rstrip('\n');return int(parsable.split(';')[0]),parsable

def predecessor_ended(proof_path,proof_sha):
    proof,digest,_=descriptor(dict(path=str(proof_path),sha256=proof_sha))
    require(proof['status']=='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP'
            and str(proof['job_id'])=='3321' and proof['code_commit']==SOURCE
            and proof['scheduler_state']=='FAILED' and proof['scheduler_exit_code']=='1:0'
            and proof['cleanup_residual']=={} and proof['original_eight_pid_start_times_preserved'] is True,
            'Own3321 positive terminal/cleanup/protected-process proof required')
    require(Path(proof_path)==ROOT/'runs/tree-fresh200-train-3321/direct-training-predecessor-proof.json',
            'Predecessor proof must stay in own3321 run')
    result,result_sha,_=descriptor(proof['run_result']);cost,cost_sha,_=descriptor(proof['cost'])
    require(str(result['job_id'])=='3321' and result['code_commit']==SOURCE
            and result['cleanup_residual']=={} and result['original_eight_pid_start_times_preserved'] is True
            and str(cost['job_id'])=='3321'
            and result_sha=='052dd4d171c523d3258856ee48494c57919d2f073cc8cff12833eec4757af00a'
            and cost_sha=='438435c67fa523367effd16c07e47c09e36fdabf97195488246c8ebe25ffb714',
            'Predecessor proof differs from exact native retained cleanup/cost records')
    journal,journal_sha,_=descriptor(proof['once_journal'])
    require(proof['once_journal']['path']==str(ROOT/'instruct300/SUBMISSION_ONCE-fresh200-direct-a18f420.json')
            and journal_sha=='086626300a58fc0bb3800d7043a3a3d9be90a75d97fdbe1bdcf76db2b2c5a35d'
            and journal['status']=='SUBMITTED' and journal['job_id']==3321 and journal['cluster_name']==CLUSTER
            and journal['manifest_sha256']=='33fd8b5351a0a28e0a7a1e5742ce5c49db604d950985f35f234ba36e2a7de504',
            'Native FAILED3321 proof must bind exact prior once journal and original V2 source')
    return dict(path=str(proof_path),sha256=digest,run_result_sha256=result_sha,cost_sha256=cost_sha,
                job_id=3321,scheduler_state=proof['scheduler_state'],scheduler_exit_code=proof['scheduler_exit_code'],
                once_journal=dict(path=proof['once_journal']['path'],sha256=journal_sha),
                actual_optimizer_committed=result.get('optimizer_committed'),scheduler_advanced=result.get('scheduler_advanced'),
                failed_unrecorded_partial_optimizer_writes_unknown=result.get('failed_unrecorded_partial_optimizer_writes_unknown'))

def submit(package_path,manifest_sha,review_path,review_sha,script_path,script_sha,predecessor_proof,predecessor_proof_sha,*,run=subprocess.run,hostname=None):
    require((socket.gethostname() if hostname is None else hostname).split('.')[0]==NODE,'Dispatcher only runs on own h100-9')
    from calibration_contract import forbid_recovery_environment
    forbid_recovery_environment(os.environ)
    require(not any(k in os.environ for k in ('TREE_PPO_CALIBRATION_ONLY','TREE_FRESH_INITIALIZATION_ONLY')),'Production dispatch forbids inherited zero-only flags')
    verify_management_scope(os.environ,hostname)
    environment=local_environment(os.environ);identity=cluster_identity(run,environment,management=True)
    package_path=Path(package_path)
    require(package_path.is_absolute() and package_path.is_relative_to(ROOT/'instruct300/candidates'),'Own candidate directory required')
    manifest,_,_=read_bound(package_path/'manifest.json',manifest_sha)
    require(manifest['scope']==manifest['execution_mode']==DIRECT_MODE
            and manifest['stage']=='fresh_mu_train200' and manifest['bootstrap_thresholds_accepted'] is False
            and manifest['production_thresholds_accepted'] is False,'Only explicit human-authorized direct200 package may dispatch')
    _,pins=production_inputs(manifest,manifest_sha256=manifest_sha,dispatch_review_descriptor=dict(path=str(review_path),sha256=review_sha))
    release,_,_=read_bound(review_path,review_sha)
    path=Path(manifest['dispatch']['once_intent'])
    require(path.parent==ROOT/'instruct300' and re.fullmatch(INTENT_RE,path.name)
            and manifest['dispatch']['cluster_name']==CLUSTER and release['dispatch_intent_path']==str(path),'Fixed reviewed NEW production intent/cluster required')
    for name,expected in manifest['files'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe production package member')
        read_regular_bound(package_path/relative,expected)
    require(Path(__file__).resolve()==(package_path/'submit_once.py').resolve(),'Dispatcher is not the reviewed package source')
    script=read_regular_bound(script_path,script_sha)
    require(release['dispatch_renderer_sha256']==manifest['files']['render_dispatch.py']
            and release['dispatch_submitter_sha256']==manifest['files']['submit_once.py'],'Exact direct dispatcher/renderer source not bound')
    # Verify rendered actual argv/script exactly, not substring directives.
    from render_dispatch import render
    wall=manifest['dispatch']['train_wall_hours']
    require(script==render(package_path,manifest_sha,review_path,review_sha,wall).encode(),'Submitted script differs from exact reviewed production renderer')
    predecessor=predecessor_ended(predecessor_proof,predecessor_proof_sha)
    lock_path=ROOT/'instruct300/FRESH200_TRAIN_DISPATCH.lock'
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Dispatch lock must be ordinary file')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=prior_journals_clear(path,run,environment,predecessor);queue=queue_clear(run,environment)
        # The immutable bytes below, not the filename, are the submitted program.
        submit_argv=['sbatch','--parsable']
        initial=dict(status='SUBMITTING',execution_mode=DIRECT_MODE,independent_acceptance=False,training_accepted=False,
            production_thresholds_accepted=False,original_sampler_actor_gate_accepted=False,
            cluster_name=CLUSTER,local_cluster_identity=identity,node_list=NODE,
            manifest_sha256=manifest_sha,review_sha256=review_sha,dispatch_intent_path=str(path),train_wall_hours=wall,
            dispatch_script_sha256=script_sha,submit_argv=submit_argv,
            submitted_input_sha256=hashlib.sha256(script).hexdigest(),submitted_input_bytes=len(script),
            submitted_input_base64=base64.b64encode(script).decode('ascii'),submit_mode='verified_bytes_via_stdin',
            selected_script_path=str(script_path),production_input_pins=pins,prior_accounting=prior,predecessor_terminal_cleanup=predecessor,
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
    for key in ('package','manifest-sha','release-path','release-sha','script','script-sha','predecessor-proof','predecessor-proof-sha'):p.add_argument('--'+key,required=True)
    args=p.parse_args();print(json.dumps(submit(args.package,args.manifest_sha,args.release_path,args.release_sha,args.script,args.script_sha,args.predecessor_proof,args.predecessor_proof_sha),sort_keys=True))

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
HISTORICAL_SOURCE='a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98'
HISTORICAL_JOURNALS={
    '3321':('SUBMISSION_ONCE-fresh200-direct-a18f420.json','33fd8b5351a0a28e0a7a1e5742ce5c49db604d950985f35f234ba36e2a7de504'),
    '3322':('SUBMISSION_ONCE-fresh200-direct-a18f420-v3.json','3f21730c9665dc25072d71c8e79b7e91becfcf5e96ad0aa136df26545cb2fad4'),
}

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

def prior_journals_clear(path,run,environment,proof_closure):
    require(not path.exists() and not path.is_symlink(),'Intent already exists; no retry even UNKNOWN')
    observations=[]
    prior_paths=set(path.parent.glob('TREE_FRESH200_TRAIN_ONCE-*.json'))|set(path.parent.glob('SUBMISSION_ONCE-fresh200-direct-*.json'))
    for prior in sorted(prior_paths):
        value,sha,_=read_bound(prior)
        require(value.get('status')=='SUBMITTED' and value.get('cluster_name')==CLUSTER
                and type(value.get('job_id')) is int and value['job_id']>0,'Previous unknown/legacy/foreign intent blocks dispatch')
        job=str(value['job_id'])
        output=value.get('parsable_stdout');require(output in (job,job+';'+CLUSTER),'Foreign prior submit suffix')
        match=[proof for proof in proof_closure.values() if proof.get('once_journal',{}).get('path')==str(prior)]
        if match:
            require(len(match)==1,'Multiple retained terminal proofs claim the same once journal')
            proof=match[0]
            require(sha==proof['once_journal']['sha256'] and job==str(proof['job_id'])
                    and value['manifest_sha256']==proof['prior_manifest_sha256'],
                    'Exact retained failed journal/source binding differs')
            observations.append(dict(path=str(prior),sha256=sha,positive_terminal_cleanup_proof=proof,
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

def predecessor_ended(manifest,proof_path,proof_sha):
    pins=manifest['terminal_proof_closure']
    require(set(pins)=={'3319','3321','3322'} and pins['3322']==dict(path=str(proof_path),sha256=proof_sha),
            'Direct V4 must bind latest3322 and retained3319/3321 terminal proof closure')
    require(pins['3319']['sha256']=='a4b9656f02427090983405b91afe3757473103a91b727c09aa51384b2f876816'
            and pins['3321']['sha256']=='484b1f5485d63c4f082c6e520e3d17914fa12cb6fefd2457509b0e2b975d51ef',
            'Retained historical terminal proof bytes cannot be replaced or relabelled')
    closure={}
    for job,pin in sorted(pins.items()):
        proof,digest,_=descriptor(pin)
        calibration=job=='3319'
        run_name=('tree-fresh200-' if calibration else 'tree-fresh200-train-')+job
        status='OWN_EXTRA_CALIBRATION_ENDED_FOR_DIRECT_TRAIN_WITH_CLEANUP' if calibration else 'OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP'
        require(pin['path']==str(ROOT/'runs'/run_name/'direct-training-predecessor-proof.json')
                and proof['status']==status and str(proof['job_id'])==job and proof['code_commit']==HISTORICAL_SOURCE
                and proof['scheduler_state']=='FAILED' and proof['scheduler_exit_code']=='1:0'
                and proof['cleanup_residual']=={} and proof['original_eight_pid_start_times_preserved'] is True,
                'Own historical positive terminal/source/cleanup/protected-process proof differs: '+job)
        result,result_sha,_=descriptor(proof['run_result']);cost,cost_sha,_=descriptor(proof['cost'])
        require(str(result['job_id'])==job and result['code_commit']==HISTORICAL_SOURCE
                and result['cleanup_residual']=={} and result['original_eight_pid_start_times_preserved'] is True
                and str(cost['job_id'])==job,'Historical native run-result/cost binding differs: '+job)
        item=dict(path=pin['path'],sha256=digest,job_id=int(job),code_commit=HISTORICAL_SOURCE,
            run_result_sha256=result_sha,cost_sha256=cost_sha,scheduler_state=proof['scheduler_state'],scheduler_exit_code=proof['scheduler_exit_code'],
            actual_optimizer_committed=result.get('optimizer_committed'),scheduler_advanced=result.get('scheduler_advanced'),
            failed_unrecorded_partial_optimizer_writes_unknown=result.get('failed_unrecorded_partial_optimizer_writes_unknown'))
        if job=='3322':
            captured=proof['positive_scheduler_observation'];fields=captured['fields']
            require(re.fullmatch('[0-9a-f]{64}',captured['observation_sha256'])
                    and captured['returncode']==0 and captured['stderr']==''
                    and captured['raw_stdout_retained'] is False
                    and all(fields.get(key)==value for key,value in dict(JobId='3322',JobState='FAILED',ExitCode='1:0',
                        EndTime='2026-10-07T15:26:24',RunTime='00:47:52').items()),
                    'Retained positive3322 parsed scheduler observation differs; missing raw stdout is never synthesized')
            item['positive_scheduler_observation']=captured
        if job in HISTORICAL_JOURNALS:
            journal_name,prior_manifest_sha=HISTORICAL_JOURNALS[job]
            journal,journal_sha,_=descriptor(proof['once_journal'])
            require(proof['once_journal']['path']==str(ROOT/'instruct300'/journal_name)
                    and journal['status']=='SUBMITTED' and journal['job_id']==int(job) and journal['cluster_name']==CLUSTER
                    and journal['manifest_sha256']==prior_manifest_sha,
                    'Historical proof must bind exact submitted once journal and original source manifest: '+job)
            item.update(once_journal=dict(path=proof['once_journal']['path'],sha256=journal_sha),prior_manifest_sha256=prior_manifest_sha)
        closure[job]=item
    return closure['3322'],closure

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
    predecessor,proof_closure=predecessor_ended(manifest,predecessor_proof,predecessor_proof_sha)
    lock_path=ROOT/'instruct300/FRESH200_TRAIN_DISPATCH.lock'
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Dispatch lock must be ordinary file')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=prior_journals_clear(path,run,environment,proof_closure);queue=queue_clear(run,environment)
        # The immutable bytes below, not the filename, are the submitted program.
        submit_argv=['sbatch','--parsable']
        initial=dict(status='SUBMITTING',execution_mode=DIRECT_MODE,independent_acceptance=False,training_accepted=False,
            production_thresholds_accepted=False,original_sampler_actor_gate_accepted=False,
            cluster_name=CLUSTER,local_cluster_identity=identity,node_list=NODE,
            manifest_sha256=manifest_sha,review_sha256=review_sha,dispatch_intent_path=str(path),train_wall_hours=wall,
            dispatch_script_sha256=script_sha,submit_argv=submit_argv,
            submitted_input_sha256=hashlib.sha256(script).hexdigest(),submitted_input_bytes=len(script),
            submitted_input_base64=base64.b64encode(script).decode('ascii'),submit_mode='verified_bytes_via_stdin',
            selected_script_path=str(script_path),production_input_pins=pins,prior_accounting=prior,
            predecessor_terminal_cleanup=predecessor,terminal_proof_closure=proof_closure,
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

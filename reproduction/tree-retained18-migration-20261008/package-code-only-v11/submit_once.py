"""One authorized same-cluster retirement migration submission; no retry path.

Missing retired3333 is recorded as unknown, never a positive terminal/cleanup
proof. The current controller must have neither the retired node nor an active
Tree GPU job. Foreign allocations may remain: Slurm must schedule the new job.
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
from production_contract import ROOT, SOURCE, CHAIN_SCOPE, DIRECT_MODE, require, read_bound, production_inputs
from slurm_identity import CLUSTER, NODE, local_environment, cluster_identity, query, create_journal, finalize_journal, verify_management_scope, timeout_output
from build_package_draft import read_regular_bound


def retirement_observation(run, environment):
    nodes = query(['sinfo','--local','-N','-h','-o','%N'],run,environment,management=True)
    require('h100-9' not in nodes.splitlines(), 'Retired original node is still registered')
    queue = query(['squeue','--local','-h','-o','%i|%j|%T|%b'],run,environment,management=True)
    for line in queue.splitlines():
        parts=line.split('|'); require(len(parts)==4,'Ambiguous controller queue')
        require(parts[0]!='3333','Previous3333 still has an active handle')
        if parts[1].startswith('tree-'):
            require(parts[3] in ('gres/gpu:0','gres/gpu=0','gpu:0'),'Another active/pending Tree GPU job exists')
    verify_management_scope(environment)
    result=run(['scontrol','--local','show','job','3333'],env=environment,text=True,capture_output=True,timeout=30)
    require(result.returncode!=0 and not result.stdout.strip()
            and result.stderr.strip()=='slurm_load_jobs error: Invalid job id specified',
            'Retired3333 handle observation differs; no submission')
    return dict(status='RETIRED_ORIGINAL_NODE_AND_JOB_HANDLE_ABSENT',observed_epoch=time.time(),
        original_job_id=3333,original_node='h100-9',original_ip='10.33.0.13',
        nodes_stdout=nodes,queue_stdout=queue,job_stdout=result.stdout,job_stderr=result.stderr,
        job_returncode=result.returncode,original_terminal_state=None,original_cleanup_verified=False,
        unpreserved_post_rolling4_updates_unknown=True,missing_handle_interpreted_as_terminal=False)


def failed_predecessor_observation(manifest, run, environment):
    pin=manifest['failed_predecessor3334']
    evidence,_,_=read_bound(pin['path'],pin['sha256'])
    require(evidence['job_id']==3334 and evidence['scheduler_state']=='FAILED'
        and evidence['native_parent_loaded'] is False and evidence['new_rank_records']==0
        and evidence['own_cgroup_live_residual_empty'] is True,
        'Actual failed3334 startup/cleanup proof required')
    for name in ('run-result.json','cost.json'):
        entry=evidence['files'][name]
        value,_,_=read_bound(entry['path'],entry['sha256'])
        if name=='run-result.json':
            require(value['status']=='FAILED' and value['cleanup_residual']=={}
                and value['own_cgroup_live_residual_empty'] is True,'3334 cleanup changed')
        else:require(value['exit_code']==1,'3334 termination exit differs')
    previous=ROOT/'instruct300/SUBMISSION_ONCE-retained18-current200-v10-6efb604.json'
    journal,_,_=read_bound(previous)
    require(journal['status']=='SUBMITTED' and journal['job_id']==3334
        and journal['manifest_sha256']==evidence['manifest_sha256'],
        'Previous once journal differs from actual failed3334')
    require(set(previous.parent.glob('SUBMISSION_ONCE-retained18-current200-*.json'))=={previous},
        'An additional retained18 submission intent exists; do not submit')
    return evidence


def submit(args, run=subprocess.run):
    verify_management_scope(os.environ,socket.gethostname())
    environment=local_environment(os.environ)
    identity=cluster_identity(run,environment,management=True)
    package=Path(args.package)
    require(package.is_absolute() and package.resolve().is_relative_to((ROOT/'instruct300/candidates').resolve()),'Own migrated package required')
    manifest,_,_=read_bound(package/'manifest.json',args.manifest_sha)
    require(manifest['packet_revision']==11 and manifest['code_commit']==SOURCE
        and manifest['dispatch']['nodes']==[NODE] and manifest['dispatch']['gpus']==8,
        'Exact retained18 package required')
    _,pins=production_inputs(manifest,manifest_sha256=args.manifest_sha,
        dispatch_review_descriptor=dict(path=args.release_path,sha256=args.release_sha))
    require(Path(__file__).resolve()==(package/'submit_once.py').resolve(),'Submitted dispatcher differs')
    for name,expected in manifest['files'].items():
        relative=Path(name); require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe package member')
        read_regular_bound(package/relative,expected)
    from current_chain_runtime import parent_metadata
    parent_metadata(manifest)
    copy,copy_sha,_=read_bound(ROOT/'.node-migration/result.json')
    require(copy['status']=='LOCAL_ASSETS_AND_PARENT_FILES_COPIED_VERIFIED'
        and copy['parent_manifest_sha256']==manifest['current_chain_resume']['parent_manifest']['sha256'],
        'Own native asset copy is incomplete')
    storage,storage_sha,_=read_bound(ROOT/'.node-migration/output-storage-verified.json',manifest['retained18_migration']['storage_receipt_sha256'])
    require(storage['shared_read_verified'] is True and storage['server_ip']=='10.33.0.18'
        and storage['server_retained'] is True,'Actual retained shared output storage required')
    from render_dispatch import render
    script=read_regular_bound(args.script,args.script_sha)
    require(script==render(package,args.manifest_sha,args.release_path,args.release_sha,168).encode(),'Exact dispatch script differs')
    path=Path(manifest['dispatch']['once_intent'])
    require(path.parent==ROOT/'instruct300' and re.fullmatch(r'SUBMISSION_ONCE-retained18-current200-v11-[0-9a-f]{7}\.json',path.name),'New retirement intent name differs')
    lock=ROOT/'instruct300/FRESH200_TRAIN_DISPATCH.lock'
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'Own dispatch lock is not ordinary')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not path.exists() and not path.is_symlink(),'Existing intent: never retry UNKNOWN or submitted')
        failed=failed_predecessor_observation(manifest,run,environment)
        observed=retirement_observation(run,environment)
        initial=dict(status='SUBMITTING',execution_mode=DIRECT_MODE,scope=CHAIN_SCOPE,
            current_chain_resume=manifest['current_chain_resume'],code_commit=SOURCE,
            independent_acceptance=False,training_accepted=False,production_thresholds_accepted=False,
            original_sampler_actor_gate_accepted=False,cluster_name=CLUSTER,node_list=NODE,
            local_cluster_identity=identity,manifest_sha256=args.manifest_sha,review_sha256=args.release_sha,
            dispatch_intent_path=str(path),train_wall_hours=168,dispatch_script_sha256=args.script_sha,
            submit_argv=['sbatch','--parsable'],submitted_input_sha256=hashlib.sha256(script).hexdigest(),
            submitted_input_bytes=len(script),submitted_input_base64=base64.b64encode(script).decode(),
            submit_mode='verified_bytes_via_stdin',production_input_pins=pins,
            migration_copy_receipt_sha256=copy_sha,shared_storage_receipt_sha256=storage_sha,
            retired_original_observation=observed,failed_predecessor_observation=failed,foreign_jobs_modified=False,
            start_epoch=time.time(),stdout=None,stderr=None,job_id=None)
        create_journal(path,initial)
        result=None
        try:
            result=run(['sbatch','--parsable'],input=script,env=environment,capture_output=True,timeout=60)
            out=result.stdout.decode('utf-8');err=result.stderr.decode('utf-8')
            require(result.returncode==0 and err=='' and re.fullmatch(r'[1-9][0-9]*(;wuhan-h100)?\n?',out),'Submission response unknown/ambiguous')
            parsable=out.rstrip('\n');job=int(parsable.split(';')[0])
            final=dict(initial,status='SUBMITTED',job_id=job,run_id='tree-current200-continue-'+str(job),
                stdout=out,stderr=err,returncode=0,parsable_stdout=parsable,end_epoch=time.time())
        except BaseException as error:
            final=dict(initial,status='UNKNOWN',error_type=type(error).__name__,error=str(error),end_epoch=time.time(),
                submitted_returncode=None if result is None else result.returncode,
                submitted_result_stdout_raw=timeout_output(None if result is None else result.stdout),
                submitted_result_stderr_raw=timeout_output(None if result is None else result.stderr))
            finalize_journal(path,final)
            raise
        finalize_journal(path,final)
        return final
    finally:
        os.close(fd)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('package','manifest-sha','release-path','release-sha','script','script-sha'):
        p.add_argument('--'+key,required=True)
    print(json.dumps(submit(p.parse_args()),sort_keys=True))

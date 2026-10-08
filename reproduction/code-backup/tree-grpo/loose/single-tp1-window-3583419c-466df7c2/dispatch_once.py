"""At most one sbatch attempt; uncertain outcome is never an automatic retry.

Candidate only. Human/root must independently release the concrete package.
"""
import argparse
import fcntl
import getpass
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from probe_package import (ROOT,atomic_json,canonical,digest,read_bound,require,same_node_path,
                           validate_package_files,validate_release)


def command(package,manifest_sha,release,release_sha):
    return ['sbatch','--parsable','-p','gpu','-w','h100-9','-N','1','--ntasks=1','--gres=gpu:8',
            '--exclusive','--cpus-per-task=32','--mem=256G','--time=00:30:00','--no-requeue',
            '--job-name=tree-single-tp1-window','--output='+str(ROOT/'logs/tree-probability-probe-%j.out'),
            str(Path(package)/'start.sh'),'--package',str(package),'--manifest-sha256',manifest_sha,
            '--release',str(release),'--release-sha256',release_sha]


def job_gpu_count(text):
    counts=[int(value) for value in re.findall(r'gres/gpu(?::[A-Za-z0-9_-]+)?=(\d+)',text)]
    counts.extend(int(value) for value in re.findall(r'gres:gpu(?::[A-Za-z0-9_-]+)?:(\d+)',text))
    return max(counts or [0])


def queue_conflicts(rows,details,expand):
    """Actual job metadata separates fixed other-node work from h100-9 risk."""
    conflicts=[]
    for row in rows:
        fields=row.split('|');require(len(fields)==5,'Malformed actual squeue metadata')
        job,name,state,nodes,gres=(field.strip() for field in fields)
        if not name.startswith('tree-'):continue
        require(re.fullmatch('[1-9][0-9]*',job),'Unresolved tree- array/heterogeneous identity; inspect before dispatch')
        text=details(job)
        require(re.search(r'(?:^|\s)JobId='+job+r'(?:\s|$)',text),'scontrol identity differs')
        count=job_gpu_count(text)
        # A nonempty requested GRES without parsed GPU count is uncertain, never assumedCPU0.
        if count==0 and 'gpu' in gres.lower() and not re.search(r'gpu(?::[^: ]+)?[:=]0(?:\s|$)',gres):
            raise ValueError('Unknown GPU budget of another tree- job')
        if count==0:continue
        requested=re.search(r'(?:^|\s)ReqNodeList=(\S+)',text)
        actual=re.search(r'(?:^|\s)NodeList=(\S+)',text)
        node_sets=[]
        for match in (actual,requested):
            if match and match.group(1) not in ('(null)','(None)','None','N/A'):
                expanded=expand(match.group(1))
                require(type(expanded) is list and expanded and all(type(node) is str and node.strip() and node==node.strip() for node in expanded),
                        'Unknown/empty/blank expanded node identity; refuse duplicate GPU dispatch')
                node_sets.append(set(expanded))
        unknown_target=not node_sets
        if unknown_target or any('h100-9' in group for group in node_sets):
            conflicts.append(dict(job_id=job,name=name,state=state,gpu_count=count,requested_or_actual_node_unknown=unknown_target,
                                  original_queue_row=row,scontrol_text=text))
    return conflicts


def refreshed_queue_gate(intent):
    argv=['squeue','--noheader','--user',getpass.getuser(),'--format=%i|%j|%T|%N|%b']
    raw=None;details={};conflicts=None;error=None
    def detail(job):
        text=subprocess.check_output(['scontrol','show','job',job],text=True,timeout=15)
        details[job]=text;return text
    try:
        raw=subprocess.check_output(argv,text=True,timeout=15)
        rows=[line for line in raw.splitlines() if line.strip()]
        conflicts=queue_conflicts(rows,detail,lambda nodes:subprocess.check_output(
            ['scontrol','show','hostnames',nodes],text=True,timeout=15).splitlines())
    except BaseException as failure:error=failure
    target=Path(str(intent)+'.queue-'+str(time.time_ns())+'.json')
    proof=atomic_json(target,dict(epoch=time.time(),host=socket.gethostname(),command=argv,raw_squeue=raw,
        scontrol=details,conflicts=conflicts,status='READ_ONLY_REFRESHED_QUEUE_BEFORE_SINGLE_ATTEMPT',
        unknown_error=repr(error) if error is not None else None,
        foreign_jobs_cancelled=False))
    require(error is None,'Current queue unknown; do not submit. Evidence: '+proof['path'])
    require(not conflicts,'Existing/pending tree- GPU job could use h100-9; do not submit another instance. Queue evidence: '+proof['path'])
    return proof


def submit_once(intent_path,argv,identity,submit,pre_submit=None):
    """CPU-testable durable single-attempt state; submit is invoked exactly once.

    Keep the original intent and unknown receipt, even if the scheduler later
    reveals no job. An operator must explicitly reconcile that evidence. This
    helper offers no reset/resume/retry API and never sends another sbatch.
    """
    path=Path(intent_path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not path.exists(),'Submission already attempted; inspect existing intent, never retry automatically')
        queue=pre_submit() if pre_submit is not None else None
        intent=dict(status='SUBMISSION_ATTEMPTED_OUTCOME_UNKNOWN',epoch=time.time(),
                    command=argv,command_sha256=digest(argv),identity=identity,automatic_retry_allowed=False,
                    refreshed_queue_evidence=queue)
        # This durable receipt precedes the potentially ambiguous external side effect.
        atomic_json(path,intent)
        result_path=path.with_suffix(path.suffix+'.result.json')
        try:
            answer=submit(argv)
            require(answer.returncode==0,'sbatch failed; outcome conservatively unknown')
            output=answer.stdout.strip()
            require(re.fullmatch('[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?',output),'Ambiguous/malformed sbatch response')
            result=dict(status='SLURM_SUBMISSION_CONFIRMED_NO_EXECUTION_ACCEPTANCE',job_id=output.split(';')[0],
                        scheduler_response=output,stderr=answer.stderr,identity=identity,command_sha256=digest(argv),
                        epoch=time.time(),automatic_retry_allowed=False,actual_gpu_probe_accepted=False)
            atomic_json(result_path,result);return result
        except BaseException as error:
            # Even nonzero/timeout/interrupt may have submitted remotely; no assumption of stopped/not submitted.
            unknown=dict(status='SUBMISSION_OUTCOME_UNKNOWN_DO_NOT_RETRY',identity=identity,command_sha256=digest(argv),
                         error_type=type(error).__name__,error=str(error),epoch=time.time(),automatic_retry_allowed=False)
            try:atomic_json(result_path,unknown)
            except BaseException:pass  # Durable original intent already guarantees refusal.
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--release',required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--intent',required=True)
    a=p.parse_args();package=same_node_path(a.package);release_path=same_node_path(a.release)
    # Single h100 cluster management connection; this module never SSH/routes elsewhere.
    require(socket.gethostname().split('.')[0]=='h100-9','This diagnostic dispatch is restricted to h100-9')
    manifest=read_bound(package/'package-manifest.json',a.manifest_sha256);validate_package_files(package,manifest)
    template=read_bound(package/'bundle-template.json',manifest['files']['bundle-template.json'])
    release=read_bound(release_path,a.release_sha256);validate_release(manifest,a.manifest_sha256,template,release)
    require(same_node_path(a.intent)==same_node_path(release['dispatch_intent_path']),
            'Do not bypass fixed single-attempt intent with another path')
    argv=command(package,a.manifest_sha256,release_path,a.release_sha256)
    identity=dict(package_manifest_sha256=a.manifest_sha256,release_sha256=a.release_sha256,
                  bundle_payload_sha256=manifest['bundle_payload_sha256'],selection_sha256=manifest['selection_sha256'],
                  runner_commit=manifest['runner_commit'],diagnostic_profile=manifest['diagnostic_profile'],
                  window_inputs_sha256=manifest['window_inputs_sha256'],window_contract=manifest['window_contract'])
    result=submit_once(same_node_path(a.intent),argv,identity,
        lambda command:subprocess.run(command,text=True,capture_output=True,timeout=30,check=False),
        pre_submit=lambda:refreshed_queue_gate(same_node_path(a.intent)))
    print(canonical(result).decode())


if __name__=='__main__':main()

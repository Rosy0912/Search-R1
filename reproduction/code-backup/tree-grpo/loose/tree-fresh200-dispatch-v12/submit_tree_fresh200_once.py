"""One new reviewed fresh zero-write job; unknown submission is never retried."""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
import types
from contextlib import contextmanager
import stat
import base64

LOCAL_CLUSTER = 'wuhan-h100'


def local_slurm_environment():
    """Discard inherited options before contacting the existing local controller."""
    exact = {'SLURM_CLUSTERS', 'SLURM_CONF', 'SLURM_CONF_SERVER',
             'SLURM_CLUSTER_NAME', 'SLURM_FEDERATION'}
    prefixes = ('SBATCH_', 'SQUEUE_', 'SACCT_', 'SCONTROL_')
    return {key: value for key, value in os.environ.items()
            if key not in exact and not key.startswith(prefixes)}


def verify_local_cluster(c, run, env):
    result = run(['scontrol', '--local', 'show', 'config'], env=env,
                 text=True, capture_output=True, timeout=30)
    c.require(result.returncode == 0 and not result.stderr.strip(),
              'Local Slurm configuration inquiry failed or unknown')
    names = re.findall(r'^ClusterName\s*=\s*([A-Za-z0-9_.-]+)\s*$', result.stdout, re.MULTILINE)
    c.require(names == [LOCAL_CLUSTER], 'Local Slurm cluster differs from pinned h100 cluster')
    federation = run(['scontrol', '--local', 'show', 'federation'], env=env,
                     text=True, capture_output=True, timeout=30)
    c.require(federation.returncode == 0 and not federation.stderr.strip()
              and not federation.stdout.strip(),
              'Local federation is configured or unknown; no sibling submission allowed')
    return dict(cluster_name=LOCAL_CLUSTER, raw_stdout=result.stdout, raw_stderr=result.stderr,
                federation_stdout=federation.stdout, federation_stderr=federation.stderr)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def finish_journal(journal, intent):
    """A failed final write leaves the original durable SUBMITTING intent."""
    temporary = journal.with_name(journal.name + '.final-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(intent, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, journal)
    fsync_directory(journal.parent)


def exception_output(value):
    if isinstance(value, bytes):
        return dict(encoding='base64', data=base64.b64encode(value).decode('ascii'))
    if value is None or isinstance(value, str): return value
    return dict(unexpected_type=type(value).__name__, representation=repr(value))


def load_common(review_path,review_sha):
    raw=Path(review_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=review_sha:raise ValueError('Helper source-review SHA differs')
    review=json.loads(raw)
    if review['status']!='FRESH200_PREPARE_AND_SUBMIT_HELPERS_SOURCE_CPU_REVIEWED':raise ValueError('Missing independent helper source review')
    if review['actual_gpu_accepted'] is not False:raise ValueError('Source receipt cannot imply GPU acceptance')
    here=Path(__file__).resolve().parent
    for name in ('prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py','fresh200_dispatch_common.py'):
        content=(here/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=review['helper_files'][name]:raise ValueError('Helper bytes changed: '+name)
        if name=='fresh200_dispatch_common.py':common_raw=content
    module=types.ModuleType('fresh200_dispatch_common');module.__file__=str(here/'fresh200_dispatch_common.py')
    exec(compile(common_raw,module.__file__,'exec'),module.__dict__);module.local_path(review_path)
    return module


def validate_dispatch(c,target,manifest,review,prep,inputs,preflight,review_refs):
    c.require(review['status']=='READY_FOR_FRESH200_BEHAVIOR_MU_ZERO_WRITE_CALIBRATION'
        and review['manifest_sha256']==review_refs['manifest'] and review['code_commit']==c.COMMIT,'Missing exact independent zero-write release')
    c.require(review['submit_helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Submit helper identity changed')
    for key in ('actual_3b_training_accepted','long_training_ready','original_sampler_actor_gate_accepted','production_thresholds_accepted'):
        c.require(review[key] is False,'Zero-write release cannot imply training/production acceptance')
    c.require(review['requires_zero_optimizer_writes'] is True and review['requires_actual_behavior_mu_binding'] is True,
              'Release lacks zero-write/actual-mu constraints')
    c.require(inputs['manifest_sha256']==review_refs['manifest'] and inputs['code_commit']==c.COMMIT
              and inputs['stage']==c.STAGE and inputs['target']==str(target),'Submission inputs differ from prepared package')
    previous = inputs.get('previous_terminal_evidence', {})
    c.require(type(previous) is dict and review.get('previous_terminal_evidence', {}) == previous,
              'Prospective prior-terminal SHA references are not bound by independent release')
    for job,entry in previous.items():
        c.require(type(job) is str and re.fullmatch(r'[1-9][0-9]*',job) is not None
                  and type(entry) is dict and set(entry) == {'path','sha256'},
                  'Prior-terminal proof reference must bind one exact numeric job/path/SHA')
        c.local_path(entry['path']); c.valid_sha(entry['sha256'])
    c.require(review['source_reviews']==prep['source_reviews']==c.validate_reviews(inputs,manifest,for_dispatch=True),'Source-review bindings differ')
    c.require(review['preparation_sha256']==review_refs['preparation'] and review['tokenizer_preflight_sha256']==review_refs['preflight']
              and review['inputs_sha256']==review_refs['inputs'] and review['helper_review_sha256']==review_refs['helper_review'],
              'Actual prep/preflight/helper/input receipts not bound by final release')
    c.require(prep['status']=='SAME_NODE_FRESH200_CODE_ASSETS_AND_CPU_CONFIG_VERIFIED_NOT_SUBMITTED'
              and prep['package']==str(target) and prep['manifest_sha256']==review_refs['manifest'] and prep['code_commit']==c.COMMIT
              and prep['inputs_sha256']==review_refs['inputs'] and prep['helper_review_sha256']==review_refs['helper_review']
              and prep['actual_hydra_resolved'] is True and prep['actual_gpu_accepted'] is False and prep['submitted'] is False
              and prep['optimizer_writes']==0 and prep['cuda_initialized'] is False,'Missing actual same-node CPU preparation')
    c.require(prep['resolved_config_sha256']==c.file_sha(prep['resolved_config_path'])
              and review['resolved_config_sha256']==prep['resolved_config_sha256'],'Actual resolved config changed')
    c.require(preflight['status']=='HISTORICAL_RAW_CPU_PREFLIGHT_OBSERVED_NOT_FRESH_TRAINING_ACCEPTED'
        and preflight['model_weight_loads']==0 and preflight['tensor_asset_loads']==0 and preflight['generation_calls']==0
        and preflight['optimizer_writes']==0 and preflight['cuda_initialized'] is False
        and preflight['original_sampler_actor_gate_accepted'] is False and preflight['fresh_tree_rollout_accepted'] is False,
        'Historical tokenizer preflight cannot imply actual fresh rollout acceptance')
    c.require(type(preflight['rows']) is int and preflight['rows']>0 and preflight['rows']==review['tokenizer_preflight_rows'],
              'Historical tokenizer report count differs from independent assessed scope')
    c.require(review['historical_preflight_limitations_accepted_for_zero_write_stage'] is True,
              'Independent release must explicitly assess historical preflight limitations')
    c.require(preflight['raw_helper_sha256']==manifest['files']['source/repo/search_r1/baselines/tree_raw_tokens.py'],
              'Tokenizer preflight used another raw token helper')
    c.require(preflight['tokenizer_identity']['tokenizer_json_sha256']==manifest['model_identity']['files']['tokenizer.json'],
              'Tokenizer identity changed')
    c.verify_package(target,manifest)
    c.verify_local_assets(manifest)


@contextmanager
def task_submit_lock(c):
    path=c.ROOT/'instruct300/TREE_FRESH200_SUBMIT.lock'
    fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        c.require(stat.S_ISREG(os.fstat(fd).st_mode),'Submission lock must be a regular file')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another Tree submission holds the whole-task lock')
        yield
    finally:os.close(fd)


def accounting_disabled(result):
    """Only the explicit disabled-accounting diagnostic permits a live fallback."""
    if result.returncode == 0 or not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
        return False
    diagnostic = (result.stdout + '\n' + result.stderr).strip()
    return re.fullmatch(r'(?:sacct: error: )?Slurm accounting storage is disabled\.?', diagnostic) is not None


def read_failed_zero_run(c, prior, job, allocation):
    """Read only the existing supervisor's fixed own-run path, on one regular FD."""
    path = c.local_path(c.ROOT/'runs'/('tree-fresh200-'+str(job))/'run-result.json')
    c.require(path.parent.resolve() == c.ROOT.resolve()/'runs'/('tree-fresh200-'+str(job)),
              'Prior own failed-run path aliases another run')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        c.require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 64*1024,
                  'Prior own failed-run receipt must be bounded regular metadata')
        raw = stream.read(); after = os.fstat(stream.fileno())
        c.require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                  (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                  'Prior own failed-run receipt changed during read')
    value = json.loads(raw)
    c.require(value.get('status') == 'FAILED' and value.get('stage') == c.STAGE
              and value.get('job_id') == str(job) and value.get('code_commit') == prior['code_commit']
              and value.get('actual_training_accepted') is False,
              'Prior own failed-run job/source/stage/result differs')
    c.require(type(value.get('allocated_gpus')) is int and value['allocated_gpus'] == 8
              and type(value.get('requested_cpus')) is int and value['requested_cpus'] == 32
              and type(value.get('allocated_cpus')) is int and value['allocated_cpus'] == int(allocation['NumCPUs'])
              and type(value.get('cleanup_residual')) is dict and value['cleanup_residual'] == {}
              and value.get('original_eight_pid_start_times_preserved') is True,
              'Prior own failed-run allocation/cleanup/protected identities are incomplete')
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),status=value['status'],
                job_id=value['job_id'],code_commit=value['code_commit'],cleanup_residual={},
                original_eight_pid_start_times_preserved=True)


def live_failed_zero_terminal(c, prior, job, run, env, journal_sha, previous_terminal_evidence):
    """Disabled sacct is never terminal evidence; require new local controller data."""
    c.require(prior.get('stage') == c.STAGE and type(prior.get('initialized_parent_outer')) is int
              and prior['initialized_parent_outer'] == 0 and type(prior.get('expected_optimizer_writes')) is int
              and prior['expected_optimizer_writes'] == 0 and prior.get('old_checkpoint_inherited') is False
              and re.fullmatch(r'[0-9a-f]{40}',prior.get('code_commit','')) is not None
              and prior.get('actual_submit_stdin_sha256') == c.WRAPPER_HASHES['launch.sbatch'],
              'Prior journal is not the known own fresh zero-stage launch')
    launch = c.local_path(prior['reviewed_launch_path_provenance_only'])
    c.require(launch.name == 'launch.sbatch' and launch.parent.parent == c.ROOT/'instruct300/candidates',
              'Prior launch does not bind the own fresh candidate scope')
    result = run(['scontrol','--local','show','job',str(job)],env=env,
                 text=True,capture_output=True,timeout=30)
    expired = (result.returncode != 0 and result.stdout == ''
               and isinstance(result.stderr,str)
               and result.stderr.strip() == 'slurm_load_jobs error: Invalid job id specified')
    retained = None
    if expired:
        entry = previous_terminal_evidence.get(str(job))
        c.require(type(entry) is dict and set(entry) == {'path','sha256'},
                  'Expired prior job requires an independently SHA-bound positive terminal proof')
        retained = c.json_bound(entry)
        c.require(retained.get('status') == 'RETAINED_POSITIVE_OWN_ZERO_TERMINAL_PROOF'
                  and type(retained.get('job_id')) is int and retained['job_id'] == job
                  and retained.get('cluster_name') == LOCAL_CLUSTER and retained.get('stage') == c.STAGE
                  and retained.get('code_commit') == prior['code_commit']
                  and retained.get('manifest_sha256') == prior['manifest_sha256']
                  and retained.get('journal_sha256') == journal_sha,
                  'Retained positive terminal proof differs from the exact prior journal/source/job')
        old_queue = retained.get('squeue', {})
        c.require(old_queue.get('returncode') == 0 and old_queue.get('stdout') == ''
                  and old_queue.get('stderr') == '', 'Retained prior terminal capture lacks empty own queue')
        live_queue = run(['squeue','--local','-h','-w','h100-9','-o','%i|%j|%T|%N'],env=env,
                         text=True,capture_output=True,timeout=30)
        c.require(live_queue.returncode == 0 and isinstance(live_queue.stdout,str)
                  and isinstance(live_queue.stderr,str) and not live_queue.stderr.strip(),
                  'Current global own-node queue is unavailable or unknown')
        c.queue_clear(live_queue.stdout)
        c.require(all(line.split('|')[0].strip() != str(job) for line in live_queue.stdout.splitlines()),
                  'Expired prior job is present in the current queue')
        current_result = result
        control = retained.get('scontrol', {})
        c.require(type(control) is dict and set(control) == {'returncode','stdout','stderr'},
                  'Retained positive control proof has an incomplete response schema')
        result = type('RetainedControl', (), control)()
    c.require(result.returncode == 0 and isinstance(result.stdout,str) and isinstance(result.stderr,str)
              and not result.stderr.strip(),'Prior live local controller state is unavailable or unknown')
    fields = {}
    for key,value in re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_\-/]*)=([^\s]+)',result.stdout):
        c.require(key not in fields,'Ambiguous prior live allocation field: '+key)
        fields[key] = value
    expected = {'JobId':str(job),'JobName':'tree-fresh200-mu-zero','JobState':'FAILED',
                'NodeList':'h100-9','BatchHost':'h100-9','NumNodes':'1','OverSubscribe':'NO',
                'TresPerNode':'gres:gpu:8','CPUs/Task':'32','MinMemoryNode':'256G',
                'TimeLimit':'03:00:00','Requeue':'0','Restarts':'0'}
    c.require(all(fields.get(key) == value for key,value in expected.items())
              and fields.get('Cluster',LOCAL_CLUSTER) == LOCAL_CLUSTER
              and re.fullmatch(r'[1-9][0-9]*',fields.get('NumCPUs','')) is not None
              and int(fields['NumCPUs']) >= 32
              and re.fullmatch(r'[0-9]+:[0-9]+',fields.get('ExitCode','')) is not None
              and fields['ExitCode'] != '0:0',
              'Prior live failed zero-stage job/node/resources/exit identity differs')
    failed_run = read_failed_zero_run(c,prior,job,fields)
    if retained is not None:
        c.require(failed_run['sha256'] == retained.get('run_result_sha256'),
                  'Current own failed-run bytes differ from retained positive terminal proof')
        return dict(cluster_name=LOCAL_CLUSTER,scheduler_state=fields['JobState'],exit_code=fields['ExitCode'],
                    evidence_source='sha_bound_retained_positive_terminal_live_expired_and_empty_queue',
                    retained_positive_proof=entry,retained_positive_scontrol_stdout=result.stdout,
                    current_scontrol_stdout=current_result.stdout,current_scontrol_stderr=current_result.stderr,
                    current_scontrol_returncode=current_result.returncode,
                    current_global_queue_stdout=live_queue.stdout,current_global_queue_stderr=live_queue.stderr,
                    failed_run=failed_run,actual_zero_or_training_accepted=False)
    return dict(cluster_name=LOCAL_CLUSTER,scheduler_state=fields['JobState'],exit_code=fields['ExitCode'],
                evidence_source='new_live_local_scontrol_and_fixed_own_failed_run',
                raw_stdout=result.stdout,raw_stderr=result.stderr,failed_run=failed_run,
                actual_zero_or_training_accepted=False)


def previous_journals_clear(c, current, run, env, previous_terminal_evidence=None):
    """Queue invisibility alone cannot resolve a prior submitted/unknown attempt."""
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED','DEADLINE'}
    previous_terminal_evidence = {} if previous_terminal_evidence is None else previous_terminal_evidence
    c.require(type(previous_terminal_evidence) is dict, 'Unknown prospective prior-terminal proof mapping')
    observed=[]
    for path in sorted((c.ROOT/'instruct300').glob('SUBMISSION_ONCE-fresh200-mu-zero-*.json')):
        if path==current:raise FileExistsError('This intent already exists; no re-attempt')
        raw=path.read_bytes(); prior=json.loads(raw)
        state=prior['status']
        c.require(state=='SUBMITTED','Prior Tree submission result is failed/unknown and needs independent reconciliation before a different intent')
        c.require(prior.get('cluster_name') == LOCAL_CLUSTER,
                  'Prior submitted cluster is missing or differs; independent reconciliation required')
        job=prior['job_id'];c.require(type(job) is int and job>0,'Prior job identity is unknown')
        result=run(['sacct','--local','-X','-n','-P','-j',str(job),'-o','JobIDRaw,Cluster,State,ExitCode'],
                   env=env,text=True,capture_output=True,timeout=30)
        if accounting_disabled(result):
            live = live_failed_zero_terminal(c,prior,job,run,env,hashlib.sha256(raw).hexdigest(),previous_terminal_evidence)
            observed.append(dict(journal=str(path),journal_sha256=hashlib.sha256(raw).hexdigest(),job_id=job,
                                 accounting_disabled_stdout=result.stdout,accounting_disabled_stderr=result.stderr,
                                 **live))
            continue
        c.require(result.returncode==0 and not result.stderr.strip(),'Prior Slurm terminal state is unknown')
        lines=[line.split('|') for line in result.stdout.splitlines() if line.strip()]
        c.require(len(lines)==1 and len(lines[0])==4 and lines[0][0]==str(job) and lines[0][1]==LOCAL_CLUSTER,
                  'Prior Slurm terminal evidence is incomplete/unparseable')
        fields=lines[0]; actual=fields[2].split(' ',1)[0]
        c.require(actual in terminal and re.fullmatch(r'[0-9]+:[0-9]+',fields[3]),'Prior own job is not terminal')
        observed.append(dict(journal=str(path),journal_sha256=hashlib.sha256(raw).hexdigest(),job_id=job,
                             cluster_name=fields[1],scheduler_state=fields[2],exit_code=fields[3],raw_stdout=result.stdout,raw_stderr=result.stderr))
    return observed


def checked_launch_bytes(c,target):
    path=target/'launch.sbatch'
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        c.require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=256*1024,
                  'Reviewed launch must be bounded regular code')
        raw=stream.read();after=os.fstat(stream.fileno())
        c.require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                  (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                  'Launch changed during same-FD reading')
    c.require(hashlib.sha256(raw).hexdigest()==c.WRAPPER_HASHES['launch.sbatch']
              and b'\0' not in raw,'Reviewed launch bytes differ')
    raw.decode('utf-8',errors='strict')
    return raw


def _submit_once_locked(c,target,manifest_sha,review_sha,refs,run):
    launch=checked_launch_bytes(c,target)
    c.protected_processes()
    env=local_slurm_environment()
    local_configuration=verify_local_cluster(c,run,env)
    queue=run(['squeue','--local','-h','-w','h100-9','-o','%i|%j|%T|%N'],env=env,text=True,capture_output=True,timeout=30)
    c.require(queue.returncode==0 and not queue.stderr.strip(),'Queue inquiry failed/unknown')
    c.queue_clear(queue.stdout)
    journal=c.ROOT/'instruct300'/('SUBMISSION_ONCE-fresh200-mu-zero-'+manifest_sha[:12]+'.json')
    previous_terminal=previous_journals_clear(c,journal,run,env,refs.get('previous_terminal_evidence', {}))
    submit_argv=['sbatch','--parsable','--nodelist=h100-9',
        '--output='+str(c.ROOT/'logs/tree-fresh200-mu-zero-%j.out'),
        '--error='+str(c.ROOT/'logs/tree-fresh200-mu-zero-%j.err')]
    intent=dict(status='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN',stage=c.STAGE,code_commit=c.COMMIT,
        epoch=time.time(),manifest_sha256=manifest_sha,review_sha256=review_sha,refs=refs,
        initialized_parent_outer=0,expected_optimizer_writes=0,old_checkpoint_inherited=False,
        previous_terminal_evidence=previous_terminal,cluster_name=LOCAL_CLUSTER,
        local_configuration_evidence=local_configuration,
        reviewed_launch_path_provenance_only=str(target/'launch.sbatch'),
        actual_submit_argv=submit_argv,actual_submit_stdin_sha256=hashlib.sha256(launch).hexdigest(),
        actual_submit_stdin_bytes=len(launch),actual_submit_stdin_base64=base64.b64encode(launch).decode('ascii'))
    # Existence alone blocks every re-attempt, independent of prior status.
    stream=journal.open('x')
    try:
        json.dump(intent,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        fsync_directory(journal.parent)
        env.update(TREE_INSTRUCT_TOOLS=str(target),TREE_INSTRUCT_PACKAGE_SHA256=manifest_sha,
            TREE_INSTRUCT_REVIEW_SHA256=review_sha)
        try:
            proc=run(submit_argv,input=launch,env=env,text=False,capture_output=True,timeout=60)
            intent.update(returncode=proc.returncode,stdout_raw=exception_output(proc.stdout),
                          stderr_raw=exception_output(proc.stderr))
            c.require(type(proc.stdout) is bytes and type(proc.stderr) is bytes,
                      'Byte-mode scheduler stdout/stderr evidence required')
            stdout=proc.stdout.decode('utf-8',errors='strict')
            stderr=proc.stderr.decode('utf-8',errors='strict')
            intent.update(stdout=stdout,stderr=stderr)
            fields=stdout.strip().split(';')
            if (proc.returncode==0 and not stderr.strip()
                    and re.fullmatch(r'[0-9]+(?:;wuhan-h100)?',stdout.strip())
                    and int(fields[0])>0):
                intent.update(status='SUBMITTED',job_id=int(fields[0]),returned_cluster=fields[1] if len(fields)==2 else None)
            elif proc.returncode==0:intent['status']='SUBMISSION_UNKNOWN_DO_NOT_RETRY'
            else:intent['status']='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
        except BaseException as exc:
            intent.update(status='SUBMISSION_UNKNOWN_DO_NOT_RETRY',exception=repr(exc))
            intent['exception_stdout']=exception_output(getattr(exc,'output',None))
            intent['exception_stderr']=exception_output(getattr(exc,'stderr',None))
        # Retain the first attempt and all raw stdout/stderr within this owned journal.
        finish_journal(journal,intent)
    finally:stream.close()
    print(json.dumps(intent,allow_nan=False),flush=True)
    c.require(intent['status']=='SUBMITTED','Submission failed or unknown: intent retained; never retry')
    return intent


def submit_once(c,target,manifest_sha,review_sha,refs,run=subprocess.run):
    with task_submit_lock(c):
        return _submit_once_locked(c,target,manifest_sha,review_sha,refs,run)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('package','manifest-sha','review','review-sha','preparation','preparation-sha','inputs','inputs-sha',
                'tokenizer-preflight','tokenizer-preflight-sha','helper-review','helper-review-sha'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args();c=load_common(args.helper_review,args.helper_review_sha);c.assert_host()
    target=c.local_path(args.package);c.require(target.parent.resolve()==(c.ROOT/'instruct300/candidates').resolve(),'Wrong candidate package path')
    manifest=c.json_bound(dict(path=str(target/'manifest.json'),sha256=args.manifest_sha));c.validate_manifest(manifest)
    review=c.json_bound(dict(path=args.review,sha256=args.review_sha))
    # launch.sbatch itself consumes this exact name; no implicit receipt copy.
    c.require(Path(args.review).resolve()==(target/'PRE_DISPATCH_RECEIPT.json').resolve(),'Release must be present at launch-consumed path')
    prep=c.json_bound(dict(path=args.preparation,sha256=args.preparation_sha))
    inputs=c.json_bound(dict(path=args.inputs,sha256=args.inputs_sha))
    preflight=c.json_bound(dict(path=args.tokenizer_preflight,sha256=args.tokenizer_preflight_sha))
    refs=dict(manifest=args.manifest_sha,preparation=args.preparation_sha,inputs=args.inputs_sha,
              preflight=args.tokenizer_preflight_sha,helper_review=args.helper_review_sha)
    validate_dispatch(c,target,manifest,review,prep,inputs,preflight,refs)
    refs['previous_terminal_evidence'] = inputs.get('previous_terminal_evidence', {})
    submit_once(c,target,args.manifest_sha,args.review_sha,refs)

if __name__=='__main__':main()

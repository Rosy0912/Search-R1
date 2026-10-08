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


def previous_journals_clear(c, current, run):
    """Queue invisibility alone cannot resolve a prior submitted/unknown attempt."""
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED','DEADLINE'}
    observed=[]
    for path in sorted((c.ROOT/'instruct300').glob('SUBMISSION_ONCE-fresh200-mu-zero-*.json')):
        if path==current:raise FileExistsError('This intent already exists; no re-attempt')
        raw=path.read_bytes(); prior=json.loads(raw)
        state=prior['status']
        c.require(state=='SUBMITTED','Prior Tree submission result is failed/unknown and needs independent reconciliation before a different intent')
        job=prior['job_id'];c.require(type(job) is int and job>0,'Prior job identity is unknown')
        result=run(['sacct','-X','-n','-P','-j',str(job),'-o','JobIDRaw,State,ExitCode'],
                   text=True,capture_output=True,timeout=30)
        c.require(result.returncode==0 and not result.stderr.strip(),'Prior Slurm terminal state is unknown')
        lines=[line.split('|') for line in result.stdout.splitlines() if line.strip()]
        c.require(len(lines)==1 and len(lines[0])==3 and lines[0][0]==str(job),
                  'Prior Slurm terminal evidence is incomplete/unparseable')
        fields=lines[0]; actual=fields[1].split(' ',1)[0]
        c.require(actual in terminal and re.fullmatch(r'[0-9]+:[0-9]+',fields[2]),'Prior own job is not terminal')
        observed.append(dict(journal=str(path),journal_sha256=hashlib.sha256(raw).hexdigest(),job_id=job,
                             scheduler_state=fields[1],exit_code=fields[2],raw_stdout=result.stdout,raw_stderr=result.stderr))
    return observed


def _submit_once_locked(c,target,manifest_sha,review_sha,refs,run):
    c.protected_processes()
    queue=run(['squeue','-h','-w','h100-9','-o','%i|%j|%T|%N'],text=True,capture_output=True,timeout=30)
    c.require(queue.returncode==0 and not queue.stderr.strip(),'Queue inquiry failed/unknown')
    c.queue_clear(queue.stdout)
    journal=c.ROOT/'instruct300'/('SUBMISSION_ONCE-fresh200-mu-zero-'+manifest_sha[:12]+'.json')
    previous_terminal=previous_journals_clear(c,journal,run)
    intent=dict(status='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN',stage=c.STAGE,code_commit=c.COMMIT,
        epoch=time.time(),manifest_sha256=manifest_sha,review_sha256=review_sha,refs=refs,
        initialized_parent_outer=0,expected_optimizer_writes=0,old_checkpoint_inherited=False,
        previous_terminal_evidence=previous_terminal)
    # Existence alone blocks every re-attempt, independent of prior status.
    stream=journal.open('x')
    try:
        json.dump(intent,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        env=os.environ.copy();env.update(TREE_INSTRUCT_TOOLS=str(target),TREE_INSTRUCT_PACKAGE_SHA256=manifest_sha,
            TREE_INSTRUCT_REVIEW_SHA256=review_sha)
        try:
            proc=run(['sbatch','--parsable','--output='+str(c.ROOT/'logs/tree-fresh200-mu-zero-%j.out'),
                '--error='+str(c.ROOT/'logs/tree-fresh200-mu-zero-%j.err'),str(target/'launch.sbatch')],
                env=env,text=True,capture_output=True,timeout=60)
            intent.update(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
            fields=proc.stdout.strip().split(';')
            if proc.returncode==0 and re.fullmatch(r'[0-9]+(?:;[A-Za-z0-9_.-]+)?',proc.stdout.strip()) and int(fields[0])>0:
                intent.update(status='SUBMITTED',job_id=int(fields[0]))
            elif proc.returncode==0:intent['status']='SUBMISSION_UNKNOWN_DO_NOT_RETRY'
            else:intent['status']='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
        except BaseException as exc:
            intent.update(status='SUBMISSION_UNKNOWN_DO_NOT_RETRY',exception=repr(exc))
        # Retain the first attempt and all raw stdout/stderr within this owned journal.
        stream.seek(0);stream.truncate();json.dump(intent,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
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
    submit_once(c,target,args.manifest_sha,args.review_sha,refs)

if __name__=='__main__':main()

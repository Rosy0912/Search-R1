#!/usr/bin/env python3
"""UNBOUND candidate. Fixed independent actual-evidence review required; no auto-release."""
import argparse,json,os,tempfile
from pathlib import Path
from first5_helpers import *
INDEPENDENT_REVIEW_SHA=None
REVIEWER_TASK='/root/fresh200_actor_safety'
IDENTITY_KEYS=('run_root','source_commit','global_step','actual_outer_updates','horizon','warmup_steps','config_sha256','pause_nonce','protocol','mini_batch_switch_outer','update_semantics','attempt_accounting','optimizer_opportunities_before_switch','optimizer_opportunities_after_switch')
def validate_review(review,collection,hf,manifest):
    require(review.get('decision')=='PASS_INDEPENDENT_ACTUAL_FRESH200_FIRST5_REVIEW','Actual independent evidence review required')
    require(review.get('reviewer_task_id')==REVIEWER_TASK and review.get('producer_task_id')!=REVIEWER_TASK and bool(review.get('producer_task_id')),'Independent review origin required')
    require(review.get('source_commit')==SOURCE and review.get('run_root')==str(RUN)and type(review.get('job_id'))is int and review['job_id']==JOB,'Review new source/run/job differs')
    expected=dict(collection_sha256=digest(collection),hf_sha256=digest(hf),checkpoint_manifest_sha256=digest(manifest))
    require(review.get('bindings')==expected,'Independent review must bind all actual new evidence bytes')
    require(review.get('whole_200_accepted')is False and review.get('release_scope')=='CONTINUE_SAME_PROCESS_FROM_ACCEPTED_OUTER_6_TO_200','Only same-process continuation authorized')
    require(review.get('mustfix')==[],'Independent mustfix unresolved')
    return expected

def build(collection_path,hf_path,review_path):
    require(INDEPENDENT_REVIEW_SHA is not None,'UNBOUND CANDIDATE: no independent actual first5 review fixed; publishing forbidden')
    bind_source();collection_path=owned(collection_path);hf_path=owned(hf_path);review_path=owned(review_path)
    require(digest(review_path)==INDEPENDENT_REVIEW_SHA,'Fixed independent actual first5 review changed')
    c=json.loads(collection_path.read_text());h=json.loads(hf_path.read_text());review=json.loads(review_path.read_text())
    mp=RUN/'checkpoints/fresh200_state/outer_5_COMPLETE/manifest.json';m=json.loads(mp.read_text())
    validate_review(review,collection_path,hf_path,mp)
    require(c.get('status')=='ACTUAL_FIRST5_COLLECTION_REQUIRES_INDEPENDENT_REVIEW'and h.get('status')=='ACTUAL_CPU_HF_REQUIRES_INDEPENDENT_REVIEW','Actual collection/HF incomplete')
    for value in (c,h):require(value.get('source_commit')==SOURCE and value.get('run_root')==str(RUN)and value.get('job_id')==JOB and value.get('native_training_accepted')is False,'Actual evidence identity differs')
    require(h['collection']==binding(collection_path),'CPU HF bound other collection')
    require(c['manifest']==binding(mp)and c['pause']['manifest_sha256']==digest(mp),'Manifest changed since collection')
    require(h['proof']['hf']['status']=='REAL_CPU_HF_FINITE_FORWARD'and h['proof']['hf']['device']=='cpu'and h['proof']['parameter_delta']['changed']is True,'Actual full CPU HF/change proof absent')
    for key in ('slurm_after_hf','resource_journal_after_cpu_hf'):require(binding(h[key]['path'])==h[key],'PostHF actual raw resource/cost evidence changed')
    live=process_proof(c['pause']['pid'],RUN);require(live['start_ticks']==c['driver_process']['start_ticks']==h['after_process']['start_ticks'],'Original paused driver replaced')
    require(not (RUN/'logs/fresh200-step5-independent-release.json').exists(),'Existing release never overwritten')
    evidence={x['path']:dict(path=x['path'],sha256=x['sha256'])for x in c['evidence']}
    for p in (collection_path,hf_path,review_path,Path(h['slurm_after_hf']['path']),Path(h['resource_journal_after_cpu_hf']['path'])):evidence[str(p)]=dict(path=str(p),sha256=digest(p))
    r={k:m[k]for k in IDENTITY_KEYS};r.update(status='PASS_INDEPENDENT_FRESH200_STEP5_REVIEW',checkpoint_manifest_sha256=digest(mp),train_log_prefix=c['train_log_prefix'],independent_review=dict(path=str(review_path),sha256=digest(review_path)),evidence=list(evidence.values()),job_id=JOB,driver_pid=c['pause']['pid'],release_scope='CONTINUE_SAME_PROCESS_FROM_ACCEPTED_OUTER_6_TO_200',whole_200_accepted=False)
    # Frozen source replays every initial/raw probability/native/driver/attempt binding
    # and actual live log prefix. This temporary candidate is never the watched path.
    from search_r1.baselines.fresh200_gate import validate_release
    fd,path=tempfile.mkstemp(prefix='first5-validation-only-',suffix='.json',dir=str(collection_path.parent))
    try:
        with os.fdopen(fd,'w')as f:json.dump(r,f,allow_nan=False)
        validate_release(path,mp,ROOT/'repo',expected_driver_pid=c['pause']['pid'])
    finally:Path(path).unlink(missing_ok=True)
    return r

def publish_atomic(path,value):
    # The waiting driver must never observe a half-written receipt; hard link
    # installs closed/fsynced bytes without replacing an existing destination.
    fd,tmp=tempfile.mkstemp(prefix='.first5-receipt-',dir=str(Path(path).parent))
    try:
        with os.fdopen(fd,'w')as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
        directory=os.open(str(Path(path).parent),os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:Path(tmp).unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--hf',type=Path,required=True);p.add_argument('--independent-review',type=Path,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    r=build(a.collection,a.hf,a.independent_review)
    if a.publish:publish_atomic(RUN/'logs/fresh200-step5-independent-release.json',r)
    else:print(json.dumps(dict(status='VALIDATED_CANDIDATE_NOT_PUBLISHED',receipt=r),allow_nan=False))
if __name__=='__main__':main()

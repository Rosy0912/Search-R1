"""Finish the actual3356 pending transaction without replaying its writes or stage."""
import copy,json,os
from pathlib import Path
from .aligned_partial_resume import OLD_RUN,OLD_SOURCE,require,exact_file,validate_manifest


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def prepare(trainer,manifest_path):
    import subprocess
    from omegaconf import OmegaConf
    manifest_path=Path(manifest_path)
    run=Path(os.environ['BAPO_RUN_ID'])
    require(run in manifest_path.parents and manifest_path.resolve()==manifest_path,'owned continuation manifest required')
    manifest=json.loads(manifest_path.read_bytes());validate_manifest(manifest)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    require(source==manifest['target_source_commit']==trainer.config.baselines.algorithm_source_commit,'continuation source differs')
    require(not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'dirty continuation source')
    require(str(run)==manifest['target_run_root'] and str(os.environ['SLURM_JOB_ID'])==manifest['target_job_id'],'continuation run/job differs')
    require(trainer.config.get('baseline_aligned',False) and trainer.config.baselines.method=='bapo','only alignedBAPO pending1 supported')
    require(trainer.config.trainer.val_before_train is False,'continuation must skip initial validation')
    directory=run/'partial-resume';directory.mkdir(exist_ok=True)
    witness=directory/'rank-witnesses';witness.mkdir(exist_ok=False)
    trainer.aligned_partial_manifest=manifest;trainer.aligned_partial_witness_directory=witness
    return manifest,witness


def restore_driver_and_commit(trainer):
    import torch,hashlib
    from .full_state import state_digest,restore_rng,rng_state
    from .fresh200_driver import Fresh200Progress
    from .aligned_outer_commit import stage_evidence,validate_commits
    from .fresh200_contract import critical_source_hashes
    m=trainer.aligned_partial_manifest;run=Path(m['target_run_root'])
    p=exact_file(m['driver'],OLD_RUN);report=json.loads(exact_file(m['driver_report'],OLD_RUN).read_bytes())
    require(p==OLD_RUN/'checkpoints/fresh200_state/attempt_1_candidate_outer_1_PARTIAL/driver.pt','exact original driver path required')
    d=torch.load(p,map_location='cpu',weights_only=False)
    require(report['status']=='PARTIAL_CURRENT_STATE_SAVED'and report['sha256']==m['driver']['sha256'] and state_digest(d)==report['state_digest'],'driver original state digest differs')
    require(d['status']=='PARTIAL'and d['source_commit']==OLD_SOURCE and d['driver_pid']==1489246 and d['protocol']=='BAPO_FRESH200_V1','original driver identity differs')
    require(all(d[k]==v and type(d[k])is int for k,v in dict(global_step=1,next_global_step=1,attempt_id=1,actual_outer_updates=0,epoch=0,attempted_outer=1,previous_completed_outer=0).items()),'original pending driver counters differ')
    require(d['rank_write_ledger']==m['rank_write_ledger'] and d['sampler']['cursor']==512,'original write ledger/cursor differs')
    progress=Fresh200Progress(baseline_aligned=True,attempts=1)
    require(progress.state_dict()==d['progress'],'only original unrecorded attempt1 supported')
    stage_binding=m['stage_metrics'];stage_path=exact_file(stage_binding,run)
    parsed=json.loads(stage_path.read_bytes())
    log=exact_file(parsed['original_log'],OLD_RUN);marker='BASELINE_EVALUATION_METRICS_JSON='
    matches=[]
    for line in log.read_text().splitlines():
        if marker in line:
            raw=line.split(marker,1)[1]
            try:value=json.loads(raw)
            except json.JSONDecodeError:continue
            if sum(v for k,v in value.items()if k.startswith('val/evaluated_count/'))==64:matches.append((raw,value))
    require(len(matches)==1 and matches[0][1]==parsed['metrics'] and hashlib.sha256(matches[0][0].encode()).hexdigest()==parsed['marker_line_sha256'],'complete originalstage64 marker differs')
    flat=parsed['metrics'];count=sum(v for k,v in flat.items()if k.startswith('val/evaluated_count/'))
    total=sum(flat[k]*flat['val/evaluated_count/'+k[len('val/test_score/'):]]for k in flat if k.startswith('val/test_score/'))
    score=float(total/count)
    require(count==64 and score==parsed['score']==14/64 and d['bapo']==dict(best=14/64,stalled=0,early=True,patience=5),'already-observed original stage state differs')
    trainer.e1_sampler.load_state_dict(d['sampler'])
    trainer.baseline_controller.stage.load_state_dict(d['bapo'])
    trainer.baseline_controller.actual_outer_updates=0
    trainer.global_steps=1;trainer.e1_epoch=d['epoch'];trainer.fresh200_progress=progress
    iterator=iter(trainer.train_dataloader)
    restore_rng(d['rng'])
    require(state_digest(trainer.e1_sampler.state_dict())==state_digest(d['sampler']) and state_digest(rng_state())==state_digest(d['rng']) and state_digest(trainer.baseline_controller.stage.state_dict())==state_digest(d['bapo']),'driver restored sampler/RNG/BAPO differs')
    w=dict(status='EXACT_ALIGNED_PARTIAL_DRIVER_RELOAD_OK',old_pid=d['driver_pid'],new_pid=os.getpid(),old_run_root=str(OLD_RUN),target_run_root=str(run),old_source_commit=OLD_SOURCE,target_source_commit=m['target_source_commit'],target_job_id=m['target_job_id'],loaded_state_digest=state_digest(d),restored_rng_digest=state_digest(rng_state()),saved_rng_digest=state_digest(d['rng']),restored_sampler_digest=state_digest(trainer.e1_sampler.state_dict()),saved_sampler_digest=state_digest(d['sampler']),restored_bapo_digest=state_digest(trainer.baseline_controller.stage.state_dict()),saved_bapo_digest=state_digest(d['bapo']),scheduler_before=0,completed_outer_before=0,attempt_id=1,sampler_cursor=512,no_generation_or_optimizer_or_stage_observe=True)
    require(w['new_pid']!=w['old_pid'],'original driver process unexpectedly alive')
    write_json(run/'partial-resume/driver-witness.json',w)
    write_json(run/'partial-resume/lineage.json',dict(manifest=m,rank_witnesses=trainer.aligned_partial_rank_witnesses,driver_witness=w))
    trainer.fresh200_rank_updates=copy.deepcopy(m['rank_write_ledger'])
    # Original prepared events/EM keep their original identities and immutable paths.
    metrics={'actor/optimizer_steps':20,'bapo/stage_validation':score,'bapo/stage_evaluated_count':64,
             'fresh200/recovered_existing_adam_writes_per_rank':20,
             'fresh200/recovery_additional_adam_writes':0,'fresh200/reused_original_stage64':True}
    ems=[x for x in m['original_closure']if x['path'].endswith('/training-em-audit/outer_0001-attempt_0001.jsonl')]
    require(len(ems)==1,'original trainingEM closure absent')
    em=exact_file(ems[0],OLD_RUN)
    records=[json.loads(line)for line in em.read_text().splitlines()]
    require(records[-1]['record_type']=='summary','original trainingEM summary missing')
    metrics.update(records[-1]['metrics'])
    stage=stage_evidence(1,1,metrics,bindings=dict(run_root=str(run),driver_pid=os.getpid(),source_commit=m['target_source_commit'],source_sha256=critical_source_hashes(),common_input_binding=trainer.fresh200_shared_input_bindings['common_input_binding'],stage_input_binding=trainer.fresh200_shared_input_bindings['stage'],order_manifest=trainer.config.baselines.alignment.order_manifest,order_manifest_sha256=trainer.config.baselines.alignment.order_sha256,recovery_original_stage=stage_binding,no_stage_observe_repeated=True))
    write_json(run/'aligned-stage-proof/outer_0001-attempt_0001.json',stage)
    # The only state transition after exact restoration: commit scheduler once.
    try:
        reports=trainer.actor_rollout_wg.commit_aligned_outer(1,1,stage)
        trainer.fresh200_rank_updates=validate_commits(reports,1,1,stage)
        metrics.update({'actor/lr':reports[0]['event']['scheduler_lr_after'][0],
            'actor/scheduler_outer_steps':1,'fresh200/outer_commit_phase':'STAGE_COMMITTED',
            'fresh200/stage_proof_sha256':stage['sha256']})
        trainer.baseline_controller.after_actor(None,metrics)
        require(trainer._fresh200_record_attempt(trainer.fresh200_rank_updates,metrics),'recovered pending transaction not accepted')
    except Exception:
        trainer._fresh200_fault_backup(trainer.fresh200_rank_updates)
        raise
    metrics.update({'fresh200/attempt_id':1,'fresh200/completed_outers':1,
        'fresh200/accepted_complete_outer':1,'fresh200/retained_adam_writes_per_rank':20,
        'fresh200/no_skip_expected_adam_writes_per_rank':4000,
        'fresh200/recovery_missing_original_reward_metrics':True})
    trainer.logger.log(data=metrics,step=1)
    print('BAPO_TRAIN_METRICS_JSON='+json.dumps(dict(step=1,metrics=metrics,attempt_id=1,accepted_complete_outer=True),sort_keys=True,allow_nan=False),flush=True)
    trainer.global_steps=2
    print('BAPO_ALIGNED_PARTIAL_CONTINUATION_JSON='+json.dumps(dict(status='ORIGINAL20_WRITES_PRESERVED_OUTER1_COMMITTED_ONCE',next_outer=2,rank_witnesses=trainer.aligned_partial_rank_witnesses,driver_witness=w,stage=stage),allow_nan=False),flush=True)
    return iterator,d['epoch']

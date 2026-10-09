"""Resume actual3357 COMPLETE5, pause for acceptance, then continue outer6."""
import json,os
from pathlib import Path
from .aligned_complete5_resume import OLD_RUN,OLD_SOURCE,require,exact_file,validate_manifest
from .aligned_partial_driver import write_json


def prepare(trainer,manifest_path):
    import subprocess
    run=Path(os.environ['BAPO_RUN_ID']);path=Path(manifest_path)
    require(path==run/'complete5-resume/manifest.json' and path.resolve()==path,'exact owned COMPLETE5 manifest required')
    manifest=json.loads(path.read_bytes());require(validate_manifest(manifest)==run,'COMPLETE5 target RUN differs')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    require(source==manifest['target_source_commit']==trainer.config.baselines.algorithm_source_commit,'COMPLETE5 target source differs')
    require(not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'dirty continuation source')
    require(os.environ.get('SLURM_JOB_ID')==manifest['target_job_id'],'COMPLETE5 target job differs')
    require(trainer.config.get('baseline_aligned',False) and trainer.config.baselines.method=='bapo' and trainer.config.trainer.val_before_train is False,'COMPLETE5 requires baseline-aligned/no initial evaluation')
    directory=run/'complete5-resume';witness=directory/'rank-witnesses';witness.mkdir(exist_ok=False)
    trainer.aligned_complete5_manifest=manifest;trainer.aligned_complete5_witness_directory=witness
    return manifest,witness


def validate_driver(manifest):
    import math,torch
    from .full_state import state_digest
    from .fresh200_driver import Fresh200Progress
    path=exact_file(manifest['driver'],OLD_RUN);side=exact_file(manifest['driver_report'],OLD_RUN)
    require(path==OLD_RUN/'checkpoints/fresh200_state/outer_5_COMPLETE/driver.pt' and side==path.with_name('driver-reload.json'),'exact original COMPLETE5 driver required')
    report=json.loads(side.read_bytes());d=torch.load(path,map_location='cpu',weights_only=False)
    require(report.get('status')=='EXACT_DRIVER_RELOAD_OK' and report.get('step')==5 and report.get('path')==str(path) and report.get('sha256')==manifest['driver']['sha256'] and report.get('state_digest')==state_digest(d),'original COMPLETE5 driver digest differs')
    require(d.get('status')=='COMPLETE' and d.get('source_commit')==OLD_SOURCE and d.get('protocol')=='BAPO_FRESH200_V1' and type(d.get('driver_pid'))is int and d['driver_pid']>1 and report.get('pid')==d['driver_pid'],'original COMPLETE5 driver identity differs')
    require(all(type(d.get(k))is int and d[k]==v for k,v in dict(global_step=5,next_global_step=6,attempt_id=5,actual_outer_updates=5,attempted_outer=5,previous_completed_outer=4,epoch=0).items()),'original COMPLETE5 counters differ')
    require(d.get('rank_write_ledger')==manifest['rank_write_ledger'] and type(d['sampler'].get('cursor'))is int and d['sampler']['cursor']==2560,'original COMPLETE5 ledger/cursor differs')
    expected=Fresh200Progress(baseline_aligned=True,attempts=5,completed_outers=5,actual_adam_writes_per_rank=[100]*8,recorded_attempts=5)
    require(json.dumps(d.get('progress'),sort_keys=True)==json.dumps(expected.state_dict(),sort_keys=True),'original COMPLETE5 progress differs')
    bapo=d['bapo'];require(type(bapo.get('best'))in(int,float) and math.isfinite(bapo['best']) and 0<=bapo['best']<=1 and type(bapo.get('stalled'))is int and bapo['stalled']>=0 and type(bapo.get('early'))is bool and bapo.get('patience')==5,'original COMPLETE5 BAPO state differs')
    require('rng'in d,'original COMPLETE5 driver RNG absent')
    return d,expected


def restore_driver_and_pause(trainer):
    from .full_state import state_digest,restore_rng,rng_state
    manifest=trainer.aligned_complete5_manifest;run=Path(manifest['target_run_root'])
    d,progress=validate_driver(manifest)
    trainer.e1_sampler.load_state_dict(d['sampler'])
    trainer.baseline_controller.stage.load_state_dict(d['bapo'])
    trainer.baseline_controller.actual_outer_updates=5
    trainer.global_steps=5;trainer.e1_epoch=d['epoch'];trainer.fresh200_progress=progress
    trainer.fresh200_rank_updates=d['rank_write_ledger']
    # Constructing a DataLoader iterator consumes its base seed; restore after it.
    iterator=iter(trainer.train_dataloader)
    restore_rng(d['rng'])
    components={'rng':rng_state(),'sampler':trainer.e1_sampler.state_dict(),'bapo':trainer.baseline_controller.stage.state_dict()}
    for name,value in components.items():require(state_digest(value)==state_digest(d[name]),'COMPLETE5 restored driver component differs: '+name)
    witness=dict(status='EXACT_ALIGNED_COMPLETE5_DRIVER_RELOAD_OK',old_pid=d['driver_pid'],new_pid=os.getpid(),old_run_root=str(OLD_RUN),target_run_root=str(run),old_source_commit=OLD_SOURCE,target_source_commit=manifest['target_source_commit'],target_job_id=manifest['target_job_id'],loaded_state_digest=state_digest(d),step=5,next_global_step=6,completed_outer_before=5,attempt_id=5,sampler_cursor=2560,no_generation_or_optimizer_or_stage_observe_or_scheduler_step=True)
    require(witness['new_pid']!=witness['old_pid'],'COMPLETE5 driver PID unexpectedly unchanged')
    for name,value in components.items():
        witness['saved_'+name+'_digest']=state_digest(d[name]);witness['restored_'+name+'_digest']=state_digest(value)
    write_json(run/'complete5-resume/driver-witness.json',witness)
    write_json(run/'complete5-resume/lineage.json',dict(manifest=manifest,rank_witnesses=trainer.aligned_complete5_rank_witnesses,driver_witness=witness))
    # Export restored weights and prove the current process's exact roundtrip.
    # Neither operation makes an optimizer/scheduler/BAPO transition.
    trainer._save_checkpoint()
    trainer._fresh200_pause_after_five()
    trainer.global_steps=6
    print('BAPO_ALIGNED_COMPLETE5_CONTINUATION_JSON='+json.dumps(dict(status='EXACT_COMPLETE5_RESTORED_AND_INDEPENDENTLY_RELEASED',next_outer=6,driver_witness=witness),allow_nan=False),flush=True)
    return iterator,d['epoch']

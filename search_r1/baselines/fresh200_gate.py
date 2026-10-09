"""Fail-closed fifth-outer pause. This module never produces PASS receipts."""
import hashlib
import json
import os
import secrets
import subprocess
import time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()



def partial_lineage(run, source, members=None, reports=None, driver_pid=None):
    """Only actual3356's frozen pending20 transaction may cross a RUN boundary."""
    from .aligned_partial_resume import validate_manifest, exact_file, OLD_RUN
    run=Path(run).resolve();path=run/'partial-resume/lineage.json'
    if not path.exists():return None
    if not path.is_file() or path.is_symlink():raise ValueError('Ordinary partial lineage required')
    if members is not None and members.get(str(path))!=digest(path):raise ValueError('Partial lineage not frozen')
    value=json.loads(path.read_text());manifest=value['manifest']
    if validate_manifest(manifest)!=run or manifest['target_source_commit']!=source:raise ValueError('Partial target differs')
    closure=manifest.get('original_closure')
    if not isinstance(closure,list) or not closure:raise ValueError('Original partial closure missing')
    originals={}
    for entry in closure:
        original=exact_file(entry,OLD_RUN)
        if str(original)in originals:raise ValueError('Duplicate partial original')
        originals[str(original)]=entry['sha256']
        if members is not None and members.get(str(original))!=entry['sha256']:raise ValueError('Original missing from frozen members')
    config_entry=manifest['original_config'];original_config=exact_file(config_entry,OLD_RUN)
    if originals.get(str(original_config))!=config_entry['sha256']:raise ValueError('Original config not in closure')
    if original_config.suffix in ('.yaml','.yml'):
        from omegaconf import OmegaConf
        old_config=OmegaConf.to_container(OmegaConf.load(original_config),resolve=True)
    else:old_config=json.loads(original_config.read_text())
    if old_config.get('baseline_aligned')is not True or old_config['baselines']['algorithm_source_commit']!=manifest['old_source_commit']:raise ValueError('Original aligned config differs')
    witnesses=value.get('rank_witnesses',[])
    if len(witnesses)!=8 or {w.get('rank')for w in witnesses}!=set(range(8)):raise ValueError('Eight exact restore witnesses required')
    pids=None if reports is None else {r['rank']:r['pid']for r in reports}
    for witness in witnesses:
        rank=witness['rank'];entry=next(r for r in manifest['ranks']if r['rank']==rank)
        for kind in ('state','report'):
            original=exact_file(entry[kind],OLD_RUN)
            if originals.get(str(original))!=entry[kind]['sha256']:raise ValueError('Native partial outside closure')
        report=json.loads(Path(entry['report']['path']).read_text())
        event=next(e for e in manifest['rank_write_ledger']if e['rank']==rank)
        from .aligned_outer_commit import _digest
        pending=dict(actual_outer=1,attempt_id=1,scheduler_before=0,write_ledger_sha256=_digest(manifest['rank_write_ledger']))
        if (type(rank)is not int or witness.get('status')!='EXACT_CROSS_JOB_ALIGNED_PARTIAL_V3_RELOAD_OK'
                or witness.get('old_pid')!=report.get('pid') or type(witness.get('new_pid'))is not int
                or witness['new_pid']<=0 or witness['new_pid']==witness['old_pid']
                or (pids is not None and witness['new_pid']!=pids[rank])
                or report.get('path')!=entry['state']['path'] or report.get('sha256')!=entry['state']['sha256']
                or report.get('latest_rank_event')!=event or report.get('native_state_version')!=3
                or report.get('status')!='PARTIAL_CURRENT_STATE_SAVED' or report.get('scheduler_last_epoch')!=0
                or report.get('aligned_pending_outer')!=pending or witness.get('aligned_pending_outer')!=pending
                or witness.get('original_state_sha256')!=entry['state']['sha256']
                or not isinstance(report.get('state_digest'),str) or len(report['state_digest'])!=64
                or witness.get('state_digest')!=report['state_digest'] or witness.get('reloaded_state_digest')!=report['state_digest']
                or witness.get('generation_rng_digest')!=report.get('generation_rng_before_digest')
                or report.get('generation_rng_before_digest')!=report.get('generation_rng_after_digest')
                or witness.get('no_adam_or_stage_observe_or_scheduler_step')is not True
                or any(type(witness.get(k))is not int or witness[k]!=v for k,v in
                       (('world_size',8),('step',1),('attempt_id',1),('scheduler_last_epoch',0),('adam_steps',20)))):
            raise ValueError('Exact partial rank restore differs')
        for key in ('old_run_root','old_source_commit','target_run_root','target_source_commit','target_job_id'):
            if witness.get(key)!=manifest[key]:raise ValueError('Partial rank lineage differs')
    witness=value.get('driver_witness',{})
    if (witness.get('status')!='EXACT_ALIGNED_PARTIAL_DRIVER_RELOAD_OK'
            or type(witness.get('old_pid'))is not int or type(witness.get('new_pid'))is not int
            or witness['old_pid']<=0 or witness['new_pid']<=0 or witness['old_pid']==witness['new_pid']
            or (driver_pid is not None and witness['new_pid']!=driver_pid)
            or witness.get('no_generation_or_optimizer_or_stage_observe')is not True
            or any(type(witness.get(k))is not int or witness[k]!=v for k,v in
                   (('scheduler_before',0),('completed_outer_before',0),('attempt_id',1),('sampler_cursor',512)))):
        raise ValueError('Exact partial driver restore missing')
    for key in ('old_run_root','old_source_commit','target_run_root','target_source_commit','target_job_id'):
        if witness.get(key)!=manifest[key]:raise ValueError('Partial driver lineage differs')
    for name in ('rng','sampler','bapo'):
        if not isinstance(witness.get('saved_'+name+'_digest'),str) or len(witness['saved_'+name+'_digest'])!=64 or witness.get('restored_'+name+'_digest')!=witness['saved_'+name+'_digest']:raise ValueError('Partial driver reload differs')
    import torch
    from .full_state import state_digest
    driver_path=exact_file(manifest['driver'],OLD_RUN)
    driver_report_path=exact_file(manifest['driver_report'],OLD_RUN)
    for entry in (manifest['driver'],manifest['driver_report']):
        if originals.get(entry['path'])!=entry['sha256']:raise ValueError('Original driver outside closure')
    driver=torch.load(driver_path,map_location='cpu',weights_only=False)
    report=json.loads(driver_report_path.read_text())
    if (report.get('path')!=str(driver_path) or report.get('sha256')!=manifest['driver']['sha256']
            or report.get('status')!='PARTIAL_CURRENT_STATE_SAVED'
            or report.get('state_digest')!=state_digest(driver) or witness.get('loaded_state_digest')!=state_digest(driver)
            or driver.get('driver_pid')!=witness['old_pid'] or driver.get('source_commit')!=manifest['old_source_commit']
            or driver.get('status')!='PARTIAL' or driver.get('rank_write_ledger')!=manifest['rank_write_ledger']
            or any(type(driver.get(k))is not int or driver[k]!=v for k,v in
                   (('global_step',1),('next_global_step',1),('attempt_id',1),('actual_outer_updates',0),('previous_completed_outer',0)))
            or driver.get('sampler',{}).get('cursor')!=512):
        raise ValueError('Original driver payload/restore binding differs')
    for name in ('rng','sampler','bapo'):
        if witness['saved_'+name+'_digest']!=state_digest(driver[name]):raise ValueError('Original driver component differs')
    stage_entry=manifest['stage_metrics'];stage_path=exact_file(stage_entry,run)
    if stage_path!=run/'partial-resume/original-stage.json':raise ValueError('Exact owned original-stage path required')
    if members is not None and members.get(str(stage_path))!=stage_entry['sha256']:raise ValueError('Original stage binding not frozen')
    parsed=json.loads(stage_path.read_text());log=exact_file(parsed['original_log'],OLD_RUN)
    if originals.get(str(log))!=parsed['original_log']['sha256']:raise ValueError('Original stage log outside closure')
    matches=[]
    for line in log.read_text().splitlines():
        marker='BASELINE_EVALUATION_METRICS_JSON='
        if marker not in line:continue
        raw=line.split(marker,1)[1]
        try:metrics=json.loads(raw)
        except json.JSONDecodeError:continue
        counts=[v for k,v in metrics.items()if k.startswith('val/evaluated_count/')]
        if counts and all(type(v)is int and v>=0 for v in counts) and sum(counts)==64:matches.append((raw,metrics))
    if len(matches)!=1 or matches[0][1]!=parsed.get('metrics') or hashlib.sha256(matches[0][0].encode()).hexdigest()!=parsed.get('marker_line_sha256'):raise ValueError('Original complete stage64 log differs')
    from .controller import weighted_validation
    score=weighted_validation(parsed['metrics'])
    if (not __import__('math').isfinite(score) or score!=parsed.get('score') or score!=14/64
            or parsed['metrics'].get('val/qa_count/all')!=64 or parsed.get('observed_in_saved_bapo')is not True
            or driver['bapo']!=dict(best=14/64,stalled=0,early=True,patience=5)):
        raise ValueError('Original observed stage64/BAPO differs')
    return dict(document=value,manifest=manifest,originals=originals,config=old_config,
                old_run=OLD_RUN,old_source=manifest['old_source_commit'],old_driver_pid=witness['old_pid'])

def validate_gate_config(config):
    from .fresh200_contract import validate_fresh_config
    if config['trainer'].get('fresh200_diagnostic_only',False):
        raise ValueError('Diagnostic-only runs cannot release formal fresh200 training')
    if config.get('baseline_aligned') is True and (
            config['trainer'].get('fresh200_gate_enabled')is not True
            or config['trainer'].get('aligned_stage_commit_enabled')is not True
            or config['actor_rollout_ref']['actor'].get('aligned_stage_commit_enabled')is not True
            or config['baselines'].get('alignment',{}).get('training_em_enabled')is not True):
        raise ValueError('Aligned first-five release requires its production gate/stage/EM profile')
    return validate_fresh_config(config)


def complete5_lineage(run,source,members=None,reports=None,driver_pid=None):
    """Bind a new verifier process to the preserved3357 COMPLETE5 transaction."""
    from .aligned_complete5_resume import validate_manifest,exact_file,require,OLD_RUN,OLD_SOURCE
    from .full_state import state_digest
    import torch
    run=Path(run).resolve();path=run/'complete5-resume/lineage.json'
    if not path.exists():return None
    require(path.is_file()and not path.is_symlink()and path.resolve()==path,'ordinary COMPLETE5 lineage required')
    if members is not None:require(members.get(str(path))==digest(path),'COMPLETE5 lineage not frozen')
    document=json.loads(path.read_text());manifest=document['manifest']
    require(validate_manifest(manifest)==run and manifest['target_source_commit']==source,'COMPLETE5 target differs')
    originals={};closure=manifest['original_closure']
    require(isinstance(closure,list)and bool(closure),'COMPLETE5 original closure absent')
    # The nested3356 closure is authorized only by its original3357 lineage.
    old_reports=[]
    for entry in sorted(manifest['ranks'],key=lambda x:x['rank']):
        report=json.loads(exact_file(entry['report'],OLD_RUN).read_text())
        require(report['path']==entry['state']['path']and report['sha256']==entry['state']['sha256'],'COMPLETE5 producer binding differs')
        require(report.get('status')=='EXACT_FULL_STATE_RELOAD_OK'and report.get('native_state_version')==3 and report.get('roundtrip')is True and report.get('state_digest')==report.get('reloaded_state_digest')and report.get('aligned_pending_outer','missing')is None and all(type(report.get(k))is int and report[k]==v for k,v in (('rank',entry['rank']),('world_size',8),('step',5),('scheduler_last_epoch',5)))and report.get('latest_rank_event')==next(e for e in manifest['rank_write_ledger']if e['rank']==entry['rank']),'COMPLETE5 original native report differs')
        old_reports.append(report)
    old_driver_report=json.loads(exact_file(manifest['driver_report'],OLD_RUN).read_text())
    old_driver=torch.load(exact_file(manifest['driver'],OLD_RUN),map_location='cpu',weights_only=False)
    require(old_driver_report.get('status')=='EXACT_DRIVER_RELOAD_OK'and old_driver_report.get('step')==5 and old_driver_report.get('path')==manifest['driver']['path']and old_driver_report.get('sha256')==manifest['driver']['sha256']and old_driver_report['state_digest']==state_digest(old_driver)and old_driver_report['pid']==old_driver['driver_pid'],'COMPLETE5 original driver differs')
    require(old_driver.get('status')=='COMPLETE'and old_driver.get('source_commit')==OLD_SOURCE and old_driver.get('protocol')=='BAPO_FRESH200_V1'and all(type(old_driver.get(k))is int and old_driver[k]==v for k,v in (('global_step',5),('next_global_step',6),('actual_outer_updates',5),('attempt_id',5),('previous_completed_outer',4)))and type(old_driver.get('sampler',{}).get('cursor'))is int and old_driver['sampler']['cursor']==2560,'COMPLETE5 original driver counters/source differ')
    from .fresh200_driver import Fresh200Progress
    expected_progress=Fresh200Progress(baseline_aligned=True,attempts=5,completed_outers=5,actual_adam_writes_per_rank=[100]*8,recorded_attempts=5).state_dict()
    require(json.dumps(old_driver.get('progress'),sort_keys=True,allow_nan=False)==json.dumps(expected_progress,sort_keys=True,allow_nan=False),'COMPLETE5 original driver accounting differs')
    nested=partial_lineage(OLD_RUN,OLD_SOURCE,None,old_reports,old_driver['driver_pid'])
    require(nested is not None,'COMPLETE5 original3356 lineage absent')
    nested_originals=nested['originals']
    for entry in closure:
        path_entry=Path(entry['path'])
        require(path_entry.is_file()and not path_entry.is_symlink()and path_entry.resolve()==path_entry and type(entry['bytes'])is int and path_entry.stat().st_size==entry['bytes'],'COMPLETE5 original file/stat differs')
        require(OLD_RUN in path_entry.parents or nested_originals.get(str(path_entry))==entry['sha256'],'COMPLETE5 original outside authorized chain')
        require(str(path_entry)not in originals,'duplicate COMPLETE5 original')
        originals[str(path_entry)]=entry['sha256']
        if members is not None:require(members.get(str(path_entry))==entry['sha256'],'COMPLETE5 original not frozen')
    for entry in [manifest['driver'],manifest['driver_report'],manifest['original_config']]+[x[k]for x in manifest['ranks']for k in ('state','report')]:
        require(originals.get(entry['path'])==entry['sha256'],'COMPLETE5 native/config outside closure')
    config_path=exact_file(manifest['original_config'],OLD_RUN)
    if config_path.suffix in ('.yaml','.yml'):
        from omegaconf import OmegaConf
        config=OmegaConf.to_container(OmegaConf.load(config_path),resolve=True)
    else:config=json.loads(config_path.read_text())
    require(config.get('baseline_aligned')is True and config['baselines']['algorithm_source_commit']==OLD_SOURCE,'COMPLETE5 original config differs')
    witnesses=document['rank_witnesses'];require(len(witnesses)==8 and all(type(w.get('rank'))is int for w in witnesses)and {w['rank']for w in witnesses}==set(range(8)),'COMPLETE5 eight witnesses absent')
    current_pids=None if reports is None else {r['rank']:r['pid']for r in reports}
    for w in witnesses:
        old=old_reports[w['rank']]
        require(w.get('status')=='EXACT_CROSS_JOB_ALIGNED_COMPLETE5_V3_RELOAD_OK'and w.get('old_pid')==old['pid']and type(w.get('new_pid'))is int and w['new_pid']>0 and w['new_pid']!=w['old_pid'],'COMPLETE5 rank process differs')
        require(w.get('state_digest')==w.get('reloaded_state_digest')==old['state_digest']and w.get('original_state_sha256')==old['sha256'],'COMPLETE5 rank digest differs')
        require('aligned_pending_outer'in w and w['aligned_pending_outer']is None and old.get('aligned_pending_outer','missing')is None,'COMPLETE5 pending transaction remains')
        require(w.get('generation_rng_digest')==old['generation_rng_before_digest']==old['generation_rng_after_digest'],'COMPLETE5 generation RNG differs')
        require(w.get('no_generation_or_optimizer_or_stage_observe_or_scheduler_step')is True,'COMPLETE5 loader performed an update')
        for key,expected in (('world_size',8),('step',5),('attempt_id',5),('scheduler_last_epoch',5),('adam_steps',100)):
            require(type(w.get(key))is int and w[key]==expected,'COMPLETE5 rank counters differ')
        for key in ('old_run_root','old_source_commit','target_run_root','target_source_commit','target_job_id'):require(w.get(key)==manifest[key],'COMPLETE5 rank identity differs')
        if current_pids is not None:
            current=next(r for r in reports if r['rank']==w['rank'])
            require(w['new_pid']==current_pids[w['rank']]and current['state_digest']==old['state_digest']and current['latest_rank_event']==old['latest_rank_event'],'COMPLETE5 current rank differs from saved original')
    witness=document['driver_witness']
    require(witness.get('status')=='EXACT_ALIGNED_COMPLETE5_DRIVER_RELOAD_OK'and witness.get('old_pid')==old_driver['driver_pid']and type(witness.get('new_pid'))is int and witness['new_pid']>0 and witness['new_pid']!=witness['old_pid'],'COMPLETE5 driver process differs')
    if driver_pid is not None:require(witness['new_pid']==driver_pid,'COMPLETE5 current driver differs')
    require(witness.get('loaded_state_digest')==state_digest(old_driver)and witness.get('no_generation_or_optimizer_or_stage_observe_or_scheduler_step')is True,'COMPLETE5 driver reload differs')
    for key,expected in (('step',5),('next_global_step',6),('completed_outer_before',5),('attempt_id',5),('sampler_cursor',2560)):
        require(type(witness.get(key))is int and witness[key]==expected,'COMPLETE5 driver counters differ')
    for key in ('old_run_root','old_source_commit','target_run_root','target_source_commit','target_job_id'):require(witness.get(key)==manifest[key],'COMPLETE5 driver identity differs')
    for name in ('rng','sampler','bapo'):
        require(witness.get('saved_'+name+'_digest')==witness.get('restored_'+name+'_digest')==state_digest(old_driver[name]),'COMPLETE5 restored driver component differs')
    require(old_driver['rank_write_ledger']==manifest['rank_write_ledger'],'COMPLETE5 original latest ledger differs')
    return dict(document=document,manifest=manifest,originals=originals,old_run=OLD_RUN,old_source=OLD_SOURCE,old_driver_pid=old_driver['driver_pid'],config=config,reports=old_reports,driver=old_driver,nested=nested)


def validate_first_five_attempts(attempts, progress, driver_pid, baseline_aligned=False):
    """Replay retained writes and accepted outers; five is not five attempts."""
    from .fresh200_driver import Fresh200Progress
    if type(driver_pid) is not int or driver_pid<=0:raise ValueError('Exact original driver PID required')
    state=Fresh200Progress(baseline_aligned=baseline_aligned)
    def equal_accounting(observed):
        # Python equality considers True == 1 and 20.0 == 20; raw receipt
        # metadata must preserve the exact typed counters produced by replay.
        return json.dumps(observed,sort_keys=True,allow_nan=False)==json.dumps(state.state_dict(),sort_keys=True,allow_nan=False)
    accepted=[]
    for attempt in attempts:
        outer=state.completed_outers+1
        attempt_id=state.begin_attempt(outer)
        if (any(type(attempt.get(key)) is not int for key in ('attempt_id','candidate_outer','driver_pid'))
                or attempt.get('attempt_id')!=attempt_id or attempt.get('candidate_outer')!=outer
                or attempt.get('driver_pid')!=driver_pid):
            raise ValueError('First-five attempt/PID/candidate identity differs')
        complete=state.record(attempt['rank_write_ledger'],outer)
        if (attempt.get('accepted_complete_outer') is not complete
                or attempt.get('status')!=attempt['rank_write_ledger'][0]['status']
                or not equal_accounting(attempt.get('progress'))):
            raise ValueError('First-five raw attempt accounting differs')
        if complete:accepted.append(attempt)
        if state.completed_outers>5:raise ValueError('Fifth gate contains later accepted updates')
    if state.completed_outers!=5 or not equal_accounting(progress):
        raise ValueError('First-five gate needs five accepted complete outers, actual writes bound')
    if baseline_aligned and (state.attempts != 5 or state.incomplete_attempts
            or state.actual_adam_writes_per_rank != [100] * 8
            or state.skipped_opportunities_per_rank != [0] * 8):
        raise ValueError('Aligned fifth gate requires exactly five complete20/no-skip attempts')
    return accepted


def write_manifest(paths, directory, config, source, run, step, reports, progress):
    horizon, warmup_steps = validate_gate_config(config)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    run=Path(run).resolve()
    if run not in directory.resolve().parents:raise ValueError('Checkpoint directory outside exact run')
    config_bytes=json.dumps(config,sort_keys=True,allow_nan=False).encode()
    config_path=directory/'resolved-config.json';config_path.write_bytes(config_bytes)
    lineage=(complete5_lineage(run,source)or partial_lineage(run,source)) if config.get('baseline_aligned')is True else None
    originals={}if lineage is None else lineage['originals']
    files=[]
    for p in sorted(set(Path(p).resolve()for p in paths)|{config_path.resolve()}):
        if not p.is_file():raise ValueError('Missing checkpoint member '+str(p))
        sha=digest(p);size=p.stat().st_size
        original_bound=originals.get(str(p))==sha
        # Empty stderr/stdout/log files are valid original diagnostic evidence.
        # Native checkpoints and structured proof files must still contain data.
        empty_original_diagnostic=original_bound and p.suffix in ('.err','.out','.log')
        if (run not in p.parents and not original_bound) or (not size and not empty_original_diagnostic):
            raise ValueError('Outside/empty checkpoint member '+str(p))
        files.append(dict(path=str(p),sha256=sha,bytes=size))
    if progress.get('completed_outers')!=5:raise ValueError('Gate requires fifth accepted outer')
    value=dict(version=2,run_root=str(Path(run).resolve()),source_commit=source,global_step=step,
               actual_outer_updates=step,horizon=horizon,warmup_steps=warmup_steps,config_sha256=digest(config_path),
               pause_nonce=secrets.token_hex(32),rank_roundtrips=reports,files=files,
               protocol='BAPO_FRESH200_V1', mini_batch_switch_outer=50,
               update_semantics='SKIP_CONTINUE_ACCEPTED_OUTERS_V1',attempt_accounting=progress,
               optimizer_opportunities_before_switch=20,optimizer_opportunities_after_switch=10)
    if config.get('baseline_aligned') is True:
        value.update(version=3, protocol='BAPO_BASELINE_ALIGNED_FRESH200_V1', baseline_aligned=True,
                     mini_batch_switch_outer=None, update_semantics='COMPLETE20_NO_SKIP_ACCEPTED_OUTERS_V1',
                     optimizer_opportunities_after_switch=20)
        if progress.get('baseline_aligned') is not True:
            raise ValueError('Aligned gate progress profile differs')
    target=directory/'manifest.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,target)
    return target


def validate_release(receipt, manifest_path, repo, expected_driver_pid=None):
    manifest_path=Path(manifest_path);m=json.loads(manifest_path.read_text());r=json.loads(Path(receipt).read_text())
    aligned=m.get('baseline_aligned') is True
    expected_status=('PASS_INDEPENDENT_BASELINE_ALIGNED_FRESH200_STEP5_REVIEW' if aligned else 'PASS_INDEPENDENT_FRESH200_STEP5_REVIEW')
    if r.get('status')!=expected_status:raise ValueError('Independent profile-specific FRESH200 step5 release missing')
    if aligned and r.get('baseline_aligned') is not True:
        raise ValueError('Aligned release profile missing')
    for key in ('run_root','source_commit','global_step','actual_outer_updates','horizon','warmup_steps','config_sha256','pause_nonce','protocol','mini_batch_switch_outer','update_semantics','attempt_accounting','optimizer_opportunities_before_switch','optimizer_opportunities_after_switch'):
        if r.get(key)!=m.get(key):raise ValueError('Release identity mismatch '+key)
    if any(type(m.get(k)) is not int or type(r.get(k)) is not int for k in ('global_step','actual_outer_updates','horizon','warmup_steps')):
        raise ValueError('Gate counters must be integers')
    if m['global_step']!=5 or m['actual_outer_updates']!=5 or m['horizon']!=200 or m['warmup_steps']!=57:
        raise ValueError('Not a fifth-outer 200/57 gate')
    if aligned:
        if (m.get('version')!=3 or m.get('update_semantics')!='COMPLETE20_NO_SKIP_ACCEPTED_OUTERS_V1'
                or m.get('protocol')!='BAPO_BASELINE_ALIGNED_FRESH200_V1' or m.get('mini_batch_switch_outer') is not None
                or m.get('optimizer_opportunities_before_switch')!=20 or m.get('optimizer_opportunities_after_switch')!=20):
            raise ValueError('Aligned fixed20 optimizer/profile contract unbound')
    elif (m.get('version')!=2 or m.get('update_semantics')!='SKIP_CONTINUE_ACCEPTED_OUTERS_V1'
            or m.get('protocol')!='BAPO_FRESH200_V1' or m.get('mini_batch_switch_outer')!=50
            or m.get('optimizer_opportunities_before_switch')!=20 or m.get('optimizer_opportunities_after_switch')!=10):
        raise ValueError('Fresh200 variable optimizer counts unbound')
    if r.get('checkpoint_manifest_sha256')!=digest(manifest_path):raise ValueError('Checkpoint manifest changed')
    source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    if source!=m['source_commit'] or dirty:raise ValueError('Current source changed')
    run=Path(m['run_root']).resolve();root=run.parent.parent
    if run not in manifest_path.resolve().parents:raise ValueError('Manifest outside exact run')
    reports=m['rank_roundtrips']
    if (len(reports)!=8 or {x.get('rank')for x in reports}!=set(range(8))
            or any(x.get('status')!='EXACT_FULL_STATE_RELOAD_OK' or x.get('step')!=5
                   or x.get('world_size')!=8 or x.get('scheduler_last_epoch')!=5
                   or x.get('model_state_dict_type')!='SHARDED_STATE_DICT'
                   or type(x.get('model_dtensor_entries')) is not int or x['model_dtensor_entries']<=0
                   or x.get('optimizer_state_entries',0)<=0 for x in reports)):
        raise ValueError('Eight full-state reload reports incomplete')
    members={x['path']:x['sha256']for x in m['files']}
    if len(members)!=len(m['files']):raise ValueError('Duplicate checkpoint members')
    configs=[x for x in m['files']if Path(x['path']).name=='resolved-config.json' and run in Path(x['path']).parents]
    if (len(configs)!=1 or configs[0]['sha256']!=m['config_sha256']
            or digest(configs[0]['path'])!=m['config_sha256']):
        raise ValueError('Config identity differs from actual resolved config')
    resolved_config = json.loads(Path(configs[0]['path']).read_text())
    if (resolved_config.get('baseline_aligned') is True) != aligned:
        raise ValueError('Resolved config and release protocol profiles differ')
    horizon, warmup_steps = validate_gate_config(resolved_config)
    if m['horizon'] != horizon or m['warmup_steps'] != warmup_steps:
        raise ValueError('Manifest counters differ from actual scheduler config')
    native_paths=set()
    for report in reports:
        state_path=Path(report.get('path',''))
        report_path=state_path.with_suffix('.json')
        if (state_path.name!=f"rank_{report['rank']}.pt" or str(state_path) in native_paths
                or members.get(str(state_path))!=report.get('sha256')
                or str(report_path) not in members or digest(report_path)!=members[str(report_path)]
                or json.loads(report_path.read_text())!=report
                or type(report.get('pid')) is not int or report['pid']<=0
                or not isinstance(report.get('state_digest'),str) or len(report['state_digest'])!=64):
            raise ValueError('Native rank state/report identity not bound')
        native_paths.add(str(state_path))
        if aligned and ('aligned_pending_outer' not in report or report['aligned_pending_outer'] is not None):
            raise ValueError('Aligned native state has missing or uncommitted pending outer')
        if aligned and (report.get('native_state_version') != 3 or report.get('roundtrip') is not True
                or report.get('reloaded_state_digest') != report.get('state_digest')
                or not isinstance(report.get('generation_rng_before_digest'), str)
                or len(report['generation_rng_before_digest']) != 64
                or report.get('generation_rng_before_digest') != report.get('generation_rng_after_digest')):
            raise ValueError('Aligned exact native generation/model/Adam/RNG reload missing')
    drivers=[x for x in m['files']if Path(x['path']).name=='driver-reload.json' and run in Path(x['path']).parents]
    driver_states=[x for x in m['files']if Path(x['path']).name=='driver.pt' and run in Path(x['path']).parents]
    if len(drivers)!=1 or len(driver_states)!=1:
        raise ValueError('Unique driver state/reload proof required')
    driver=json.loads(Path(drivers[0]['path']).read_text())
    import torch
    from search_r1.baselines.full_state import state_digest
    driver_state=torch.load(driver_states[0]['path'],map_location='cpu',weights_only=False)
    if (driver.get('status')!='EXACT_DRIVER_RELOAD_OK' or driver.get('step')!=5
            or driver.get('path')!=driver_states[0]['path']
            or driver.get('sha256')!=driver_states[0]['sha256']
            or digest(driver_states[0]['path'])!=driver_states[0]['sha256']
            or driver.get('state_digest')!=state_digest(driver_state)
            or type(driver.get('pid')) is not int or driver['pid']<=0
            or driver.get('pid')!=driver_state.get('driver_pid')
            or (expected_driver_pid is not None and
                (type(expected_driver_pid) is not int or expected_driver_pid<=0
                 or driver.get('pid')!=expected_driver_pid))
            or driver_state.get('global_step')!=5 or driver_state.get('next_global_step')!=6
            or driver_state.get('actual_outer_updates')!=5
            or driver_state.get('progress')!=m.get('attempt_accounting')):
        raise ValueError('Actual driver sampler/RNG/BAPO reload not proven')
    complete_lineage=complete5_lineage(run,source,members,reports,driver['pid']) if aligned else None
    lineage=(complete_lineage or partial_lineage(run,source,members,reports,driver['pid'])) if aligned else None
    originals={}if lineage is None else lineage['originals']
    if complete_lineage is not None:
        if any(state_digest(driver_state[name])!=state_digest(complete_lineage['driver'][name])for name in ('rng','sampler','bapo','progress')):
            raise ValueError('COMPLETE5 current driver differs from preserved components')
    sampling_proof=(None if aligned else validate_first_sampling_evidence(members,resolved_config,source,run,driver['pid'],reports))
    evidence=r.get('evidence',[]);bindings={x['path']:x['sha256']for x in evidence}
    if len(bindings)!=len(evidence):raise ValueError('Duplicate release evidence')
    if str(run/'source-commit.txt') not in bindings:raise ValueError('Run source not directly bound')
    if str(run/'train.log') in bindings:raise ValueError('Use immutable step5 prefix, not mutable whole train.log')
    prefix=r.get('train_log_prefix',{})
    snapshot=Path(prefix.get('snapshot_path',''))
    length=prefix.get('bytes')
    if (prefix.get('live_path')!=str(run/'train.log') or type(length) is not int or length<=0
            or not snapshot.is_absolute() or root not in snapshot.resolve().parents
            or not snapshot.is_file() or snapshot.stat().st_size!=length
            or digest(snapshot)!=prefix.get('sha256') or bindings.get(str(snapshot))!=prefix.get('sha256')):
        raise ValueError('Immutable step5 log snapshot missing/unbound')
    with (run/'train.log').open('rb') as handle:
        h=hashlib.sha256();remaining=length
        while remaining:
            block=handle.read(min(1048576,remaining))
            if not block:raise ValueError('Actual live log shorter than signed prefix')
            h.update(block);remaining-=len(block)
    if h.hexdigest()!=prefix['sha256']:raise ValueError('Actual live prefix differs from snapshot')
    events=[];attempts=[]
    timeline=snapshot
    timeline_run=run;timeline_source=source;timeline_pid=driver['pid']
    if complete_lineage is not None:
        timeline=complete_lineage['old_run']/'train.log'
        if members.get(str(timeline))!=digest(timeline):raise ValueError('COMPLETE5 original timeline missing/unbound')
        timeline_run=complete_lineage['old_run'];timeline_source=complete_lineage['old_source'];timeline_pid=complete_lineage['old_driver_pid']
    for line in timeline.read_text().split('\n'):
        if 'BAPO_TRAIN_METRICS_JSON=' in line:
            events.append(json.loads(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1]))
        if 'FRESH200_ATTEMPT_JSON=' in line:
            attempts.append(json.loads(line.split('FRESH200_ATTEMPT_JSON=',1)[1]))
    accepted=validate_first_five_attempts(attempts,m['attempt_accounting'],timeline_pid,baseline_aligned=aligned)
    if aligned:
        if complete_lineage is None:
            validate_aligned_first_five_evidence(members,resolved_config,source,run,driver['pid'],reports,attempts,events)
        else:
            validate_complete5_first_five(members,resolved_config,source,run,driver['pid'],complete_lineage,attempts,events)
        if driver_state.get('rank_write_ledger')!=attempts[-1]['rank_write_ledger']:
            raise ValueError('Aligned driver latest rank ledger differs from actual fifth attempt')
        final_ranks={event['rank']:event for event in attempts[-1]['rank_write_ledger']}
        for report in reports:
            if report.get('latest_rank_event')!=final_ranks[report['rank']]:
                raise ValueError('Aligned native reload latest rank event differs')
    for attempt in attempts:
        path=Path(attempt.get('ledger_path',''))
        if (attempt.get('source_commit')!=timeline_source
                or not path.is_absolute() or timeline_run not in path.resolve().parents or str(path) not in members
                or digest(path)!=members[str(path)] or json.loads(path.read_text())!=attempt):
            raise ValueError('Raw per-attempt all-rank ledger missing or unbound')
    complete_events=[event for event in events if event.get('accepted_complete_outer') is True]
    if [x.get('step')for x in complete_events]!=[1,2,3,4,5]:raise ValueError('Prefix must contain exact five accepted outer summaries')
    if len(events)!=len(attempts):raise ValueError('Each attempt must have one metrics summary')
    for event,attempt in zip(events,attempts):
        if (event.get('step')!=attempt['candidate_outer'] or event.get('attempt_id')!=attempt['attempt_id']
                or event.get('accepted_complete_outer') is not attempt['accepted_complete_outer']
                or event.get('metrics',{}).get('actor/optimizer_steps')!=attempt['rank_write_ledger'][0]['optimizer_steps']):
            raise ValueError('Actual metrics differ from rank write ledger')
    if not aligned and not any(a['attempt_id']==sampling_proof['attempt_id'] and a['candidate_outer']==sampling_proof['actual_outer']
               and a['accepted_complete_outer'] is True for a in attempts):
        raise ValueError('Verified raw sampling proof must belong to an accepted actual attempt')
    if any(x.get('metrics',{}).get('bapo/actual_outer_updates')!=x['step']for x in complete_events):
        raise ValueError('Signed prefix actualouter mismatch')
    if (run/'source-commit.txt').read_text().strip()!=m['source_commit']:
        raise ValueError('Run-start source marker changed')
    for item in m['files']:
        p=Path(item['path'])
        if ((run not in p.resolve().parents and originals.get(str(p))!=item['sha256']) or digest(p)!=item['sha256'] or bindings.get(str(p))!=item['sha256']):
            raise ValueError('Checkpoint member changed/unbound '+str(p))
    if bindings.get(str(manifest_path.resolve()))!=digest(manifest_path):raise ValueError('Manifest not directly bound')
    review=r.get('independent_review',{})
    for item in evidence+[review]:
        p=Path(item.get('path',''))
        if not p.is_absolute() or (root not in p.resolve().parents and originals.get(str(p))!=item.get('sha256')) or not p.is_file() or digest(p)!=item.get('sha256'):
            raise ValueError('Missing/changed owned evidence')
    return r


def wait_for_release(receipt, manifest_path, repo, interval=5):
    print('FRESH200_STEP5_PAUSED_JSON='+json.dumps(dict(manifest_path=str(manifest_path),manifest_sha256=digest(manifest_path),pid=os.getpid())),flush=True)
    while not Path(receipt).exists():time.sleep(interval)
    # An invalid present receipt is a hard failure, never a silent release.
    result=validate_release(receipt,manifest_path,repo,expected_driver_pid=os.getpid())
    print('FRESH200_STEP5_RELEASED_JSON='+json.dumps(dict(receipt=str(receipt),receipt_sha256=digest(receipt),pid=os.getpid())),flush=True)
    return result


def validate_first_sampling_evidence(members, config, source, run, driver_pid, rank_reports=None):
    """Replay actual transported probabilities, bind raw native initial evidence."""
    import torch
    from verl import DataProto
    from .fresh200_contract import verify_sampling_policy_binding, critical_source_hashes
    def member(name):
        found=[Path(p) for p in members if Path(p).name==name]
        if len(found)!=1 or digest(found[0])!=members[str(found[0])]:
            raise ValueError('Unique immutable first sampling/initial evidence missing: '+name)
        return found[0]
    path=member('formal-first-sampling-batch.pt');report=json.loads(member('formal-first-sampling-batch.json').read_text())
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (report.get('path')!=str(path) or report.get('sha256')!=members[str(path)]
            or report.get('sampling_verifier_status')!='PASS_ACTUAL_BATCH_BINDING_NOT_RELEASE'
            or payload.get('source_commit')!=source or payload.get('source_sha256')!=critical_source_hashes()
            or payload.get('driver_pid')!=driver_pid or payload.get('run_root')!=str(run)
            or type(payload.get('attempt_id')) is not int or payload['attempt_id']<1
            or type(payload.get('actual_outer')) is not int or not 1<=payload['actual_outer']<=5):
        raise ValueError('First formal raw sampling identity differs')
    for key in ('version','status','source_commit','source_sha256','run_root','driver_pid','attempt_id','actual_outer','sampling_binding_sha256','max_selected_logprob_abs_diff','selected_tokens','sampling_logprob_atol','probability_contract','hf_native_drift_is_identity_gate','behavior_policy_role'):
        if json.dumps(report.get(key),sort_keys=True,allow_nan=False)!=json.dumps(payload.get(key),sort_keys=True,allow_nan=False):
            raise ValueError('First raw proof/report typed identity differs: '+key)
    if (payload.get('probability_contract')!='ACTUAL_NATIVE_BEHAVIOR_V2_HF_DIAGNOSTIC_SEPARATE'
            or payload.get('hf_native_drift_is_identity_gate') is not False
            or payload.get('behavior_policy_role')!='actual_native_stochastic_sampling_policy'):
        raise ValueError('First sampling actual behavior contract missing')
    data=DataProto.from_dict(tensors=payload['tensors'],non_tensors=payload['non_tensor_batch'],meta_info=payload['meta_info'])
    if (type(data.meta_info.get('attempt_id')) is not int or data.meta_info['attempt_id']!=payload['attempt_id']
            or type(data.meta_info.get('actual_outer')) is not int or data.meta_info['actual_outer']!=payload['actual_outer']
            or data.meta_info.get('fresh200_shared_input_bindings')!=payload.get('shared_input_bindings')):
        raise ValueError('First batch actual attempt/shared inputs differ')
    binding=verify_sampling_policy_binding(data,config['actor_rollout_ref']['actor']['policy_safety'])
    mask=data.batch['sampling_mask'].bool()
    gap=float((data.batch['frozen_hf_log_probs'][mask].double()-data.batch['sampling_log_probs'][mask].double()).abs().max())
    if (binding!=payload.get('sampling_binding_sha256') or binding!=report.get('sampling_binding_sha256')
            or gap!=payload.get('max_selected_logprob_abs_diff') or gap!=report.get('max_selected_logprob_abs_diff')
            or int(mask.sum())!=report.get('selected_tokens')):
        raise ValueError('First raw sampler numerical replay differs')
    alignment=json.loads(member('native-initial-official-alignment.json').read_text())
    official=member('native-initial-official-hf.pt')
    if alignment.get('path')!=str(official) or alignment.get('sha256')!=members[str(official)]:
        raise ValueError('Actual official initial alignment is not directly bound')
    checkpoint_pids={r['rank']:r['pid'] for r in (rank_reports or [])}
    for rank in range(8):
        state=member(f'native-initial-rank_{rank}.pt');initial=json.loads(member(f'native-initial-rank_{rank}.json').read_text())
        if (initial.get('path')!=str(state) or initial.get('sha256')!=members[str(state)]
                or initial.get('official_alignment')!=alignment or initial.get('source_sha256')!=payload['source_sha256']
                or type(initial.get('optimizer_state_entries')) is not int or initial['optimizer_state_entries']!=0
                or type(initial.get('scheduler_last_epoch')) is not int or initial['scheduler_last_epoch']!=0
                or initial.get('lr')!=[1e-6/57] or initial.get('status')!='ACTUAL_INITIAL_STATE_CAPTURED_NOT_RELEASE'
                or type(initial.get('native_process',{}).get('rank')) is not int
                or initial.get('native_process',{}).get('rank')!=rank
                or (rank_reports is not None and (type(initial.get('native_process',{}).get('pid')) is not int
                    or initial['native_process']['pid']!=checkpoint_pids.get(rank)))):
            raise ValueError('Eight rank actual fresh initial evidence is not directly bound')
    return report


def write_aligned_step5_bindings(directory, config, source, run, driver_pid, shared_inputs):
    """Persist factual aligned input identity; never an approval/PASS receipt."""
    from .fresh200_contract import critical_source_hashes
    run=Path(run).resolve();directory=Path(directory).resolve()
    if run not in directory.parents or type(driver_pid) is not int or driver_pid<=0:
        raise ValueError('Aligned binding needs exact original driver/run')
    complete=complete5_lineage(run,source)
    lineage=complete or partial_lineage(run,source)
    claim=run/'.aligned-startup-claim.json'
    claim_value=json.loads(claim.read_text())
    if (claim_value.get('source_commit')!=source or claim_value.get('scope')!=('same-experiment-complete5-verify-only-resume' if complete is not None else ('new-run-only-no-resume' if lineage is None else 'same-experiment-pending20-stage-only-resume'))
            or type(claim_value.get('launcher_pid')) is not int or claim_value['launcher_pid']<=0):
        raise ValueError('Aligned original new-run claim differs')
    inputs={}
    for name,settings,path_key,sha_key in (
            ('shared_input_manifest',config['baselines']['holdout'],'manifest_path','manifest_sha256'),
            ('retriever_identity_manifest',config['retriever'],'identity_manifest','identity_manifest_sha256'),
            ('order_manifest',config['baselines']['alignment'],'order_manifest','order_sha256')):
        path=Path(settings[path_key]).resolve()
        if digest(path)!=settings[sha_key]:raise ValueError('Aligned pinned input changed: '+name)
        inputs[name]=dict(path=str(path),sha256=settings[sha_key])
    value=dict(version=1,status='ALIGNED_STEP5_FACTUAL_BINDINGS_NOT_RELEASE',
        source_commit=source,source_sha256=critical_source_hashes(),run_root=str(run),driver_pid=driver_pid,
        actual_outer_updates=5,next_global_step=6,
        new_run_claim=dict(path=str(claim),sha256=digest(claim)),inputs=inputs,shared_inputs=shared_inputs)
    target=directory/'aligned-step5-bindings.json'
    with target.open('x') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
    return target


def validate_aligned_first_five_evidence(members,config,source,run,driver_pid,reports,attempts,events,*,expected_source_hashes=None):
    """Bind aligned HF-old/data/initial-state/EM evidence, without legacy LP admission."""
    from .fresh200_contract import critical_source_hashes, fresh200_lr_factor
    import math
    lineage=partial_lineage(run,source,members,reports,driver_pid)
    old_run=run if lineage is None else lineage['old_run']
    old_source=source if lineage is None else lineage['old_source']
    old_driver=driver_pid if lineage is None else lineage['old_driver_pid']
    old_config=config if lineage is None else lineage['config']
    def member(name):
        found=[Path(path)for path in members if Path(path).name==name and
               (old_run if lineage is not None and name.startswith('native-initial-') else run) in Path(path).parents]
        if len(found)!=1 or digest(found[0])!=members[str(found[0])]:
            raise ValueError('Unique aligned first-five evidence missing: '+name)
        return found[0]
    value=json.loads(member('aligned-step5-bindings.json').read_text())
    if (value.get('version')!=1 or value.get('status')!='ALIGNED_STEP5_FACTUAL_BINDINGS_NOT_RELEASE'
            or value.get('source_commit')!=source or value.get('source_sha256')!=(critical_source_hashes()if expected_source_hashes is None else expected_source_hashes)
            or value.get('run_root')!=str(run) or value.get('driver_pid')!=driver_pid
            or type(value.get('driver_pid'))is not int or type(value.get('actual_outer_updates'))is not int
            or type(value.get('next_global_step'))is not int
            or value.get('actual_outer_updates')!=5 or value.get('next_global_step')!=6):
        raise ValueError('Aligned factual driver/source/run evidence differs')
    claim=member('.aligned-startup-claim.json');record=json.loads(claim.read_text())
    if (value.get('new_run_claim')!=dict(path=str(claim),sha256=members[str(claim)])
            or record.get('source_commit')!=source or record.get('scope')!=('new-run-only-no-resume' if lineage is None else 'same-experiment-pending20-stage-only-resume')
            or type(record.get('launcher_pid'))is not int or record['launcher_pid']<=0):
        raise ValueError('Aligned new-run source claim is unbound')
    from .baseline_alignment import prepare_aligned_inputs
    from omegaconf import OmegaConf
    if prepare_aligned_inputs(OmegaConf.create(config))!=value.get('shared_inputs'):
        raise ValueError('Aligned actual shared inputs/stage64 differ')
    old_shared=value['shared_inputs'] if lineage is None else prepare_aligned_inputs(OmegaConf.create(old_config))
    if lineage is not None:
        before=old_shared.get('common_input_binding',{});after=value['shared_inputs'].get('common_input_binding',{})
        if (before.get('content_identity')!=after.get('content_identity') or before.get('train_rows')!=after.get('train_rows')
                or old_shared.get('content_identity')!=value['shared_inputs'].get('content_identity')
                or old_config['baselines']['alignment']['order_sha256']!=config['baselines']['alignment']['order_sha256']):
            raise ValueError('Recovered experiment common input/order identity changed')
    for name,settings,path_key,sha_key in (
            ('shared_input_manifest',config['baselines']['holdout'],'manifest_path','manifest_sha256'),
            ('retriever_identity_manifest',config['retriever'],'identity_manifest','identity_manifest_sha256'),
            ('order_manifest',config['baselines']['alignment'],'order_manifest','order_sha256')):
        expected=dict(path=str(Path(settings[path_key]).resolve()),sha256=settings[sha_key])
        if value.get('inputs',{}).get(name)!=expected or digest(expected['path'])!=expected['sha256']:
            raise ValueError('Aligned pinned input manifest changed: '+name)
    alignment=json.loads(member('native-initial-official-alignment.json').read_text())
    official=member('native-initial-official-hf.pt')
    if (alignment.get('path')!=str(official) or alignment.get('sha256')!=members[str(official)]
            or Path(alignment.get('official_model_root','')).resolve()!=Path(config['actor_rollout_ref']['model']['path']).resolve()):
        raise ValueError('Aligned initial actor/reference official identity differs')
    pids=({report['rank']:report['pid']for report in reports} if lineage is None else
          {w['rank']:w['old_pid']for w in lineage['document']['rank_witnesses']})
    initial_source_hashes=value['source_sha256'] if lineage is None else json.loads(member('native-initial-rank_0.json').read_text())['source_sha256']
    from .fresh200_native_evidence import verify_native_process
    for rank in range(8):
        state=member(f'native-initial-rank_{rank}.pt')
        initial=json.loads(member(f'native-initial-rank_{rank}.json').read_text())
        if (initial.get('path')!=str(state) or initial.get('sha256')!=members[str(state)]
                or initial.get('official_alignment')!=alignment or initial.get('source_sha256')!=initial_source_hashes
                or initial.get('official_model_manifest_sha256')!='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
                or type(initial.get('optimizer_state_entries'))is not int or initial['optimizer_state_entries']!=0
                or type(initial.get('scheduler_last_epoch'))is not int or initial['scheduler_last_epoch']!=0
                or initial.get('lr')!=[1e-6/57] or initial.get('status')!='ACTUAL_INITIAL_STATE_CAPTURED_NOT_RELEASE'
                or type(initial.get('native_process',{}).get('rank'))is not int
                or type(initial.get('native_process',{}).get('world_size'))is not int
                or type(initial.get('native_process',{}).get('pid'))is not int
                or initial.get('native_process',{}).get('rank')!=rank
                or initial.get('native_process',{}).get('world_size')!=8
                or initial.get('native_process',{}).get('pid')!=pids[rank]):
            raise ValueError('Aligned eight-rank fresh initial state is unbound')
        verify_native_process(initial['native_process'],old_source)
        if initial['native_process'].get('run_root')!=str(old_run):
            raise ValueError('Aligned native initial process belongs to another run')
    # The audit verifier replays exact authored terminal answers against targets
    # and checks source/run/snapshot/attempt identities; it never grants release.
    from .training_em import verify_training_em_audit
    for attempt,event in zip(attempts,events):
        outer=attempt['candidate_outer'];ordinal=attempt['attempt_id']
        def owned_record(scope):
            origin=old_run if lineage is not None and outer==ordinal==1 and scope=='aligned-outer-prepared' else run
            path=origin/scope/f'outer_{outer:04d}-attempt_{ordinal:04d}.json'
            if str(path)not in members or digest(path)!=members[str(path)]:
                raise ValueError('Aligned immutable stage/prepare proof missing: '+scope)
            return json.loads(path.read_text())
        stage=owned_record('aligned-stage-proof')
        stage_body={key:item for key,item in stage.items()if key!='sha256'}
        if lineage is not None and outer==ordinal==1 and (stage.get('recovery_original_stage')!=lineage['manifest']['stage_metrics'] or stage.get('no_stage_observe_repeated')is not True or stage.get('score')!=14/64):
            raise ValueError('Recovered stage must bind original observed64 without repeating it')
        from .aligned_outer_commit import _digest
        if (stage.get('sha256')!=_digest(stage_body)
                or type(stage.get('actual_outer'))is not int or type(stage.get('attempt_id'))is not int
                or type(stage.get('driver_pid'))is not int
                or stage.get('actual_outer')!=outer or stage.get('attempt_id')!=ordinal
                or type(stage.get('evaluated_count'))is not int or stage['evaluated_count']!=64
                or type(stage.get('score'))not in (int,float) or not math.isfinite(stage['score'])
                or not 0.<=stage['score']<=1.
                or stage.get('score')!=event['metrics'].get('bapo/stage_validation')
                or stage.get('run_root')!=str(run) or stage.get('driver_pid')!=driver_pid
                or stage.get('source_commit')!=source or stage.get('source_sha256')!=value['source_sha256']
                or stage.get('common_input_binding')!=value['shared_inputs'].get('common_input_binding')
                or stage.get('stage_input_binding')!=value['shared_inputs'].get('stage')
                or stage.get('order_manifest')!=config['baselines']['alignment']['order_manifest']
                or stage.get('order_manifest_sha256')!=config['baselines']['alignment']['order_sha256']
                or event['metrics'].get('actor/scheduler_outer_steps')!=outer
                or event['metrics'].get('bapo/stage_evaluated_count')!=64
                or event['metrics'].get('fresh200/stage_proof_sha256')!=stage['sha256']):
            raise ValueError('Aligned stage64/source/input/scheduler proof differs')
        prepared=owned_record('aligned-outer-prepared')
        recovered=lineage is not None and outer==ordinal==1
        if (prepared.get('status')!='WRITES_PREPARED_NOT_ACCEPTED_OUTER'
                or prepared.get('actual_outer')!=outer or prepared.get('attempt_id')!=ordinal
                or prepared.get('driver_pid')!=(old_driver if recovered else driver_pid) or prepared.get('source_commit')!=(old_source if recovered else source)
                or prepared.get('previous_completed_outer')!=outer-1):
            raise ValueError('Aligned prepared write identity differs')
        provisional={item.get('rank'):item for item in prepared.get('rank_write_ledger',[])}
        if len(provisional)!=8 or set(provisional)!=set(range(8)):
            raise ValueError('Aligned prepared rank coverage differs')
        em_run=old_run if recovered else run
        path=em_run/'training-em-audit'/f'outer_{outer:04d}-attempt_{ordinal:04d}.jsonl'
        if members.get(str(path))!=digest(path):raise ValueError('Unique aligned first-five evidence missing: original EM journal')
        expected=em_run/'training-em-audit'/path.name
        em_config=old_config if recovered else config
        if path!=expected:raise ValueError('Training EM audit outside original run path')
        binding=dict(data_manifest_path=em_config['baselines']['holdout']['manifest_path'],
            data_manifest_sha256=em_config['baselines']['holdout']['manifest_sha256'],
            retrieval_manifest_path=em_config['retriever']['identity_manifest'],
            retrieval_manifest_sha256=em_config['retriever']['identity_manifest_sha256'],
            order_manifest_path=em_config['baselines']['alignment']['order_manifest'],
            order_manifest_sha256=em_config['baselines']['alignment']['order_sha256'],
            common_input_binding=(old_shared if recovered else value['shared_inputs']).get('common_input_binding'))
        verified=verify_training_em_audit(path,members[str(path)],expected=dict(
            source_commit=old_source if recovered else source,run_root=str(em_run),driver_pid=old_driver if recovered else driver_pid,actual_outer=outer,attempt_id=ordinal,
            expected_questions=512,siblings=5,bindings=binding,
            policy_snapshot_id=f'{em_run}:attempt_{ordinal}:before_outer_{outer}'))
        if any(event['metrics'].get(key)!=number for key,number in verified['metrics'].items()):
            raise ValueError('Training EM audit aggregate differs from actual attempt metrics')
        for rank_event in attempt['rank_write_ledger']:
            rank=rank_event['rank']
            prior=provisional[rank]
            transitions={'outer_commit_phase','accepted_complete_outer','stage_validation',
                         'scheduler_before','scheduler_after','scheduler_lr_before','scheduler_lr_after'}
            if (rank_event.get('outer_commit_phase')!='STAGE_COMMITTED'
                    or rank_event.get('accepted_complete_outer')is not True
                    or rank_event.get('stage_validation')!=stage
                    or rank_event.get('scheduler_before')!=outer-1 or rank_event.get('scheduler_after')!=outer
                    or prior.get('outer_commit_phase')!='WRITES_PREPARED'
                    or prior.get('accepted_complete_outer')is not False
                    or prior.get('scheduler_before')!=outer-1 or prior.get('scheduler_after')!=outer-1
                    or any(rank_event.get(key)!=item for key,item in prior.items()if key not in transitions)):
                raise ValueError('Aligned stage-before-scheduler/write commit differs')
            for key,completed in (('scheduler_lr_before',outer-1),('scheduler_lr_after',outer)):
                values=rank_event.get(key)
                expected_lr=1e-6*fresh200_lr_factor(completed)
                if (not isinstance(values,list) or not values or any(type(lr)not in (int,float)
                        or not math.isfinite(lr) or not math.isclose(lr,expected_lr,rel_tol=1e-14,abs_tol=0.)
                        for lr in values)):
                    raise ValueError('Aligned exact warmup57 LR commit differs')
            if recovered:
                found=[Path(p)for p in lineage['originals']if Path(p).name==f'rank_{rank}-outer_{outer}-attempt_{ordinal}.prepared.json']
                if len(found)!=1:raise ValueError('Unique recovered prepared rank event missing')
                prior_path=found[0]
            else:prior_path=member(f'rank_{rank}-outer_{outer}-attempt_{ordinal}.prepared.json')
            if json.loads(prior_path.read_text())!=prior:
                raise ValueError('Aligned prepared rank event differs from actual driver record')
            rank_path=member(f'rank_{rank}-outer_{outer}-attempt_{ordinal}.json')
            if json.loads(rank_path.read_text())!=rank_event:
                raise ValueError('Aligned rank diagnostic event differs from driver ledger')
            monitors=rank_event.get('policy_monitor',[])
            if (len(monitors)!=2 or [item.get('stage')for item in monitors]!=['before_adam','after_adam']
                    or any(item.get('denominator_role')!='frozen_hf_old_policy'
                        or item.get('diagnostic_only')is not True
                        or item.get('monitoring_scope')!='full_frozen_rollout'
                        or item.get('sampling_policy_kl_status')!='unavailable_on_retokenized_training_actions'
                        or item.get('raw_sampler_diagnostic',{}).get('diagnostic_only')is not True
                        or type(item.get('raw_sampler_diagnostic',{}).get('available'))is not bool
                        for item in monitors)):
                raise ValueError('Aligned HF-old and raw-sampler diagnostic roles/evidence missing')
    return value


def validate_complete5_first_five(members,config,source,run,driver_pid,lineage,attempts,events):
    """Validate old factual five rounds and the new process's unchanged inputs."""
    from .aligned_complete5_resume import require
    from .fresh200_contract import critical_source_hashes
    from .baseline_alignment import prepare_aligned_inputs
    from omegaconf import OmegaConf
    old_run=lineage['old_run'];old_source=lineage['old_source']
    old_binding_paths=[Path(p)for p in members if Path(p).name=='aligned-step5-bindings.json'and old_run in Path(p).parents]
    require(len(old_binding_paths)==1,'original COMPLETE5 factual binding absent')
    old_binding=json.loads(old_binding_paths[0].read_text())
    old_repo=old_run.parent.parent/'repo'
    require(subprocess.check_output(['git','-C',str(old_repo),'rev-parse','HEAD'],text=True).strip()==old_source and not subprocess.check_output(['git','-C',str(old_repo),'status','--porcelain'],text=True).strip(),'original COMPLETE5 source changed/dirty')
    old_hashes=old_binding['source_sha256']
    import ast
    contract=ast.parse((old_repo/'search_r1/baselines/fresh200_contract.py').read_text())
    function=next(n for n in contract.body if isinstance(n,ast.FunctionDef)and n.name=='critical_source_hashes')
    names=next(ast.literal_eval(n.value)for n in function.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='names'for t in n.targets))
    require(isinstance(old_hashes,dict)and set(old_hashes)==set(names),'original COMPLETE5 critical source inventory differs')
    for name,sha in old_hashes.items():
        path=old_repo/name
        require(not Path(name).is_absolute()and '..'not in Path(name).parts and path.is_file()and not path.is_symlink()and old_repo.resolve()in path.resolve().parents and digest(path)==sha,'original COMPLETE5 critical source bytes differ')
    validate_aligned_first_five_evidence(members,lineage['config'],old_source,old_run,lineage['old_driver_pid'],lineage['reports'],attempts,events,expected_source_hashes=old_hashes)
    paths=[Path(p)for p in members if Path(p).name=='aligned-step5-bindings.json'and run in Path(p).parents]
    require(len(paths)==1,'current COMPLETE5 factual binding absent')
    value=json.loads(paths[0].read_text());claim=run/'.aligned-startup-claim.json'
    require(members.get(str(claim))==digest(claim),'current COMPLETE5 claim not frozen')
    record=json.loads(claim.read_text())
    require(record.get('scope')=='same-experiment-complete5-verify-only-resume'and record.get('source_commit')==source and type(record.get('launcher_pid'))is int and record['launcher_pid']>0,'current COMPLETE5 claim differs')
    require(value.get('version')==1 and value.get('status')=='ALIGNED_STEP5_FACTUAL_BINDINGS_NOT_RELEASE'and value.get('source_commit')==source and value.get('source_sha256')==critical_source_hashes()and value.get('run_root')==str(run)and type(value.get('driver_pid'))is int and value['driver_pid']==driver_pid and type(value.get('actual_outer_updates'))is int and value['actual_outer_updates']==5 and type(value.get('next_global_step'))is int and value['next_global_step']==6 and value.get('new_run_claim')==dict(path=str(claim),sha256=members[str(claim)]),'current COMPLETE5 factual identity differs')
    current_shared=prepare_aligned_inputs(OmegaConf.create(config));old_shared=prepare_aligned_inputs(OmegaConf.create(lineage['config']))
    require(value.get('shared_inputs')==current_shared,'current COMPLETE5 shared input proof differs')
    require(current_shared.get('content_identity')==old_shared.get('content_identity')and current_shared.get('common_input_binding',{}).get('content_identity')==old_shared.get('common_input_binding',{}).get('content_identity')and current_shared.get('common_input_binding',{}).get('train_rows')==old_shared.get('common_input_binding',{}).get('train_rows')and config['baselines']['alignment']['order_sha256']==lineage['config']['baselines']['alignment']['order_sha256'],'COMPLETE5 continuation input/order identity changed')
    for name,settings,path_key,sha_key in (
            ('shared_input_manifest',config['baselines']['holdout'],'manifest_path','manifest_sha256'),
            ('retriever_identity_manifest',config['retriever'],'identity_manifest','identity_manifest_sha256'),
            ('order_manifest',config['baselines']['alignment'],'order_manifest','order_sha256')):
        expected=dict(path=str(Path(settings[path_key]).resolve()),sha256=settings[sha_key])
        require(value.get('inputs',{}).get(name)==expected and digest(expected['path'])==expected['sha256'],'current COMPLETE5 pinned input changed')
